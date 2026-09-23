# hbond_xyz_wcpp.py
import numpy as np
import os
import math
import re 
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from . import _core as hbond_core

def _app_minimum_image(r1, r2, xlo, xhi, ylo, yhi, zlo, zhi, px, py, pz):
    box = np.array([xhi - xlo, yhi - ylo, zhi - zlo])
    delta = r2 - r1
    for i, periodic in enumerate([px, py, pz]):
        if periodic:
            while delta[i] >  box[i] / 2.0: delta[i] -= box[i]
            while delta[i] < -box[i] / 2.0: delta[i] += box[i]
    return r1 + delta

def _process_frame_atom_unified(frame_str, dh_map, acceptor_names, fallback_box, fallback_pbc, file_format, use_dynamic):
    lines = frame_str.strip().split('\n')
    if len(lines) < 2: return [], fallback_box, fallback_pbc, {} 
        
    lines_filtered = []
    current_box = fallback_box
    current_pbc = fallback_pbc
    has_mol_column = False
    
    # Extração de listas de átomos permitidos a partir do dh_map
    donor_names = set(dh_map.keys())
    hydrogen_names = set(h for h_list in dh_map.values() for h in h_list)
    allowed_atoms = set(acceptor_names) | donor_names | hydrogen_names
    
    if file_format == 'lammpstrj':
        try:
            start_idx = 0
            for idx, l in enumerate(lines):
                if use_dynamic and l.startswith("ITEM: BOX BOUNDS"):
                    parts = l.split()[3:]
                    if len(parts) >= 3:
                        current_pbc = tuple(('p' in p) for p in parts)
                    try:
                        x_line = lines[idx+1].split()
                        y_line = lines[idx+2].split()
                        z_line = lines[idx+3].split()
                        current_box = (float(x_line[0]), float(x_line[1]), 
                                       float(y_line[0]), float(y_line[1]), 
                                       float(z_line[0]), float(z_line[1]))
                    except:
                        pass 
                elif l.startswith("ITEM: ATOMS"):
                    header_parts = l.split()
                    if 'mol' in header_parts or 'molecule' in header_parts:
                        has_mol_column = True
                    start_idx = idx + 1
                    break
            
            for line in lines[start_idx:]:
                parts = line.split()
                if len(parts) >= 4:
                    if has_mol_column and len(parts) >= 6:
                        atom_type = parts[2] if not parts[1].replace('.','',1).isdigit() else parts[1] 
                        mol_id = parts[1]
                        atom_type = parts[2]
                        x, y, z = parts[3], parts[4], parts[5]
                    else:
                        mol_id = None
                        atom_type = parts[1] 
                        x, y, z = parts[2], parts[3], parts[4]

                    if atom_type in allowed_atoms:
                        lines_filtered.append([atom_type, x, y, z, mol_id])
        except Exception as e:
            print(f"Info: Erro ao ler lammpstrj. Pulando frame... ({e})")
            return [], current_box, current_pbc, {} 

    else: # xyz / extxyz
        try:
            if use_dynamic:
                line2 = lines[1]
                match_lattice = re.search(r'Lattice="([^"]+)"', line2)
                if match_lattice:
                    v = [float(x) for x in match_lattice.group(1).split()]
                    if len(v) >= 9:
                        current_box = (0.0, v[0], 0.0, v[4], 0.0, v[8])
                
                match_pbc = re.search(r'pbc="([^"]+)"', line2)
                if match_pbc:
                    p_flags = match_pbc.group(1).split()
                    if len(p_flags) >= 3:
                        current_pbc = tuple(p.upper() == 'T' for p in p_flags)

            for line in lines[2:]:
                parts = line.split()
                if not parts: continue
                mol_id = None
                for p in parts:
                    if p.startswith("mol="):
                        mol_id = p.split("=")[1]
                        has_mol_column = True
                
                atom_type = parts[0]
                coords = []
                for p in parts[1:]:
                    if '=' not in p:
                        try:
                            coords.append(float(p))
                        except ValueError:
                            pass
                if len(coords) >= 3:
                    if atom_type in allowed_atoms:
                        lines_filtered.append([atom_type, str(coords[0]), str(coords[1]), str(coords[2]), mol_id])
        except Exception as e:
            print(f"Info: Erro ao ler xyz. Pulando frame... ({e})")
            return [], current_box, current_pbc, {}

    results = []
    mol_types = {} 
    auto_mol_id = 0
    i = 0
    (lx_min, lx_max, ly_min, ly_max, lz_min, lz_max) = current_box
    (px, py, pz) = current_pbc

    while i < len(lines_filtered):
        line = lines_filtered[i]
        atom_name = line[0]
        explicit_mol = line[4]

        if explicit_mol is not None:
            current_mol_id = int(explicit_mol)
        else:
            if atom_name in dh_map.keys() or atom_name in acceptor_names:
                auto_mol_id += 1
            current_mol_id = auto_mol_id

        is_donor = atom_name in dh_map.keys()
        is_acceptor = atom_name in acceptor_names

        if is_donor or is_acceptor:
            mol_types[current_mol_id] = atom_name

        if is_donor and is_acceptor:
            coords = {"x": float(line[1]), "y": float(line[2]), "z": float(line[3])}
            results.append({"ID": current_mol_id, "type": 'acceptor', **coords})
            results.append({"ID": current_mol_id, "type": 'donor', **coords})
            i += 1 
            
            h_count = 0
            allowed_hydrogens = dh_map[atom_name] 
            
            while i < len(lines_filtered):
                next_line = lines_filtered[i]
                if next_line[0] in allowed_hydrogens:
                    if explicit_mol is not None and next_line[4] != explicit_mol:
                        break
                    h_count += 1
                    results.append({
                        "ID": current_mol_id, "type": f"H{h_count}",
                        "x": float(next_line[1]), "y": float(next_line[2]), "z": float(next_line[3])
                    })
                    i += 1
                else:
                    break
        else:
            if is_acceptor:
                 results.append({"ID": current_mol_id, "type": 'acceptor', "x": float(line[1]), "y": float(line[2]), "z": float(line[3])})
            elif is_donor:
                 results.append({"ID": current_mol_id, "type": 'donor', "x": float(line[1]), "y": float(line[2]), "z": float(line[3])})
            i += 1
    
    donors_map = {atom["ID"]: np.array([atom["x"], atom["y"], atom["z"]]) for atom in results if atom["type"] == "donor"}

    for h_atom in results:
        if h_atom["type"].startswith("H"):
            h_id = h_atom["ID"] 
            if h_id in donors_map:
                donor_coords = donors_map[h_id]
                h_coords = np.array([h_atom["x"], h_atom["y"], h_atom["z"]])
                dist = np.linalg.norm(h_coords - donor_coords)
                if dist > 1.5: 
                    h_fixed = _app_minimum_image(donor_coords, h_coords, lx_min, lx_max, ly_min, ly_max, lz_min, lz_max, px, py, pz)
                    h_atom["x"], h_atom["y"], h_atom["z"] = h_fixed[0], h_fixed[1], h_fixed[2]
                            
    return results, current_box, current_pbc, mol_types

def _group_single_frame(frame_data):
    if not frame_data: return []
    frame_data.sort(key=lambda x: x['ID'])
    packs = []
    current_id = -1
    current_pack = []
    for atom in frame_data:
        if atom['ID'] != current_id:
            if current_pack: packs.append(current_pack)
            current_pack = [atom]
            current_id = atom['ID']
        else:
            current_pack.append(atom)
    if current_pack: packs.append(current_pack)
    return packs

# =============================================================================
# WORKERS PARALELOS (TEXTO E GROMACS)
# =============================================================================

def _worker_pipeline_text(args):
    frame_str, dh_map, acceptor, box_base, pbc_base, set_len, set_angle, file_format, use_dynamic = args
    
    atom_data, frame_box, frame_pbc, mol_types = _process_frame_atom_unified(
        frame_str, dh_map, acceptor, box_base, pbc_base, file_format, use_dynamic
    )
    
    n_donors = sum(1 for a in atom_data if a['type'] == 'donor')
    n_acceptors = sum(1 for a in atom_data if a['type'] == 'acceptor')
    
    packs_frame = _group_single_frame(atom_data)
    
    donors_list, acceptors_list, hydrogens_list = [], [], []
    for mol_pack in packs_frame:
        mol_id = mol_pack[0]["ID"]
        for atom in mol_pack:
            if atom["type"] == 'donor':
                donors_list.append([mol_id, atom["x"], atom["y"], atom["z"]])
            elif atom["type"] == 'acceptor':
                acceptors_list.append([mol_id, atom["x"], atom["y"], atom["z"]])
            elif atom["type"].startswith("H"):
                hydrogens_list.append([mol_id, atom["x"], atom["y"], atom["z"]])
                
    d_array = np.array(donors_list, dtype=np.float64) if donors_list else np.empty((0, 4))
    a_array = np.array(acceptors_list, dtype=np.float64) if acceptors_list else np.empty((0, 4))
    h_array = np.array(hydrogens_list, dtype=np.float64) if hydrogens_list else np.empty((0, 4))
    
    ids, don, acc, p_d, p_a = hbond_core.calculate_frame_hbonds(
        d_array, a_array, h_array, list(frame_box), list(frame_pbc), set_len, set_angle
    )
    
    mapped_pairs = [(int(d_id), int(a_id)) for d_id, a_id in zip(p_d, p_a)]
    
    d_types = [mol_types[d] for d, a in mapped_pairs]
    a_types = [mol_types[a] for d, a in mapped_pairs]
    
    coords_dict = {int(a[0]): (float(a[1]), float(a[2]), float(a[3])) for a in acceptors_list}
    
    return mapped_pairs, d_types, a_types, n_donors, n_acceptors, coords_dict, frame_box, frame_pbc

def _worker_pipeline_gmx(args):
    d_data, a_data, h_data, f_box, f_pbc, set_len, set_angle, n_donors, n_acceptors, d_types_map, a_types_map = args
    
    ids, don, acc, p_d, p_a = hbond_core.calculate_frame_hbonds(
        d_data, a_data, h_data, list(f_box), list(f_pbc), set_len, set_angle
    )
    
    mapped_pairs = [(int(d_id), int(a_id)) for d_id, a_id in zip(p_d, p_a)]
    
    d_types = [d_types_map[d] for d, a in mapped_pairs]
    a_types = [a_types_map[a] for d, a in mapped_pairs]
    
    coords_dict = {int(a_data[i, 0]): (float(a_data[i, 1]), float(a_data[i, 2]), float(a_data[i, 3])) for i in range(a_data.shape[0])}
    
    return mapped_pairs, d_types, a_types, n_donors, n_acceptors, coords_dict, f_box, f_pbc

# =============================================================================
# FASE 1: O CONSTRUTOR (Calcula e Salva no Parquet)
# =============================================================================
class HBondAnalysis:
    def __init__(self, filename, topology_file=None, file_format=None,
                 donor_hydrogen=None, acceptor=None, 
                 box_limits=None, pbc=None,
                 set_len=3.5, set_angle=30, n_cpus=None, output_prefix="hbond_data"):   
        
        if not os.path.exists(filename): 
            raise FileNotFoundError(f"Arquivo de trajetória não encontrado: {filename}")
        
        self.filename = filename
        self.topology_file = topology_file 
        
        if donor_hydrogen is None or not isinstance(donor_hydrogen, list):
            raise ValueError("Você deve fornecer 'donor_hydrogen' como uma lista de tuplas. Ex: [('OW', ['HW1', 'HW2']), ('oh', 'ho')]")
            
        self.dh_map = {}
        for d_name, h_names in donor_hydrogen:
            if d_name not in self.dh_map:
                self.dh_map[d_name] = []
            if isinstance(h_names, (list, tuple)):
                self.dh_map[d_name].extend(h_names)
            else:
                self.dh_map[d_name].append(h_names)
                
        self.acceptor = (acceptor,) if isinstance(acceptor, str) else tuple(acceptor)
        self.set_len = set_len
        self.set_angle = set_angle
        self.n_cpus = n_cpus if n_cpus is not None else max(1, os.cpu_count() - 1)
        self.output_prefix = output_prefix

        self.file_format = file_format
        if not self.file_format:
            if filename.endswith('.lammpstrj'): self.file_format = 'lammpstrj'
            elif filename.endswith('.xtc') or filename.endswith('.trr'): self.file_format = 'gromacs'
            else: self.file_format = 'xyz'

        if self.file_format == 'gromacs' and self.topology_file is None:
            raise ValueError("Erro: Para ler arquivos do GROMACS (.xtc), você DEVE fornecer um 'topology_file' (ex: arquivo .top).")
        
        if self.topology_file and not os.path.exists(self.topology_file):
            raise FileNotFoundError(f"Arquivo de topologia não encontrado: {self.topology_file}")
            
        parsed_box, parsed_pbc = self._parse_header()
        self.pbc = pbc if pbc is not None else (parsed_pbc if parsed_pbc is not None else (False, False, False))
        self.use_dynamic = (box_limits is None)
        
        if box_limits is not None:
            self.box_limits = box_limits
            print(f"Info: Usando caixa FIXA definida pelo usuário -> Box: {self.box_limits} | PBC: {self.pbc}")
        elif parsed_box is not None:
            self.box_limits = parsed_box
            print(f"Info: Configuração base extraída do arquivo -> Box: {self.box_limits} | PBC: {self.pbc}")
            print(f"Info: Leitura DINÂMICA de caixa ativada para os próximos frames.")
        else:
            print("WARNING: Arquivo simples sem info de Lattice/Box e 'box_limits' não foi fornecido.")
            self.box_limits = self._auto_detect_box()
            print(f"Info: Leitura DINÂMICA de caixa ativada.")

        # --- A MÁGICA DO GROMACS: Lê o .top e desenrola a topologia inteira aqui ---
        if self.file_format == 'gromacs':
            print(f"Info: Processando o arquivo {self.topology_file} para mapeamento GROMACS...")
            self._parse_top_topology()

    def _parse_top_topology(self):
        mol_templates = {}
        system_mols = []
        
        with open(self.topology_file, 'r') as f:
            lines = f.readlines()
            
        state = None
        current_mol = None
        
        for line in lines:
            line = line.split(';')[0].strip() # Remove comentários
            if not line: continue
            
            if line.startswith('[') and line.endswith(']'):
                state = line[1:-1].strip()
                continue
                
            if state == 'moleculetype':
                parts = line.split()
                if parts:
                    current_mol = parts[0]
                    mol_templates[current_mol] = []
                    state = 'wait_atoms'
            elif state == 'atoms':
                parts = line.split()
                # Coluna 2 (parts[1]) é o 'type' do campo de força (ex: OW, HW, ca)
                if len(parts) >= 5:
                    mol_templates[current_mol].append(parts[1])
            elif state == 'molecules':
                parts = line.split()
                if len(parts) >= 2:
                    system_mols.append((parts[0], int(parts[1])))

        allowed_donors = set(self.dh_map.keys())
        allowed_acceptors = set(self.acceptor)
        
        d_idx, d_mols, d_types_map = [], [], {}
        a_idx, a_mols, a_types_map = [], [], {}
        h_idx, h_mols = [], []
        
        global_atom_idx = 0
        global_sub_id = 0 
        
        for mol_name, count in system_mols:
            template = mol_templates[mol_name]
            for _ in range(count):
                t_idx = 0
                while t_idx < len(template):
                    atom_type = template[t_idx]
                    is_d = atom_type in allowed_donors
                    is_a = atom_type in allowed_acceptors
                    
                    if is_d or is_a:
                        global_sub_id += 1 
                        if is_d:
                            d_idx.append(global_atom_idx + t_idx)
                            d_mols.append(global_sub_id)
                            d_types_map[global_sub_id] = atom_type
                            
                            allowed_h = self.dh_map[atom_type]
                            lookahead = 1
                            while t_idx + lookahead < len(template):
                                next_type = template[t_idx + lookahead]
                                if next_type in allowed_h:
                                    h_idx.append(global_atom_idx + t_idx + lookahead)
                                    h_mols.append(global_sub_id)
                                    lookahead += 1
                                else:
                                    break
                        if is_a:
                            a_idx.append(global_atom_idx + t_idx)
                            a_mols.append(global_sub_id)
                            a_types_map[global_sub_id] = atom_type
                            
                    t_idx += 1  # <--- CORREÇÃO: Um Tab para a direita! (Dentro do while)
                    
                # global_atom_idx fica alinhado com o 't_idx = 0'
                global_atom_idx += len(template)

        self.gmx_d_idx = np.array(d_idx, dtype=np.int32)
        self.gmx_d_mols = np.array(d_mols, dtype=np.int32)
        self.gmx_a_idx = np.array(a_idx, dtype=np.int32)
        self.gmx_a_mols = np.array(a_mols, dtype=np.int32)
        self.gmx_h_idx = np.array(h_idx, dtype=np.int32)
        self.gmx_h_mols = np.array(h_mols, dtype=np.int32)
        self.gmx_d_types_map = d_types_map
        self.gmx_a_types_map = a_types_map
        self.gmx_n_donors = len(d_idx)
        self.gmx_n_acceptors = len(a_idx)

    def _parse_header(self):
        # ... Mantido igual para XYZ e LAMMPS ...
        box, pbc = None, None
        if self.file_format == 'gromacs':
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (True, True, True)

        with open(self.filename, 'r') as f:
            if self.file_format == 'lammpstrj':
                for _ in range(100):
                    line = f.readline()
                    if not line: break
                    if line.startswith("ITEM: BOX BOUNDS"):
                        parts = line.split()[3:]
                        pbc = tuple(('p' in p) for p in parts) if len(parts) >= 3 else (True, True, True)
                        x_line = f.readline().split()
                        y_line = f.readline().split()
                        z_line = f.readline().split()
                        box = (float(x_line[0]), float(x_line[1]), float(y_line[0]), float(y_line[1]), float(z_line[0]), float(z_line[1]))
                        break
            else: 
                f.readline() 
                line2 = f.readline()
                if not line2: return None, None
                match_lattice = re.search(r'Lattice="([^"]+)"', line2)
                if match_lattice:
                    v = [float(x) for x in match_lattice.group(1).split()]
                    if len(v) >= 9:
                        box = (0.0, v[0], 0.0, v[4], 0.0, v[8])
                match_pbc = re.search(r'pbc="([^"]+)"', line2)
                if match_pbc:
                    p_flags = match_pbc.group(1).split()
                    pbc = tuple(p.upper() == 'T' for p in p_flags) if len(p_flags) >= 3 else None
        return box, pbc

    def _auto_detect_box(self):
        min_x, max_x, min_y, max_y, min_z, max_z = math.inf, -math.inf, math.inf, -math.inf, math.inf, -math.inf
        try:
            with open(self.filename, 'r') as f:
                while True:
                    line1 = f.readline()
                    if not line1: break 
                    try: num_atoms = int(line1.strip())
                    except ValueError: continue
                    f.readline()
                    for _ in range(num_atoms):
                        parts = f.readline().split()
                        if len(parts) >= 4:
                            try:
                                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                min_x, max_x = min(min_x, x), max(max_x, x)
                                min_y, max_y = min(min_y, y), max(max_y, y)
                                min_z, max_z = min(min_z, z), max(max_z, z)
                            except: continue
            return (min_x, max_x, min_y, max_y, min_z, max_z)
        except: return (0,0,0,0,0,0)

    def _frame_generator_text(self):
        with open(self.filename, 'r') as f:
            if self.file_format == 'lammpstrj':
                frame_lines = []
                for line in f:
                    if line.startswith("ITEM: TIMESTEP"):
                        if frame_lines:
                            yield "".join(frame_lines)
                            frame_lines = []
                    frame_lines.append(line)
                if frame_lines:
                    yield "".join(frame_lines)
            else:
                while True:
                    line_header = f.readline()
                    if not line_header: break
                    try: num_atoms = int(line_header.strip())
                    except ValueError: continue
                    frame_lines = [line_header, f.readline()]
                    for _ in range(num_atoms):
                        frame_lines.append(f.readline())
                    yield "".join(frame_lines)

    def _arg_generator_text(self):
        for frame_str in self._frame_generator_text():
            yield (frame_str, self.dh_map, self.acceptor, 
                   self.box_limits, self.pbc, self.set_len, self.set_angle, self.file_format, self.use_dynamic)

    def _arg_generator_gmx(self):
        # Aciona o motor C++ com XDRFILE!
        reader = hbond_core.XTCReader(self.filename)
        while True:
            frame_data = reader.read_next_frame(
                self.gmx_d_idx, self.gmx_d_mols,
                self.gmx_a_idx, self.gmx_a_mols,
                self.gmx_h_idx, self.gmx_h_mols
            )
            if frame_data is None: # Fim do XTC
                break
            
            f_box, f_pbc, d_data, a_data, h_data = frame_data
            
            yield (d_data, a_data, h_data, f_box, f_pbc, 
                   self.set_len, self.set_angle, 
                   self.gmx_n_donors, self.gmx_n_acceptors,
                   self.gmx_d_types_map, self.gmx_a_types_map)

    def run(self):
        print(f"Iniciando construção do Banco de Dados Parquet (C++ + {self.n_cpus} CPUs)...")
        
        schema_hb = pa.schema([
            ('frame', pa.int32()), 
            ('donor_id', pa.int32()), ('donor_type', pa.string()),
            ('acceptor_id', pa.int32()), ('acceptor_type', pa.string()),
            ('acceptor_x', pa.float32()), ('acceptor_y', pa.float32()), ('acceptor_z', pa.float32())
        ])
        
        schema_stats = pa.schema([
            ('frame', pa.int32()), ('n_donors_sys', pa.int32()), 
            ('n_acceptors_sys', pa.int32()), ('total_hbonds', pa.int32()),
            ('box_xmin', pa.float32()), ('box_xmax', pa.float32()),
            ('box_ymin', pa.float32()), ('box_ymax', pa.float32()),
            ('box_zmin', pa.float32()), ('box_zmax', pa.float32()),
            ('pbc_x', pa.bool_()), ('pbc_y', pa.bool_()), ('pbc_z', pa.bool_())
        ])

        writer_hb = pq.ParquetWriter(f"{self.output_prefix}_hbonds.parquet", schema_hb)
        writer_stats = pq.ParquetWriter(f"{self.output_prefix}_stats.parquet", schema_stats)

        # SELETOR DE WORKER
        if self.file_format == 'gromacs':
            generator = self._arg_generator_gmx()
            worker_func = _worker_pipeline_gmx
        else:
            generator = self._arg_generator_text()
            worker_func = _worker_pipeline_text

        frame_count = 0

        with ProcessPoolExecutor(max_workers=self.n_cpus) as executor:
            for frame_idx, (mapped_pairs, d_types, a_types, n_donors, n_acceptors, coords_dict, f_box, f_pbc) in enumerate(executor.map(worker_func, generator, chunksize=5)):
                d_ids, a_ids, a_xs, a_ys, a_zs = [], [], [], [], []
                for d, a in mapped_pairs:
                    d_ids.append(d)
                    a_ids.append(a)
                    cx, cy, cz = coords_dict[a]
                    a_xs.append(cx)
                    a_ys.append(cy)
                    a_zs.append(cz)

                if d_ids:
                    batch_hb = pa.RecordBatch.from_arrays([
                        pa.array([frame_idx]*len(d_ids), type=pa.int32()),
                        pa.array(d_ids, type=pa.int32()), pa.array(d_types, type=pa.string()),
                        pa.array(a_ids, type=pa.int32()), pa.array(a_types, type=pa.string()),
                        pa.array(a_xs, type=pa.float32()), pa.array(a_ys, type=pa.float32()), pa.array(a_zs, type=pa.float32())
                    ], schema=schema_hb)
                    writer_hb.write_batch(batch_hb)

                batch_stats = pa.RecordBatch.from_arrays([
                    pa.array([frame_idx], type=pa.int32()), pa.array([n_donors], type=pa.int32()),
                    pa.array([n_acceptors], type=pa.int32()), pa.array([len(d_ids)], type=pa.int32()),
                    pa.array([f_box[0]], type=pa.float32()), pa.array([f_box[1]], type=pa.float32()),
                    pa.array([f_box[2]], type=pa.float32()), pa.array([f_box[3]], type=pa.float32()),
                    pa.array([f_box[4]], type=pa.float32()), pa.array([f_box[5]], type=pa.float32()),
                    pa.array([f_pbc[0]], type=pa.bool_()), pa.array([f_pbc[1]], type=pa.bool_()), pa.array([f_pbc[2]], type=pa.bool_())
                ], schema=schema_stats)
                writer_stats.write_batch(batch_stats)
                
                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"Frames salvos no disco: {frame_count}...")
        
        writer_hb.close()
        writer_stats.close()
        print(f"Geração concluída! Dados salvos em '{self.output_prefix}_hbonds.parquet' e '{self.output_prefix}_stats.parquet'")


# =============================================================================
# FASE 2: O EXPLORADOR (Leitura em Pandas para Filtros Rápidos)
# =============================================================================
class HBondExplorer:
    def __init__(self, prefix="hbond_data"):
        self.file_hbonds = f"{prefix}_hbonds.parquet"
        self.file_stats = f"{prefix}_stats.parquet"
        
        if not os.path.exists(self.file_stats):
            raise FileNotFoundError(f"Bancos de dados não encontrados para o prefixo '{prefix}'. Rode HBondAnalysis.run() primeiro.")
            
        self.df_stats = pd.read_parquet(self.file_stats)

    def get_system_averages(self):
        avg_hbonds = self.df_stats['total_hbonds'].mean()
        avg_donors = self.df_stats['n_donors_sys'].mean()
        return avg_hbonds / avg_donors if avg_donors > 0 else 0

    def get_hbonds_by_frame(self, frame_idx):
        return pd.read_parquet(self.file_hbonds, filters=[('frame', '==', frame_idx)])

    def get_frame_box(self, frame_idx):
        frame_data = self.df_stats[self.df_stats['frame'] == frame_idx].iloc[0]
        return frame_data.to_dict()

    def filter_volume(self, frame_idx, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None):
        df = self.get_hbonds_by_frame(frame_idx)
        
        if x_min is not None: df = df[df['acceptor_x'] >= x_min]
        if x_max is not None: df = df[df['acceptor_x'] <= x_max]
        
        if y_min is not None: df = df[df['acceptor_y'] >= y_min]
        if y_max is not None: df = df[df['acceptor_y'] <= y_max]
        
        if z_min is not None: df = df[df['acceptor_z'] >= z_min]
        if z_max is not None: df = df[df['acceptor_z'] <= z_max]
        
        return df

    def filter_sphere(self, frame_idx, radius, center_x=None, center_y=None, center_z=None):
        f_box = self.get_frame_box(frame_idx)
        
        cg_x = (f_box['box_xmin'] + f_box['box_xmax']) / 2.0
        cg_y = (f_box['box_ymin'] + f_box['box_ymax']) / 2.0
        cg_z = (f_box['box_zmin'] + f_box['box_zmax']) / 2.0
        
        center_x = center_x if center_x is not None else cg_x
        center_y = center_y if center_y is not None else cg_y
        center_z = center_z if center_z is not None else cg_z
        
        df = self.get_hbonds_by_frame(frame_idx)
        dist_sq = (df['acceptor_x'] - center_x)**2 + (df['acceptor_y'] - center_y)**2 + (df['acceptor_z'] - center_z)**2
        return df[dist_sq <= radius**2]
