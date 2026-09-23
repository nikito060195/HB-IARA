#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "xdrfile_xtc.h" // A Biblioteca Independente!
#include <cmath>
#include <vector>
#include <unordered_map>

namespace py = pybind11;

inline double apply_pbc(double delta, double box_length, bool pbc_flag) {
    if (pbc_flag) {
        while (delta >  box_length / 2.0) delta -= box_length;
        while (delta < -box_length / 2.0) delta += box_length;
    }
    return delta;
}

py::tuple calculate_frame_hbonds(
    py::array_t<double> donors,      
    py::array_t<double> acceptors,   
    py::array_t<double> hydrogens,   
    std::vector<double> box,         
    std::vector<bool> pbc,           
    double set_len,
    double set_angle
) {
    auto d_data = donors.unchecked<2>();
    auto a_data = acceptors.unchecked<2>();
    auto h_data = hydrogens.unchecked<2>();

    double box_x = box[1] - box[0];
    double box_y = box[3] - box[2];
    double box_z = box[5] - box[4];

    double set_len_sq = set_len * set_len;
    double cos_set_angle = std::cos(set_angle * M_PI / 180.0);

    std::unordered_map<int, std::pair<int, int>> counts;
    std::vector<std::pair<int, int>> hbond_pairs;
    
    for (py::ssize_t i = 0; i < d_data.shape(0); i++) {
        counts[static_cast<int>(d_data(i, 0))] = {0, 0};
    }
    for (py::ssize_t i = 0; i < a_data.shape(0); i++) {
        int a_id = static_cast<int>(a_data(i, 0));
        if (counts.find(a_id) == counts.end()) {
            counts[a_id] = {0, 0};
        }
    }

    for (py::ssize_t i = 0; i < d_data.shape(0); i++) {
        int d_id = static_cast<int>(d_data(i, 0));
        double dx = d_data(i, 1), dy = d_data(i, 2), dz = d_data(i, 3);

        for (py::ssize_t j = 0; j < a_data.shape(0); j++) {
            int a_id = static_cast<int>(a_data(j, 0));
            if (d_id == a_id) continue;

            double diff_x = apply_pbc(a_data(j, 1) - dx, box_x, pbc[0]);
            double diff_y = apply_pbc(a_data(j, 2) - dy, box_y, pbc[1]);
            double diff_z = apply_pbc(a_data(j, 3) - dz, box_z, pbc[2]);
            
            double dist_sq = diff_x*diff_x + diff_y*diff_y + diff_z*diff_z;
            
            if (dist_sq <= set_len_sq) {
                for (py::ssize_t k = 0; k < h_data.shape(0); k++) {
                    if (static_cast<int>(h_data(k, 0)) == d_id) {
                        double hx = h_data(k, 1), hy = h_data(k, 2), hz = h_data(k, 3);
                        
                        double ba_x = apply_pbc(hx - dx, box_x, pbc[0]);
                        double ba_y = apply_pbc(hy - dy, box_y, pbc[1]);
                        double ba_z = apply_pbc(hz - dz, box_z, pbc[2]);
                        
                        double dot_product = ba_x*diff_x + ba_y*diff_y + ba_z*diff_z;
                        double norm_ba_sq = ba_x*ba_x + ba_y*ba_y + ba_z*ba_z;
                        
                        if (norm_ba_sq > 0 && dist_sq > 0) {
                            double cos_theta = dot_product / std::sqrt(norm_ba_sq * dist_sq);
                            if (cos_theta >= cos_set_angle) {
                                counts[d_id].first += 1;
                                counts[a_id].second += 1;
                                hbond_pairs.push_back({d_id, a_id});
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<int> out_ids, out_don, out_acc;
    out_ids.reserve(counts.size());
    out_don.reserve(counts.size());
    out_acc.reserve(counts.size());
    
    for (auto const& pair : counts) {
        out_ids.push_back(pair.first);
        out_don.push_back(pair.second.first);
        out_acc.push_back(pair.second.second);
    }

    std::vector<int> out_pair_d, out_pair_a;
    out_pair_d.reserve(hbond_pairs.size());
    out_pair_a.reserve(hbond_pairs.size());

    for (auto const& p : hbond_pairs) {
        out_pair_d.push_back(p.first);
        out_pair_a.push_back(p.second);
    }

    return py::make_tuple(
        py::cast(out_ids), py::cast(out_don), py::cast(out_acc),
        py::cast(out_pair_d), py::cast(out_pair_a)
    );
}

// =============================================================================
// O Leitor Autônomo XTC (Sem Depender do GROMACS Instalado!)
// =============================================================================
class XTCReader {
private:
    XDRFILE *xd; // Ponteiro nativo da xdrfile
    int natoms;
    int step;
    float time;
    matrix box;
    rvec *x = nullptr;
    float prec;

public:
    XTCReader(std::string filename) {
        // Lê o número de átomos do XTC primeiro
        if (read_xtc_natoms(const_cast<char*>(filename.c_str()), &natoms) != exdrOK) {
            throw std::runtime_error("Erro ao ler o numero de atomos do arquivo XTC.");
        }
        
        // Aloca a memória exata necessária
        x = (rvec*)malloc(natoms * sizeof(rvec));
        
        // Abre o arquivo de forma independente
        xd = xdrfile_open(filename.c_str(), "r");
        if (!xd) {
            throw std::runtime_error("Nao foi possivel abrir o arquivo XTC.");
        }
    }

    ~XTCReader() {
        if (xd) xdrfile_close(xd);
        if (x) free(x);
    }

    py::object read_next_frame(
        py::array_t<int> d_idx_arr, py::array_t<int> d_mol_arr,
        py::array_t<int> a_idx_arr, py::array_t<int> a_mol_arr,
        py::array_t<int> h_idx_arr, py::array_t<int> h_mol_arr
    ) {
        // Tenta ler o frame. Se retornar erro (ex: chegou no final), encerra.
        if (read_xtc(xd, natoms, &step, &time, box, x, &prec) != exdrOK) {
            return py::none();
        }

        auto d_idx = d_idx_arr.unchecked<1>();
        auto d_mol = d_mol_arr.unchecked<1>();
        auto a_idx = a_idx_arr.unchecked<1>();
        auto a_mol = a_mol_arr.unchecked<1>();
        auto h_idx = h_idx_arr.unchecked<1>();
        auto h_mol = h_mol_arr.unchecked<1>();
        
        // 1. Matriz de Doadores (* 10.0 converte GROMACS nm -> Angstroms)
        py::array_t<double> d_data({d_idx.shape(0), (py::ssize_t)4});
        auto d_ptr = d_data.mutable_unchecked<2>();
        for(py::ssize_t i = 0; i < d_idx.shape(0); ++i) {
            d_ptr(i, 0) = d_mol(i);
            d_ptr(i, 1) = x[d_idx(i)][0] * 10.0;
            d_ptr(i, 2) = x[d_idx(i)][1] * 10.0;
            d_ptr(i, 3) = x[d_idx(i)][2] * 10.0;
        }

        // 2. Matriz de Aceitadores
        py::array_t<double> a_data({a_idx.shape(0), (py::ssize_t)4});
        auto a_ptr = a_data.mutable_unchecked<2>();
        for(py::ssize_t i = 0; i < a_idx.shape(0); ++i) {
            a_ptr(i, 0) = a_mol(i);
            a_ptr(i, 1) = x[a_idx(i)][0] * 10.0;
            a_ptr(i, 2) = x[a_idx(i)][1] * 10.0;
            a_ptr(i, 3) = x[a_idx(i)][2] * 10.0;
        }

        // 3. Matriz de Hidrogênios
        py::array_t<double> h_data({h_idx.shape(0), (py::ssize_t)4});
        auto h_ptr = h_data.mutable_unchecked<2>();
        for(py::ssize_t i = 0; i < h_idx.shape(0); ++i) {
            h_ptr(i, 0) = h_mol(i);
            h_ptr(i, 1) = x[h_idx(i)][0] * 10.0;
            h_ptr(i, 2) = x[h_idx(i)][1] * 10.0;
            h_ptr(i, 3) = x[h_idx(i)][2] * 10.0;
        }
        
        // Pega a matriz da caixa na diagonal
        std::vector<double> current_box = {
            0.0, box[0][0] * 10.0,
            0.0, box[1][1] * 10.0,
            0.0, box[2][2] * 10.0
        };
        std::vector<bool> current_pbc = {true, true, true};

        return py::make_tuple(current_box, current_pbc, d_data, a_data, h_data);
    }
};

PYBIND11_MODULE(_core, m) {
    m.def("calculate_frame_hbonds", &calculate_frame_hbonds);
    py::class_<XTCReader>(m, "XTCReader")
        .def(py::init<std::string>())
        .def("read_next_frame", &XTCReader::read_next_frame);
}
