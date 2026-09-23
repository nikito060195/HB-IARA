import hbiara as hb
import os
import pandas as pd

def main():
    # =========================================================================
    # CONFIGURAÇÕES INICIAIS DA TRAJETÓRIA GROMACS
    # =========================================================================
    xtc_file = "../dados_testes/bulk.xtc"  
    top_file = "../dados_testes/bulk.top"
    
    prefix = "analise_gromacs_bulk" 
    
    # =========================================================================
    # FASE 1: CONSTRUÇÃO DO BANCO DE DADOS (PARQUET)
    # =========================================================================
    if not os.path.exists(f"{prefix}_stats.parquet"):
        print("Bancos de dados não encontrados. Iniciando o motor C++ com XDRFile...")
        
        # Mapeamento usando a Coluna 2 (type) do arquivo .top!
        # Como vimos, a água usa "OW" para oxigênio e "HW" engloba HW1 e HW2.
        doadores_e_hidrogenios = [
            ("opls_116", ["opls_117"]), 
            # ("oh", ["ho"]) # Exemplo: Descomente e ajuste se a membrana de GO doar ligações
        ]
        
        # Quem recebe? A própria água. 
        # (Se quiser ver a água doando para o epóxi do GO, adicione o tipo do epóxi aqui, ex: "op")
        aceitadores_permitidos = ["opls_116"] 
        
        construtor = hb.HBondAnalysis(
            filename=xtc_file,
            topology_file=top_file,     # OBRIGATÓRIO PARA GROMACS
            file_format="gromacs",      # Força a engine a usar a classe autônoma XTCReader
            donor_hydrogen=doadores_e_hidrogenios, 
            acceptor=aceitadores_permitidos,
            set_len=3.5,
            set_angle=30,
            n_cpus=5,                   # Processamento paralelo ativado
            box_limits=None, 
            pbc=None,
            output_prefix=prefix 
        )
        construtor.run()
    else:
        print("Bancos de dados Parquet encontrados! Pulando processamento C++...")

    # =========================================================================
    # FASE 2: EXPLORAÇÃO E DATA SCIENCE
    # =========================================================================
    explorador = hb.HBondExplorer(prefix=prefix)
    
    print("\n" + "="*50)
    print(" ANÁLISE GLOBAL DO SISTEMA (MÉDIA GERAL)")
    print("="*50)
    
    status_df = explorador.df_stats
    avg_total_hbonds = status_df['total_hbonds'].mean()
    sd_total_hbonds = status_df['total_hbonds'].std()
    
    print(f"-> Total absoluto de ligações no sistema: {avg_total_hbonds:.3f} ± {sd_total_hbonds:.3f} por frame")

    print("\n" + "="*50)
    print(" DENSIDADE DE LIGAÇÕES POR COMBINAÇÃO (LIGAÇÕES / MOLÉCULA)")
    print("="*50)
    
    df_hbonds = pd.read_parquet(explorador.file_hbonds)
    
    if df_hbonds.empty:
        print("Nenhuma ligação de hidrogênio encontrada.")
    else:
        # Descobre a quantidade EXATA de doadores no sistema
        qtd_doadores = df_hbonds.groupby('donor_type')['donor_id'].nunique().to_dict()
        
        # Agrupa por frame, donor_type e acceptor_type e conta as ligações (absoluto)
        contagem = df_hbonds.groupby(['frame', 'donor_type', 'acceptor_type']).size().reset_index(name='hbond_count')
        
        # Converte a contagem absoluta para DENSIDADE (ligações / molécula doadora)
        contagem['density'] = contagem.apply(
            lambda row: row['hbond_count'] / qtd_doadores.get(row['donor_type'], 1), 
            axis=1
        )
        
        # Calcula a Média e o Desvio Padrão da densidade ao longo do tempo (frames)
        resumo_combinacoes = contagem.groupby(['donor_type', 'acceptor_type'])['density'].agg(['mean', 'std']).reset_index()
        
        # Imprime os resultados termodinâmicos finais
        for index, row in resumo_combinacoes.iterrows():
            d_type = row['donor_type']
            a_type = row['acceptor_type']
            media = row['mean']
            desvio = 0.0 if pd.isna(row['std']) else row['std']
            
            total_mol = qtd_doadores.get(d_type, 0)
            print(f"-> {d_type:^5} (Doou) -> {a_type:^5} (Recebeu) : {media:.3f} ± {desvio:.3f} ligações/molécula")

        # ---------------------------------------------------------------------
        # NOVA LÓGICA DE PARTICIPAÇÃO TOTAL (MÉDIA ± DESVIO PADRÃO)
        # ---------------------------------------------------------------------
        print("\n PARTICIPAÇÃO TOTAL (DOADAS + RECEBIDAS) POR ESPÉCIE")
        
        # Cria uma lista das espécies únicas encontradas no sistema
        especies = set(df_hbonds['donor_type']).union(set(df_hbonds['acceptor_type']))
        
        # Descobre a base real de moléculas (unindo quem atuou como doador e quem atuou como aceitador)
        qtd_moleculas_total = {}
        for esp in especies:
            ids_doador = set(df_hbonds[df_hbonds['donor_type'] == esp]['donor_id'])
            ids_aceitador = set(df_hbonds[df_hbonds['acceptor_type'] == esp]['acceptor_id'])
            qtd_moleculas_total[esp] = len(ids_doador.union(ids_aceitador))
        
        for esp in especies:
            n_mols = qtd_moleculas_total[esp]
            if n_mols == 0: continue
            
            # Conta absoluta por frame de vezes que a espécie DOOU e RECEBEU
            doadas_por_frame = df_hbonds[df_hbonds['donor_type'] == esp].groupby('frame').size()
            recebidas_por_frame = df_hbonds[df_hbonds['acceptor_type'] == esp].groupby('frame').size()
            
            # Soma as duas matrizes alinhadas por frame (fill_value=0 evita erro se não doou num frame)
            total_absoluto_por_frame = doadas_por_frame.add(recebidas_por_frame, fill_value=0)
            
            # Converte em densidade
            densidade_total_por_frame = total_absoluto_por_frame / n_mols
            
            # Extrai estatísticas corretas baseadas no conjunto de frames
            media_total = densidade_total_por_frame.mean()
            std_total = densidade_total_por_frame.std()
            std_total = 0.0 if pd.isna(std_total) else std_total
            
            print(f"-> A espécie {esp:^5} participa de {media_total:.3f} ± {std_total:.3f} ligações/molécula (Base: {n_mols} mols)")


    print("\n" + "="*50)
    print(" ANÁLISE ESPACIAL - FRAME 0")
    print("="*50)
    
    frame_alvo = 0 
    
    esfera = explorador.filter_sphere(
        frame_idx=frame_alvo, 
        radius=10.0, 
        center_x=None, center_y=None, center_z=None
    )
    print(f"-> Ligações na esfera de raio 10 (Centro da Caixa): {len(esfera)}")
    
    print("\n DADOS CRUS (DataFrame Pandas da Esfera)")
    colunas_para_mostrar = ['donor_id', 'donor_type', 'acceptor_id', 'acceptor_type', 'acceptor_x', 'acceptor_y', 'acceptor_z']
    print(esfera[colunas_para_mostrar].head(3).to_string(index=False))
if __name__ == "__main__":
    main()
