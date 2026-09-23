import hbiara as hb
import os
import pandas as pd

def main():
    # =========================================================================
    # CONFIGURAÇÕES INICIAIS DA TRAJETÓRIA
    # =========================================================================
    file = "../dados_testes/md_committee_all.extxyz"  
    prefix = "meu_teste_bulk_thiago_v3" 
    
    # =========================================================================
    # FASE 1: CONSTRUÇÃO DO BANCO DE DADOS (PARQUET)
    # =========================================================================
    if not os.path.exists(f"{prefix}_stats.parquet"):
        print("Bancos de dados não encontrados. Iniciando o motor C++...")
        construtor = hb.HBondAnalysis(
            filename=file,
            donor_hydrogen=[("O", ["H"])], # Sintaxe estrita de tuplas
            acceptor=["O"],
            set_len=3.5,
            set_angle=30,
            n_cpus=5,
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
    
    # 1. Carrega o banco de dados completo de ligações
    df_hbonds = pd.read_parquet(explorador.file_hbonds)
    
    if df_hbonds.empty:
        print("Nenhuma ligação de hidrogênio encontrada.")
    else:
        # 2. Descobre a quantidade EXATA de moléculas de cada espécie no sistema
        # Usando os IDs únicos que o C++ registrou ao longo da simulação
        qtd_doadores = df_hbonds.groupby('donor_type')['donor_id'].nunique().to_dict()
        
        # 3. Agrupa por frame, donor_type e acceptor_type e conta as ligações (absoluto)
        contagem = df_hbonds.groupby(['frame', 'donor_type', 'acceptor_type']).size().reset_index(name='hbond_count')
        
        # 4. Converte a contagem absoluta para DENSIDADE (ligações / molécula doadora)
        contagem['density'] = contagem.apply(
            lambda row: row['hbond_count'] / qtd_doadores[row['donor_type']], 
            axis=1
        )
        
        # 5. Calcula a Média e o Desvio Padrão da densidade ao longo do tempo (frames)
        resumo_combinacoes = contagem.groupby(['donor_type', 'acceptor_type'])['density'].agg(['mean', 'std']).reset_index()
        
        # 6. Imprime os resultados com a termodinâmica correta
        for index, row in resumo_combinacoes.iterrows():
            d_type = row['donor_type']
            a_type = row['acceptor_type']
            media = row['mean']
            desvio = 0.0 if pd.isna(row['std']) else row['std']
            
            total_mol = qtd_doadores[d_type]
            print(f"-> {d_type:^5} (Doou) -> {a_type:^5} (Recebeu) : {media:.3f} ± {desvio:.3f} ligações/molécula (Base: {total_mol} mols)")
    
    # 1. Carrega o banco de dados completo de ligações
    df_hbonds = pd.read_parquet(explorador.file_hbonds)
    
    # 2. Agrupa por frame, donor_type e acceptor_type e conta quantas ligações existem
    contagem_por_frame = df_hbonds.groupby(['frame', 'donor_type', 'acceptor_type']).size().reset_index(name='hbond_count')
    
    # 3. Calcula a média e o desvio padrão dessas contagens ao longo de todos os frames
    resumo_combinacoes = contagem_por_frame.groupby(['donor_type', 'acceptor_type'])['hbond_count'].agg(['mean', 'std']).reset_index()
    
    # 4. Imprime os resultados formatados
    if resumo_combinacoes.empty:
        print("Nenhuma ligação de hidrogênio encontrada com os critérios atuais.")
    else:
        for index, row in resumo_combinacoes.iterrows():
            d_type = row['donor_type']
            a_type = row['acceptor_type']
            media = row['mean']
            desvio = row['std']
            # Se o desvio for NaN (ex: quando há apenas 1 frame), substituímos por 0.0
            desvio = 0.0 if pd.isna(desvio) else desvio 
            print(f"-> {d_type:^5} (Doou)  ------>  {a_type:^5} (Recebeu) : {media:.3f} ± {desvio:.3f} ligações/frame")


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
