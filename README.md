# HB-IARA (Iteractive Archiving for Rapid Analysis of Hydrogen Bonds)
Instalação do backend via setup.py.
Importe a biblioteca hbiara no seu script de cálculo. 
Método de cálculo:
       H
      /
     /
    /
   /
  / \ θ
 /   |
D-------------A
       r

# Organização dos arquivos da biblioteca
HB-IARA/
├── backend/                       <-- Tudo o que é C/C++ fica escondido aqui
│   ├── hbond_core.cpp
│   ├── xdrfile.c
│   ├── xdrfile.h
│   └── ... (outros arquivos xdr)
│
├── exemplos/                      <-- Scripts de teste
│   ├── hbond_xyz_how_to_use.py
│   └── hbond_how_to_use_gromacs.py
│
├── dados_teste/                   <-- Arquivos xtc, top, xyz, ...
│
│── hbiara/                        <-- Pasta oficial da biblioteca Python
│   ├── __init__.py                <-- Expõe as classes principais ao utilizador
│   └── analysis.py                <-- O orquestrador
│
├── pyproject.toml
├── setup.py
├── requirements.txt
├── README.md
├── LICENSE
└── LICENSE_XDR

# Instalação recomendada
Para a biblioteca HB-IARA, o OS precisa de um compilador C/C++ e dos cabeçalhos de desenvolvimento do Python.
Para sistemas Ubuntu/Debian, faça:

sudo apt update

sudo apt install -y build-essential python3-dev

Recomenda-se ainda, para uma boa prática de isolamento, criar um ambiente para evitar conflitos de bibliotecas no sistema:
# Criar o ambiente virtual (exemplo com venv)
python3 -m venv .venv
# Ativar o ambiente virtual
source .venv/bin/activate

# Atualizar as ferramentas básicas de empacotamento
pip install --upgrade pip setuptools wheel

# Instalação de dependências
Os pacotes necessários para execução da biblioteca estão contidos no arquivo requirements.txt, e podem ser instalados via:

pip install -r requirements.txt

# Instalação final da biblioteca
Com os passos anteriores finalizados, a compilação e instalação pode ser feita de duas maneiras.
Modo de desenvolvimento, para o usuário que queira testar ou editar a biblioteca, faça:

pip install -e . --no-build-isolation

Para instalação padrão, para o usuário que queira apenas rodar as análises oferecidas pela biblioteca, faça:

pip install .

# Verificação final
A verificação da integridade da instalação pode ser feita via:

python -c "import hbiara; print('hbond_core compilado com sucesso')"
