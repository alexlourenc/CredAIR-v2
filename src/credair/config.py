"""Constantes do projeto: caminhos das camadas, semente e listas de variáveis."""

from pathlib import Path

RANDOM_STATE = 42

# Raiz do projeto: src/credair/config.py → sobe três níveis
RAIZ = Path(__file__).resolve().parents[2]
BRONZE = RAIZ / "data" / "bronze" / "dataset.csv"
PRATA = RAIZ / "data" / "silver" / "credito_limpo.parquet"
OURO = RAIZ / "data" / "gold" / "credito_modelagem.parquet"
FIGURAS = RAIZ / "reports" / "figures"

# Escolaridade é ordinal: a ordem das categorias tem significado
ORDEM_ESCOLARIDADE = ["Ensino Fundamental", "Ensino Médio", "Ensino Superior", "Pós-graduação"]

# Variáveis do modelo (seção 6 do notebook)
ALVO = "inadimplente"
NUMERICAS = ["idade", "renda_mensal", "tempo_emprego_anos", "score_credito", "valor_solicitado",
             "prazo_meses", "dividas_ativas", "comprometimento_renda"]
CATEGORICAS = ["escolaridade", "finalidade"]
EXCLUIDAS = ["estado", "parcela_estimada", "valor_sobre_renda"]
VARIAVEIS_FINAIS = NUMERICAS  # escolha feita pela validação cruzada (seção 7)
