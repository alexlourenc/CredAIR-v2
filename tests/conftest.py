import pandas as pd
import pytest

from credair.config import BRONZE


@pytest.fixture
def propostas():
    # Amostra pequena com um exemplo de cada problema de qualidade (P1 a P5)
    return pd.DataFrame({
        "idade": [30, 30, 250, 40, 25],
        "renda_mensal": [3000.0, 3000.0, 4000.0, 300.0, 2500.0],
        "tempo_emprego_anos": [5, 5, 10, 45, 3],
        "score_credito": [700, 700, 550, 400, 820],
        "valor_solicitado": [12000.0, 12000.0, 5000.0, 8000.0, 6000.0],
        "prazo_meses": [24, 24, 12, 36, 12],
        "dividas_ativas": [0, 0, 2, 3, 1],
        "estado": ["SP", " sp ", "MG", "RJ", " BA"],
        "escolaridade": ["Ensino Médio", "Ensino Médio", "Ensino Superior", "Pós-graduação", "Ensino Fundamental"],
        "finalidade": ["Veículo", "Veículo", "Reforma", "Outros", "Educação"],
        "inadimplente": [0, 0, 1, 1, 0],
    })


@pytest.fixture(scope="session")
def bronze():
    return pd.read_csv(BRONZE)
