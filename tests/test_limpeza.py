import numpy as np

from credair.config import ORDEM_ESCOLARIDADE
from credair.limpeza import limpar


def test_padroniza_estado_e_remove_duplicata_disfarcada(propostas):
    # " sp " vira "SP" e a linha 2 passa a ser cópia da linha 1
    limpo, log = limpar(propostas)
    assert len(limpo) == 4
    assert set(limpo["estado"]) == {"SP", "MG", "RJ", "BA"}
    assert log.set_index("problema").loc["P2", "registros"] == 1


def test_valores_invalidos_viram_ausentes(propostas):
    limpo, _ = limpar(propostas)
    # P3: idade de 250 anos; P4: 45 anos de emprego aos 40; P5: renda de R$ 300
    assert np.isnan(limpo.loc[1, "idade"])
    assert np.isnan(limpo.loc[2, "tempo_emprego_anos"])
    assert np.isnan(limpo.loc[2, "renda_mensal"])
    # As demais colunas das linhas com problema são preservadas
    assert limpo.loc[1, "score_credito"] == 550


def test_tipos_da_camada_prata(propostas):
    limpo, _ = limpar(propostas)
    assert list(limpo["escolaridade"].cat.categories) == ORDEM_ESCOLARIDADE
    assert limpo["escolaridade"].cat.ordered
    assert limpo["estado"].dtype == "category"
    assert limpo["inadimplente"].dtype == "int8"


def test_nao_altera_o_dado_original(propostas):
    copia = propostas.copy()
    limpar(propostas)
    assert propostas.equals(copia)


def test_bronze_real(bronze):
    # Números do diagnóstico registrados no notebook (seção 2)
    limpo, log = limpar(bronze)
    assert log.set_index("problema").loc["P2", "registros"] == 19
    assert len(limpo) == len(bronze) - 19
    assert not limpo.duplicated().any()
    assert limpo["idade"].max() <= 100
    assert limpo["renda_mensal"].min() >= 600
    assert limpo["estado"].nunique() == 27
