import numpy as np
import pytest

from credair.atributos import criar_atributos
from credair.limpeza import limpar


def test_calcula_variaveis_derivadas(propostas):
    ouro = criar_atributos(propostas)
    # 12.000 em 24 meses = parcela de 500; 500 / 3.000 de renda = 1/6
    assert ouro.loc[0, "parcela_estimada"] == pytest.approx(500)
    assert ouro.loc[0, "comprometimento_renda"] == pytest.approx(500 / 3000)
    assert ouro.loc[0, "valor_sobre_renda"] == pytest.approx(4)


def test_renda_ausente_propaga_ausente(propostas):
    prata, _ = limpar(propostas)
    ouro = criar_atributos(prata)
    assert np.isnan(ouro.loc[2, "comprometimento_renda"])
    assert np.isnan(ouro.loc[2, "valor_sobre_renda"])


def test_transformacao_linha_a_linha(propostas):
    # Regra contra vazamento: o resultado de uma linha não depende das outras
    completo = criar_atributos(propostas)
    sozinha = criar_atributos(propostas.iloc[[3]])
    assert sozinha.loc[3, "comprometimento_renda"] == completo.loc[3, "comprometimento_renda"]


def test_nao_altera_o_dado_original(propostas):
    copia = propostas.copy()
    criar_atributos(propostas)
    assert propostas.equals(copia)
