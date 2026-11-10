import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from credair.atributos import criar_atributos
from credair.config import ALVO, CATEGORICAS, NUMERICAS, RANDOM_STATE, VARIAVEIS_FINAIS
from credair.limpeza import limpar
from credair.modelo import escolher_limiar, metricas, modelo_logistico


def test_pipeline_aceita_valores_ausentes(propostas):
    prata, _ = limpar(propostas)
    ouro = criar_atributos(prata)
    modelo = modelo_logistico(NUMERICAS, CATEGORICAS).fit(ouro[NUMERICAS + CATEGORICAS], ouro[ALVO])
    prob = modelo.predict_proba(ouro[NUMERICAS + CATEGORICAS])[:, 1]
    assert prob.shape == (len(ouro),)
    assert np.all((prob >= 0) & (prob <= 1))


def test_pipeline_sem_categoricas():
    modelo = modelo_logistico(["score_credito"], [])
    assert [nome for nome, *_ in modelo.named_steps["preparar"].transformers] == ["num"]


def test_metricas_com_limiar():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.4, 0.35, 0.8])
    padrao = metricas(y, prob)
    assert padrao["Recall"] == 0.5
    assert padrao["Precisão"] == 1.0
    baixo = metricas(y, prob, limiar=0.3)
    assert baixo["Recall"] == 1.0
    assert baixo["ROC-AUC"] == padrao["ROC-AUC"] == 0.75


@pytest.fixture(scope="module")
def divisao(bronze):
    ouro = criar_atributos(limpar(bronze)[0])
    return train_test_split(ouro[VARIAVEIS_FINAIS], ouro[ALVO], test_size=0.2,
                            stratify=ouro[ALVO], random_state=RANDOM_STATE)


def test_reproduz_resultados_do_notebook(divisao):
    # Resultados publicados no notebook (seção 8)
    X_treino, X_teste, y_treino, y_teste = divisao
    modelo = modelo_logistico(VARIAVEIS_FINAIS, []).fit(X_treino, y_treino)
    prob = modelo.predict_proba(X_teste)[:, 1]
    limiar = escolher_limiar(modelo_logistico(VARIAVEIS_FINAIS, []), X_treino, y_treino)
    assert limiar == pytest.approx(0.31)
    resultado = metricas(y_teste, prob, limiar)
    assert resultado["ROC-AUC"] == pytest.approx(0.781, abs=0.001)
    assert resultado["Recall"] == pytest.approx(0.738, abs=0.001)
