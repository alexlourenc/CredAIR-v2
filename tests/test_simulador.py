import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from credair.atributos import criar_atributos
from credair.config import ALVO, BRONZE, RAIZ, RANDOM_STATE, VARIAVEIS_FINAIS
from credair.limpeza import limpar
from credair.modelo import modelo_logistico
from credair.simulador import VERSOES, avaliar, exportar_parametros

PROPOSTA = {"idade": 30, "renda_mensal": 3000, "tempo_emprego_anos": 5, "score_credito": 700,
            "valor_solicitado": 12000, "prazo_meses": 24, "dividas_ativas": 0}


@pytest.fixture(scope="module")
def parametros():
    return exportar_parametros()


@pytest.fixture(scope="module")
def treinados():
    # As mesmas versões treinadas com o scikit-learn, para comparar com a conta feita pelo simulador
    ouro = criar_atributos(limpar(pd.read_csv(BRONZE))[0])
    X_treino, _, y_treino, _ = train_test_split(ouro[VARIAVEIS_FINAIS], ouro[ALVO], test_size=0.2,
                                                stratify=ouro[ALVO], random_state=RANDOM_STATE)
    return {chave: modelo_logistico(variaveis, []).fit(X_treino[variaveis], y_treino)
            for chave, (_, variaveis) in VERSOES.items()}


def test_json_do_vercel_esta_atualizado(parametros):
    # Se este teste falhar, rode `python -m credair.simulador` para regenerar o arquivo
    salvo = json.loads((RAIZ / "api" / "modelos.json").read_text(encoding="utf-8"))
    assert salvo["padrao"] == parametros["padrao"]
    assert salvo["versoes"].keys() == parametros["versoes"].keys()
    for chave, versao in parametros["versoes"].items():
        for campo in ("mediana", "media", "desvio", "coeficientes", "intercepto", "limiar", "auc_cv"):
            # Tolerância pequena: o treino pode variar na última casa decimal entre máquinas
            assert salvo["versoes"][chave][campo] == pytest.approx(versao[campo], rel=1e-6), (chave, campo)


def test_usa_os_resultados_do_notebook(parametros):
    final = parametros["versoes"]["final"]
    assert final["limiar"] == pytest.approx(0.31)
    assert final["auc_cv"] == pytest.approx(0.794, abs=0.001)
    ponto = next(p for p in final["curva"] if p["limiar"] == 0.31)
    assert ponto["recall"] == pytest.approx(0.738, abs=0.001)


@pytest.mark.parametrize("versao", list(VERSOES))
@pytest.mark.parametrize("ajuste", [{}, {"score_credito": 400, "dividas_ativas": 5},
                                    {"idade": 250, "renda_mensal": 300}, {"prazo_meses": 6}])
def test_mesma_probabilidade_do_scikit_learn(treinados, parametros, versao, ajuste):
    # A conta feita com os parâmetros exportados reproduz o pipeline treinado
    proposta = {**PROPOSTA, **ajuste}
    bronze = pd.DataFrame([proposta]).assign(estado=None, escolaridade=None, finalidade=None)
    ouro = criar_atributos(limpar(bronze)[0])
    variaveis = VERSOES[versao][1]
    esperado = treinados[versao].predict_proba(ouro[variaveis])[0, 1]
    assert avaliar(proposta, parametros, versao)["probabilidade"] == pytest.approx(esperado)


def test_avalia_proposta(parametros):
    r = avaliar(PROPOSTA, parametros)
    assert r["versao"] == "final"
    assert 0 < r["probabilidade"] < 1
    assert r["limiar"] == r["limiar_sugerido"] == pytest.approx(0.31)
    assert r["alto_risco"] == (r["probabilidade"] >= r["limiar"])
    # 12.000 em 24 meses = parcela de 500; 500 / 3.000 de renda = 1/6
    assert r["parcela_estimada"] == pytest.approx(500)
    assert r["comprometimento_renda"] == pytest.approx(1 / 6)
    assert r["avisos"] == []


def test_contribuicoes_somam_o_log_odds(parametros):
    r = avaliar(PROPOSTA, parametros)
    log_odds = parametros["versoes"]["final"]["intercepto"] + sum(f["contribuicao"] for f in r["fatores"])
    assert 1 / (1 + np.exp(-log_odds)) == pytest.approx(r["probabilidade"])


def test_score_baixo_aumenta_o_risco(parametros):
    risco = avaliar({**PROPOSTA, "score_credito": 400}, parametros)
    assert risco["probabilidade"] > avaliar(PROPOSTA, parametros)["probabilidade"]


def test_comprometimento_faz_diferenca(parametros):
    # Parcela pesada (R$ 20 mil em 6 meses para renda de R$ 3 mil): só o modelo final percebe o tamanho do risco
    pesada = {**PROPOSTA, "valor_solicitado": 20000, "prazo_meses": 6}
    final = avaliar(pesada, parametros, "final")["probabilidade"]
    sem = avaliar(pesada, parametros, "sem_comprometimento")["probabilidade"]
    assert final > sem + 0.3


def test_limiar_escolhido_muda_a_decisao(parametros):
    prob = avaliar(PROPOSTA, parametros)["probabilidade"]
    assert avaliar(PROPOSTA, parametros, limiar=prob - 0.01)["alto_risco"]
    assert not avaliar(PROPOSTA, parametros, limiar=prob + 0.01)["alto_risco"]


def test_desempenho_do_limiar_vem_da_curva(parametros):
    r = avaliar(PROPOSTA, parametros, limiar=0.5)
    ponto = next(p for p in parametros["versoes"]["final"]["curva"] if p["limiar"] == 0.5)
    assert r["desempenho"] == ponto
    # Baixar o limiar sinaliza mais propostas e pega mais inadimplentes
    baixo = avaliar(PROPOSTA, parametros, limiar=0.2)["desempenho"]
    assert baixo["sinalizadas"] > ponto["sinalizadas"]
    assert baixo["recall"] > ponto["recall"]


def test_aplica_a_limpeza_do_notebook(parametros):
    # P3 e P5: idade de 250 anos e renda de R$ 300 viram ausentes e o modelo usa a mediana
    r = avaliar({**PROPOSTA, "idade": 250, "renda_mensal": 300}, parametros)
    assert len(r["avisos"]) == 2
    assert r["comprometimento_renda"] is None
    assert 0 < r["probabilidade"] < 1


@pytest.mark.parametrize("proposta, versao, limiar", [
    ({**PROPOSTA, "idade": None}, None, None),
    ({**PROPOSTA, "prazo_meses": 0}, None, None),
    ({**PROPOSTA, "renda_mensal": "abc"}, None, None),
    (PROPOSTA, "estado", None),
    (PROPOSTA, None, 1.5),
])
def test_rejeita_pedido_invalido(parametros, proposta, versao, limiar):
    with pytest.raises(ValueError):
        avaliar(proposta, parametros, versao, limiar)
