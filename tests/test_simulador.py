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
                                    {"renda_mensal": 300}, {"prazo_meses": 6}])
def test_mesma_probabilidade_do_scikit_learn(treinados, parametros, versao, ajuste):
    # A conta feita com os parâmetros exportados reproduz o pipeline treinado
    proposta = {**PROPOSTA, **ajuste}
    bronze = pd.DataFrame([proposta]).assign(estado=None, escolaridade=None, finalidade=None)
    ouro = criar_atributos(limpar(bronze, marcar_invalidos=False)[0])
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


# Proposta da revisão (D16): parcela pesada, em que só a renda varia
PESADA = {"idade": 30, "tempo_emprego_anos": 5, "score_credito": 650, "valor_solicitado": 30000,
          "prazo_meses": 6, "dividas_ativas": 2}


def test_renda_menor_nunca_reduz_o_risco(parametros):
    # Com os outros campos fixos, a probabilidade não diminui quando a renda cai
    probs = [avaliar({**PESADA, "renda_mensal": r}, parametros)["probabilidade"] for r in (3000, 1000, 650, 599, 300)]
    assert all(depois >= antes for antes, depois in zip(probs, probs[1:]))


def test_renda_abaixo_de_600_e_usada_como_informada(parametros):
    # Na base, a regra P5 anulava rendas artificiais; numa proposta nova, a renda baixa conta como renda baixa
    r = avaliar({**PESADA, "renda_mensal": 599}, parametros)
    assert r["alto_risco"]
    assert r["comprometimento_renda"] == pytest.approx(5000 / 599)


@pytest.mark.parametrize("renda", [300, 599, 3000, 40000])
def test_comprometimento_sempre_calculado(parametros, renda):
    assert avaliar({**PROPOSTA, "renda_mensal": renda}, parametros)["comprometimento_renda"] is not None


def test_avisa_fora_da_faixa_do_treino(parametros):
    # Renda de R$ 300 e prazo de 72 meses são válidos, mas o modelo nunca viu valores assim
    r = avaliar({**PROPOSTA, "renda_mensal": 300, "prazo_meses": 72}, parametros)
    assert len(r["avisos"]) == 2
    assert avaliar(PROPOSTA, parametros)["avisos"] == []


@pytest.mark.parametrize("ajuste", [
    {"idade": 17}, {"idade": 101}, {"idade": float("nan")},
    {"renda_mensal": 0}, {"renda_mensal": -100}, {"renda_mensal": float("inf")},
    {"tempo_emprego_anos": -1}, {"tempo_emprego_anos": 30},
    {"score_credito": -1}, {"score_credito": 1001},
    {"valor_solicitado": 0}, {"valor_solicitado": -5000}, {"valor_solicitado": float("-inf")},
    {"prazo_meses": 0}, {"prazo_meses": float("nan")},
    {"dividas_ativas": -1}, {"dividas_ativas": 1.5},
    {"idade": None}, {"renda_mensal": "abc"}, {"dividas_ativas": True},
])
def test_rejeita_proposta_impossivel(parametros, ajuste):
    # Cada regra de validação (D16) recusa a proposta com ValueError
    with pytest.raises(ValueError):
        avaliar({**PROPOSTA, **ajuste}, parametros)


@pytest.mark.parametrize("versao, limiar", [("estado", None), (None, 1.5), (None, 0)])
def test_rejeita_versao_ou_limiar_invalido(parametros, versao, limiar):
    with pytest.raises(ValueError):
        avaliar(PROPOSTA, parametros, versao, limiar)
