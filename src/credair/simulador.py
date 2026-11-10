"""Simulação de propostas novas com os modelos validados (Fase 14).

O treino usa o scikit-learn, mas a avaliação de uma proposta usa só os parâmetros aprendidos
(medianas, médias, desvios, coeficientes e limiar). Assim a função do Vercel não precisa do
scikit-learn, que deixaria o pacote da função acima do limite de tamanho.
"""

import numpy as np
import pandas as pd

from credair.atributos import criar_atributos
from credair.config import ALVO, BRONZE, RANDOM_STATE, VARIAVEIS_FINAIS
from credair.limpeza import limpar

# Campos que o simulador pede: as variáveis do modelo final, menos a derivada (comprometimento_renda)
CAMPOS = ["idade", "renda_mensal", "tempo_emprego_anos", "score_credito", "valor_solicitado",
          "prazo_meses", "dividas_ativas"]

# Versões validadas na seção 7 do notebook que o simulador permite comparar
VERSOES = {
    "final": ("Modelo final", VARIAVEIS_FINAIS),
    "sem_comprometimento": ("Sem comprometimento da renda", CAMPOS),
    "so_score": ("Só score (referência)", ["score_credito"]),
}

# Limiares que a página pode escolher, com o desempenho de cada um no conjunto de teste
GRADE_LIMIARES = np.round(np.arange(0.05, 0.96, 0.01), 2)


def _parametros(modelo) -> dict:
    # Extrai do pipeline treinado o que a avaliação precisa: imputação, padronização e coeficientes
    numericas = modelo.named_steps["preparar"].named_transformers_["num"]
    regressao = modelo.named_steps["modelo"]
    return {
        "mediana": numericas.named_steps["imputar"].statistics_.tolist(),
        "media": numericas.named_steps["padronizar"].mean_.tolist(),
        "desvio": numericas.named_steps["padronizar"].scale_.tolist(),
        "coeficientes": regressao.coef_[0].tolist(),
        "intercepto": float(regressao.intercept_[0]),
    }


def _curva(y_teste, prob) -> list[dict]:
    # Precisão, recall e fração de propostas sinalizadas em cada limiar da grade
    curva = []
    for limiar in GRADE_LIMIARES:
        pred = prob >= limiar
        sinalizadas = int(pred.sum())
        curva.append({
            "limiar": float(limiar),
            "precisao": float(y_teste[pred].mean()) if sinalizadas else None,
            "recall": float(pred[y_teste == 1].mean()),
            "sinalizadas": sinalizadas / len(y_teste),
        })
    return curva


def exportar_parametros() -> dict:
    # Refaz o treino do notebook (mesma divisão 80/20) para cada versão e exporta tudo num dicionário
    from sklearn.model_selection import cross_val_score, train_test_split

    from credair.modelo import escolher_limiar, modelo_logistico, validacao_cruzada

    ouro = criar_atributos(limpar(pd.read_csv(BRONZE))[0])
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        ouro[VARIAVEIS_FINAIS], ouro[ALVO], test_size=0.2, stratify=ouro[ALVO], random_state=RANDOM_STATE)

    versoes = {}
    for chave, (nome, variaveis) in VERSOES.items():
        modelo = modelo_logistico(variaveis, []).fit(X_treino[variaveis], y_treino)
        prob = modelo.predict_proba(X_teste[variaveis])[:, 1]
        auc_cv = cross_val_score(modelo_logistico(variaveis, []), X_treino[variaveis], y_treino,
                                 cv=validacao_cruzada(), scoring="roc_auc").mean()
        versoes[chave] = {
            "nome": nome,
            "variaveis": variaveis,
            **_parametros(modelo),
            "limiar": escolher_limiar(modelo_logistico(variaveis, []), X_treino[variaveis], y_treino),
            "auc_cv": float(auc_cv),
            "curva": _curva(y_teste.to_numpy(), prob),
        }
    return {"padrao": "final", "versoes": versoes}


def avaliar(proposta: dict, parametros: dict, versao: str | None = None, limiar: float | None = None) -> dict:
    # Aplica limpeza, atributos e modelo a uma proposta e explica o resultado
    faltando = [c for c in CAMPOS if proposta.get(c) is None]
    if faltando:
        raise ValueError(f"Campos obrigatórios ausentes: {', '.join(faltando)}")
    if float(proposta["prazo_meses"]) < 1:
        raise ValueError("O prazo deve ser de pelo menos 1 mês")
    versao = versao or parametros["padrao"]
    if versao not in parametros["versoes"]:
        raise ValueError(f"Versão desconhecida: {versao}")
    v = parametros["versoes"][versao]
    limiar = v["limiar"] if limiar is None else float(limiar)
    if not 0 < limiar < 1:
        raise ValueError("O limiar deve estar entre 0 e 1")

    # Colunas fora do modelo ficam vazias: a limpeza precisa delas, mas elas não afetam o resultado
    bronze = pd.DataFrame([{c: float(proposta[c]) for c in CAMPOS}])
    bronze[["estado", "escolaridade", "finalidade"]] = None
    prata, log = limpar(bronze)
    ouro = criar_atributos(prata)

    # Mesmas etapas do pipeline: imputação pela mediana, padronização e regressão logística
    x = ouro[v["variaveis"]].to_numpy(dtype=float)[0]
    x = np.where(np.isnan(x), v["mediana"], x)
    z = (x - np.array(v["media"])) / np.array(v["desvio"])
    contribuicoes = np.array(v["coeficientes"]) * z
    prob = float(1 / (1 + np.exp(-(v["intercepto"] + contribuicoes.sum()))))

    # Contribuição de cada variável para o log-odds, em relação a uma proposta média do treino
    fatores = sorted(({"variavel": nome, "contribuicao": float(c)} for nome, c in zip(v["variaveis"], contribuicoes)),
                     key=lambda f: abs(f["contribuicao"]), reverse=True)

    # Avisos dos problemas P3 a P5: o valor informado vira ausente e o modelo usa a mediana do treino
    avisos = log[log["problema"].isin(["P3", "P4", "P5"]) & (log["registros"] > 0)]["tratamento"].tolist()

    # Desempenho no teste do limiar escolhido (o ponto mais próximo da grade)
    desempenho = min(v["curva"], key=lambda p: abs(p["limiar"] - limiar))

    comprometimento = ouro.loc[0, "comprometimento_renda"]
    return {
        "versao": versao,
        "nome_versao": v["nome"],
        "probabilidade": prob,
        "limiar": limiar,
        "limiar_sugerido": v["limiar"],
        "alto_risco": prob >= limiar,
        "parcela_estimada": float(ouro.loc[0, "parcela_estimada"]),
        "comprometimento_renda": None if pd.isna(comprometimento) else float(comprometimento),
        "fatores": fatores,
        "avisos": avisos,
        "desempenho": desempenho,
    }


if __name__ == "__main__":
    # Gera o arquivo de parâmetros usado pela função do Vercel: python -m credair.simulador
    import json

    from credair.config import RAIZ

    destino = RAIZ / "api" / "modelos.json"
    destino.write_text(json.dumps(exportar_parametros(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Parâmetros salvos em {destino.relative_to(RAIZ)}")
