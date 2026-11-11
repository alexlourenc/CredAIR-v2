"""Limpeza dos dados: camada bronze → camada prata (seção 3 do notebook)."""

import pandas as pd

from credair.config import ALVO, ORDEM_ESCOLARIDADE


def limpar(df: pd.DataFrame, marcar_invalidos: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Aplica as regras de limpeza e devolve (dados_limpos, log_de_tratamentos)
    df = df.copy()
    log = []

    # P1 — padroniza a UF: remove espaços e usa maiúsculas
    afetados = (df["estado"] != df["estado"].str.strip().str.upper()).sum()
    df["estado"] = df["estado"].str.strip().str.upper()
    log.append(("P1", "Espaços removidos em estado", afetados))

    # P2 — remove duplicatas (depois de padronizar, para pegar duplicatas "disfarçadas")
    antes = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    log.append(("P2", "Linhas duplicadas removidas", antes - len(df)))

    # P3 a P5 corrigem valores artificiais da base; uma proposta nova chega validada e é usada como informada (D16)
    if marcar_invalidos:
        df = _marcar_invalidos(df, log)

    # Tipos: categorias explícitas (escolaridade é ordinal)
    df["escolaridade"] = pd.Categorical(df["escolaridade"], categories=ORDEM_ESCOLARIDADE, ordered=True)
    df["estado"] = df["estado"].astype("category")
    df["finalidade"] = df["finalidade"].astype("category")
    if ALVO in df:  # proposta nova (simulador) ainda não tem o alvo
        df[ALVO] = df[ALVO].astype("int8")

    return df, pd.DataFrame(log, columns=["problema", "tratamento", "registros"])


def _marcar_invalidos(df: pd.DataFrame, log: list) -> pd.DataFrame:
    # Regras P3 a P5: valores impossíveis ou artificiais viram ausentes (imputados depois, no pipeline)

    # P3 — idade fora do intervalo plausível vira ausente
    invalida = ~df["idade"].between(18, 100)
    df["idade"] = df["idade"].astype("float").mask(invalida)
    log.append(("P3", "Idade inválida → ausente", int(invalida.sum())))

    # P4 — tempo de emprego maior ou igual à idade vira ausente
    invalido = df["tempo_emprego_anos"] >= df["idade"]
    df["tempo_emprego_anos"] = df["tempo_emprego_anos"].astype("float").mask(invalido)
    log.append(("P4", "Tempo de emprego ≥ idade → ausente", int(invalido.sum())))

    # P5 — renda abaixo de R$ 600 (sequência artificial) vira ausente
    invalida = df["renda_mensal"] < 600
    df["renda_mensal"] = df["renda_mensal"].mask(invalida)
    log.append(("P5", "Renda < R$ 600 → ausente", int(invalida.sum())))
    return df
