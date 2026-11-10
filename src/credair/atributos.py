"""Engenharia de atributos: camada prata → camada ouro (seção 5 do notebook)."""

import pandas as pd


def criar_atributos(df: pd.DataFrame) -> pd.DataFrame:
    # Cria as variáveis derivadas (transformações determinísticas, linha a linha)
    df = df.copy()
    df["parcela_estimada"] = df["valor_solicitado"] / df["prazo_meses"]
    df["comprometimento_renda"] = df["parcela_estimada"] / df["renda_mensal"]
    df["valor_sobre_renda"] = df["valor_solicitado"] / df["renda_mensal"]
    return df
