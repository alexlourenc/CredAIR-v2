"""Pipeline do modelo, métricas e escolha do limiar (seções 6 a 8 do notebook)."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credair.config import RANDOM_STATE


def modelo_logistico(numericas, categoricas, **kwargs):
    # Monta o pipeline completo: pré-processamento + regressão logística
    transformadores = [("num", Pipeline([("imputar", SimpleImputer(strategy="median")),
                                         ("padronizar", StandardScaler())]), numericas)]
    if categoricas:
        transformadores.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categoricas))
    return Pipeline([
        ("preparar", ColumnTransformer(transformadores)),
        ("modelo", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **kwargs)),
    ])


def validacao_cruzada():
    # Validação cruzada estratificada com 5 partes, usada para comparar modelos e escolher o limiar
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def metricas(y_real, prob, limiar=0.5):
    # Calcula as métricas de avaliação para um limiar de decisão
    pred = (prob >= limiar).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_real, prob),
        "Acurácia": accuracy_score(y_real, pred),
        "Precisão": precision_score(y_real, pred, zero_division=0),
        "Recall": recall_score(y_real, pred, zero_division=0),
        "F1": f1_score(y_real, pred, zero_division=0),
    }


def escolher_limiar(modelo, X_treino, y_treino, cv=None, grade=None):
    # Limiar que maximiza o F1 nas previsões fora da amostra do treino (o teste não é usado)
    cv = cv if cv is not None else validacao_cruzada()
    grade = grade if grade is not None else np.round(np.arange(0.15, 0.71, 0.01), 2)
    p_oof = cross_val_predict(modelo, X_treino, y_treino, cv=cv, method="predict_proba")[:, 1]
    f1s = [f1_score(y_treino, (p_oof >= t).astype(int)) for t in grade]
    return float(grade[int(np.argmax(f1s))])
