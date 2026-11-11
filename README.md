# CredAIR — Análise de Risco de Crédito

Projeto de Ciência de Dados que usa o histórico de propostas de crédito da **CredAIR** (financeira fictícia) para entender os fatores ligados à inadimplência e construir um modelo de **Regressão Logística** que identifique propostas de maior risco.

> Desafio 1 da trilha **Fast Track - IA | Turma 1**. Enunciado original: [desafios/analise-credito](https://github.com/LucasKiraly/fast-track-ia-t1/tree/main/desafios/analise-credito).
> Os dados são **sintéticos** e foram criados apenas para fins educacionais.

**Simulador online:** **[cred-air-v2.vercel.app](https://cred-air-v2.vercel.app)**

## Principais resultados

| Modelo | ROC-AUC (teste) | Recall da classe inadimplente |
|---|---|---|
| Referência: classe majoritária | 0,500 | 0% |
| Referência: só score de crédito | 0,695 | 24% |
| **Regressão Logística (8 variáveis)** | **0,781** | **74%** (limiar 0,31) |

- **Fator de risco mais forte:** `comprometimento_renda`, a variável criada no projeto, que mede a fração da renda comprometida com a parcela. Odds ratio de 2,78 por desvio-padrão. Ela eleva a AUC da validação cruzada de 0,769 para 0,793.
- **Fator de proteção mais forte:** `score_credito`. Odds ratio de 0,44 por desvio-padrão.
- **O "paradoxo do prazo":** prazos curtos têm mais inadimplência porque geram parcelas pesadas. Descontado o comprometimento da renda, o efeito do prazo inverte o sinal.
- `estado`, `escolaridade` e `finalidade` não mostraram associação significativa e ficaram fora do modelo final.

Detalhes no [notebook](notebooks/analise_credito.ipynb), no [registro de decisões](docs/ROADMAP.md) e nos [slides da apresentação](docs/apresentacao.pdf) (PDF).

## Estrutura do projeto

```text
CredAIR-v2/
├── data/
│   ├── bronze/         # 🥉 Dado original, como recebido. Nunca é alterado.
│   ├── silver/         # 🥈 Dado limpo (gerado pelo notebook, fora do Git)
│   └── gold/           # 🥇 Dado pronto para modelagem (gerado pelo notebook, fora do Git)
├── notebooks/
│   └── analise_credito.ipynb   # Notebook principal da análise (entregável)
├── src/credair/        # Pacote com as regras validadas no notebook
│   ├── config.py       #   caminhos, semente e listas de variáveis
│   ├── limpeza.py      #   bronze → prata
│   ├── atributos.py    #   prata → ouro
│   ├── modelo.py       #   pipeline, métricas e escolha do limiar
│   └── simulador.py    #   exportação dos modelos e avaliação de propostas novas
├── api/                # Função do simulador no Vercel + parâmetros dos modelos (modelos.json)
├── public/             # Página do simulador
├── tests/              # Testes automatizados (pytest)
├── reports/figures/    # Gráficos exportados para a apresentação
├── docs/               # Roadmap e registro de decisões
├── .github/workflows/  # Testes automáticos a cada push (GitHub Actions)
├── pyproject.toml      # Torna o pacote credair instalável
├── requirements.txt    # Dependências com versões fixadas (inclui o pacote credair)
└── vercel.json         # Configuração da publicação no Vercel
```

## Arquitetura de dados: modelo medalhão

| Camada | Conteúdo | Gerada na |
|---|---|---|
| 🥉 Bronze | `dataset.csv` exatamente como recebido (SHA-256 `5f50b73c…8d6724`) | (fonte) |
| 🥈 Prata | Dados limpos: duplicatas removidas, UFs padronizadas, valores inválidos marcados como ausentes | Seção 3 do notebook |
| 🥇 Ouro | Tabela para modelagem, com as variáveis derivadas | Seção 5 do notebook |

Transformações que "aprendem" com os dados (imputação, padronização, *one-hot encoding*) **não** ficam na camada ouro: são ajustadas apenas no conjunto de treino, dentro do `Pipeline` do scikit-learn, para evitar vazamento de dados.

## Como executar

Pré-requisitos: **Python 3.12+** (o numpy fixado no `requirements.txt` não tem versão para o 3.11) e **VS Code** com a extensão Jupyter.

```bash
# 1. Clonar o repositório
git clone https://github.com/alexlourenc/CredAIR-v2.git
cd CredAIR-v2

# 2. Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 3. Instalar as dependências (inclui o pacote credair, da pasta src/)
pip install -r requirements.txt
```

Depois, abra `notebooks/analise_credito.ipynb` no VS Code, selecione o kernel do `.venv` e use **Restart** e **Run All**. O notebook gera as camadas prata e ouro em `data/` e os gráficos em `reports/figures/`.

> **Windows:** se aparecer `OpenBLAS error: Memory allocation still failed`, rode `$env:OPENBLAS_NUM_THREADS = "1"` no terminal antes de treinar o modelo.

### Testes

```bash
pytest
```

São 45 testes: cada regra de limpeza e de engenharia de atributos, os resultados publicados acima (ROC-AUC de 0,781 e limiar de 0,31) e a equivalência entre o simulador e o scikit-learn. Eles também rodam automaticamente no GitHub a cada push.

## Simulador de propostas

O simulador (pastas `api/` e `public/`) é uma página onde se informa uma proposta e se vê a probabilidade de inadimplência, a decisão e o que mais pesou na avaliação. Ele aplica as mesmas regras de limpeza e de atributos do pacote `credair` e permite:

- **ajustar o limiar de decisão** e ver, no conjunto de teste, quantos inadimplentes o modelo identifica e quantas propostas sinaliza;
- **comparar três versões validadas do modelo**: o modelo final, o modelo sem `comprometimento_renda` e a referência só com o score.

Para testar no computador:

```bash
python api/simular.py          # depois abra http://localhost:8000
python api/simular.py 8001     # se a porta 8000 já estiver em uso
```

Se o modelo mudar, regenere os parâmetros usados pelo simulador (um teste avisa quando o arquivo fica desatualizado):

```bash
python -m credair.simulador
```
