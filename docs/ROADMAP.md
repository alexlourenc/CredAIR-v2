# Roadmap e registro de decisões

## Visão geral

O projeto evolui em três blocos. Cada um só começa quando o anterior estiver concluído.

| Bloco | Objetivo | Fases |
|---|---|---|
| **A. Análise (entregável do desafio)** | Notebook completo e executável | 1 a 12 |
| **B. Código reutilizável** | Transformar o que foi validado no notebook em um pacote Python testado | 13 |
| **C. Aplicação** | Disponibilizar o modelo para uso: simulador de propostas | 14 |

**Por que nessa ordem?** O notebook é onde **exploramos e decidimos**; o pacote é onde **consolidamos** o que já foi decidido. Criar a aplicação antes de entender os dados seria construir em cima de suposições.

## Fases

| # | Fase | Commit | Status |
|---|---|---|---|
| 1 | Entendimento do problema e do enunciado | — | ✅ |
| 2 | Ambiente: pastas, `.venv`, dependências e Git | `chore: estrutura inicial do projeto` | ✅ |
| 3 | Dataset original na camada 🥉 bronze | `chore: adiciona dataset original na camada bronze` | ✅ |
| 4 | Entendimento dos dados e da variável resposta | `feat: inicia analise com entendimento do problema` | ✅ |
| 5 | Diagnóstico de qualidade dos dados | `feat: adiciona diagnostico de qualidade dos dados` | ✅ |
| 6 | Limpeza → camada 🥈 prata | `feat: realiza limpeza dos dados e gera camada prata` | ✅ |
| 7 | Análise exploratória | `feat: adiciona analise exploratoria` | ✅ |
| 8 | Engenharia de atributos → camada 🥇 ouro | `feat: cria atributos e gera camada ouro` | ✅ |
| 9 | Preparação para modelagem | `feat: prepara dados para modelagem` | ✅ |
| 10 | Modelagem (referências + Regressão Logística) | `feat: desenvolve modelo de regressao logistica` | ✅ |
| 11 | Avaliação e interpretação | `feat: adiciona avaliacao e interpretacao do modelo` | ✅ |
| 12 | Conclusão e execução do início ao fim | `docs: adiciona conclusao da analise` | ✅ |
| 13 | Pacote `src/credair` + testes + GitHub Actions | `refactor: extrai pipeline para modulo` | ✅ |
| 14 | Simulador publicado no Vercel | `feat: cria aplicacao de simulacao` | ✅ |

## Registro de decisões

Cada decisão importante fica registrada com o contexto, a escolha feita e o motivo.

### D01 — Ferramenta: VS Code + notebook Jupyter
- **Decisão:** desenvolver no VS Code, com notebooks pela extensão Jupyter.
- **Motivo:** o mesmo editor atende à análise (notebook) e ao código da aplicação (`.py`, testes, Git integrado).

### D02 — Dataset mantido como recebido, com o nome original
- **Contexto:** o enunciado cita `credito.csv`, mas o arquivo publicado se chama `dataset.csv`.
- **Decisão:** manter o nome e o conteúdo originais em `data/bronze/`, nunca alterados.
- **Integridade (SHA-256):** `5f50b73c2ece0d0ad9baeafef685e4e555c2cabfa2e80e0e3eac38546c8d6724`.
- **Proteção:** o `.gitattributes` marca `data/bronze/**` como `-text`. Sem isso, o Git no Windows converte as quebras de linha (LF → CRLF) e o hash da cópia local muda, embora os dados sejam os mesmos.

### D03 — Dependências com versões fixadas
- **Decisão:** `requirements.txt` com `==` em cada biblioteca, dentro de um ambiente virtual (`.venv`) criado com Python 3.12 dentro da pasta do projeto.
- **Motivo:** reprodutibilidade. O enunciado exige que o notebook rode do início ao fim; versões diferentes podem mudar resultados ou quebrar o código.

### D04 — Estrutura mínima, sem pastas antecipadas
- **Decisão:** `src/`, `tests/`, `api/` e `public/` só foram criadas nas fases 13 e 14.
- **Motivo:** cada pasta surge quando houver conteúdo real para ela.

### D05 — Modelo medalhão (bronze, prata, ouro) em versão leve
- **Decisão:** bronze (dado recebido), prata (dado limpo, Parquet) e ouro (tabela de modelagem, Parquet). Prata e ouro ficam fora do Git, porque são recriadas pelo notebook.
- **Motivo:** cada camada tem uma responsabilidade, o que facilita auditar e reaproveitar as regras na aplicação. O Parquet preserva os tipos das colunas, como a ordem da escolaridade.
- **⚠️ Regra contra vazamento de dados:** a camada ouro guarda apenas transformações **determinísticas, linha a linha**. Imputação, padronização e *one-hot encoding* aprendem estatísticas dos dados e ficam no `Pipeline`, ajustadas **somente no treino**.

### D06 — Valores inválidos marcados como ausentes (P3 a P5)
- **Contexto:** 15 idades de 210 a 750 anos, 8 tempos de emprego ≥ idade e 10 rendas artificiais (R$ 100 a R$ 550, de 50 em 50).
- **Decisão:** converter em `NaN` e imputar pela **mediana dentro do pipeline**.
- **Alternativas descartadas:** remover a linha (perderia as outras colunas, que estão corretas) e dividir a idade por 10. As 15 idades são múltiplas de 10 e a divisão daria idades plausíveis em 14 casos, mas é uma suposição não verificável e um caso passaria a indicar início de trabalho aos 11 anos.

### D07 — Tempo de emprego implausível mantido, e o impacto medido (P6)
- **Contexto:** cerca de 690 registros implicam início de trabalho antes dos 14 anos, concentrados nos jovens (50% das pessoas de 18 a 25 anos). Revisando na fase 9, cerca de 425 implicam início **antes dos 10 anos**, e o tempo de emprego vai de 1 a 25 anos em todas as faixas de idade: a variável parece ter sido gerada sem relação com a idade.
- **Decisão:** manter os valores e medir o impacto.
- **Motivo:** removê-los apagaria metade dos jovens (viés contra uma faixa etária); anulá-los e imputar a mediana (13 anos) criaria novas incoerências. Na validação cruzada, o modelo com esses casos marcados como ausentes teve AUC 0,793, contra 0,794 com os valores originais: não faz diferença.
- **Limitação registrada:** o coeficiente de `tempo_emprego_anos` deve ser lido com cautela.

### D08 — Variável derivada `comprometimento_renda`
- **Construção:** `(valor_solicitado / prazo_meses) / renda_mensal`, a fração da renda comprometida com a parcela, sem juros.
- **Motivo:** explica o "paradoxo do prazo". É a 2ª variável mais discriminante sozinha (AUC 0,667), à frente das três que combina. Na validação cruzada, eleva a AUC de 0,769 para 0,793 com escolaridade e finalidade no modelo, e de 0,771 para 0,794 só com variáveis numéricas, a configuração do modelo final.
- **Descartadas:** `valor_sobre_renda` (correlação 0,69 com a escolhida e ignora o prazo) e `parcela_estimada` (mais fraca e ignora a renda).

### D09 — `estado` excluído do modelo
- **Motivo:** sem associação significativa (qui-quadrado, p = 0,60), 26 *dummies* com risco de sobreajuste e **risco ético e regulatório** de discriminação regional. Testado na validação cruzada: não melhora o modelo (AUC 0,788).

### D10 — Modelo final só com variáveis numéricas (parcimônia)
- **Motivo:** retirar `escolaridade` e `finalidade` não altera o desempenho (AUC 0,794 contra 0,793, dentro do desvio). O modelo mais simples é mais explicável e menos sujeito a sobreajuste.

### D11 — Configurações comparadas com validação cruzada; teste usado só no final
- **Motivo:** escolher o modelo pelo teste o tornaria otimista. Validação cruzada estratificada com 5 partes no treino, métrica ROC-AUC (independe do limiar).

### D12 — Limiar de decisão escolhido no treino (F1 máximo nas previsões fora da amostra)
- **Resultado:** limiar de 0,31. No teste, o recall sobe de 45,2% para 73,8% e a precisão cai de 67,7% para 55,8% (arredondados, 45% → 74% e 68% → 56%). A tabela ilustrativa do notebook, no limiar 0,30, mostra 55,0%: são limiares diferentes.
- **Motivo:** o falso negativo (aprovar um mau pagador) é o erro mais caro. O limiar final é uma **decisão de negócio** e deve ser revisto com os custos reais.

### D13 — Regras do notebook consolidadas no pacote `credair`
- **Decisão:** `limpar`, `criar_atributos`, `modelo_logistico`, `metricas` e `escolher_limiar` em `src/credair/`, com as constantes em `config.py`. O notebook importa o pacote e mostra o código das funções com `inspect.getsource`.
- **Testes:** `pytest` com dados pequenos montados à mão (um exemplo de cada problema P1 a P5) e um **teste de regressão** que refaz o pipeline a partir da bronze e confere ROC-AUC 0,781, recall 73,8% e limiar 0,31. Os testes rodam no GitHub Actions a cada push.

### D14 — Simulador publicado no Vercel, sem o scikit-learn na execução
- **Decisão:** página estática (`public/`) e uma função Python no Vercel (`api/simular.py`) que importa o pacote `credair`.
- **Por que sem o scikit-learn:** com ele, a função passaria do limite de tamanho do Vercel. Uma regressão logística treinada só precisa de medianas, médias, desvios, coeficientes e limiar, que são exportados para `api/modelos.json` (`python -m credair.simulador`). A conta feita com esses parâmetros reproduz exatamente o `predict_proba` do scikit-learn, o que um teste confere. O scikit-learn virou dependência opcional do pacote (`[project.optional-dependencies]`).

### D15 — Simulador com limiar ajustável e versões validadas, sem seleção livre de variáveis
- **Decisão:** a página permite ajustar o limiar (mostrando recall, precisão e % de propostas sinalizadas no teste) e comparar três versões já validadas na seção 7: modelo final, sem `comprometimento_renda` e só score.
- **Motivo:** o limiar é uma decisão de negócio, e o simulador deixa essa troca visível. A escolha livre de variáveis foi descartada: exigiria treinar no servidor e permitiria modelos não validados, inclusive com `estado`.

### D16 — Proposta nova é validada, e não "limpa" com as regras P3 a P5
- **Contexto (revisão externa):** o simulador reaproveitava a `limpar()` inteira. Numa proposta nova, a regra P5 trocava uma renda abaixo de R$ 600 pela mediana do treino, e quem ganhava menos recebia **menos** risco: com valor de R$ 30 mil em 6 meses, renda de R$ 650 dava 100% de risco e renda de R$ 599, 28%. Entradas impossíveis (idade 16, valor negativo, prazo `NaN`, renda infinita) também passavam sem erro.
- **Decisão:** a `limpar()` ganhou o parâmetro `marcar_invalidos` (padrão `True`, o comportamento do notebook, sem mudança). O simulador chama `limpar(..., marcar_invalidos=False)`, depois de uma `validar_proposta()` que recusa com `ValueError` (HTTP 400 na API) qualquer campo não numérico ou não finito, idade fora de 18 a 100, renda ou valor ≤ 0, tempo de emprego negativo ou ≥ idade, score fora de 0 a 1000, prazo < 1 e dívidas que não sejam um inteiro ≥ 0.
- **Por que essa opção:** é a mais simples. Na base, P3 a P5 corrigem valores **artificiais**; numa proposta, um valor fora da faixa é **erro de digitação** (recusado) ou um dado **real** (usado como informado). Uma renda baixa precisa contar como renda baixa.
- **Avisos:** os antigos avisos de P3 a P5 saíram. No lugar, o simulador avisa quando um campo válido está **fora da faixa vista no treino** (por exemplo, renda abaixo de R$ 900 ou prazo acima de 60 meses): o modelo responde, mas extrapola.
- **Sem mudança no modelo:** variáveis, pipeline, divisão e semente continuam iguais; o `api/modelos.json` regenerado é idêntico, e AUC 0,781 e limiar 0,31 se mantêm.
- **Testes:** a probabilidade nunca cai quando a renda cai (3.000, 1.000, 650, 599 e 300); renda de R$ 599 dá alto risco; o comprometimento é sempre calculado; um teste parametrizado cobre cada regra de validação, e a API devolve 400 para idade 16, valor negativo e prazo `NaN`.

## Próximos passos (fora do escopo, porque mudariam os resultados publicados)
- `comprometimento_renda` é muito assimétrica (mediana 0,11, máximo 4,02): testar logaritmo ou um teto na validação cruzada.
- Explicar o odds ratio em unidades de negócio (por exemplo, por +10 pontos percentuais de comprometimento), e não só por desvio-padrão.
- Verificar a calibração das probabilidades (curva de calibração), já que o simulador mostra a probabilidade em destaque.
- Escolher o limiar por custo (por exemplo, falso negativo = 5× falso positivo) em vez de F1.
- Intervalo de confiança da AUC de teste por *bootstrap*: o teste tem só 997 propostas.
