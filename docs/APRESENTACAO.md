# Roteiro da apresentação: CredAIR, análise de risco de crédito

**Formato:** reunião individual no Microsoft Teams, com cerca de 15 minutos de fala.
**Fio condutor:** contar a **história da análise**, deixando claro o **porquê de cada decisão**. O enunciado avalia se as decisões foram entendidas, e não apenas os resultados.

Os gráficos citados estão em `reports/figures/`. As decisões (D01 a D15) estão em [`ROADMAP.md`](ROADMAP.md).

**Antes de começar, deixe abertos:** o simulador ([cred-air-v2.vercel.app](https://cred-air-v2.vercel.app)), o notebook já executado, o repositório no GitHub (página inicial e aba *Actions*) e os gráficos.

---

## 1. O problema (1 min)
- A CredAIR quer usar o histórico de propostas para identificar as de **maior risco de inadimplência**.
- **Variável resposta:** `inadimplente` (0/1): **classificação binária supervisionada**.
- **Os erros não custam o mesmo:** aprovar um mau pagador (falso negativo) custa o capital emprestado; recusar um bom pagador (falso positivo) custa os juros e o cliente.
- 📊 `01_distribuicao_alvo.png`: 34,4% de inadimplentes. Quem chuta sempre "adimplente" acerta 66% sem aprender nada: **a acurácia engana**.

## 2. Como organizei o projeto (1 min)
- Fases com **um commit por etapa** (Conventional Commits).
- **Medalhão leve:** 🥉 bronze (como recebido, com hash SHA-256 registrado), 🥈 prata (limpo) e 🥇 ouro (pronto para o modelo).
- **Regra contra vazamento de dados:** a camada ouro só tem transformações linha a linha; imputação e escala ficam no `Pipeline`, ajustadas **só no treino**.

## 3. Qualidade dos dados (2 min)
- **Diagnosticar antes de tratar.** Problemas encontrados:
  - 19 duplicatas e 101 UFs com espaços (`' MG '` virava um estado diferente de `'MG'`: 85 "estados" em vez de 27);
  - 15 idades impossíveis (210 a 750), 8 tempos de emprego ≥ idade e 10 rendas artificiais (R$ 100 a R$ 550, de 50 em 50).
- **Decisão: marcar como ausente** e imputar no pipeline, em vez de apagar a linha ou "adivinhar" o valor.
  - *Por que não dividir a idade por 10?* Todas são múltiplas de 10 e a divisão parece resolver, mas é uma suposição que não dá para verificar, e um caso passaria a indicar início de trabalho aos 11 anos.
- **Revisão que fiz no meio do caminho:** cerca de 425 registros implicam começar a trabalhar **antes dos 10 anos**, e o tempo de emprego parece ter sido gerado sem relação com a idade. Mantive os valores (removê-los apagaria metade dos jovens) e **medi o impacto**: AUC 0,793 com a correção contra 0,794 sem ela. Ficou registrado como limitação.

## 4. O que os dados mostram (2 min)
- 📊 `03_taxa_por_quintil.png`: **score** (56% → 16% de inadimplência) e **renda** (53% → 21%) são os sinais mais fortes.
- 📊 `04_taxa_por_categoria.png`: mais **dívidas ativas**, mais risco (27% → 61%). **Prazos curtos** têm mais inadimplência (54% em 6 meses): um resultado contraintuitivo.
- **Estado, escolaridade e finalidade:** o qui-quadrado não mostrou associação (p de 0,23 a 0,60). O estado varia de 29% a 42%, mas isso é compatível com o acaso.
- *Ressalva:* associação não é causalidade.

## 5. A variável que criei (2 min), o ponto alto da análise
- **Hipótese:** o prazo curto não é arriscado por si mesmo; ele gera uma **parcela pesada**.
- **`comprometimento_renda` = (valor ÷ prazo) ÷ renda**, a fração da renda que vai para a parcela.
- 📊 `06_comprometimento_renda.png`: a inadimplência sobe de 21% para **71%** do menor para o maior decil.
- É a 2ª variável mais forte sozinha (AUC 0,667), à frente das três que combina, e **eleva a AUC do modelo de 0,769 para 0,793**.

## 6. Modelagem (2 min)
- Treino/teste 80/20 **estratificado**; o teste é usado uma única vez, no final.
- **Referências:** classe majoritária (AUC 0,50) e regra só com score (0,68 na validação cruzada).
- **Validação cruzada (5 partes)** para comparar as configurações:
  - incluir `estado` não ajuda (0,788) e ainda tem **risco ético**: ficou fora;
  - escolaridade e finalidade não acrescentam nada: ficou o **modelo mais simples** (parcimônia), com 8 variáveis numéricas.

## 7. Resultados (2 min)
- 📊 `07_curva_roc.png`: **AUC de 0,781 no teste**, contra 0,695 da regra só com score. Próxima da validação cruzada (0,794): o modelo **generaliza**.
- 📊 `08_matriz_confusao_05.png`: no limiar 0,5, o modelo pega só **45%** dos inadimplentes (155 de 343).
- 📊 `09_precisao_recall_limiar.png` e `10_matriz_confusao_sugerido.png`: com o **limiar 0,31**, escolhido no treino, pega **74%** (253 de 343). Os calotes aprovados caem de 188 para 90, e os alertas falsos sobem de 74 para 200.
- **O limiar é uma decisão de negócio:** a sugestão é mandar os casos sinalizados para análise manual ou ajustar prazo e valor, em vez de recusar automaticamente.
- 📊 `11_coeficientes.png` (odds ratio por desvio-padrão): comprometimento da renda **2,78×**, score **0,44×**, dívidas **1,60×**.
- *O prazo "troca de sinal" no modelo:* descontado o comprometimento, prazo longo aumenta levemente o risco. O efeito do prazo curto era, na verdade, o efeito da parcela pesada.

## 8. Do notebook ao produto: demonstração ao vivo (2 min)
- **Pacote `credair` com 45 testes**, rodando no GitHub Actions a cada push. Mostre o ✅ na aba *Actions*.
- **Simulador publicado:** a página aplica as **mesmas regras** do notebook, sem copiar código. Para caber no limite do Vercel, o simulador não usa o scikit-learn: exporta só os parâmetros do modelo, e um teste prova que a conta dá exatamente o mesmo resultado.
- **Roteiro da demonstração:**
  1. Simule a proposta padrão: o modelo final dá **27%, baixo risco**. No comparativo, as duas versões mais simples dão **alerta** (33%): só o modelo final percebe que a parcela é leve (9% da renda) e evita recusar um bom cliente.
  2. Troque para **renda R$ 3.000, valor R$ 20.000 e prazo 6 meses**: a parcela consome **111% da renda**, e o modelo final dá **93% de risco**.
  3. Mostre o comparativo: **sem a minha variável, 56%; só com o score, 33%**. Só o modelo com o comprometimento percebe o tamanho do risco.
  4. **Arraste o limiar** e mostre o recall e a % de propostas sinalizadas mudando: a troca entre pegar caloteiros e recusar bons clientes.
  5. Digite **idade 250**: aparece o aviso da limpeza, a mesma regra aplicada à base.

## 9. Conclusões, limitações e cuidados (1 min)
- **Ação prática:** para clientes no limite, oferecer **prazo maior ou valor menor** reduz o comprometimento e o risco. **Teto de comprometimento** na política de crédito: a inadimplência chega a 94% quando a parcela supera a renda.
- **Limitações:** dados sintéticos; tempo de emprego pouco confiável; viés de seleção (só propostas já analisadas); parcela sem juros; nenhuma dimensão temporal.
- **Cuidados em produção:** o modelo apoia, não decide sozinho; justiça e LGPD; monitorar *drift* e retreinar; aplicar as mesmas regras do treino.

---

## Perguntas prováveis e respostas curtas

| Pergunta | Resposta |
|---|---|
| Por que não usou acurácia? | Com 34% de inadimplentes, quem chuta sempre "adimplente" tem 66% de acurácia. A AUC e o recall mostram se o modelo realmente separa as classes. |
| O que é a AUC? | A chance de, sorteando um inadimplente e um bom pagador, o modelo dar ao inadimplente a nota de risco maior. 0,5 é chute e 1,0 é perfeito; o nosso tem 0,78. |
| Por que imputar com a mediana? | É robusta a valores extremos, e só 33 valores estavam ausentes (menos de 1% da base). |
| Por que padronizar? | A regularização penaliza os coeficientes; sem escala, variáveis com números grandes, como a renda, seriam penalizadas de forma desigual. A escala também torna os coeficientes comparáveis. |
| O que é vazamento de dados? | Quando informação do teste "vaza" para o treino, como calcular a mediana com a base toda. As métricas ficam otimistas demais. O `Pipeline` evita isso. |
| Como o modelo encontra os pesos? | Ajustes sucessivos até minimizar o erro de previsão no treino (máxima verossimilhança), com uma regularização que evita pesos exagerados. O limiar não é aprendido: fui eu que escolhi, por validação cruzada. |
| Por que excluir o estado? | Não tem associação significativa, piora levemente o modelo e pode **discriminar regiões**. |
| Por que o limiar de 0,31? | Maximiza o F1 nas previsões de validação cruzada do treino. Prioriza pegar mais inadimplentes, que é o erro mais caro, mas a decisão final é do negócio. |
| O modelo está sobreajustado? | Não. A AUC no teste (0,781) é próxima da validação cruzada (0,794), e o desvio entre as partes é pequeno (±0,016). |
| Por que não corrigiu o tempo de emprego? | Testei a correção: AUC 0,793 contra 0,794. Não muda o resultado, e removê-lo apagaria metade dos jovens. Ficou registrado como limitação. |
| Como garante que o simulador usa o mesmo modelo? | As regras ficam no pacote `credair`, usado pelo notebook, pelos testes e pelo simulador. Um teste compara a probabilidade do simulador com a do scikit-learn em 12 combinações. |
| Por que o simulador não deixa escolher as variáveis? | Exigiria treinar no servidor (acima do limite do Vercel) e permitiria modelos não validados, inclusive com o estado. Ele oferece as três versões validadas no notebook. |
