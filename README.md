# 💳 Detecção de Fraude em Cartão de Crédito com Machine Learning

Projeto desenvolvido para o desafio da DIO com o objetivo de construir um pipeline completo de classificação para identificar transações fraudulentas em cartões de crédito em uma base altamente desbalanceada.

---

## 🎯 O Problema e o Desafio do Desbalanceamento

A base de dados utilizada contém transações reais realizadas por titulares de cartões europeus (`creditcard.csv`), carregada diretamente via URL do Google Storage sem necessidade de armazenar arquivos pesados no repositório.

* **Variáveis:** `Time` (tempo decorrido), `Amount` (valor da transação), `V1` a `V28` (componentes principais anonimizados via PCA) e `Class` (alvo: `0` para normal e `1` para fraude).
* **Desbalanceamento:** Apenas **0,17%** das transações são fraudes, enquanto **99,83%** são legítimas.

### Por que a acurácia engana nesse cenário?
Em uma base onde 99,83% dos registros são da classe normal, um modelo ingênuo que classifique **todas** as transações como legítimas alcançará **99,83% de acurácia**, mas terá **0% de eficácia** na detecção de fraudes. Por isso, a avaliação foi direcionada para as métricas da **classe minoritária (Fraude = 1)**:
* **Recall (Revocação):** Prioridade principal — mede o percentual de fraudes reais que o modelo conseguiu capturar.
* **Precisão:** Mede quantas das transações bloqueadas como fraude eram realmente fraudulentas (evitando atrito com clientes legítimos).
* **F1-Score e PR-AUC:** Equilibram precisão e recall, sendo a curva Precisão-Recall muito mais informativa que a curva ROC nesse contexto.

---

## 🛠️ Preparação dos Dados

1. **Carregamento Direto:** Leitura do dataset via link público no Pandas, mantendo o repositório leve e reprodutível.
2. **Transformação Logarítmica (`np.log1p`):** Aplicada sobre a coluna `Amount` para reduzir a assimetria de valores extremos e lidar corretamente com transações de valor zero, gerando a feature `Log_Amount`.
3. **Divisão Estratificada (`stratify=y`):** Separação em treino (80%) e teste (20%) preservando a proporção exata de 0,17% de fraudes em ambos os conjuntos.
4. **Padronização (`StandardScaler`):** Aplicada apenas sobre `Time` e `Log_Amount` (visto que `V1` a `V28` já passaram por PCA), ajustada exclusivamente nos dados de treino para evitar vazamento de dados (*data leakage*).

---

## 📊 Comparação entre os Modelos

Foram treinados três algoritmos distintos com tratamento nativo de pesos para a classe minoritária (`class_weight='balanced'` no Random Forest e `scale_pos_weight` no XGBoost):

| Modelo | Recall (Fraude) | Precisão (Fraude) | F1-Score (Fraude) | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Regressão Logística (Baseline)** | ~0.64 | ~0.83 | ~0.72 | ~0.74 |
| **Random Forest (`balanced`)** | ~0.76 | ~0.95 | ~0.84 | ~0.85 |
| **XGBoost (`scale_pos_weight`)** | **~0.85** | **~0.76** | **~0.80** | **~0.87** |

*(Nota: Os valores exatos das métricas, matrizes de confusão e curvas ROC / Precisão-Recall estão salvos nas células executadas do notebook `.ipynb` neste repositório).*

---

## 🎚️ Ajuste de Pipeline, Limiar de Decisão e Explicabilidade (SHAP)

* **Pipeline e Otimização:** O `ColumnTransformer` com `StandardScaler` foi encapsulado junto ao `XGBClassifier` em um `Pipeline` do `scikit-learn` e otimizado com `GridSearchCV` utilizando `scoring='recall'`.
* **Limiar de Decisão (`0.5` → `0.3`):** Ao reduzir o ponto de corte de probabilidade de 50% para 30%, o modelo passou a exigir menos certeza para sinalizar uma suspeita, elevando o **Recall** para capturar fraudes mais sutis em troca de uma redução controlada na precisão.
* **Interpretabilidade com SHAP:** A análise via `TreeExplainer` evidenciou que componentes como **`V14`**, **`V17`**, **`V12`** e **`V10`** são os fatores com maior peso para empurrar a predição em direção à fraude, permitindo justificar individualmente cada bloqueio preventivo.

---

## 🔄 Diferenciais Implementados em Relação ao Roteiro Base

* **Prevenção de Data Leakage:** Aplicação da divisão estratificada (`train_test_split`) antes do ajuste do `StandardScaler` e encapsulamento via `ColumnTransformer` dentro do `Pipeline` para a validação cruzada.
* **Balanceamento no XGBoost:** Inclusão explícita do cálculo dinâmico de `scale_pos_weight` (razão entre transações normais e fraudulentas no treino) para penalizar erros na classe minoritária durante o treinamento.
* **Análise Comparativa Visual:** Plotagem lado a lado da **Curva Precisão-Recall** e da **Curva ROC** para demonstrar na prática por que a curva ROC mascara o erro em bases desbalanceadas.

---

## 📁 Estrutura do Repositório

* `deteccao_fraude_cartao.ipynb`: Notebook completo desenvolvido no Google Colab com código, tabelas de métricas, gráficos comparativos e análise SHAP salvos.
* `README.md`: Documentação técnica do projeto e análise dos resultados.
