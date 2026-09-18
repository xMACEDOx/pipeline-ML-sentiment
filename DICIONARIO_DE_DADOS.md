# Dicionário de Dados — Review Sentiment MVP

> Documento de referência técnica: dicionário de dados de cada tabela/camada do pipeline, regras de qualidade aplicadas, linhagem e classificação de sensibilidade. Complementa o `README.md` (como rodar) e `case-ml-reviews-sentimento.md` (decisões e justificativas).

## 1. Visão geral e linhagem

```
Hugging Face (fancyzhx/amazon_polarity)
        │  src/ingest.py
        ▼
  BRONZE  (data/bronze/reviews_raw/*.parquet)   — bruto, fiel à fonte
        │  src/process.py
        ▼
  SILVER  (data/silver/reviews_clean.parquet)   — limpo, normalizado
        │  src/build_features.py
        ▼
  GOLD    (data/gold/train.parquet, test.parquet) — pronto para modelagem
        │  src/train.py
        ▼
  MODELO  (models/sentiment_pipeline.joblib)     — TF-IDF + Regressão Logística
        │  api/main.py
        ▼
  API     (POST /predict)                        — consumo por outros sistemas
```

Cada seta é um script determinístico (não há transformação manual/ad-hoc em nenhuma etapa) — rodar o mesmo script sobre o mesmo dado de entrada produz o mesmo dado de saída, exceto onde explicitamente marcado como não-determinístico (ver seção 6).

## 2. Camada Bronze — `data/bronze/reviews_raw/part-*.parquet`

**Origem:** streaming do dataset `fancyzhx/amazon_polarity` (Hugging Face Hub), split `train`. **Gerado por:** `src/ingest.py`. **Regra de qualidade aplicada:** nenhuma — a Bronze é intencionalmente não tratada (ver seção 6, "Bronze fiel à fonte").

| Coluna | Tipo | Descrição | Exemplo | Nulável |
|---|---|---|---|---|
| `label` | int64 | Rótulo de sentimento original do dataset. `0` = negativo, `1` = positivo. Atribuído pelos criadores do dataset a partir da nota em estrelas do review original (1–2★ → negativo, 4–5★ → positivo; reviews de 3★ já foram excluídos na origem). | `0` | Sim (não filtrado na ingestão) |
| `title` | string | Título do review, como escrito pelo autor. | `"commercial shlock"` | Sim |
| `content` | string | Corpo do review, como escrito pelo autor — texto bruto, sem nenhuma limpeza. | `"This is a poorly written and poorly directed f..."` | Sim |

**Granularidade:** 1 linha = 1 review. **Chave:** não há chave única declarada (duplicatas são possíveis e são tratadas na Silver). **Particionamento físico:** múltiplos arquivos `part-0000.parquet`, `part-0001.parquet`, ... — cada um até `CHUNK_SIZE` linhas (hoje 50.000), não corresponde a nenhuma partição lógica dos dados (é só um limite de memória na escrita).

## 3. Camada Silver — `data/silver/reviews_clean.parquet`

**Origem:** Bronze. **Gerado por:** `src/process.py`.

| Coluna | Tipo | Descrição | Exemplo | Nulável |
|---|---|---|---|---|
| `sentiment` | string | Rótulo traduzido para texto legível. `"negative"` ou `"positive"` (mapeado 1:1 de `label` 0/1). | `"negative"` | Não (linhas com `label` nulo são descartadas) |
| `text` | string | `title` + `content` concatenados, em minúsculas, sem tags HTML, sem URLs, sem caracteres fora de `[a-z0-9 .,!?']`, espaços colapsados. | `"this is a poorly written and poorly directed f..."` | Não (linhas com `content` nulo, ou que ficaram vazias após a limpeza, são descartadas) |

**Regras de qualidade aplicadas** (todas em `src/process.py`, via SQL/DuckDB):
1. Remoção de linhas com `content` ou `label` nulos.
2. Remoção de tags HTML (regex `<[^>]+>`).
3. Remoção de URLs (regex `https?://\S+`).
4. Remoção de caracteres fora do conjunto básico (regex `[^a-z0-9\s.,!?']`).
5. Normalização de espaços em branco (múltiplos espaços/quebras de linha → um espaço).
6. Deduplicação por texto exato (mantém a primeira ocorrência de cada `text` idêntico).

**Granularidade:** 1 linha = 1 review único (pós-deduplicação). **Chave:** `text` é efetivamente única após a deduplicação.

## 4. Camada Gold — `data/gold/train.parquet` e `data/gold/test.parquet`

**Origem:** Silver. **Gerado por:** `src/build_features.py`. **Schema:** idêntico ao da Silver (`sentiment`, `text`) — nenhuma coluna nova é criada aqui; a vetorização (features numéricas de fato) acontece só dentro do treino, para não vazar vocabulário do teste (ver seção 6).

| Coluna | Tipo | Descrição |
|---|---|---|
| `sentiment` | string | Igual à Silver. |
| `text` | string | Igual à Silver. |

**Regra de split:** `abs(hash(text)) % 100 >= 20` → `train.parquet`; `< 20` → `test.parquet`. Determinístico (mesmo `text` sempre cai no mesmo lado), aproximadamente 80/20, sem sobreposição entre os dois arquivos.

## 5. Artefatos do modelo — `models/`

Não são tabelas de dados, mas fazem parte da linhagem e precisam de dicionário próprio, já que alimentam a API.

**`sentiment_pipeline.joblib`** — objeto serializado (scikit-learn `Pipeline`) contendo o vetorizador TF-IDF já ajustado e o classificador (Regressão Logística) já treinado, como uma unidade só. Gerado por `src/train.py`. **Não versionado por nome** — cada execução do `train.py` sobrescreve o arquivo anterior (ver seção 6, gap de governança).

**`train_metrics.json`** — métricas da validação cruzada do treino final.

| Campo | Tipo | Descrição |
|---|---|---|
| `trained_at` | string (ISO 8601, UTC) | Timestamp de quando o treino rodou. |
| `n_train_rows` | int | Linhas usadas no treino. |
| `n_folds` | int | Número de folds da validação cruzada (5). |
| `cv_accuracy_mean` / `cv_accuracy_std` | float | Média/desvio-padrão da acurácia entre os folds. |
| `cv_f1_macro_mean` / `cv_f1_macro_std` | float | Média/desvio-padrão do F1-macro entre os folds. |

**`model_comparison.json`** — resultado da comparação entre os três candidatos (Regressão Logística, XGBoost, rede neural), gerado por `src/compare_models.py`.

| Campo | Tipo | Descrição |
|---|---|---|
| `compared_at` | string (ISO 8601) | Timestamp da comparação. |
| `n_train_rows` / `n_valid_rows` | int | Linhas do split interno de comparação (80/20). |
| `max_features` | int | Tamanho do vocabulário TF-IDF usado na comparação. |
| `results.<modelo>.accuracy` / `.f1_macro` | float | Métricas de cada candidato no holdout de validação. |

**`evaluation_metrics.json`** — avaliação final contra o `test.parquet`, gerado por `src/evaluate.py`. É o único artefato que reflete performance esperada em produção (dado nunca visto pelo modelo).

| Campo | Tipo | Descrição |
|---|---|---|
| `evaluated_at` | string (ISO 8601) | Timestamp da avaliação. |
| `n_test_rows` | int | Linhas do `test.parquet`. |
| `accuracy` / `f1_macro` | float | Métricas gerais. |
| `per_class.<negative\|positive>.precision` / `.recall` / `.f1` | float | Métricas por classe. |
| `confusion_matrix.labels` | array[string] | Ordem das classes na matriz (`["negative", "positive"]`). |
| `confusion_matrix.matrix` | array[array[int]] | Matriz de confusão 2×2 (linhas = real, colunas = previsto). |

**`confusion_matrix.png`** — visualização da matriz de confusão acima, gerada pelo mesmo `evaluate.py`.

## 6. Regras de negócio e decisões de governança relevantes

**Bronze fiel à fonte:** a ingestão (`ingest.py`) não filtra nem balanceia por `label` — decidir o volume total (`TOTAL_SAMPLES`) é uma decisão de ingestão; decidir *quais* registros entram com base no conteúdo/rótulo seria uma decisão de modelagem, e não pertence a essa camada. Qualquer balanceamento de classes, se necessário, deve ser feito explicitamente na Gold, nunca silenciosamente na Bronze.

**Vazamento de dado (data leakage) — controlado em dois pontos:**
1. A vetorização TF-IDF nunca é ajustada (`fit`) sobre a Gold inteira — só dentro do treino, e só com a fatia de treino (a `text` da Gold permanece como texto puro, sem features numéricas, justamente para impor essa disciplina).
2. Dentro da validação cruzada (`train.py`), o TF-IDF é reajustado a cada fold, dentro do `Pipeline` — nunca ajustado uma vez só sobre o treino inteiro antes da validação.

**Não-determinismo conhecido:** o `ingest.py` usa `shuffle(seed=42, buffer_size=20_000)` sobre um *stream* — a amostra não é 100% reprodutível entre execuções (o buffer de shuffle não garante a mesma ordem/seleção exata a cada streaming). Rodar `ingest.py` de novo produz uma amostra diferente do dataset original, ainda que do mesmo tamanho e mesma distribuição esperada.

**Versionamento de modelo — gap conhecido:** hoje `train.py` sobrescreve `sentiment_pipeline.joblib` a cada execução, sem histórico. Não há como comparar o modelo em produção com uma versão anterior, nem fazer rollback. Recomendação para evolução futura: nomear artefatos com timestamp/hash (ex: `sentiment_pipeline_2026-09-15.joblib`) ou usar um registro de modelos (na AWS, o SageMaker Model Registry cumpre esse papel — ver arquitetura AWS).

## 7. Classificação de sensibilidade dos dados

O dataset (`fancyzhx/amazon_polarity`) contém apenas `label`, `title` e `content` — **não há nome, e-mail, ID de usuário ou qualquer identificador de quem escreveu o review**. Classificação: **dado público, não sensível, sem PII**.

Ressalva para evolução futura do projeto: se a ingestão um dia passar a consumir reviews reais da própria empresa (em vez do dataset público), essa classificação precisa ser revista — reviews internos costumam vir associados a e-mail/ID de cliente, o que mudaria a camada Bronze para "dado sensível" e exigiria controles adicionais (mascaramento, controle de acesso por camada, retenção definida) antes de chegar à Silver.

## 8. Contrato da API (`api/main.py`)

| Endpoint | Método | Request | Response | Descrição |
|---|---|---|---|---|
| `/health` | GET | — | `{"status": "ok", "model_loaded": bool}` | Checagem de saúde do serviço. |
| `/predict` | POST | `{"text": string}` (mínimo 1 caractere) | `{"sentiment": "positive"\|"negative", "confidence": float}` | Classifica o sentimento de um texto de review. `confidence` é a probabilidade que o modelo atribuiu à classe prevista (0 a 1). |

**Nota de nomenclatura:** o campo de entrada é `text` (mesmo nome usado nas tabelas Silver/Gold) e o de saída é `sentiment` (mesmo nome/valores usados desde a Silver) — de propósito, para manter um vocabulário único do dado bruto até a resposta da API, sem sinônimos que confundam quem for integrar com o serviço.
