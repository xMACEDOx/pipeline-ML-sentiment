"""
Etapa 3 (Gold): le data/silver/reviews_clean.parquet, faz o split treino/teste
e salva em data/gold/train.parquet e data/gold/test.parquet.

O split usa hash(text) em vez de um sorteio aleatorio: e determinístico (rodar
de novo da o mesmo resultado, sem precisar guardar seed em memoria) e escala
pro dataset inteiro via DuckDB, sem carregar tudo no pandas. Aplicado sobre
qualquer volume de linhas, mantem a proporcao ~80/20 em cada classe (a mesma
ideia do `stratify` do scikit-learn, só que calculada em SQL).

A vetorizacao (TF-IDF) fica dentro do pipeline de treino (Etapa 4), e nao aqui,
pra evitar vazamento de dados do teste no vocabulario do vetorizador.

Uso:
    python src/build_features.py
"""
from pathlib import Path

import duckdb

SILVER_PATH = Path(__file__).resolve().parent.parent / "data" / "silver" / "reviews_clean.parquet"
GOLD_DIR = Path(__file__).resolve().parent.parent / "data" / "gold"
TRAIN_PATH = GOLD_DIR / "train.parquet"
TEST_PATH = GOLD_DIR / "test.parquet"
TEST_SIZE_PCT = 20  # 20% pra teste, por classe


def main() -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    silver = f"read_parquet('{SILVER_PATH}')"

    con.execute(f"""
        COPY (
            SELECT sentiment, text FROM {silver}
            WHERE abs(hash(text)) % 100 >= {TEST_SIZE_PCT}
        ) TO '{TRAIN_PATH}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            SELECT sentiment, text FROM {silver}
            WHERE abs(hash(text)) % 100 < {TEST_SIZE_PCT}
        ) TO '{TEST_PATH}' (FORMAT PARQUET)
    """)

    n_train = con.sql(f"SELECT count(*) FROM read_parquet('{TRAIN_PATH}')").fetchone()[0]
    n_test = con.sql(f"SELECT count(*) FROM read_parquet('{TEST_PATH}')").fetchone()[0]
    print(f"Gold salvo em {GOLD_DIR} — train: {n_train}, test: {n_test}")
    print("Distribuicao treino:")
    print(con.sql(f"""
        SELECT sentiment, count(*) * 1.0 / sum(count(*)) OVER () AS proporcao
        FROM read_parquet('{TRAIN_PATH}') GROUP BY sentiment
    """).df())


if __name__ == "__main__":
    main()