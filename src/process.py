"""
Etapa 3 (Silver): le data/bronze/reviews_raw/*.parquet, limpa e normaliza o
texto e o rotulo, remove nulos/duplicatas, salva em data/silver/reviews_clean.parquet.

Tudo via SQL no DuckDB, direto sobre os arquivos parquet do Bronze - nao
carrega o dataset inteiro num DataFrame do pandas antes de processar, entao
funciona do mesmo jeito com 50 mil ou 3,6 milhoes de linhas.

Uso:
    python src/process.py
"""
from pathlib import Path

import duckdb

BRONZE_GLOB = str(Path(__file__).resolve().parent.parent / "data" / "bronze" / "reviews_raw" / "*.parquet")
SILVER_DIR = Path(__file__).resolve().parent.parent / "data" / "silver"
OUTPUT_PATH = SILVER_DIR / "reviews_clean.parquet"

HTML_TAG_RE = r"<[^>]+>"
URL_RE = r"https?://\S+"
NON_BASIC_RE = r"[^a-z0-9\s.,!?']"
WHITESPACE_RE = r"\s+"


def process() -> None:
    con = duckdb.connect()
    query = f"""
        COPY (
            WITH base AS (
                SELECT
                    label,
                    regexp_replace(
                        regexp_replace(
                            regexp_replace(
                                lower(coalesce(title, '') || ' ' || content),
                                ?, ' ', 'g'
                            ),
                            ?, ' ', 'g'
                        ),
                        ?, ' ', 'g'
                    ) AS text_raw
                FROM read_parquet('{BRONZE_GLOB}')
                WHERE content IS NOT NULL AND label IS NOT NULL
            ),
            cleaned AS (
                SELECT
                    CASE label WHEN 0 THEN 'negative' WHEN 1 THEN 'positive' END AS sentiment,
                    trim(regexp_replace(text_raw, ?, ' ', 'g')) AS text
                FROM base
            ),
            deduped AS (
                SELECT sentiment, text,
                       row_number() OVER (PARTITION BY text ORDER BY text) AS rn
                FROM cleaned
                WHERE length(text) > 0
            )
            SELECT sentiment, text FROM deduped WHERE rn = 1
        ) TO '{OUTPUT_PATH}' (FORMAT PARQUET)
    """
    con.execute(query, [HTML_TAG_RE, URL_RE, NON_BASIC_RE, WHITESPACE_RE])


def main() -> None:
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    process()

    con = duckdb.connect()
    n = con.sql(f"SELECT count(*) FROM read_parquet('{OUTPUT_PATH}')").fetchone()[0]
    print(f"Silver salvo em {OUTPUT_PATH} ({n} linhas)")
    print(con.sql(f"SELECT sentiment, count(*) AS n FROM read_parquet('{OUTPUT_PATH}') GROUP BY sentiment").df())


if __name__ == "__main__":
    main()