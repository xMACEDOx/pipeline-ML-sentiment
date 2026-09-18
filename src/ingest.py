"""
Etapa 3 (Bronze): baixa o dataset publico "amazon_polarity" (Hugging Face
`datasets`) e salva em data/bronze/reviews_raw/part-*.parquet, sem NENHUMA
transformacao alem de limitar o volume (opcional).

Grava em chunks (arquivos part-*.parquet) em vez de um CSV unico: assim a
memoria usada fica limitada ao tamanho do chunk, nao ao tamanho do dataset
inteiro - da pra puxar TOTAL_SAMPLES=None (o dataset completo, ~3.6 milhoes
de linhas) sem estourar RAM. Parquet tambem e o formato que o DuckDB
(src/eda.py, src/process.py, src/build_features.py) le e escreve nativamente.

Importante: a Bronze precisa ficar fiel a fonte. Balancear classes ou
escolher registros com base no rotulo e uma decisao de modelagem, nao de
ingestao - por isso este script nao olha pro `label` pra decidir o que
entra, so limita quantos exemplos totais vem (ou nem isso, se voce pedir o
dataset inteiro).

Precisa de internet (baixa da Hugging Face Hub). Rode localmente:
    python src/ingest.py
"""
from itertools import islice
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

BRONZE_DIR = Path(__file__).resolve().parent.parent / "data" / "bronze" / "reviews_raw"
CHUNK_SIZE = 50_000  # quantas linhas ficam em memoria por vez, antes de gravar um part-*.parquet
TOTAL_SAMPLES = 50_000  # None = dataset inteiro (~3.6 milhoes de linhas); um numero = limite de linhas


def _write_chunk(rows: list[dict], part_number: int) -> None:
    table = pa.Table.from_pylist(rows, schema=pa.schema([
        ("label", pa.int64()),
        ("title", pa.string()),
        ("content", pa.string()),
    ]))
    out_path = BRONZE_DIR / f"part-{part_number:04d}.parquet"
    pq.write_table(table, out_path)


def fetch_raw_sample(total_samples: int | None = TOTAL_SAMPLES, chunk_size: int = CHUNK_SIZE) -> int:
    """Faz streaming do amazon_polarity (sem baixar o dataset inteiro de uma vez)
    e grava em disco a cada `chunk_size` exemplos, sem olhar pro rotulo pra
    decidir o que fica. Retorna o total de linhas gravadas."""
    ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=50_000)  # so pra amostra nao vir de um unico trecho do stream

    stream = islice(ds, total_samples) if total_samples is not None else ds

    buffer: list[dict] = []
    part_number = 0
    total_written = 0
    for example in stream:
        buffer.append({
            "label": example["label"],
            "title": example["title"],
            "content": example["content"],
        })
        if len(buffer) >= chunk_size:
            _write_chunk(buffer, part_number)
            total_written += len(buffer)
            part_number += 1
            buffer = []

    if buffer:
        _write_chunk(buffer, part_number)
        total_written += len(buffer)

    return total_written


def main() -> None:
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    total = fetch_raw_sample()
    n_parts = len(list(BRONZE_DIR.glob("part-*.parquet")))
    print(f"Bronze salvo em {BRONZE_DIR} ({total} linhas em {n_parts} arquivo(s) parquet)")


if __name__ == "__main__":
    main()