from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
from agent import config

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDATABASE = os.getenv("PGDATABASE", "olist_db")
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "")
RAW_SCHEMA = os.getenv("RAW_SCHEMA", "raw")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
CHUNK_SIZE = 50_000

FILE_TABLE_MAP: dict[str, str] = {
    "olist_orders_dataset.csv": "orders",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_products_dataset.csv": "products",
    "olist_customers_dataset.csv": "customers",
    "olist_sellers_dataset.csv": "sellers",
    "product_category_name_translation.csv": "category_translation",
    "olist_geolocation_dataset.csv": "geolocation",
}

PARSE_DATES: dict[str, list[str]] = {
    "olist_orders_dataset.csv": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "olist_order_items_dataset.csv": ["shipping_limit_date"],
    "olist_order_reviews_dataset.csv": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
}


def normalize_columns(columns: Iterable[str]) -> list[str]:
    return [str(col).strip().lower() for col in columns]


def make_engine():
    url = f"postgresql+psycopg://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
    return create_engine(url, future=True)


def ensure_schema(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS analytics"))


def load_file(engine, csv_path: Path, table_name: str) -> None:
    filename = csv_path.name
    parse_dates = PARSE_DATES.get(filename, [])
    mode = "replace"

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, parse_dates=parse_dates):
        chunk.columns = normalize_columns(chunk.columns)
        chunk.to_sql(
            table_name,
            con=engine,
            schema=RAW_SCHEMA,
            if_exists=mode,
            index=False,
            method="multi",
            chunksize=1000,
        )
        mode = "append"
        print(f"Loaded chunk into {RAW_SCHEMA}.{table_name} from {filename}")


def main() -> None:
    engine = make_engine()
    ensure_schema(engine)

    missing = [name for name in FILE_TABLE_MAP if not (DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing CSV files in data directory: " + ", ".join(missing)
        )

    for filename, table_name in FILE_TABLE_MAP.items():
        load_file(engine, DATA_DIR / filename, table_name)

    print("All Olist CSV files were loaded into Postgres raw schema.")
    print("Next step: run sql/001_create_analytics_views.sql")


if __name__ == "__main__":
    main()
