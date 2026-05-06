import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDATABASE = os.getenv("PGDATABASE", "olist_db")
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "")
RAW_SCHEMA = os.getenv("RAW_SCHEMA", "raw")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
CHUNK_SIZE = 50_000
