import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def normalize_columns(columns):
    """Chuẩn hóa tên cột về dạng chữ thường, bỏ khoảng trắng"""
    return [str(col).strip().lower() for col in columns]

def get_db_config():
    """Đọc thông tin kết nối DB từ .env"""
    return {
            "PGHOST" : os.getenv("PGHOST", "localhost"),
            "PGPORT" : int(os.getenv("PGPORT", "5432")),
            "PGDATABASE" : os.getenv("PGDATABASE", "olist_db"),
            "PGUSER" : os.getenv("PGUSER", "postgres"),
            "PGPASSWORD" : os.getenv("PGPASSWORD", ""),
            "RAW_SCHEMA": os.getenv("RAW_SCHEMA", "raw"),
            "DATA_DIR" : Path(os.getenv("DATA_DIR", "./data")),
            "CHUNK_SIZE": 50_000
            }
