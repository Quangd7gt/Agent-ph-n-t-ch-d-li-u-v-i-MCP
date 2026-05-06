from __future__ import annotations
import os

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from agent.visualization import plot_bar_chart, generate_html_report

# 1. Load biến môi trường
load_dotenv()
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDATABASE = os.getenv("PGDATABASE", "olist_db")
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "")
RAW_SCHEMA = os.getenv("RAW_SCHEMA", "raw")
ANALYTICS_SCHEMA = "analytics"

# 2. Tạo engine kết nối DB
def make_engine():
    url = f"postgresql+psycopg://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
    return create_engine(url, future=True)

# 3. Agent pipeline
class OlistAgent:
    def __init__(self):

        self.engine = make_engine()
        self.gemma = None

    def get_schema(self, table: str) -> pd.DataFrame:
        """Lấy schema của bảng qua SQLAlchemy"""
        query = text(f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table;
        """)
        return pd.read_sql(query, self.engine, params={"schema": RAW_SCHEMA, "table": table})

    def run_query(self, sql: str) -> pd.DataFrame:
        """Thực thi SQL và trả về DataFrame"""
        return pd.read_sql(sql, self.engine)

    def analyze_top_products(self, year: int, month: int, top_n: int = 10) -> pd.DataFrame:
        """Ví dụ workflow: top sản phẩm bán chạy"""
        top_n = max(1, min(int(top_n), 50))
        sql = text(f"""
        SELECT p.product_category_name, COUNT(*) AS total_sales
        FROM {RAW_SCHEMA}.order_items oi
        JOIN {RAW_SCHEMA}.products p ON oi.product_id = p.product_id
        JOIN {RAW_SCHEMA}.orders o ON oi.order_id = o.order_id
        WHERE EXTRACT(MONTH FROM o.order_purchase_timestamp) = :month
          AND EXTRACT(YEAR FROM o.order_purchase_timestamp) = :year
        GROUP BY p.product_category_name
        ORDER BY total_sales DESC
        LIMIT :top_n;
        """)
        df = pd.read_sql(sql, self.engine, params={"year": year, "month": month, "top_n": top_n})
        return df

    def analyze_favorite_products(self, year: int, top_n: int = 10) -> pd.DataFrame:
        """Workflow: top sản phẩm yêu thích (điểm đánh giá cao nhất)"""
        top_n = max(1, min(int(top_n), 50))
        sql = text(f"""
        SELECT product_category_name,
               COUNT(DISTINCT order_id) as total_orders,
               ROUND(AVG(review_score_avg), 2) as avg_review_score
        FROM {ANALYTICS_SCHEMA}.fct_order_items
        WHERE EXTRACT(YEAR FROM order_purchase_timestamp) = :year
          AND review_score_avg IS NOT NULL
        GROUP BY product_category_name
        HAVING COUNT(DISTINCT order_id) > 50
        ORDER BY avg_review_score DESC
        LIMIT :top_n;
        """)
        df = pd.read_sql(sql, self.engine, params={"year": year, "top_n": top_n})
        return df

    def generate_report(self, df: pd.DataFrame, title: str, chart_type: str = "bar", y_col: str = "total_sales"):
        if chart_type != "bar":
            raise ValueError("Only bar chart reports are currently supported.")
        fig = plot_bar_chart(df, x="product_category_name", y=y_col, title=title)
        # dùng Gemma để viết phần mô tả báo cáo
        if self.gemma is None:
            from model import GemmaModel

            self.gemma = GemmaModel(model_name=os.getenv("GEMMA_MODEL", "google/gemma-2b"))
        summary = self.gemma.generate_text(f"Hãy viết đoạn tóm tắt ngắn gọn cho báo cáo {title}.")
        html = generate_html_report(df, fig, title, summary)
        return html

