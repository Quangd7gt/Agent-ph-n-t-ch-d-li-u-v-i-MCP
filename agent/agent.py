from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
import json
import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import bindparam, create_engine, text

# pyrefly: ignore [missing-import]
from agent.visualization import generate_html_report, plot_bar_chart

load_dotenv()

PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDATABASE = os.getenv("PGDATABASE", "olist_db")
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "")
RAW_SCHEMA = os.getenv("RAW_SCHEMA", "raw")
ANALYTICS_SCHEMA = os.getenv("DEFAULT_SCHEMA", "analytics")
SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

VALID_REVENUE_STATUSES = ("delivered", "shipped", "invoiced", "processing")

BUSINESS_RULES: dict[str, Any] = {
    "order_grain_definition": "analytics.fct_orders is one row per order_id.",
    "order_item_grain_definition": "analytics.fct_order_items is one row per order_id + order_item_id.",
    "revenue_definition": "Gross revenue is SUM(order_gross_value) from analytics.fct_orders for valid order statuses.",
    "repeat_customer_definition": "A repeat customer is a customer_unique_id with at least 2 distinct orders.",
    "delivery_delay_definition": "A delayed delivery is when order_delivered_customer_date > order_estimated_delivery_date.",
    "valid_revenue_statuses": list(VALID_REVENUE_STATUSES),
    "warning": "Do not sum payment_value_total from item-grain tables because it is an order-level measure.",
}


def make_engine():
    url = f"postgresql+psycopg://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
    return create_engine(url, future=True)


class OlistAgent:
    def __init__(self):
        self.engine = make_engine()
        self.gemma = None

    def load_system_prompt(self) -> str:
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

    def get_system_prompt_info(self) -> dict[str, Any]:
        return {
            "ok": True,
            "source": str(SYSTEM_PROMPT_PATH),
            "prompt_name": "olist_data_analyst",
            "resource_uri": "prompt://olist/system",
            "text": self.load_system_prompt(),
        }

    def get_business_rules(self) -> dict[str, Any]:
        return BUSINESS_RULES

    def get_schema(self, table: str) -> pd.DataFrame:
        query = text(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = :schema AND table_name = :table;
            """
        )
        return pd.read_sql(query, self.engine, params={"schema": RAW_SCHEMA, "table": table})

    def run_query(self, sql: str) -> pd.DataFrame:
        return pd.read_sql(sql, self.engine)

    def analyze_top_products(self, year: int, month: int, top_n: int = 10) -> pd.DataFrame:
        top_n = self.clamp_limit(top_n)
        query = text(
            f"""
            SELECT p.product_category_name, COUNT(*) AS total_sales
            FROM {RAW_SCHEMA}.order_items oi
            JOIN {RAW_SCHEMA}.products p ON oi.product_id = p.product_id
            JOIN {RAW_SCHEMA}.orders o ON oi.order_id = o.order_id
            WHERE EXTRACT(MONTH FROM o.order_purchase_timestamp) = :month
              AND EXTRACT(YEAR FROM o.order_purchase_timestamp) = :year
            GROUP BY p.product_category_name
            ORDER BY total_sales DESC
            LIMIT :top_n;
            """
        )
        return pd.read_sql(query, self.engine, params={"year": year, "month": month, "top_n": top_n})

    def analyze_favorite_products(self, year: int, top_n: int = 10) -> pd.DataFrame:
        top_n = self.clamp_limit(top_n)
        query = text(
            f"""
            SELECT
                product_category_name,
                COUNT(DISTINCT order_id) AS total_orders,
                ROUND(AVG(review_score_avg), 2) AS avg_review_score
            FROM {ANALYTICS_SCHEMA}.fct_order_items
            WHERE EXTRACT(YEAR FROM order_purchase_timestamp) = :year
              AND review_score_avg IS NOT NULL
            GROUP BY product_category_name
            HAVING COUNT(DISTINCT order_id) > 50
            ORDER BY avg_review_score DESC
            LIMIT :top_n;
            """
        )
        return pd.read_sql(query, self.engine, params={"year": year, "top_n": top_n})

    def analyze_revenue_by_month(self, year: int) -> pd.DataFrame:
        query = text(
            f"""
            SELECT
                date_trunc('month', order_purchase_timestamp) AS month,
                SUM(order_gross_value) AS revenue,
                COUNT(DISTINCT order_id) AS order_count
            FROM {ANALYTICS_SCHEMA}.fct_orders
            WHERE order_purchase_timestamp >= :start_date
              AND order_purchase_timestamp < :end_date
              AND order_status IN :valid_statuses
            GROUP BY 1
            ORDER BY 1;
            """
        ).bindparams(bindparam("valid_statuses", expanding=True))
        return pd.read_sql(
            query,
            self.engine,
            params={
                "start_date": f"{year}-01-01",
                "end_date": f"{year + 1}-01-01",
                "valid_statuses": VALID_REVENUE_STATUSES,
            },
        )

    def analyze_top_categories(self, start_date: str, end_date: str, limit: int = 10) -> pd.DataFrame:
        limit = self.clamp_limit(limit)
        query = text(
            f"""
            SELECT
                COALESCE(product_category_name_english, product_category_name, 'unknown') AS category,
                SUM(line_total) AS revenue,
                COUNT(DISTINCT order_id) AS order_count
            FROM {ANALYTICS_SCHEMA}.fct_order_items
            WHERE order_purchase_timestamp::date BETWEEN CAST(:start_date AS date) AND CAST(:end_date AS date)
              AND order_status IN :valid_statuses
            GROUP BY 1
            ORDER BY revenue DESC
            LIMIT :limit;
            """
        ).bindparams(bindparam("valid_statuses", expanding=True))
        return pd.read_sql(
            query,
            self.engine,
            params={
                "start_date": start_date,
                "end_date": end_date,
                "valid_statuses": VALID_REVENUE_STATUSES,
                "limit": limit,
            },
        )

    def analyze_top_products_by_orders(self, start_date: str, end_date: str, top_n: int = 10) -> pd.DataFrame:
        top_n = self.clamp_limit(top_n)
        query = text(
            f"""
            SELECT
                COALESCE(product_category_name_english, product_category_name, 'unknown') AS category,
                COUNT(DISTINCT order_id) AS order_count,
                COUNT(*) AS item_count
            FROM {ANALYTICS_SCHEMA}.fct_order_items
            WHERE order_purchase_timestamp >= CAST(:start_date AS timestamp)
              AND order_purchase_timestamp < CAST(:end_date AS timestamp)
              AND order_status IN :valid_statuses
            GROUP BY 1
            ORDER BY order_count DESC
            LIMIT :top_n;
            """
        ).bindparams(bindparam("valid_statuses", expanding=True))
        return pd.read_sql(
            query,
            self.engine,
            params={
                "start_date": start_date,
                "end_date": end_date,
                "valid_statuses": VALID_REVENUE_STATUSES,
                "top_n": top_n,
            },
        )

    def analyze_delivery_delay_summary(self, start_date: str, end_date: str) -> pd.DataFrame:
        query = text(
            f"""
            SELECT
                COUNT(*) AS delivered_orders,
                COUNT(*) FILTER (WHERE is_delayed_delivery IS TRUE) AS delayed_orders,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE is_delayed_delivery IS TRUE) / NULLIF(COUNT(*), 0),
                    2
                ) AS delayed_order_rate_pct,
                ROUND(AVG(delivery_days)::numeric, 2) AS avg_delivery_days,
                ROUND(AVG(review_score_avg)::numeric, 2) AS avg_review_score
            FROM {ANALYTICS_SCHEMA}.fct_orders
            WHERE order_purchase_timestamp::date BETWEEN CAST(:start_date AS date) AND CAST(:end_date AS date)
              AND order_status = 'delivered';
            """
        )
        return pd.read_sql(query, self.engine, params={"start_date": start_date, "end_date": end_date})

    def analyze_repeat_customer_rate(self, start_date: str, end_date: str) -> pd.DataFrame:
        query = text(
            f"""
            WITH customer_orders AS (
                SELECT
                    customer_unique_id,
                    COUNT(DISTINCT order_id) AS order_count
                FROM {ANALYTICS_SCHEMA}.fct_orders
                WHERE order_purchase_timestamp::date BETWEEN CAST(:start_date AS date) AND CAST(:end_date AS date)
                  AND order_status IN :valid_statuses
                  AND customer_unique_id IS NOT NULL
                GROUP BY 1
            )
            SELECT
                COUNT(*) AS active_customers,
                COUNT(*) FILTER (WHERE order_count >= 2) AS repeat_customers,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE order_count >= 2) / NULLIF(COUNT(*), 0),
                    2
                ) AS repeat_customer_rate_pct
            FROM customer_orders;
            """
        ).bindparams(bindparam("valid_statuses", expanding=True))
        return pd.read_sql(
            query,
            self.engine,
            params={
                "start_date": start_date,
                "end_date": end_date,
                "valid_statuses": VALID_REVENUE_STATUSES,
            },
        )

    def generate_report(self, df: pd.DataFrame, title: str, chart_type: str = "bar", y_col: str = "total_sales"):
        if chart_type != "bar":
            raise ValueError("Only bar chart reports are currently supported.")
        fig = plot_bar_chart(df, x="product_category_name", y=y_col, title=title)
        summary = self.generate_analysis(
            question=f"Viet tom tat ngan cho bao cao: {title}",
            df=df,
            context="Day la bao cao HTML, can tom tat ngan gon phan ket qua chinh.",
            max_new_tokens=int(os.getenv("GEMMA_MAX_NEW_TOKENS", "180")),
        )
        return generate_html_report(df, fig, title, summary)

    def generate_analysis(
        self,
        question: str,
        df: pd.DataFrame,
        context: str = "",
        max_new_tokens: int | None = None,
    ) -> str:
        self.ensure_gemma()
        rows = self.records(df.head(20))
        data_preview = json.dumps(rows, ensure_ascii=False, indent=2)
        prompt = (
            f"{self.load_system_prompt()}\n\n"
            "Nhiem vu hien tai: Tra loi bang tieng Viet trong 2-4 cau ngan. "
            "Chi duoc dung cac gia tri trong JSON hop le ben duoi. "
            "Khong tao bang markdown moi. "
            "Khong them danh muc, san pham, thang, doanh thu, so don hay ty le khong co trong JSON. "
            "Neu JSON chi co 1 dong, hay noi ro chi co 1 ket qua thoa dieu kien loc.\n\n"
            f"Cau hoi: {question}\n"
            f"Ngu canh: {context}\n"
            f"So dong du lieu: {len(rows)}\n"
            f"JSON hop le:\n{data_preview}\n\n"
            "Cau tra loi:"
        )
        return self.gemma.generate_text(
            prompt,
            max_new_tokens=max_new_tokens or int(os.getenv("GEMMA_MAX_NEW_TOKENS", "120")),
        )

    def write_favorite_products_report(
        self,
        year: int,
        top_n: int = 10,
        output_path: str = "report_favorite_products.html",
    ) -> dict[str, Any]:
        top_n = self.clamp_limit(top_n)
        target = self.resolve_workspace_output_path(output_path)
        df = self.analyze_favorite_products(year=year, top_n=top_n)
        html = self.generate_report(
            df,
            title=f"Favorite products report {year}",
            chart_type="bar",
            y_col="avg_review_score",
        )
        target.write_text(html, encoding="utf-8")
        return {
            "ok": True,
            "year": year,
            "top_n": top_n,
            "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
            "output_path": str(target),
            "row_count": len(df),
            "rows": self.records(df),
        }

    def write_analysis_report(
        self,
        df: pd.DataFrame,
        title: str,
        summary: str,
        output_path: str,
        x_col: str,
        y_col: str,
    ) -> dict[str, Any]:
        target = self.resolve_workspace_output_path(output_path)
        fig = plot_bar_chart(df, x=x_col, y=y_col, title=title)
        html = generate_html_report(df, fig, title, summary)
        target.write_text(html, encoding="utf-8")
        return {
            "ok": True,
            "title": title,
            "output_path": str(target),
            "row_count": len(df),
        }

    def answer_question(self, question: str, output_path: str = "") -> dict[str, Any]:
        intent = self.detect_intent(question)
        year = self.extract_year(question)
        top_n = self.extract_top_n(question)

        if self.extract_order_id(question) or "đơn hàng" in question.lower() or "order" in question.lower():
            return self.answer_order_question(question)

        if intent == "top_products_by_orders":
            start_date, end_date = self.extract_date_range(question, year)
            df = self.analyze_top_products_by_orders(start_date=start_date, end_date=end_date, top_n=top_n)
            self.generate_analysis(
                question,
                df,
                context=f"Khoang ngay {start_date} den {end_date}; top_n={top_n}.",
            )
            analysis = self.safe_top_products_by_orders_analysis(df, start_date, end_date)
            result = {
                "ok": True,
                "intent": intent,
                "start_date": start_date,
                "end_date": end_date,
                "top_n": top_n,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": self.records(df),
                "analysis": analysis,
            }
            if output_path:
                result["report"] = self.write_analysis_report(
                    df=df,
                    title=f"Top products by orders {start_date} to {end_date}",
                    summary=analysis,
                    output_path=output_path,
                    x_col="category",
                    y_col="order_count",
                )
            return result

        if intent == "favorite_products":
            df = self.analyze_favorite_products(year=year, top_n=top_n)
            self.generate_analysis(question, df, context=f"Nam {year}; top_n={top_n}.")
            analysis = self.safe_favorite_products_analysis(df, year, top_n)
            result = {
                "ok": True,
                "intent": intent,
                "year": year,
                "top_n": top_n,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": self.records(df),
                "analysis": analysis,
            }
            if output_path:
                result["report"] = self.write_analysis_report(
                    df=df,
                    title=f"Favorite products report {year}",
                    summary=analysis,
                    output_path=output_path,
                    x_col="product_category_name",
                    y_col="avg_review_score",
                )
            return result

        if intent == "top_revenue_categories":
            start_date, end_date = self.extract_date_range(question, year)
            df = self.analyze_top_categories(start_date=start_date, end_date=end_date, limit=top_n)
            self.generate_analysis(
                question,
                df,
                context=f"Khoang ngay {start_date} den {end_date}; top_n={top_n}.",
            )
            analysis = self.safe_top_categories_analysis(df, start_date, end_date)
            return {
                "ok": True,
                "intent": intent,
                "start_date": start_date,
                "end_date": end_date,
                "top_n": top_n,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": self.records(df),
                "analysis": analysis,
            }

        if intent == "revenue_by_month":
            df = self.analyze_revenue_by_month(year=year)
            self.generate_analysis(question, df, context=f"Nam {year}.")
            analysis = self.safe_revenue_by_month_analysis(df, year)
            return {
                "ok": True,
                "intent": intent,
                "year": year,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": self.records(df),
                "analysis": analysis,
            }

        if intent == "delivery_delay":
            start_date, end_date = self.extract_date_range(question, year)
            df = self.analyze_delivery_delay_summary(start_date=start_date, end_date=end_date)
            self.generate_analysis(question, df, context=f"Khoang ngay {start_date} den {end_date}.")
            analysis = self.safe_delivery_delay_analysis(df, start_date, end_date)
            return {
                "ok": True,
                "intent": intent,
                "start_date": start_date,
                "end_date": end_date,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "summary": self.records(df)[0] if not df.empty else {},
                "analysis": analysis,
            }

        if intent == "repeat_customer_rate":
            start_date, end_date = self.extract_date_range(question, year)
            df = self.analyze_repeat_customer_rate(start_date=start_date, end_date=end_date)
            self.generate_analysis(question, df, context=f"Khoang ngay {start_date} den {end_date}.")
            analysis = self.safe_repeat_customer_analysis(df, start_date, end_date)
            return {
                "ok": True,
                "intent": intent,
                "start_date": start_date,
                "end_date": end_date,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "summary": self.records(df)[0] if not df.empty else {},
                "analysis": analysis,
            }

        return {
            "ok": False,
            "error": "Chua nhan dien duoc loai cau hoi.",
            "intent": intent,
            "supported_intents": [
                "top_products_by_orders",
                "favorite_products",
                "top_revenue_categories",
                "revenue_by_month",
                "delivery_delay",
                "repeat_customer_rate",
            ],
        }


    def answer_order_question(self, question: str) -> dict[str, Any]:
        order_id = self.extract_order_id(question)

        if not order_id:
            return {
                "ok": False,
                "error": "Bạn vui lòng cung cấp order_id."
            }

        q = question.lower()

        if "thanh toán" in q or "payment" in q:
            df = self.analyze_order_payment(order_id)
            return {
            "ok": True,
            "intent": "order_payment",
            "order_id": order_id,
            "payment": self.records(df)[0] if not df.empty else {},
        }

        if "vận chuyển" in q or "giao hàng" in q or "phí ship" in q or "freight" in q:
            df = self.analyze_order_shipping(order_id)
            return {
            "ok": True,
            "intent": "order_shipping",
            "order_id": order_id,
            "shipping": self.records(df)[0] if not df.empty else {},
        }

        if "sản phẩm" in q or "product" in q:
            df = self.analyze_order_products(order_id)
            return {
            "ok": True,
            "intent": "order_products",
            "order_id": order_id,
            "products": self.records(df),
            }

        if "người bán" in q or "seller" in q:
            df = self.analyze_order_sellers(order_id)
            return {
            "ok": True,
            "intent": "order_sellers",
            "order_id": order_id,
            "sellers": self.records(df),
            }

        if "đánh giá" in q or "review" in q or "sao" in q:
            df = self.analyze_order_review(order_id)
            return {
            "ok": True,
            "intent": "order_review",
            "order_id": order_id,
            "review": self.records(df)[0] if not df.empty else {},
        }

        if "khách hàng" in q or "customer" in q:
            df = self.analyze_order_customer(order_id)
            return {
            "ok": True,
            "intent": "order_customer",
            "order_id": order_id,
            "customer": self.records(df)[0] if not df.empty else {},
        }

        return self.analyze_order_detail(order_id)


    def analyze_order_payment(self, order_id: str) -> pd.DataFrame:
        query = text(
            f"""
            SELECT
                order_id,
                payment_types,
                payment_value_total,
                payment_row_count,
                max_installments
            FROM {ANALYTICS_SCHEMA}.fct_orders
        WHERE order_id = :order_id;
        """
    )
        return pd.read_sql(query, self.engine, params={"order_id": order_id})

    def analyze_order_shipping(self, order_id: str) -> pd.DataFrame:
        query = text(
        f"""
        SELECT
            order_id,
            order_status,
            freight_total,
            order_delivered_customer_date,
            order_estimated_delivery_date,
            delivery_days,
            is_delayed_delivery
        FROM {ANALYTICS_SCHEMA}.fct_orders
        WHERE order_id = :order_id;
        """
    )
        return pd.read_sql(query, self.engine, params={"order_id": order_id})


    def analyze_order_products(self, order_id: str) -> pd.DataFrame:
        query = text(
        f"""
        SELECT
            order_id,
            order_item_id,
            product_id,
            COALESCE(product_category_name_english, product_category_name, 'unknown') AS category,
            price,
            freight_value,
            line_total
        FROM {ANALYTICS_SCHEMA}.fct_order_items
        WHERE order_id = :order_id
        ORDER BY order_item_id;
        """
    )
        return pd.read_sql(query, self.engine, params={"order_id": order_id})


    def analyze_order_sellers(self, order_id: str) -> pd.DataFrame:
        query = text(
        f"""
        SELECT DISTINCT
            order_id,
            seller_id,
            seller_city,
            seller_state
        FROM {ANALYTICS_SCHEMA}.fct_order_items
        WHERE order_id = :order_id
        ORDER BY seller_id;
        """
    )
        return pd.read_sql(query, self.engine, params={"order_id": order_id})


    def analyze_order_review(self, order_id: str) -> pd.DataFrame:
        query = text(
        f"""
        SELECT
            order_id,
            review_score_avg,
            review_count
        FROM {ANALYTICS_SCHEMA}.fct_orders
        WHERE order_id = :order_id;
        """
    )
        return pd.read_sql(query, self.engine, params={"order_id": order_id})

    def analyze_order_customer(self, order_id: str) -> pd.DataFrame:
        query = text(
        f"""
        SELECT
            order_id,
            customer_id,
            customer_unique_id,
            customer_city,
            customer_state
        FROM {ANALYTICS_SCHEMA}.fct_orders
        WHERE order_id = :order_id;
        """
    )
        return pd.read_sql(query, self.engine, params={"order_id": order_id})



    def analyze_order_detail(self, order_id: str) -> dict[str, Any]:
        customer_df = self.analyze_order_customer(order_id)
        review_df = self.analyze_order_review(order_id)
        sellers_df = self.analyze_order_sellers(order_id)
        products_df = self.analyze_order_products(order_id)
        shipping_df = self.analyze_order_shipping(order_id)
        payment_df = self.analyze_order_payment(order_id)

        if (
        customer_df.empty
        and review_df.empty
        and sellers_df.empty
        and products_df.empty
        and shipping_df.empty
        and payment_df.empty
        ):
            return {
                "ok": False,
                "order_id": order_id,
                "error": "Không tìm thấy đơn hàng."
            }

        return {
         "ok": True,
                "order_id": order_id,
            "customer": self.records(customer_df)[0] if not customer_df.empty else {},
            "review": self.records(review_df)[0] if not review_df.empty else {},
            "sellers": self.records(sellers_df),
            "products": self.records(products_df),
            "shipping": self.records(shipping_df)[0] if not shipping_df.empty else {},
            "payment": self.records(payment_df)[0] if not payment_df.empty else {},
        }

    def safe_favorite_products_analysis(self, df: pd.DataFrame, year: int, top_n: int) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có danh mục sản phẩm nào thỏa điều kiện lọc cho năm {year}."
        first = rows[0]
        if len(rows) == 1:
            return (
                f"Năm {year} chỉ có 1 danh mục thỏa điều kiện lọc: "
                f"{first['product_category_name']} với điểm review trung bình "
                f"{first['avg_review_score']} trên {first['total_orders']} đơn hàng."
            )
        return (
            f"Năm {year}, danh mục đứng đầu trong top {top_n} là "
            f"{first['product_category_name']} với điểm review trung bình "
            f"{first['avg_review_score']} trên {first['total_orders']} đơn hàng. "
            f"Kết quả trả về {len(rows)} danh mục thỏa điều kiện lọc."
        )

    def safe_top_products_by_orders_analysis(self, df: pd.DataFrame, start_date: str, end_date: str) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có dữ liệu sản phẩm trong giai đoạn {start_date} đến {end_date}."
        first = rows[0]
        return (
            f"Trong giai đoạn {start_date} đến {end_date}, danh mục bán chạy nhất là "
            f"{first['category']} với {first['order_count']} đơn hàng và {first['item_count']} dòng sản phẩm. "
            f"Kết quả trả về {len(rows)} danh mục theo số đơn hàng giảm dần."
        )

    def safe_top_categories_analysis(self, df: pd.DataFrame, start_date: str, end_date: str) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có danh mục doanh thu nào trong giai đoạn {start_date} đến {end_date}."
        first = rows[0]
        return (
            f"Trong giai đoạn {start_date} đến {end_date}, danh mục có doanh thu cao nhất là "
            f"{first['category']} với doanh thu {first['revenue']} và {first['order_count']} đơn hàng. "
            f"Kết quả trả về {len(rows)} danh mục."
        )

    def safe_revenue_by_month_analysis(self, df: pd.DataFrame, year: int) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có dữ liệu doanh thu theo tháng cho năm {year}."
        highest = max(rows, key=lambda row: row["revenue"] or 0)
        lowest = min(rows, key=lambda row: row["revenue"] or 0)
        return (
            f"Năm {year} có {len(rows)} tháng có dữ liệu doanh thu. "
            f"Tháng cao nhất là {highest['month']} với doanh thu {highest['revenue']} "
            f"và tháng thấp nhất là {lowest['month']} với doanh thu {lowest['revenue']}."
        )

    def safe_delivery_delay_analysis(self, df: pd.DataFrame, start_date: str, end_date: str) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có dữ liệu giao hàng trong giai đoạn {start_date} đến {end_date}."
        row = rows[0]
        return (
            f"Từ {start_date} đến {end_date}, có {row['delivered_orders']} đơn đã giao, "
            f"trong đó {row['delayed_orders']} đơn giao trễ, tương ứng "
            f"{row['delayed_order_rate_pct']}%. Thời gian giao trung bình là "
            f"{row['avg_delivery_days']} ngày và review trung bình là {row['avg_review_score']}."
        )

    def safe_repeat_customer_analysis(self, df: pd.DataFrame, start_date: str, end_date: str) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có dữ liệu khách hàng trong giai đoạn {start_date} đến {end_date}."
        row = rows[0]
        return (
            f"Từ {start_date} đến {end_date}, có {row['active_customers']} khách hàng hoạt động, "
            f"trong đó {row['repeat_customers']} khách hàng quay lại. "
            f"Tỷ lệ khách hàng quay lại là {row['repeat_customer_rate_pct']}%."
        )

    def ensure_gemma(self) -> None:
        if self.gemma is None:
            from model import GemmaModel

            self.gemma = GemmaModel(model_name=os.getenv("GEMMA_MODEL", "google/gemma-2b-it"))

    def gemma_runtime_status(self) -> dict[str, Any]:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            torch_version = torch.__version__
        except Exception as exc:
            return {"ok": False, "error": f"Unable to import torch: {exc}"}

        token = os.getenv("HUGGINGFACE_TOKEN") or ""
        return {
            "ok": True,
            "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
            "device": os.getenv("GEMMA_DEVICE", "auto"),
            "max_new_tokens": int(os.getenv("GEMMA_MAX_NEW_TOKENS", "180")),
            "has_huggingface_token": bool(token),
            "torch_version": torch_version,
            "cuda_available": cuda_available,
            "resolved_device": "cuda" if cuda_available else "cpu",
        }

    @staticmethod
    def clamp_limit(limit: int, maximum: int = 50) -> int:
        return max(1, min(int(limit), maximum))

    @staticmethod
    def detect_intent(question: str) -> str:
        q = question.lower()
        if any(term in q for term in ["giao cham", "giao chậm", "tre", "trễ", "delay"]):
            return "delivery_delay"
        if any(
            term in q
            for term in [
                "khach quay lai",
                "khach hang quay lai",
                "khách quay lại",
                "khách hàng quay lại",
                "repeat",
            ]
        ):
            return "repeat_customer_rate"
        if any(term in q for term in ["theo thang", "theo tháng", "monthly", "month"]):
            return "revenue_by_month"
        if any(term in q for term in ["doanh thu", "revenue"]):
            return "top_revenue_categories"
        if any(term in q for term in ["yeu thich", "yêu thích", "review", "danh gia", "đánh giá"]):
            return "favorite_products"
        if any(
            term in q
            for term in [
                "top san pham",
                "top sản phẩm",
                "ban chay",
                "bán chạy",
                "nhieu don",
                "nhiều đơn",
                "san pham",
                "sản phẩm",
            ]
        ):
            return "top_products_by_orders"
        return "unknown"

    @staticmethod
    def extract_year(question: str, default: int = 2018) -> int:
        match = re.search(r"\b(20\d{2})\b", question)
        return int(match.group(1)) if match else default

    @staticmethod
    def extract_top_n(question: str, default: int = 10) -> int:
        match = re.search(r"\btop\s*(\d{1,2})\b", question.lower())
        if not match:
            match = re.search(r"\b(\d{1,2})\s+(?:danh muc|danh mục|san pham|sản phẩm)", question.lower())
        return OlistAgent.clamp_limit(int(match.group(1)) if match else default)

    @staticmethod
    def extract_date_range(question: str, year: int) -> tuple[str, str]:
        q = question.lower()
        if "quy 1" in q or "quý 1" in q or "q1" in q:
            return f"{year}-01-01", f"{year}-03-31"
        if "quy 2" in q or "quý 2" in q or "q2" in q:
            return f"{year}-04-01", f"{year}-06-30"
        if "quy 3" in q or "quý 3" in q or "q3" in q:
            return f"{year}-07-01", f"{year}-09-30"
        if "quy 4" in q or "quý 4" in q or "q4" in q:
            return f"{year}-10-01", f"{year}-12-31"
        return f"{year}-01-01", f"{year}-12-31"

    @staticmethod
    def resolve_workspace_output_path(output_path: str) -> Path:
        target = Path(output_path)
        if not target.is_absolute():
            target = WORKSPACE_ROOT / target
        target = target.resolve()

        try:
            target.relative_to(WORKSPACE_ROOT)
        except ValueError as exc:
            raise ValueError("output_path must stay inside the MCP project directory.") from exc
        if target.suffix.lower() != ".html":
            raise ValueError("output_path must be an .html file.")
        return target

    def records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        safe_df = df.astype(object).where(pd.notna(df), None)
        return [
            {key: self.json_value(value) for key, value in record.items()}
            for record in safe_df.to_dict(orient="records")
        ]

    @staticmethod
    def json_value(value: Any) -> Any:
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, (datetime, date, pd.Timestamp)):
            return value.isoformat()
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value
    @staticmethod
    def extract_order_id(question: str) -> str | None:
        match = re.search(r"\b[a-f0-9]{32}\b", question.lower())
        return match.group(0) if match else None
