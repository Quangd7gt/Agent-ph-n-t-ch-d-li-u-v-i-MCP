from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
import json
import os
from pathlib import Path
import re
from time import perf_counter
import unicodedata
from typing import Any

from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import bindparam, create_engine, text

# pyrefly: ignore [missing-import]
from agent.question_understanding import QuestionUnderstanding, QuestionUnderstandingEngine
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
    "order_grain_definition": "analytics.fct_orders có 1 dòng cho mỗi order_id.",
    "order_item_grain_definition": "analytics.fct_order_items có 1 dòng cho mỗi order_id + order_item_id.",
    "revenue_definition": "Doanh thu gộp là SUM(order_gross_value) từ analytics.fct_orders với các trạng thái đơn hàng hợp lệ.",
    "repeat_customer_definition": "Khách hàng quay lại là customer_unique_id có ít nhất 2 đơn hàng khác nhau.",
    "delivery_delay_definition": "Đơn hàng giao trễ là đơn có order_delivered_customer_date > order_estimated_delivery_date.",
    "valid_revenue_statuses": list(VALID_REVENUE_STATUSES),
    "warning": "Không cộng payment_value_total từ bảng item-grain vì đây là chỉ số cấp đơn hàng.",
}


def make_engine():
    url = f"postgresql+psycopg://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
    return create_engine(url, future=True)


class Agent:
    def __init__(self):
        self.engine = make_engine()
        self.gemma = None
        self._revenue_by_month_cache: dict[int, dict[str, Any]] = {}

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

    def answer_business_rule_question(self, question: str) -> dict[str, Any]:
        q = self.normalize_text(question)
        matched_rules: dict[str, Any] = {}
        answer_parts: list[str] = []
        intents: list[str] = []

        if any(term in q for term in ["doanh thu", "revenue", "gross"]):
            intents.append("revenue_rule")
            matched_rules["revenue_definition"] = BUSINESS_RULES["revenue_definition"]
            matched_rules["valid_revenue_statuses"] = BUSINESS_RULES["valid_revenue_statuses"]
            matched_rules["warning"] = BUSINESS_RULES["warning"]
            answer_parts.append(
                "Doanh thu được tính bằng SUM(order_gross_value) từ analytics.fct_orders "
                "cho các trạng thái hợp lệ: delivered, shipped, invoiced, processing."
            )

        if any(term in q for term in ["grain", "cap do", "muc du lieu", "bang", "table", "fct_orders", "fct_order_items"]):
            intents.append("table_grain_rule")
            matched_rules["order_grain_definition"] = BUSINESS_RULES["order_grain_definition"]
            matched_rules["order_item_grain_definition"] = BUSINESS_RULES["order_item_grain_definition"]
            answer_parts.append(
                "analytics.fct_orders có grain 1 dòng cho mỗi order_id; "
                "analytics.fct_order_items có grain 1 dòng cho mỗi order_id + order_item_id."
            )

        if any(term in q for term in ["khach hang quay lai", "repeat", "returning customer", "mua lai"]):
            intents.append("repeat_customer_rule")
            matched_rules["repeat_customer_definition"] = BUSINESS_RULES["repeat_customer_definition"]
            answer_parts.append(
                "Khách hàng quay lại là customer_unique_id có ít nhất 2 đơn hàng khác nhau."
            )

        if any(term in q for term in ["giao hang tre", "giao tre", "delay", "delayed", "tre"]):
            intents.append("delivery_delay_rule")
            matched_rules["delivery_delay_definition"] = BUSINESS_RULES["delivery_delay_definition"]
            answer_parts.append(
                "Đơn hàng được coi là giao trễ khi order_delivered_customer_date lớn hơn "
                "order_estimated_delivery_date."
            )

        if any(term in q for term in ["payment", "thanh toan", "payment_value_total"]):
            intents.append("payment_warning_rule")
            matched_rules["warning"] = BUSINESS_RULES["warning"]
            answer_parts.append(
                "Không cộng payment_value_total từ bảng item-grain vì đây là chỉ số cấp đơn hàng "
                "và có thể bị lặp số liệu."
            )

        if not matched_rules:
            return {
                "ok": True,
                "intent": "all_business_rules",
                "question": question,
                "rules": BUSINESS_RULES,
                "answer": "Không tìm thấy nhóm rule cụ thể, trả về toàn bộ business rules Agent đang sử dụng.",
            }

        return {
            "ok": True,
            "intent": "+".join(dict.fromkeys(intents)),
            "question": question,
            "rules": matched_rules,
            "answer": " ".join(answer_parts),
        }

    def get_schema(self, table: str) -> pd.DataFrame:
        query = text(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = :schema AND table_name = :table;
            """
        )
        return pd.read_sql(query, self.engine, params={"schema": RAW_SCHEMA, "table": table})

    def list_data_tables(self) -> pd.DataFrame:
        query = text(
            """
            SELECT
                table_schema,
                table_name,
                table_type
            FROM information_schema.tables
            WHERE table_schema IN :schemas
            ORDER BY table_schema, table_name;
            """
        ).bindparams(bindparam("schemas", expanding=True))
        return pd.read_sql(
            query,
            self.engine,
            params={"schemas": [ANALYTICS_SCHEMA, RAW_SCHEMA]},
        )

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
                COALESCE(product_category_name_english, product_category_name, 'unknown') AS product_category_name,
                COUNT(DISTINCT order_id) AS total_orders,
                ROUND(AVG(review_score_avg), 2) AS avg_review_score
            FROM {ANALYTICS_SCHEMA}.fct_order_items
            WHERE EXTRACT(YEAR FROM order_purchase_timestamp) = :year
              AND order_status IN :valid_statuses
              AND review_score_avg IS NOT NULL
            GROUP BY COALESCE(product_category_name_english, product_category_name, 'unknown')
            HAVING COUNT(DISTINCT order_id) > 50
            ORDER BY avg_review_score DESC, total_orders DESC
            LIMIT :top_n;
            """
        ).bindparams(bindparam("valid_statuses", expanding=True))
        return pd.read_sql(
            query,
            self.engine,
            params={"year": year, "top_n": top_n, "valid_statuses": VALID_REVENUE_STATUSES},
        )

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

    def analyze_revenue_by_month_fast(self, year: int) -> dict[str, Any]:
        if year < 2000 or year > 2100:
            return {"ok": False, "error": "year must be between 2000 and 2100."}
        cached = self._revenue_by_month_cache.get(year)
        if cached is not None:
            return {**cached, "cache_hit": True}

        start_time = perf_counter()
        query = text(
            f"""
            WITH valid_orders AS (
                SELECT
                    order_id,
                    date_trunc('month', order_purchase_timestamp) AS month_start
                FROM {RAW_SCHEMA}.orders
                WHERE order_purchase_timestamp >= :start_date
                  AND order_purchase_timestamp < :end_date
                  AND order_status = ANY(:valid_statuses)
            )
            SELECT
                to_char(vo.month_start, 'YYYY-MM') AS month,
                ROUND(COALESCE(SUM(oi.price + oi.freight_value), 0)::numeric, 2)::float8 AS revenue,
                COUNT(DISTINCT vo.order_id)::int AS order_count
            FROM valid_orders vo
            LEFT JOIN {RAW_SCHEMA}.order_items oi
              ON oi.order_id = vo.order_id
            GROUP BY vo.month_start
            ORDER BY vo.month_start;
            """
        )
        with self.engine.connect() as conn:
            rows = conn.execute(
                query,
                {
                    "start_date": f"{year}-01-01",
                    "end_date": f"{year + 1}-01-01",
                    "valid_statuses": list(VALID_REVENUE_STATUSES),
                },
            ).mappings().all()

        elapsed_ms = round((perf_counter() - start_time) * 1000, 2)
        result = {
            "ok": True,
            "year": year,
            "source": f"{RAW_SCHEMA}.orders + {RAW_SCHEMA}.order_items",
            "valid_statuses": list(VALID_REVENUE_STATUSES),
            "row_count": len(rows),
            "elapsed_ms": elapsed_ms,
            "cache_hit": False,
            "rows": [
                {key: self.json_value(value) for key, value in row.items()}
                for row in rows
            ],
        }
        self._revenue_by_month_cache[year] = result
        return result

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

    def analyze_top_revenue_products(self, start_date: str, end_date: str, top_n: int = 10) -> pd.DataFrame:
        top_n = self.clamp_limit(top_n)
        query = text(
            f"""
            SELECT
                product_id,
                COALESCE(product_category_name_english, product_category_name, 'unknown') AS category,
                SUM(line_total) AS revenue,
                COUNT(DISTINCT order_id) AS order_count,
                COUNT(*) AS item_count
            FROM {ANALYTICS_SCHEMA}.fct_order_items
            WHERE order_purchase_timestamp::date BETWEEN CAST(:start_date AS date) AND CAST(:end_date AS date)
              AND order_status IN :valid_statuses
            GROUP BY product_id, COALESCE(product_category_name_english, product_category_name, 'unknown')
            ORDER BY revenue DESC
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

    def analyze_top_categories_by_orders(self, start_date: str, end_date: str, top_n: int = 10) -> pd.DataFrame:
        top_n = self.clamp_limit(top_n)
        query = text(
            f"""
            SELECT
                COALESCE(product_category_name_english, product_category_name, 'unknown') AS category,
                COUNT(DISTINCT order_id) AS order_count,
                COUNT(*) AS item_count
            FROM {ANALYTICS_SCHEMA}.fct_order_items
            WHERE order_purchase_timestamp >= CAST(:start_date AS timestamp)
              AND order_purchase_timestamp < CAST(:end_date AS timestamp) + INTERVAL '1 day'
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

    def analyze_top_products_by_orders(self, start_date: str, end_date: str, top_n: int = 10) -> pd.DataFrame:
        top_n = self.clamp_limit(top_n)
        query = text(
            f"""
            SELECT
                product_id,
                COALESCE(product_category_name_english, product_category_name, 'unknown') AS category,
                COUNT(DISTINCT order_id) AS order_count,
                COUNT(*) AS item_count
            FROM {ANALYTICS_SCHEMA}.fct_order_items
            WHERE order_purchase_timestamp >= CAST(:start_date AS timestamp)
              AND order_purchase_timestamp < CAST(:end_date AS timestamp) + INTERVAL '1 day'
              AND order_status IN :valid_statuses
            GROUP BY product_id, COALESCE(product_category_name_english, product_category_name, 'unknown')
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

    def analyze_category_rankings(self, year: int) -> pd.DataFrame:
        query = text(
            f"""
            WITH category_metrics AS (
                SELECT
                    COALESCE(product_category_name_english, product_category_name, 'unknown') AS category,
                    ROUND(SUM(line_total)::numeric, 2) AS revenue,
                    COUNT(DISTINCT order_id)::int AS order_count,
                    ROUND(AVG(review_score_avg)::numeric, 2) AS avg_review_score
                FROM {ANALYTICS_SCHEMA}.fct_order_items
                WHERE EXTRACT(YEAR FROM order_purchase_timestamp) = :year
                  AND order_status IN :valid_statuses
                  AND review_score_avg IS NOT NULL
                GROUP BY 1
                HAVING COUNT(DISTINCT order_id) > 50
            )
            SELECT
                category,
                revenue,
                order_count,
                avg_review_score,
                RANK() OVER (ORDER BY avg_review_score DESC, order_count DESC) AS review_rank,
                RANK() OVER (ORDER BY revenue DESC) AS revenue_rank,
                RANK() OVER (ORDER BY order_count DESC) AS order_rank
            FROM category_metrics
            ORDER BY review_rank, revenue_rank, order_rank;
            """
        ).bindparams(bindparam("valid_statuses", expanding=True))
        return pd.read_sql(
            query,
            self.engine,
            params={"year": year, "valid_statuses": VALID_REVENUE_STATUSES},
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

    def analyze_category_performance(self, year: int, top_n: int = 5) -> pd.DataFrame:
        top_n = self.clamp_limit(top_n)
        query = text(
            f"""
            WITH category_metrics AS (
                SELECT
                    COALESCE(product_category_name_english, product_category_name, 'unknown') AS category,
                    SUM(line_total)::numeric AS revenue,
                    COUNT(DISTINCT order_id) AS order_count,
                    ROUND(AVG(review_score_avg)::numeric, 2) AS avg_review_score
                FROM {ANALYTICS_SCHEMA}.fct_order_items
                WHERE order_purchase_timestamp >= :start_date
                  AND order_purchase_timestamp < :end_date
                  AND order_status IN :valid_statuses
                GROUP BY 1
            ),
            scored AS (
                SELECT
                    *,
                    CASE
                        WHEN MAX(revenue) OVER () = MIN(revenue) OVER () THEN 1.0
                        ELSE (revenue - MIN(revenue) OVER ()) / NULLIF(MAX(revenue) OVER () - MIN(revenue) OVER (), 0)
                    END AS revenue_score,
                    CASE
                        WHEN MAX(order_count) OVER () = MIN(order_count) OVER () THEN 1.0
                        ELSE (order_count - MIN(order_count) OVER ())::numeric
                             / NULLIF(MAX(order_count) OVER () - MIN(order_count) OVER (), 0)
                    END AS order_score,
                    CASE
                        WHEN MAX(COALESCE(avg_review_score, 0)) OVER () = MIN(COALESCE(avg_review_score, 0)) OVER () THEN 1.0
                        ELSE (COALESCE(avg_review_score, 0) - MIN(COALESCE(avg_review_score, 0)) OVER ())
                             / NULLIF(MAX(COALESCE(avg_review_score, 0)) OVER () - MIN(COALESCE(avg_review_score, 0)) OVER (), 0)
                    END AS review_score
                FROM category_metrics
            )
            SELECT
                category,
                ROUND(revenue, 2) AS revenue,
                order_count,
                avg_review_score,
                ROUND((0.4 * revenue_score + 0.4 * order_score + 0.2 * review_score)::numeric, 4) AS performance_score
            FROM scored
            ORDER BY performance_score DESC, revenue DESC
            LIMIT :top_n;
            """
        ).bindparams(bindparam("valid_statuses", expanding=True))
        return pd.read_sql(
            query,
            self.engine,
            params={
                "start_date": f"{year}-01-01",
                "end_date": f"{year + 1}-01-01",
                "valid_statuses": VALID_REVENUE_STATUSES,
                "top_n": top_n,
            },
        )

    def analyze_monthly_revenue_extremes(self, year: int) -> pd.DataFrame:
        query = text(
            f"""
            SELECT
                to_char(date_trunc('month', order_purchase_timestamp), 'YYYY-MM') AS month,
                ROUND(SUM(order_gross_value)::numeric, 2) AS revenue,
                COUNT(DISTINCT order_id) AS order_count,
                ROUND(AVG(order_gross_value)::numeric, 2) AS avg_order_value
            FROM {ANALYTICS_SCHEMA}.fct_orders
            WHERE order_purchase_timestamp >= :start_date
              AND order_purchase_timestamp < :end_date
              AND order_status IN :valid_statuses
            GROUP BY 1, date_trunc('month', order_purchase_timestamp)
            ORDER BY date_trunc('month', order_purchase_timestamp);
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

    def analyze_quarterly_revenue(self, year: int, quarters: list[int] | None = None) -> pd.DataFrame:
        valid_quarters = tuple(q for q in (quarters or [1, 2, 3, 4]) if 1 <= int(q) <= 4)
        if not valid_quarters:
            valid_quarters = (1, 2, 3, 4)
        query = text(
            f"""
            SELECT
                EXTRACT(QUARTER FROM order_purchase_timestamp)::int AS quarter,
                ROUND(SUM(order_gross_value)::numeric, 2) AS revenue,
                COUNT(DISTINCT order_id) AS order_count,
                ROUND(AVG(order_gross_value)::numeric, 2) AS avg_order_value
            FROM {ANALYTICS_SCHEMA}.fct_orders
            WHERE order_purchase_timestamp >= :start_date
              AND order_purchase_timestamp < :end_date
              AND EXTRACT(QUARTER FROM order_purchase_timestamp)::int IN :quarters
              AND order_status IN :valid_statuses
            GROUP BY 1
            ORDER BY 1;
            """
        ).bindparams(
            bindparam("valid_statuses", expanding=True),
            bindparam("quarters", expanding=True),
        )
        return pd.read_sql(
            query,
            self.engine,
            params={
                "start_date": f"{year}-01-01",
                "end_date": f"{year + 1}-01-01",
                "valid_statuses": VALID_REVENUE_STATUSES,
                "quarters": valid_quarters,
            },
        )

    def analyze_delivery_review_impact(self, start_date: str, end_date: str) -> pd.DataFrame:
        query = text(
            f"""
            WITH grouped AS (
                SELECT
                    CASE
                        WHEN is_delayed_delivery IS TRUE THEN 'delayed'
                        ELSE 'on_time'
                    END AS delivery_status,
                    COUNT(*) AS order_count,
                    ROUND(SUM(order_gross_value)::numeric, 2) AS total_revenue,
                    ROUND(AVG(order_gross_value)::numeric, 2) AS avg_order_value,
                    ROUND(AVG(review_score_avg)::numeric, 2) AS avg_review_score,
                    ROUND(AVG(delivery_days)::numeric, 2) AS avg_delivery_days
                FROM {ANALYTICS_SCHEMA}.fct_orders
                WHERE order_purchase_timestamp::date BETWEEN CAST(:start_date AS date) AND CAST(:end_date AS date)
                  AND order_status = 'delivered'
                  AND review_score_avg IS NOT NULL
                  AND order_gross_value IS NOT NULL
                GROUP BY 1
            )
            SELECT
                delivery_status,
                order_count,
                total_revenue,
                ROUND(
                    100.0 * total_revenue / NULLIF(SUM(total_revenue) OVER (), 0),
                    2
                ) AS revenue_share_pct,
                avg_order_value,
                avg_review_score,
                avg_delivery_days
            FROM grouped
            ORDER BY delivery_status;
            """
        )
        return pd.read_sql(query, self.engine, params={"start_date": start_date, "end_date": end_date})

    def analyze_state_market_importance(self, year: int, top_n: int = 10) -> pd.DataFrame:
        top_n = self.clamp_limit(top_n)
        query = text(
            f"""
            WITH state_metrics AS (
                SELECT
                    customer_state AS state,
                    SUM(order_gross_value)::numeric AS revenue,
                    COUNT(DISTINCT order_id) AS order_count,
                    COUNT(DISTINCT customer_unique_id) AS customer_count,
                    ROUND(AVG(review_score_avg)::numeric, 2) AS avg_review_score
                FROM {ANALYTICS_SCHEMA}.fct_orders
                WHERE order_purchase_timestamp >= :start_date
                  AND order_purchase_timestamp < :end_date
                  AND order_status IN :valid_statuses
                  AND customer_state IS NOT NULL
                GROUP BY 1
            ),
            scored AS (
                SELECT
                    *,
                    CASE
                        WHEN MAX(revenue) OVER () = MIN(revenue) OVER () THEN 1.0
                        ELSE (revenue - MIN(revenue) OVER ()) / NULLIF(MAX(revenue) OVER () - MIN(revenue) OVER (), 0)
                    END AS revenue_score,
                    CASE
                        WHEN MAX(order_count) OVER () = MIN(order_count) OVER () THEN 1.0
                        ELSE (order_count - MIN(order_count) OVER ())::numeric
                             / NULLIF(MAX(order_count) OVER () - MIN(order_count) OVER (), 0)
                    END AS order_score,
                    CASE
                        WHEN MAX(customer_count) OVER () = MIN(customer_count) OVER () THEN 1.0
                        ELSE (customer_count - MIN(customer_count) OVER ())::numeric
                             / NULLIF(MAX(customer_count) OVER () - MIN(customer_count) OVER (), 0)
                    END AS customer_score
                FROM state_metrics
            )
            SELECT
                state,
                ROUND(revenue, 2) AS revenue,
                order_count,
                customer_count,
                avg_review_score,
                ROUND((0.4 * revenue_score + 0.3 * order_score + 0.3 * customer_score)::numeric, 4) AS importance_score
            FROM scored
            ORDER BY importance_score DESC, revenue DESC
            LIMIT :top_n;
            """
        ).bindparams(bindparam("valid_statuses", expanding=True))
        return pd.read_sql(
            query,
            self.engine,
            params={
                "start_date": f"{year}-01-01",
                "end_date": f"{year + 1}-01-01",
                "valid_statuses": VALID_REVENUE_STATUSES,
                "top_n": top_n,
            },
        )

    def analyze_customer_experience_priority_categories(self, year: int, top_n: int = 10) -> pd.DataFrame:
        top_n = self.clamp_limit(top_n)
        query = text(
            f"""
            WITH category_metrics AS (
                SELECT
                    COALESCE(product_category_name_english, product_category_name, 'unknown') AS category,
                    SUM(line_total)::numeric AS revenue,
                    COUNT(DISTINCT order_id) AS order_count,
                    ROUND(AVG(review_score_avg)::numeric, 2) AS avg_review_score,
                    ROUND(
                        100.0 * COUNT(DISTINCT order_id) FILTER (
                            WHERE order_delivered_customer_date > order_estimated_delivery_date
                        ) / NULLIF(COUNT(DISTINCT order_id), 0),
                        2
                    ) AS delayed_order_rate_pct
                FROM {ANALYTICS_SCHEMA}.fct_order_items
                WHERE order_purchase_timestamp >= :start_date
                  AND order_purchase_timestamp < :end_date
                  AND order_status IN :valid_statuses
                GROUP BY 1
            ),
            scored AS (
                SELECT
                    *,
                    CASE
                        WHEN MAX(order_count) OVER () = MIN(order_count) OVER () THEN 1.0
                        ELSE (order_count - MIN(order_count) OVER ())::numeric
                             / NULLIF(MAX(order_count) OVER () - MIN(order_count) OVER (), 0)
                    END AS volume_score,
                    CASE
                        WHEN MAX(COALESCE(avg_review_score, 0)) OVER () = MIN(COALESCE(avg_review_score, 0)) OVER () THEN 0.0
                        ELSE (MAX(COALESCE(avg_review_score, 0)) OVER () - COALESCE(avg_review_score, 0))
                             / NULLIF(MAX(COALESCE(avg_review_score, 0)) OVER () - MIN(COALESCE(avg_review_score, 0)) OVER (), 0)
                    END AS low_review_score,
                    CASE
                        WHEN MAX(COALESCE(delayed_order_rate_pct, 0)) OVER () = MIN(COALESCE(delayed_order_rate_pct, 0)) OVER () THEN 0.0
                        ELSE (COALESCE(delayed_order_rate_pct, 0) - MIN(COALESCE(delayed_order_rate_pct, 0)) OVER ())
                             / NULLIF(MAX(COALESCE(delayed_order_rate_pct, 0)) OVER () - MIN(COALESCE(delayed_order_rate_pct, 0)) OVER (), 0)
                    END AS delay_score
                FROM category_metrics
            )
            SELECT
                category,
                ROUND(revenue, 2) AS revenue,
                order_count,
                avg_review_score,
                delayed_order_rate_pct,
                ROUND((0.4 * volume_score + 0.3 * low_review_score + 0.3 * delay_score)::numeric, 4) AS priority_score
            FROM scored
            ORDER BY priority_score DESC, order_count DESC
            LIMIT :top_n;
            """
        ).bindparams(bindparam("valid_statuses", expanding=True))
        return pd.read_sql(
            query,
            self.engine,
            params={
                "start_date": f"{year}-01-01",
                "end_date": f"{year + 1}-01-01",
                "valid_statuses": VALID_REVENUE_STATUSES,
                "top_n": top_n,
            },
        )

    def analyze_customer_experience_priority_states(self, year: int, top_n: int = 10) -> pd.DataFrame:
        top_n = self.clamp_limit(top_n)
        query = text(
            f"""
            WITH state_metrics AS (
                SELECT
                    customer_state AS state,
                    SUM(order_gross_value)::numeric AS revenue,
                    COUNT(DISTINCT order_id) AS order_count,
                    ROUND(AVG(review_score_avg)::numeric, 2) AS avg_review_score,
                    ROUND(
                        100.0 * COUNT(*) FILTER (WHERE is_delayed_delivery IS TRUE) / NULLIF(COUNT(*), 0),
                        2
                    ) AS delayed_order_rate_pct
                FROM {ANALYTICS_SCHEMA}.fct_orders
                WHERE order_purchase_timestamp >= :start_date
                  AND order_purchase_timestamp < :end_date
                  AND order_status IN :valid_statuses
                  AND customer_state IS NOT NULL
                GROUP BY 1
            ),
            scored AS (
                SELECT
                    *,
                    CASE
                        WHEN MAX(order_count) OVER () = MIN(order_count) OVER () THEN 1.0
                        ELSE (order_count - MIN(order_count) OVER ())::numeric
                             / NULLIF(MAX(order_count) OVER () - MIN(order_count) OVER (), 0)
                    END AS volume_score,
                    CASE
                        WHEN MAX(COALESCE(avg_review_score, 0)) OVER () = MIN(COALESCE(avg_review_score, 0)) OVER () THEN 0.0
                        ELSE (MAX(COALESCE(avg_review_score, 0)) OVER () - COALESCE(avg_review_score, 0))
                             / NULLIF(MAX(COALESCE(avg_review_score, 0)) OVER () - MIN(COALESCE(avg_review_score, 0)) OVER (), 0)
                    END AS low_review_score,
                    CASE
                        WHEN MAX(COALESCE(delayed_order_rate_pct, 0)) OVER () = MIN(COALESCE(delayed_order_rate_pct, 0)) OVER () THEN 0.0
                        ELSE (COALESCE(delayed_order_rate_pct, 0) - MIN(COALESCE(delayed_order_rate_pct, 0)) OVER ())
                             / NULLIF(MAX(COALESCE(delayed_order_rate_pct, 0)) OVER () - MIN(COALESCE(delayed_order_rate_pct, 0)) OVER (), 0)
                    END AS delay_score
                FROM state_metrics
            )
            SELECT
                state,
                ROUND(revenue, 2) AS revenue,
                order_count,
                avg_review_score,
                delayed_order_rate_pct,
                ROUND((0.4 * volume_score + 0.3 * low_review_score + 0.3 * delay_score)::numeric, 4) AS priority_score
            FROM scored
            ORDER BY priority_score DESC, order_count DESC
            LIMIT :top_n;
            """
        ).bindparams(bindparam("valid_statuses", expanding=True))
        return pd.read_sql(
            query,
            self.engine,
            params={
                "start_date": f"{year}-01-01",
                "end_date": f"{year + 1}-01-01",
                "valid_statuses": VALID_REVENUE_STATUSES,
                "top_n": top_n,
            },
        )

    def analyze_customer_experience_priorities(self, year: int, top_n: int = 10) -> dict[str, Any]:
        category_df = self.analyze_customer_experience_priority_categories(year, top_n)
        state_df = self.analyze_customer_experience_priority_states(year, top_n)
        return {
            "year": year,
            "top_n": top_n,
            "categories": self.records(category_df),
            "states": self.records(state_df),
        }

    def generate_report(self, df: pd.DataFrame, title: str, chart_type: str = "bar", y_col: str = "total_sales"):
        if chart_type != "bar":
            raise ValueError("Only bar chart reports are currently supported.")
        fig = plot_bar_chart(df, x="product_category_name", y=y_col, title=title)
        summary = self.generate_analysis(
            question=f"Viết tóm tắt ngắn cho báo cáo: {title}",
            df=df,
            context="Đây là báo cáo HTML, cần tóm tắt ngắn gọn phần kết quả chính.",
            max_new_tokens=int(os.getenv("GEMMA_MAX_NEW_TOKENS", "180")),
        )
        return generate_html_report(df, fig, title, summary)

    @staticmethod
    def analysis_system_instructions() -> str:
        return (
            "Bạn là agent phân tích dữ liệu Olist. "
            "Nhiệm vụ duy nhất là trả lời câu hỏi hiện tại dựa trên JSON được cung cấp. "
            "Không liệt kê câu hỏi mẫu, không copy prompt, không tạo Cảnh/Kịch bản, "
            "không tự thêm loại phân tích khác, không bịa số liệu."
        )

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
            f"{self.analysis_system_instructions()}\n\n"
            "Nhiệm vụ hiện tại: Trả lời bằng tiếng Việt có dấu trong 2-4 câu ngắn. "
            "Tuyệt đối không trả lời tiếng Việt không dấu. "
            "Chỉ được dùng các giá trị trong JSON hợp lệ bên dưới. "
            "Không tạo bảng markdown mới. "
            "Không chia câu trả lời thành Cảnh/Kịch bản và không liệt kê các loại phân tích khác. "
            "Chỉ trả lời đúng câu hỏi hiện tại theo dữ liệu JSON hiện tại. "
            "Không thêm danh mục, sản phẩm, tháng, doanh thu, số đơn hay tỷ lệ không có trong JSON. "
            "Nếu JSON chỉ có 1 dòng, hãy nói rõ chỉ có 1 kết quả thỏa điều kiện lọc.\n\n"
            f"Câu hỏi: {question}\n"
            f"Ngữ cảnh: {context}\n"
            f"Số dòng dữ liệu: {len(rows)}\n"
            f"JSON hợp lệ:\n{data_preview}\n\n"
            "Câu trả lời:"
        )
        return self.gemma.generate_text(
            prompt,
            max_new_tokens=max_new_tokens or int(os.getenv("GEMMA_MAX_NEW_TOKENS", "120")),
        )

    def generate_analysis_from_payload(
        self,
        question: str,
        payload: dict[str, Any],
        context: str = "",
        max_new_tokens: int | None = None,
    ) -> str:
        self.ensure_gemma()
        data_preview = json.dumps(payload, ensure_ascii=False, indent=2)
        prompt = (
            f"{self.analysis_system_instructions()}\n\n"
            "Nhiệm vụ hiện tại: Trả lời bằng tiếng Việt có dấu trong 3-6 câu ngắn. "
            "Tuyệt đối không trả lời tiếng Việt không dấu. "
            "Chỉ được dùng số liệu trong JSON hợp lệ bên dưới. "
            "Không chia câu trả lời thành Cảnh/Kịch bản và không liệt kê các loại phân tích khác. "
            "Chỉ trả lời đúng câu hỏi hiện tại theo dữ liệu JSON hiện tại. "
            "Không thêm chỉ số, bang, danh mục, bang/khu vực, doanh thu, số đơn, tỷ lệ hay review không có trong JSON. "
            "Nếu đưa ra khuyến nghị, phải gắn với các chỉ số có trong JSON.\n\n"
            f"Câu hỏi: {question}\n"
            f"Ngữ cảnh: {context}\n"
            f"JSON hợp lệ:\n{data_preview}\n\n"
            "Câu trả lời:"
        )
        return self.gemma.generate_text(
            prompt,
            max_new_tokens=max_new_tokens or int(os.getenv("GEMMA_MAX_NEW_TOKENS", "160")),
        )

    def generate_analysis_or_fallback(
        self,
        question: str,
        df: pd.DataFrame,
        context: str,
        fallback: str,
        max_new_tokens: int | None = None,
    ) -> str:
        try:
            analysis = self.generate_analysis(question, df, context=context, max_new_tokens=max_new_tokens)
            return fallback if self.generated_analysis_should_fallback(question, analysis, df) else analysis
        except Exception:
            return fallback

    def generate_payload_analysis_or_fallback(
        self,
        question: str,
        payload: dict[str, Any],
        context: str,
        fallback: str,
        max_new_tokens: int | None = None,
    ) -> str:
        try:
            analysis = self.generate_analysis_from_payload(
                question,
                payload,
                context=context,
                max_new_tokens=max_new_tokens,
            )
            return fallback if self.generated_payload_analysis_should_fallback(question, analysis, payload) else analysis
        except Exception:
            return fallback

    @staticmethod
    def generated_analysis_should_fallback(question: str, analysis: str, df: pd.DataFrame) -> bool:
        if not analysis or not analysis.strip():
            return True
        normalized = Agent.normalize_text(analysis)
        if Agent.generated_text_looks_prompt_copied(normalized):
            return True

        columns = set(df.columns)
        question_normalized = Agent.normalize_text(question)
        if {"state", "importance_score"}.issubset(columns):
            rows = df.head(1).astype(object).where(pd.notna(df.head(1)), None).to_dict(orient="records")
            top_state = str(rows[0].get("state") or "").lower() if rows else ""
            if top_state and top_state not in normalized:
                return True
            if "danh muc" in normalized and "danh muc" not in question_normalized:
                return True
        return False

    @staticmethod
    def generated_payload_analysis_should_fallback(question: str, analysis: str, payload: dict[str, Any]) -> bool:
        if not analysis or not analysis.strip():
            return True
        normalized = Agent.normalize_text(analysis)
        if Agent.generated_text_looks_prompt_copied(normalized):
            return True
        return False

    @staticmethod
    def generated_text_looks_prompt_copied(normalized_text: str) -> bool:
        bad_patterns = [
            r"\b(?:canh|kich ban)\s*\d+\b",
            r"\bcau hoi phan tich du lieu\b",
            r"\bcac nhom cau hoi\b",
            r"\bphan tich 5 danh muc san pham tot nhat\b",
            r"\bso sanh cac thang co doanh thu cao nhat\b",
            r"\bso sanh doanh thu theo quy\b",
        ]
        return any(re.search(pattern, normalized_text) for pattern in bad_patterns)

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

    def understand_question(self, question: str) -> QuestionUnderstanding:
        enable_gemma = os.getenv("QUESTION_UNDERSTANDING_WITH_GEMMA", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        engine = QuestionUnderstandingEngine(
            llm_generate=self.generate_question_understanding if enable_gemma else None,
            enable_llm=enable_gemma,
        )
        return engine.understand(question)

    def generate_question_understanding(self, prompt: str) -> str:
        self.ensure_gemma()
        return self.gemma.generate_text(
            prompt,
            max_new_tokens=int(os.getenv("QUESTION_UNDERSTANDING_MAX_NEW_TOKENS", "260")),
        )

    @staticmethod
    def attach_understanding(
        result: dict[str, Any],
        understanding: QuestionUnderstanding,
    ) -> dict[str, Any]:
        result.setdefault("question_understanding", understanding.to_dict())
        return result

    @staticmethod
    def date_range_from_understanding(
        understanding: QuestionUnderstanding,
        question: str,
        year: int,
    ) -> tuple[str, str]:
        if understanding.start_date and understanding.end_date:
            return understanding.start_date, understanding.end_date
        return Agent.extract_date_range(question, year)

    def answer_question(self, question: str, output_path: str = "") -> dict[str, Any]:
        understanding = self.understand_question(question)
        intent = understanding.intent
        normalized_question = understanding.normalized_question

        def finalize(result: dict[str, Any]) -> dict[str, Any]:
            return self.attach_understanding(result, understanding)

        if intent == "schema_tables":
            df = self.list_data_tables()
            rows = self.records(df)
            return finalize({
                "ok": True,
                "intent": intent,
                "database": PGDATABASE,
                "schemas": [ANALYTICS_SCHEMA, RAW_SCHEMA],
                "rows": rows,
                "analysis": (
                    f"Database {PGDATABASE} hiện có {len(rows)} bảng/view dữ liệu "
                    f"trong các schema {ANALYTICS_SCHEMA} và {RAW_SCHEMA}."
                ),
                "analysis_source": "rule_based",
            })

        if understanding.needs_clarification:
            return finalize({
                "ok": False,
                "intent": intent,
                "needs_clarification": True,
                "reason": understanding.reason or "Cần bổ sung thông tin trước khi phân tích.",
                "clarifying_question": understanding.clarifying_question or "Bạn vui lòng bổ sung thông tin cần thiết.",
                "examples": [
                    "doanh thu năm 2018 thế nào?",
                    "doanh thu quý 1 năm 2018 thế nào?",
                    "đơn e481f51cbdc54678b7cc49136f2d6af7 được thanh toán bằng phương thức nào?",
                ],
            })

        if understanding.is_order_detail:
            return finalize(self.answer_order_question(question))

        year = understanding.year or self.extract_year(question)
        top_n = understanding.top_n or self.extract_top_n(question)
        gemma_source = "gemma_with_safe_fallback"

        if intent == "category_performance":
            df = self.analyze_category_performance(year=year, top_n=top_n)
            safe_summary = self.safe_category_performance_analysis(df, year)
            analysis = self.generate_analysis_or_fallback(
                question=question,
                df=df,
                context=(
                    f"Dữ liệu hiệu suất danh mục năm {year} đã được tính bằng SQL. "
                    "Hãy phân tích dựa trên revenue, order_count, avg_review_score và performance_score. "
                    "Chỉ dùng số liệu trong JSON; không bịa thêm danh mục hoặc doanh thu."
                ),
                fallback=safe_summary,
                max_new_tokens=240,
            )
            return finalize({
                "ok": True,
                "intent": intent,
                "year": year,
                "top_n": top_n,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": self.records(df),
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            })

        if intent == "monthly_revenue_extremes":
            df = self.analyze_monthly_revenue_extremes(year=year)
            safe_summary = self.safe_monthly_revenue_extremes_analysis(df, year)
            analysis = self.generate_analysis_or_fallback(
                question=question,
                df=df,
                context=(
                    f"Dữ liệu doanh thu theo tháng năm {year} đã được tính bằng SQL. "
                    "Hãy chỉ ra tháng cao nhất, tháng thấp nhất, mức chênh lệch và ý nghĩa kinh doanh. "
                    "Nếu có tháng ít đơn bất thường, hãy nhắc cần kiểm tra độ đầy đủ dữ liệu."
                ),
                fallback=safe_summary,
                max_new_tokens=240,
            )
            return finalize({
                "ok": True,
                "intent": intent,
                "year": year,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": self.records(df),
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            })

        if intent == "quarterly_revenue_comparison":
            quarters = self.extract_quarters(question)
            df = self.analyze_quarterly_revenue(year=year, quarters=quarters)
            safe_summary = self.safe_quarterly_revenue_analysis(df, year)
            analysis = self.generate_analysis_or_fallback(
                question=question,
                df=df,
                context=(
                    f"Dữ liệu doanh thu theo quý năm {year} đã được tính bằng SQL. "
                    "Hãy so sánh doanh thu, số đơn và giá trị đơn trung bình giữa các quý trong JSON."
                ),
                fallback=safe_summary,
                max_new_tokens=220,
            )
            return finalize({
                "ok": True,
                "intent": intent,
                "year": year,
                "quarters": quarters or [1, 2, 3, 4],
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": self.records(df),
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            })

        if intent == "delivery_review_impact":
            start_date, end_date = self.date_range_from_understanding(understanding, question, year)
            df = self.analyze_delivery_review_impact(start_date=start_date, end_date=end_date)
            safe_summary = self.safe_delivery_review_impact_analysis(df, start_date, end_date)
            analysis = self.generate_analysis_or_fallback(
                question=question,
                df=df,
                context=(
                    f"Dữ liệu đã được tính toán chính xác bằng SQL cho giai đoạn {start_date} đến {end_date}. "
                    "Hãy phân tích giao hàng trễ ảnh hưởng/liên quan thế nào đến doanh thu, tỷ trọng doanh thu, "
                    "giá trị đơn hàng trung bình, số đơn, thời gian giao hàng và review. "
                    "Chỉ dùng số liệu trong JSON. Nếu chỉ là so sánh mô tả thì nói rõ không khẳng định nhân quả tuyệt đối."
                ),
                fallback=safe_summary,
                max_new_tokens=280,
            )
            return finalize({
                "ok": True,
                "intent": intent,
                "start_date": start_date,
                "end_date": end_date,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": self.records(df),
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            })

        if intent == "state_market_importance":
            df = self.analyze_state_market_importance(year=year, top_n=top_n)
            safe_summary = self.safe_state_market_importance_analysis(df, year)
            analysis = self.generate_analysis_or_fallback(
                question=question,
                df=df,
                context=(
                    f"Dữ liệu mức độ quan trọng thị trường theo bang năm {year} đã được tính bằng SQL. "
                    "Hãy phân tích theo revenue, order_count, customer_count, avg_review_score và importance_score."
                ),
                fallback=safe_summary,
                max_new_tokens=240,
            )
            return finalize({
                "ok": True,
                "intent": intent,
                "year": year,
                "top_n": top_n,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": self.records(df),
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            })

        if intent == "state_revenue_low_review":
            df = self.analyze_customer_experience_priority_states(year=year, top_n=top_n)
            safe_summary = self.safe_state_revenue_low_review_analysis(df, year)
            analysis = self.generate_analysis_or_fallback(
                question=question,
                df=df,
                context=(
                    f"Dữ liệu bang có doanh thu cao nhưng review thấp năm {year} đã được tính bằng SQL. "
                    "Hãy phân tích nơi cần ưu tiên cải thiện dựa trên revenue, order_count, avg_review_score, "
                    "delayed_order_rate_pct và priority_score."
                ),
                fallback=safe_summary,
                max_new_tokens=240,
            )
            return finalize({
                "ok": True,
                "intent": intent,
                "year": year,
                "top_n": top_n,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": self.records(df),
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            })

        if intent == "customer_experience_priority":
            payload = self.analyze_customer_experience_priorities(year=year, top_n=top_n)
            safe_summary = self.safe_customer_experience_priority_analysis(payload)
            analysis = self.generate_payload_analysis_or_fallback(
                question=question,
                payload=payload,
                context=(
                    f"Dữ liệu ưu tiên cải thiện trải nghiệm khách hàng năm {year} gồm hai nhóm: danh mục và bang. "
                    "Hãy phân tích ngắn gọn điểm cần ưu tiên, dựa trên priority_score, review và tỷ lệ giao trễ."
                ),
                fallback=safe_summary,
                max_new_tokens=260,
            )
            return finalize({
                "ok": True,
                "intent": intent,
                "year": year,
                "top_n": top_n,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "data": payload,
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            })

        if intent == "favorite_vs_revenue":
            df = self.analyze_category_rankings(year=year)
            rows = self.category_rank_comparison_rows(df, ["review_rank", "revenue_rank"], top_n=top_n)
            safe_summary = self.safe_favorite_vs_revenue_analysis(df, year, top_n)
            payload = {"year": year, "top_n": top_n, "rows": rows}
            analysis = self.generate_payload_analysis_or_fallback(
                question=question,
                payload=payload,
                context=(
                    f"Dữ liệu so sánh danh mục được yêu thích và danh mục doanh thu cao năm {year}. "
                    "Hãy trả lời có trùng nhau hay không và giải thích bằng các hạng review_rank, revenue_rank, order_rank."
                ),
                fallback=safe_summary,
                max_new_tokens=260,
            )
            return finalize({
                "ok": True,
                "intent": intent,
                "year": year,
                "top_n": top_n,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": rows,
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            })

        if intent == "orders_vs_review":
            df = self.analyze_category_rankings(year=year)
            rows = self.category_rank_comparison_rows(df, ["order_rank", "review_rank"], top_n=top_n)
            safe_summary = self.safe_orders_vs_review_analysis(df, year, top_n)
            payload = {"year": year, "top_n": top_n, "rows": rows}
            analysis = self.generate_payload_analysis_or_fallback(
                question=question,
                payload=payload,
                context=(
                    f"Dữ liệu so sánh danh mục nhiều đơn và danh mục review tốt năm {year}. "
                    "Hãy trả lời hai nhóm có trùng nhau không và giải thích bằng số đơn, review trung bình và các thứ hạng."
                ),
                fallback=safe_summary,
                max_new_tokens=240,
            )
            return finalize({
                "ok": True,
                "intent": intent,
                "year": year,
                "top_n": top_n,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": rows,
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            })

        if intent == "top_products_by_orders":
            start_date, end_date = self.date_range_from_understanding(understanding, question, year)
            df = self.analyze_top_products_by_orders(start_date=start_date, end_date=end_date, top_n=top_n)
            safe_summary = self.safe_top_products_by_orders_analysis(df, start_date, end_date)
            analysis = self.generate_analysis_or_fallback(
                question=question,
                df=df,
                context=(
                    f"Dữ liệu top sản phẩm theo số đơn trong giai đoạn {start_date} đến {end_date}. "
                    "Hãy phân tích product_id dẫn đầu, danh mục của sản phẩm, nhóm theo sau và ý nghĩa nhu cầu thị trường."
                ),
                fallback=safe_summary,
                max_new_tokens=220,
            )
            result = {
                "ok": True,
                "intent": intent,
                "start_date": start_date,
                "end_date": end_date,
                "top_n": top_n,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": self.records(df),
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            }
            if output_path:
                result["report"] = self.write_analysis_report(
                    df=df,
                    title=f"Top products by orders {start_date} to {end_date}",
                    summary=analysis,
                    output_path=output_path,
                    x_col="product_id",
                    y_col="order_count",
                )
            return finalize(result)

        if intent == "top_categories_by_orders":
            start_date, end_date = self.date_range_from_understanding(understanding, question, year)
            df = self.analyze_top_categories_by_orders(start_date=start_date, end_date=end_date, top_n=top_n)
            safe_summary = self.safe_top_categories_by_orders_analysis(df, start_date, end_date)
            analysis = self.generate_analysis_or_fallback(
                question=question,
                df=df,
                context=(
                    f"Dữ liệu top danh mục theo số đơn trong giai đoạn {start_date} đến {end_date}. "
                    "Hãy phân tích danh mục dẫn đầu, nhóm theo sau và ý nghĩa nhu cầu thị trường."
                ),
                fallback=safe_summary,
                max_new_tokens=220,
            )
            return finalize({
                "ok": True,
                "intent": intent,
                "start_date": start_date,
                "end_date": end_date,
                "top_n": top_n,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": self.records(df),
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            })

        if intent == "favorite_products":
            df = self.analyze_favorite_products(year=year, top_n=top_n)
            safe_summary = self.safe_favorite_products_analysis(df, year, top_n)
            analysis = self.generate_analysis_or_fallback(
                question=question,
                df=df,
                context=(
                    f"Dữ liệu danh mục được yêu thích năm {year} đã lọc theo review và số đơn. "
                    "Hãy phân tích top danh mục dựa trên avg_review_score và total_orders."
                ),
                fallback=safe_summary,
                max_new_tokens=220,
            )
            result = {
                "ok": True,
                "intent": intent,
                "year": year,
                "top_n": top_n,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": self.records(df),
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
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
            return finalize(result)

        if intent == "top_revenue_categories":
            start_date, end_date = self.date_range_from_understanding(understanding, question, year)
            df = self.analyze_top_categories(start_date=start_date, end_date=end_date, limit=top_n)
            safe_summary = self.safe_top_categories_analysis(df, start_date, end_date)
            analysis = self.generate_analysis_or_fallback(
                question=question,
                df=df,
                context=(
                    f"Dữ liệu top danh mục theo doanh thu trong giai đoạn {start_date} đến {end_date}. "
                    "Hãy phân tích danh mục doanh thu cao nhất, khoảng cách với nhóm sau nếu có, và ý nghĩa kinh doanh."
                ),
                fallback=safe_summary,
                max_new_tokens=220,
            )
            return finalize({
                "ok": True,
                "intent": intent,
                "start_date": start_date,
                "end_date": end_date,
                "top_n": top_n,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": self.records(df),
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            })

        if intent == "top_revenue_products":
            start_date, end_date = self.date_range_from_understanding(understanding, question, year)
            df = self.analyze_top_revenue_products(start_date=start_date, end_date=end_date, top_n=top_n)
            safe_summary = self.safe_top_revenue_products_analysis(df, start_date, end_date)
            analysis = self.generate_analysis_or_fallback(
                question=question,
                df=df,
                context=(
                    f"Dữ liệu top sản phẩm theo doanh thu trong giai đoạn {start_date} đến {end_date}. "
                    "Hãy phân tích product_id doanh thu cao nhất, category, doanh thu, số đơn và item_count. "
                    "Không gọi đây là danh mục nếu JSON đang ở cấp product_id."
                ),
                fallback=safe_summary,
                max_new_tokens=240,
            )
            return finalize({
                "ok": True,
                "intent": intent,
                "start_date": start_date,
                "end_date": end_date,
                "top_n": top_n,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": self.records(df),
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            })

        if intent == "revenue_by_month":
            df = self.analyze_revenue_by_month(year=year)
            safe_summary = self.safe_revenue_by_month_analysis(df, year)
            analysis = self.generate_analysis_or_fallback(
                question=question,
                df=df,
                context=(
                    f"Dữ liệu doanh thu theo tháng năm {year} đã được tính bằng SQL. "
                    "Hãy phân tích xu hướng, tháng nổi bật và mối liên hệ giữa revenue với order_count."
                ),
                fallback=safe_summary,
                max_new_tokens=240,
            )
            return finalize({
                "ok": True,
                "intent": intent,
                "year": year,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "rows": self.records(df),
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            })

        if intent == "delivery_delay":
            start_date, end_date = self.date_range_from_understanding(understanding, question, year)
            df = self.analyze_delivery_delay_summary(start_date=start_date, end_date=end_date)
            safe_summary = self.safe_delivery_delay_analysis(df, start_date, end_date)
            analysis = self.generate_analysis_or_fallback(
                question=question,
                df=df,
                context=(
                    f"Dữ liệu tổng quan giao hàng trễ trong giai đoạn {start_date} đến {end_date}. "
                    "Hãy phân tích tỷ lệ giao trễ, thời gian giao trung bình và review trung bình."
                ),
                fallback=safe_summary,
                max_new_tokens=200,
            )
            return finalize({
                "ok": True,
                "intent": intent,
                "start_date": start_date,
                "end_date": end_date,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "summary": self.records(df)[0] if not df.empty else {},
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            })

        if intent == "repeat_customer_rate":
            start_date, end_date = self.date_range_from_understanding(understanding, question, year)
            df = self.analyze_repeat_customer_rate(start_date=start_date, end_date=end_date)
            safe_summary = self.safe_repeat_customer_analysis(df, start_date, end_date)
            analysis = self.generate_analysis_or_fallback(
                question=question,
                df=df,
                context=(
                    f"Dữ liệu tỷ lệ khách hàng quay lại trong giai đoạn {start_date} đến {end_date}. "
                    "Hãy phân tích active_customers, repeat_customers và repeat_customer_rate_pct."
                ),
                fallback=safe_summary,
                max_new_tokens=200,
            )
            return finalize({
                "ok": True,
                "intent": intent,
                "start_date": start_date,
                "end_date": end_date,
                "model": os.getenv("GEMMA_MODEL", "google/gemma-2b-it"),
                "summary": self.records(df)[0] if not df.empty else {},
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            })

        return finalize({
            "ok": False,
            "error": "Chưa nhận diện được loại câu hỏi.",
            "intent": intent,
            "supported_intents": [
                "category_performance",
                "monthly_revenue_extremes",
                "quarterly_revenue_comparison",
                "delivery_review_impact",
                "state_market_importance",
                "state_revenue_low_review",
                "customer_experience_priority",
                "favorite_vs_revenue",
                "orders_vs_review",
                "top_products_by_orders",
                "top_categories_by_orders",
                "favorite_products",
                "top_revenue_categories",
                "top_revenue_products",
                "revenue_by_month",
                "delivery_delay",
                "repeat_customer_rate",
                "schema_tables",
            ],
        })


    def answer_order_question(self, question: str) -> dict[str, Any]:
        order_id = self.extract_order_id(question)

        if not order_id:
            return {
                "ok": False,
                "error": "Bạn vui lòng cung cấp order_id."
            }

        q = Agent.normalize_text(question)
        gemma_source = "gemma_with_safe_fallback"

        if "thanh toan" in q or "payment" in q:
            df = self.analyze_order_payment(order_id)
            payment = self.records(df)[0] if not df.empty else {}
            safe_summary = self.safe_order_payment_analysis(order_id, payment)
            analysis = self.generate_payload_analysis_or_fallback(
                question=question,
                payload={"order_id": order_id, "payment": payment},
                context="Hãy phân tích thông tin thanh toán đơn hàng. Chỉ dùng dữ liệu trong JSON.",
                fallback=safe_summary,
                max_new_tokens=180,
            )
            return {
                "ok": True,
                "intent": "order_payment",
                "order_id": order_id,
                "payment": payment,
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            }

        if "van chuyen" in q or "giao hang" in q or "phi ship" in q or "freight" in q:
            df = self.analyze_order_shipping(order_id)
            shipping = self.records(df)[0] if not df.empty else {}
            safe_summary = self.safe_order_shipping_analysis(order_id, shipping)
            analysis = self.generate_payload_analysis_or_fallback(
                question=question,
                payload={"order_id": order_id, "shipping": shipping},
                context="Hãy phân tích thông tin vận chuyển đơn hàng, trạng thái, phí ship, thời gian giao và giao trễ nếu có.",
                fallback=safe_summary,
                max_new_tokens=200,
            )
            return {
                "ok": True,
                "intent": "order_shipping",
                "order_id": order_id,
                "shipping": shipping,
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            }

        if "san pham" in q or "product" in q:
            df = self.analyze_order_products(order_id)
            products = self.records(df)
            safe_summary = self.safe_order_products_analysis(order_id, products)
            analysis = self.generate_payload_analysis_or_fallback(
                question=question,
                payload={"order_id": order_id, "products": products},
                context="Hãy phân tích các dòng sản phẩm trong đơn hàng, danh mục, giá, phí vận chuyển và line_total.",
                fallback=safe_summary,
                max_new_tokens=220,
            )
            return {
                "ok": True,
                "intent": "order_products",
                "order_id": order_id,
                "products": products,
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            }

        if "nguoi ban" in q or "seller" in q:
            df = self.analyze_order_sellers(order_id)
            sellers = self.records(df)
            safe_summary = self.safe_order_sellers_analysis(order_id, sellers)
            analysis = self.generate_payload_analysis_or_fallback(
                question=question,
                payload={"order_id": order_id, "sellers": sellers},
                context="Hãy phân tích người bán liên quan đến đơn hàng, thành phố và bang/khu vực của seller.",
                fallback=safe_summary,
                max_new_tokens=180,
            )
            return {
                "ok": True,
                "intent": "order_sellers",
                "order_id": order_id,
                "sellers": sellers,
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            }

        if "danh gia" in q or "review" in q or "sao" in q:
            df = self.analyze_order_review(order_id)
            review = self.records(df)[0] if not df.empty else {}
            safe_summary = self.safe_order_review_analysis(order_id, review)
            analysis = self.generate_payload_analysis_or_fallback(
                question=question,
                payload={"order_id": order_id, "review": review},
                context="Hãy phân tích điểm đánh giá của đơn hàng. Chỉ dùng review_score_avg và review_count trong JSON.",
                fallback=safe_summary,
                max_new_tokens=160,
            )
            return {
                "ok": True,
                "intent": "order_review",
                "order_id": order_id,
                "review": review,
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            }

        if "khach hang" in q or "customer" in q:
            df = self.analyze_order_customer(order_id)
            customer = self.records(df)[0] if not df.empty else {}
            safe_summary = self.safe_order_customer_analysis(order_id, customer)
            analysis = self.generate_payload_analysis_or_fallback(
                question=question,
                payload={"order_id": order_id, "customer": customer},
                context="Hãy phân tích thông tin khách hàng của đơn hàng, gồm thành phố và bang/khu vực. Không suy đoán thông tin cá nhân khác.",
                fallback=safe_summary,
                max_new_tokens=160,
            )
            return {
                "ok": True,
                "intent": "order_customer",
                "order_id": order_id,
                "customer": customer,
                "safe_summary": safe_summary,
                "analysis": analysis,
                "analysis_source": gemma_source,
            }

        return self.analyze_order_detail(order_id, question=question)


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



    def analyze_order_detail(self, order_id: str, question: str | None = None) -> dict[str, Any]:
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

        result = {
            "ok": True,
            "intent": "order_detail",
            "order_id": order_id,
            "customer": self.records(customer_df)[0] if not customer_df.empty else {},
            "review": self.records(review_df)[0] if not review_df.empty else {},
            "sellers": self.records(sellers_df),
            "products": self.records(products_df),
            "shipping": self.records(shipping_df)[0] if not shipping_df.empty else {},
            "payment": self.records(payment_df)[0] if not payment_df.empty else {},
        }
        safe_summary = self.safe_order_detail_analysis(order_id, result)
        analysis = self.generate_payload_analysis_or_fallback(
            question=question or f"Phân tích chi tiết đơn hàng {order_id}",
            payload=result,
            context=(
                "Hãy phân tích tổng quan chi tiết đơn hàng dựa trên customer, review, sellers, "
                "products, shipping và payment. Chỉ dùng số liệu trong JSON."
            ),
            fallback=safe_summary,
            max_new_tokens=260,
        )
        result["safe_summary"] = safe_summary
        result["analysis"] = analysis
        result["analysis_source"] = "gemma_with_safe_fallback"
        return result

    def safe_order_payment_analysis(self, order_id: str, payment: dict[str, Any]) -> str:
        if not payment:
            return f"Không tìm thấy thông tin thanh toán cho đơn {order_id}."
        payment_types = payment.get("payment_types") or "không rõ"
        payment_total = payment.get("payment_value_total")
        row_count = payment.get("payment_row_count")
        installments = payment.get("max_installments")
        return (
            f"Đơn {order_id} thanh toán bằng {payment_types} với tổng giá trị thanh toán {payment_total}. "
            f"Dữ liệu có {row_count} dòng thanh toán; số kỳ trả góp cao nhất là {installments}."
        )

    def safe_order_shipping_analysis(self, order_id: str, shipping: dict[str, Any]) -> str:
        if not shipping:
            return f"Không tìm thấy thông tin vận chuyển cho đơn {order_id}."
        status = shipping.get("order_status") or "không rõ"
        freight_total = shipping.get("freight_total")
        delivery_days = shipping.get("delivery_days")
        delayed = shipping.get("is_delayed_delivery")
        if delayed is True:
            delayed_text = "bị giao trễ"
        elif delayed is False:
            delayed_text = "không ghi nhận giao trễ"
        else:
            delayed_text = "chưa đủ dữ liệu để xác định giao trễ"
        return (
            f"Đơn {order_id} có trạng thái {status}, phí vận chuyển {freight_total} "
            f"và thời gian giao hàng {delivery_days} ngày. Theo dữ liệu hiện có, đơn này {delayed_text}."
        )

    def safe_order_products_analysis(self, order_id: str, products: list[dict[str, Any]]) -> str:
        if not products:
            return f"Không tìm thấy sản phẩm nào cho đơn {order_id}."
        categories = sorted({row.get("category") or "unknown" for row in products})
        line_total = round(sum((row.get("line_total") or 0) for row in products), 2)
        category_text = ", ".join(categories[:5])
        if len(categories) > 5:
            category_text += f" và {len(categories) - 5} danh mục khác"
        return (
            f"Đơn {order_id} có {len(products)} dòng sản phẩm thuộc {len(categories)} danh mục: "
            f"{category_text}. Tổng line_total của các dòng sản phẩm là {line_total}."
        )

    def safe_order_sellers_analysis(self, order_id: str, sellers: list[dict[str, Any]]) -> str:
        if not sellers:
            return f"Không tìm thấy người bán nào cho đơn {order_id}."
        states = sorted({row.get("seller_state") or "unknown" for row in sellers})
        state_text = ", ".join(states)
        return (
            f"Đơn {order_id} có {len(sellers)} người bán liên quan. "
            f"Các người bán nằm ở {len(states)} bang/khu vực: {state_text}."
        )

    def safe_order_review_analysis(self, order_id: str, review: dict[str, Any]) -> str:
        if not review:
            return f"Không tìm thấy đánh giá cho đơn {order_id}."
        score = review.get("review_score_avg")
        review_count = review.get("review_count")
        if score is None:
            return f"Đơn {order_id} chưa có điểm đánh giá trong dữ liệu hiện có."
        return f"Đơn {order_id} có điểm đánh giá trung bình {score} dựa trên {review_count} đánh giá."

    def safe_order_customer_analysis(self, order_id: str, customer: dict[str, Any]) -> str:
        if not customer:
            return f"Không tìm thấy thông tin khách hàng cho đơn {order_id}."
        city = customer.get("customer_city") or "không rõ"
        state = customer.get("customer_state") or "không rõ"
        return f"Đơn {order_id} thuộc về khách hàng ở {city}, bang/khu vực {state}."

    def safe_order_detail_analysis(self, order_id: str, result: dict[str, Any]) -> str:
        products = result.get("products") or []
        sellers = result.get("sellers") or []
        payment = result.get("payment") or {}
        shipping = result.get("shipping") or {}
        review = result.get("review") or {}
        customer = result.get("customer") or {}

        product_count = len(products)
        seller_count = len(sellers)
        payment_types = payment.get("payment_types") or "không rõ"
        payment_total = payment.get("payment_value_total")
        status = shipping.get("order_status") or "không rõ"
        delayed = shipping.get("is_delayed_delivery")
        if delayed is True:
            delayed_text = "có giao trễ"
        elif delayed is False:
            delayed_text = "không ghi nhận giao trễ"
        else:
            delayed_text = "chưa đủ dữ liệu giao trễ"
        review_score = review.get("review_score_avg")
        review_text = review_score if review_score is not None else "chưa có"
        customer_state = customer.get("customer_state") or "không rõ"

        return (
            f"Đơn {order_id} có {product_count} dòng sản phẩm từ {seller_count} người bán, "
            f"khách hàng thuộc bang/khu vực {customer_state}. Đơn đang ở trạng thái {status}, "
            f"{delayed_text}, thanh toán bằng {payment_types} với tổng giá trị {payment_total}. "
            f"Điểm review trung bình hiện có là {review_text}."
        )

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

    def category_rank_comparison_rows(
        self,
        df: pd.DataFrame,
        rank_columns: list[str],
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        if df.empty:
            return []
        mask = pd.Series(False, index=df.index)
        for column in rank_columns:
            mask = mask | (df[column] <= 1)
        mask = mask | (df["revenue_rank"] <= min(top_n, 10))
        selected = df.loc[mask].copy()
        selected = selected.sort_values(["revenue_rank", "review_rank", "order_rank"])
        return self.records(selected)

    def safe_favorite_vs_revenue_analysis(self, df: pd.DataFrame, year: int, top_n: int) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có dữ liệu đủ điều kiện để so sánh danh mục được yêu thích và doanh thu trong năm {year}."
        favorite = min(rows, key=lambda row: row["review_rank"])
        top_revenue = min(rows, key=lambda row: row["revenue_rank"])
        same_category = favorite["category"] == top_revenue["category"]
        if same_category:
            conclusion = (
                f"Có. Năm {year}, {favorite['category']} vừa là danh mục được yêu thích nhất "
                f"vừa là danh mục có doanh thu cao nhất trong nhóm đủ điều kiện."
            )
        else:
            conclusion = (
                f"Không. Năm {year}, danh mục được yêu thích nhất là {favorite['category']} "
                f"với review trung bình {favorite['avg_review_score']} và {favorite['order_count']} đơn, "
                f"nhưng danh mục doanh thu cao nhất là {top_revenue['category']} với doanh thu {top_revenue['revenue']}."
            )
        return (
            f"{conclusion} "
            f"{favorite['category']} chỉ xếp hạng {favorite['revenue_rank']} theo doanh thu "
            f"với doanh thu {favorite['revenue']}, trong khi {top_revenue['category']} có review trung bình "
            f"{top_revenue['avg_review_score']} và xếp hạng {top_revenue['review_rank']} theo review.\n\n"
            "Cách phân tích: lọc các danh mục năm "
            f"{year} có hơn 50 đơn và có review; tính doanh thu bằng SUM(line_total), "
            "đếm đơn bằng COUNT(DISTINCT order_id), tính yêu thích bằng AVG(review_score_avg); "
            "sau đó so sánh hạng review với hạng doanh thu."
        )

    def safe_orders_vs_review_analysis(self, df: pd.DataFrame, year: int, top_n: int) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có dữ liệu đủ điều kiện để so sánh số đơn và review trong năm {year}."
        top_orders = min(rows, key=lambda row: row["order_rank"])
        top_review = min(rows, key=lambda row: row["review_rank"])
        same_category = top_orders["category"] == top_review["category"]
        if same_category:
            conclusion = (
                f"Năm {year}, cùng một danh mục đứng đầu cả về số đơn và review: {top_orders['category']}."
            )
        else:
            conclusion = (
                f"Năm {year}, danh mục nhiều đơn nhất và danh mục review tốt nhất không trùng nhau. "
                f"Danh mục nhiều đơn nhất là {top_orders['category']} với {top_orders['order_count']} đơn, "
                f"còn danh mục review tốt nhất là {top_review['category']} với review trung bình {top_review['avg_review_score']}."
            )
        return (
            f"{conclusion} "
            f"{top_orders['category']} có review trung bình {top_orders['avg_review_score']} "
            f"và xếp hạng {top_orders['review_rank']} theo review; "
            f"{top_review['category']} có {top_review['order_count']} đơn "
            f"và xếp hạng {top_review['order_rank']} theo số đơn.\n\n"
            "Cách phân tích: dùng cùng một tập danh mục đủ điều kiện trong năm "
            f"{year}, rồi xếp hạng riêng theo COUNT(DISTINCT order_id) và AVG(review_score_avg). "
            "Kết luận dựa trên việc hai hạng 1 có cùng category hay không."
        )

    def safe_top_products_by_orders_analysis(self, df: pd.DataFrame, start_date: str, end_date: str) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có dữ liệu sản phẩm trong giai đoạn {start_date} đến {end_date}."
        first = rows[0]
        second = rows[1] if len(rows) > 1 else None
        second_text = (
            f" Sản phẩm đứng thứ hai là {second['product_id']} thuộc danh mục {second['category']} "
            f"với {second['order_count']} đơn."
            if second else ""
        )
        return (
            f"Kết luận: trong giai đoạn {start_date} đến {end_date}, sản phẩm bán chạy nhất là "
            f"{first['product_id']} thuộc danh mục {first['category']} với {first['order_count']} đơn hàng "
            f"và {first['item_count']} dòng sản phẩm. "
            f"Kết quả trả về {len(rows)} sản phẩm theo số đơn hàng giảm dần.{second_text} "
            "Nên kiểm tra thêm doanh thu và review "
            "trước khi quyết định ưu tiên vận hành."
        )

    def safe_top_categories_by_orders_analysis(self, df: pd.DataFrame, start_date: str, end_date: str) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có dữ liệu danh mục trong giai đoạn {start_date} đến {end_date}."
        first = rows[0]
        second = rows[1] if len(rows) > 1 else None
        second_text = (
            f" Danh mục đứng thứ hai là {second['category']} với {second['order_count']} đơn."
            if second else ""
        )
        return (
            f"Kết luận: trong giai đoạn {start_date} đến {end_date}, danh mục bán chạy nhất là "
            f"{first['category']} với {first['order_count']} đơn hàng và {first['item_count']} dòng sản phẩm. "
            f"Kết quả trả về {len(rows)} danh mục theo số đơn hàng giảm dần.{second_text} "
            "Điều này cho thấy nhu cầu tập trung vào một số nhóm danh mục lớn."
        )

    def safe_top_categories_analysis(self, df: pd.DataFrame, start_date: str, end_date: str) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có danh mục doanh thu nào trong giai đoạn {start_date} đến {end_date}."
        first = rows[0]
        second = rows[1] if len(rows) > 1 else None
        gap_text = ""
        if second and second.get("revenue"):
            gap = (first["revenue"] or 0) - (second["revenue"] or 0)
            gap_text = f" Cao hơn nhóm thứ hai ({second['category']}) khoảng {round(gap, 2)} doanh thu."
        return (
            f"Kết luận: trong giai đoạn {start_date} đến {end_date}, danh mục có doanh thu cao nhất là "
            f"{first['category']} với doanh thu {first['revenue']} và {first['order_count']} đơn hàng. "
            f"Kết quả trả về {len(rows)} danh mục.{gap_text} "
            "Nên đọc kết quả này như phân tích doanh thu theo danh mục/sản phẩm, không phải theo bang hay khách hàng."
        )

    def safe_top_revenue_products_analysis(self, df: pd.DataFrame, start_date: str, end_date: str) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có dữ liệu doanh thu sản phẩm trong giai đoạn {start_date} đến {end_date}."
        first = rows[0]
        second = rows[1] if len(rows) > 1 else None
        gap_text = ""
        if second and second.get("revenue"):
            gap = (first["revenue"] or 0) - (second["revenue"] or 0)
            gap_text = f" Cao hơn sản phẩm thứ hai ({second['product_id']}) khoảng {round(gap, 2)} doanh thu."
        return (
            f"Kết luận: trong giai đoạn {start_date} đến {end_date}, sản phẩm có doanh thu cao nhất là "
            f"{first['product_id']} thuộc danh mục {first['category']} với doanh thu {first['revenue']}, "
            f"{first['order_count']} đơn hàng và {first['item_count']} dòng sản phẩm. "
            f"Kết quả trả về {len(rows)} sản phẩm.{gap_text}"
        )

    def safe_revenue_by_month_analysis(self, df: pd.DataFrame, year: int) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có dữ liệu doanh thu theo tháng cho năm {year}."
        highest = max(rows, key=lambda row: row["revenue"] or 0)
        lowest = min(rows, key=lambda row: row["revenue"] or 0)
        gap = (highest["revenue"] or 0) - (lowest["revenue"] or 0)
        pct_gap = round(100.0 * gap / lowest["revenue"], 2) if lowest["revenue"] else None
        pct_text = f", tương đương cao hơn {pct_gap}%" if pct_gap is not None else ""
        return (
            f"Kết luận: năm {year} có {len(rows)} tháng có dữ liệu doanh thu. "
            f"Tháng cao nhất là {highest['month']} với doanh thu {highest['revenue']} "
            f"và tháng thấp nhất là {lowest['month']} với doanh thu {lowest['revenue']}. "
            f"Chênh lệch tuyệt đối là {round(gap, 2)}{pct_text}. "
            "Nếu tháng thấp nhất chỉ có rất ít đơn, cần xem đó có phải tháng dữ liệu không đầy đủ trước khi kết luận xu hướng kinh doanh."
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

    def safe_category_performance_analysis(self, df: pd.DataFrame, year: int) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có dữ liệu danh mục sản phẩm cho năm {year}."
        first = rows[0]
        return (
            f"Kết luận: năm {year}, danh mục có điểm tổng hợp cao nhất là {first['category']} "
            f"với doanh thu {first['revenue']}, {first['order_count']} đơn hàng, "
            f"review trung bình {first['avg_review_score']} và điểm {first['performance_score']}. "
            "Điểm này kết hợp doanh thu, số đơn và review, nên phù hợp hơn việc chỉ xếp hạng theo doanh thu. "
            "Các danh mục còn lại trong bảng nên được so sánh theo từng metric để biết nhóm nào mạnh về quy mô và nhóm nào mạnh về chất lượng."
        )

    def safe_monthly_revenue_extremes_analysis(self, df: pd.DataFrame, year: int) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có dữ liệu doanh thu theo tháng cho năm {year}."
        highest = max(rows, key=lambda row: row["revenue"] or 0)
        lowest = min(rows, key=lambda row: row["revenue"] or 0)
        gap = (highest["revenue"] or 0) - (lowest["revenue"] or 0)
        pct_gap = round(100.0 * gap / lowest["revenue"], 2) if lowest["revenue"] else None
        suffix = f", cao hơn {pct_gap}%" if pct_gap is not None else ""
        return (
            f"Kết luận: năm {year}, tháng doanh thu cao nhất là {highest['month']} với {highest['revenue']} "
            f"trên {highest['order_count']} đơn; tháng thấp nhất là {lowest['month']} với "
            f"{lowest['revenue']} trên {lowest['order_count']} đơn. Chênh lệch tuyệt đối là {gap}{suffix}. "
            "Điểm cần chú ý là số đơn của tháng thấp nhất: nếu số đơn quá nhỏ, khả năng cao đây là giai đoạn dữ liệu không đầy đủ "
            "hoặc chỉ có một phần tháng, thay vì phản ánh nhu cầu thực sự giảm mạnh."
        )

    def safe_quarterly_revenue_analysis(self, df: pd.DataFrame, year: int) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có dữ liệu doanh thu theo quý cho năm {year}."
        highest = max(rows, key=lambda row: row["revenue"] or 0)
        lowest = min(rows, key=lambda row: row["revenue"] or 0)
        return (
            f"Kết luận: năm {year}, quý {highest['quarter']} có doanh thu cao nhất với {highest['revenue']} "
            f"và {highest['order_count']} đơn; quý {lowest['quarter']} thấp nhất với "
            f"{lowest['revenue']} và {lowest['order_count']} đơn. "
            "Nên so sánh thêm số đơn và giá trị đơn trung bình để phân biệt tăng trưởng do nhiều đơn hơn hay do đơn hàng có giá trị lớn hơn."
        )

    def safe_delivery_review_impact_analysis(self, df: pd.DataFrame, start_date: str, end_date: str) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có dữ liệu giao hàng/review trong giai đoạn {start_date} đến {end_date}."
        by_status = {row["delivery_status"]: row for row in rows}
        delayed = by_status.get("delayed")
        on_time = by_status.get("on_time")
        if not delayed or not on_time:
            return f"Dữ liệu chỉ có {len(rows)} nhóm giao hàng trong giai đoạn {start_date} đến {end_date}."

        gap = round((on_time["avg_review_score"] or 0) - (delayed["avg_review_score"] or 0), 2)
        delayed_rate = round(
            100.0 * (delayed["order_count"] or 0)
            / ((delayed["order_count"] or 0) + (on_time["order_count"] or 0)),
            2,
        )
        delivery_gap = round((delayed["avg_delivery_days"] or 0) - (on_time["avg_delivery_days"] or 0), 2)

        revenue_text = ""
        if "total_revenue" in delayed and "total_revenue" in on_time:
            revenue_text = (
                f" Doanh thu nhóm giao trễ là {delayed.get('total_revenue')} "
                f"({delayed.get('revenue_share_pct')}% tổng doanh thu hai nhóm), "
                f"trong khi nhóm đúng hạn là {on_time.get('total_revenue')} "
                f"({on_time.get('revenue_share_pct')}%)."
            )

        return (
            f"Kết luận: giao hàng trễ có liên quan đến điểm review thấp hơn trong giai đoạn {start_date} đến {end_date}. "
            f"Nhóm giao đúng hạn có review trung bình "
            f"{on_time['avg_review_score']} trên {on_time['order_count']} đơn; nhóm giao trễ có "
            f"{delayed['avg_review_score']} trên {delayed['order_count']} đơn. Chênh lệch review là {gap} điểm. "
            f"Nhóm giao trễ chiếm khoảng {delayed_rate}% tổng số đơn trong hai nhóm và giao lâu hơn trung bình {delivery_gap} ngày."
            f"{revenue_text} "
            "Đây là so sánh mô tả, chưa khẳng định quan hệ nhân quả tuyệt đối."
        )

    def safe_state_market_importance_analysis(self, df: pd.DataFrame, year: int) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có dữ liệu thị trường theo bang cho năm {year}."
        first = rows[0]
        return (
            f"Kết luận: năm {year}, bang quan trọng nhất là {first['state']} với doanh thu {first['revenue']}, "
            f"{first['order_count']} đơn, {first['customer_count']} khách hàng và điểm quan trọng "
            f"{first['importance_score']}. "
            "Điểm quan trọng kết hợp doanh thu, số đơn và số khách hàng, nên kết quả này phản ánh quy mô thị trường tổng hợp. "
            "Nếu mục tiêu là cải thiện chất lượng dịch vụ, cần xem thêm review và tỷ lệ giao trễ chứ không chỉ doanh thu."
        )

    def safe_state_revenue_low_review_analysis(self, df: pd.DataFrame, year: int) -> str:
        rows = self.records(df)
        if not rows:
            return f"Không có dữ liệu doanh thu và review theo bang cho năm {year}."
        first = rows[0]
        return (
            f"Kết luận: năm {year}, bang nổi bật nhất theo tiêu chí doanh thu cao nhưng review thấp là "
            f"{first['state']}. Bang này có doanh thu {first['revenue']}, "
            f"{first['order_count']} đơn, review trung bình {first['avg_review_score']}, "
            f"tỷ lệ giao trễ {first['delayed_order_rate_pct']}% và điểm ưu tiên {first['priority_score']}. "
            "Cách xếp hạng này không chọn bang chỉ vì doanh thu cao; nó ưu tiên nơi có quy mô đủ lớn nhưng chất lượng trải nghiệm còn yếu. "
            "Nên kiểm tra các nguyên nhân vận hành như giao trễ, seller hoặc danh mục hàng bán nhiều tại bang này trước khi đưa ra hành động."
        )

    def safe_customer_experience_priority_analysis(self, payload: dict[str, Any]) -> str:
        categories = payload.get("categories") or []
        states = payload.get("states") or []
        year = payload.get("year")
        if not categories and not states:
            return f"Không có dữ liệu ưu tiên cải thiện trải nghiệm cho năm {year}."
        parts = []
        if categories:
            first_category = categories[0]
            parts.append(
                f"danh mục {first_category['category']} có điểm ưu tiên {first_category['priority_score']}, "
                f"review {first_category['avg_review_score']} và tỷ lệ giao trễ {first_category['delayed_order_rate_pct']}%"
            )
        if states:
            first_state = states[0]
            parts.append(
                f"bang {first_state['state']} có điểm ưu tiên {first_state['priority_score']}, "
                f"review {first_state['avg_review_score']} và tỷ lệ giao trễ {first_state['delayed_order_rate_pct']}%"
            )
        return f"Năm {year}, nên ưu tiên " + "; đồng thời ".join(parts) + "."

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
    def normalize_text(text_value: str) -> str:
        text_value = unicodedata.normalize("NFKD", text_value)
        text_value = "".join(ch for ch in text_value if not unicodedata.combining(ch))
        text_value = text_value.replace("đ", "d").replace("Đ", "D")
        return text_value.lower()

    @staticmethod
    def detect_intent(question: str) -> str:
        q = Agent.normalize_text(question)
        has_revenue = "doanh thu" in q or "revenue" in q
        asks_category = any(term in q for term in ["danh muc", "category", "nhom san pham"])
        asks_product = any(term in q for term in ["san pham", "product", "product_id"])
        has_favorite = any(
            term in q
            for term in [
                "yeu thich",
                "duoc yeu",
                "yeu nhat",
                "duoc thich",
                "thich nhat",
                "ua thich",
                "review tot",
                "review cao",
                "danh gia tot",
                "danh gia cao",
            ]
        )
        has_review = has_favorite or any(term in q for term in ["review", "danh gia", "sao"])
        has_order_volume = any(
            term in q
            for term in [
                "nhieu don",
                "so don",
                "don nhat",
                "ban chay",
                "ban duoc",
                "order_count",
                "orders",
                "top",
            ]
        )
        wants_comparison = any(term in q for term in ["so sanh", "khac", "dong thoi", "co phai", "voi"])
        if (
            any(term in q for term in ["liet ke", "danh sach", "co nhung", "cac bang", "nhung bang"])
            and any(term in q for term in ["bang", "table", "schema", "du lieu", "database"])
        ):
            return "schema_tables"
        if has_favorite and has_revenue and any(term in q for term in ["dong thoi", "co phai", "cao khong", "doanh thu cao"]):
            return "favorite_vs_revenue"
        if wants_comparison and has_order_volume and has_review:
            return "orders_vs_review"
        if (
        has_revenue
        and Agent.is_quarter_question(q)
        and any(term in q for term in ["so sanh", "khac nhau", "chenh lech", "cao nhat", "thap nhat", "noi bat"])
            ):
            return "quarterly_revenue_comparison"
        
        if (
            asks_category
            and has_revenue
            and any(term in q for term in ["review", "danh gia", "so don", "don hang"])
        ):
            return "category_performance"
        if (
            has_revenue
            and "thang" in q
            and any(term in q for term in ["cao nhat", "thap nhat", "bat thuong", "khac nhau", "so sanh"])
        ):
            return "monthly_revenue_extremes"
        if (
            "giao hang" in q
            and any(term in q for term in ["tre", "cham", "delay"])
            and any(term in q for term in ["danh gia", "review", "sao", "anh huong"])
        ):
            return "delivery_review_impact"
        if (
            ("bang" in q or "state" in q)
            and any(term in q for term in ["thi truong", "market", "quan trong"])
            and has_revenue
        ):
            return "state_market_importance"
        if (
            ("bang" in q or "state" in q or "khu vuc" in q)
            and has_revenue
            and any(term in q for term in ["review", "danh gia", "diem danh gia", "sao"])
        ):
            return "state_revenue_low_review"
        if any(term in q for term in ["cai thien trai nghiem", "trai nghiem khach hang", "uu tien"]):
            return "customer_experience_priority"
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
        if has_revenue and asks_product and not asks_category:
            return "top_revenue_products"
        if has_revenue:
            return "top_revenue_categories"
        if has_review:
            return "favorite_products"
        if asks_category and has_order_volume:
            return "top_categories_by_orders"
        if asks_product and has_order_volume:
            return "top_products_by_orders"
        if has_order_volume and any(term in q for term in ["don hang", "order"]):
            return "revenue_by_month" if any(term in q for term in ["thang", "month"]) else "top_categories_by_orders"
        return "unknown"

    @staticmethod
    def extract_year(question: str, default: int = 2018) -> int:
        match = re.search(r"\b(20\d{2})\b", question)
        return int(match.group(1)) if match else default

    @staticmethod
    def has_explicit_time_scope(question: str) -> bool:
        q = Agent.normalize_text(question)
        return bool(
            re.search(r"\b20\d{2}\b", question)
            or re.search(r"\b\d{4}-\d{2}-\d{2}\b", question)
            or re.search(r"\b(?:q|quy|qui)\s*[1-4]\b", q)
            or re.search(r"\bthang\s*(?:1[0-2]|0?[1-9])\b", q)
        )

    @staticmethod
    def time_clarification(question: str, intent: str) -> dict[str, Any] | None:
        time_sensitive_intents = {
            "category_performance",
            "monthly_revenue_extremes",
            "quarterly_revenue_comparison",
            "delivery_review_impact",
            "state_market_importance",
            "customer_experience_priority",
            "favorite_vs_revenue",
            "orders_vs_review",
            "top_products_by_orders",
            "top_categories_by_orders",
            "favorite_products",
            "top_revenue_categories",
            "top_revenue_products",
            "revenue_by_month",
            "delivery_delay",
            "repeat_customer_rate",
        }
        if intent not in time_sensitive_intents:
            return None
        if Agent.has_explicit_time_scope(question):
            return None
        return {
            "ok": False,
            "intent": intent,
            "needs_clarification": True,
            "reason": "Câu hỏi phân tích cần có phạm vi thời gian rõ ràng trước khi truy vấn dữ liệu.",
            "clarifying_question": (
                "Bạn muốn phân tích trong khoảng thời gian nào? "
                "Ví dụ: năm 2018, quý 1 năm 2018, tháng 11 năm 2017, "
                "hoặc từ 2017-01-01 đến 2017-12-31."
            ),
            "examples": [
                "doanh thu năm 2018 thế nào?",
                "doanh thu quý 1 năm 2018 thế nào?",
                "doanh thu tháng 11 năm 2017 thế nào?",
            ],
        }

    @staticmethod
    def is_order_detail_question(question: str, intent: str) -> bool:
        if Agent.extract_order_id(question):
            return True
        if intent != "unknown":
            return False

        q = Agent.normalize_text(question)
        order_terms = ["don hang", "don nay", "ma don", "order_id", "order id", "order"]
        detail_terms = [
            "thanh toan",
            "payment",
            "van chuyen",
            "giao hang",
            "phi ship",
            "freight",
            "review",
            "danh gia",
            "khach hang",
            "customer",
            "nguoi ban",
            "seller",
            "chi tiet",
            "cu the",
        ]
        aggregate_terms = [
            "theo",
            "nam",
            "thang",
            "quy",
            "top",
            "tong",
            "so don",
            "nhieu don",
            "ty le",
            "ti le",
            "bao nhieu",
            "xu huong",
            "doanh thu",
            "trung binh",
        ]
        return (
            any(term in q for term in order_terms)
            and any(term in q for term in detail_terms)
            and not any(term in q for term in aggregate_terms)
            and not Agent.has_explicit_time_scope(question)
        )

    @staticmethod
    def extract_top_n(question: str, default: int = 10) -> int:
        q = Agent.normalize_text(question)
        match = re.search(r"\btop\s*(\d{1,2})\b", q)
        if not match:
            match = re.search(r"\b(\d{1,2})\s+(?:danh muc|san pham|bang|khu vuc|state)", q)
        return Agent.clamp_limit(int(match.group(1)) if match else default)

    @staticmethod
    def extract_date_range(question: str, year: int) -> tuple[str, str]:
        q = Agent.normalize_text(question)
        explicit_dates = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", question)
        if len(explicit_dates) >= 2:
            return explicit_dates[0], explicit_dates[1]
        if "quy 1" in q or "qui 1" in q or "quý 1" in q or "q1" in q:
            return f"{year}-01-01", f"{year}-03-31"
        if "quy 2" in q or "qui 2" in q or "quý 2" in q or "q2" in q:
            return f"{year}-04-01", f"{year}-06-30"
        if "quy 3" in q or "qui 3" in q or "quý 3" in q or "q3" in q:
            return f"{year}-07-01", f"{year}-09-30"
        if "quy 4" in q or "qui 4" in q or "quý 4" in q or "q4" in q:
            return f"{year}-10-01", f"{year}-12-31"
        return f"{year}-01-01", f"{year}-12-31"

    @staticmethod
    def extract_quarters(question: str) -> list[int]:
        q = Agent.normalize_text(question)
        quarters = {int(match.group(1)) for match in re.finditer(r"\b(?:quy|qui|q)\s*([1-4])\b", q)}
        return sorted(quarters)

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
    @staticmethod
    def is_quarter_question(q: str) -> bool:
        return bool(
        re.search(r"\bq[1-4]\b", q)
        or re.search(r"\b(?:quy|qui)\s*[1-4]\b", q)
        or any(
            term in q
            for term in [
                "theo quy",
                "theo qui",
                "cac quy",
                "cac qui",
                "giua cac quy",
                "giua cac qui",
                "quy nao",
                "qui nao",
                "quy cao nhat",
                "qui cao nhat",
                "quy thap nhat",
                "qui thap nhat",
            ]
        )
    )


OlistAgent = Agent
