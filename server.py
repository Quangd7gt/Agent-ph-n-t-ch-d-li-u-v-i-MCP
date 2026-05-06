from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from psycopg import connect, sql
from psycopg.rows import dict_row

load_dotenv()
DEFAULT_SCHEMA = os.getenv("DEFAULT_SCHEMA", "analytics")
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(message)s",
)

mcp = FastMCP("olist-postgres-mcp")

PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDATABASE = os.getenv("PGDATABASE", "olist_db")
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "")
MAX_RETURN_ROWS = int(os.getenv("MAX_RETURN_ROWS", "200"))
DEFAULT_SCHEMA = os.getenv("DEFAULT_SCHEMA", "analytics")

FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "truncate",
    "create", "grant", "revoke", "comment", "copy", "call",
    "vacuum", "refresh", "merge", "set", "begin", "commit", "rollback"
}
SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
LEADING_SQL_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
LIMIT_RE = re.compile(r"\blimit\s+\d+\b", re.IGNORECASE)

BUSINESS_RULES = {
    "order_grain_definition": "analytics.fct_orders is one row per order_id.",
    "order_item_grain_definition": "analytics.fct_order_items is one row per order_id + order_item_id.",
    "revenue_definition": "Gross revenue is SUM(order_gross_value) from analytics.fct_orders for valid order statuses.",
    "repeat_customer_definition": "A repeat customer is a customer_unique_id with at least 2 distinct orders.",
    "delivery_delay_definition": "A delayed delivery is when order_delivered_customer_date > order_estimated_delivery_date.",
    "valid_revenue_statuses": ["delivered", "shipped", "invoiced", "processing"],
    "warning": "Do not sum payment_value_total from item-grain tables because it is an order-level measure."
}

VALID_REVENUE_STATUSES = ("delivered", "shipped", "invoiced", "processing")


def get_conn():
    return connect(
        host=PGHOST,
        port=PGPORT,
        dbname=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
        row_factory=dict_row,
    )


def is_safe_identifier(name: str) -> bool:
    return bool(SAFE_IDENTIFIER_RE.match(name))


def has_multiple_statements(query: str) -> bool:
    q = query.strip()
    if q.endswith(";"):
        q = q[:-1]
    return ";" in q


def detect_forbidden(query: str) -> list[str]:
    lowered = query.lower()
    hits = [kw for kw in FORBIDDEN_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", lowered)]
    return sorted(set(hits))


def validate_select_query(query: str) -> dict[str, Any]:
    q = query.strip()
    if not q:
        return {"ok": False, "reason": "Query is empty."}
    if has_multiple_statements(q):
        return {"ok": False, "reason": "Multiple SQL statements are not allowed."}
    if not LEADING_SQL_RE.search(q):
        return {"ok": False, "reason": "Only SELECT or WITH ... SELECT queries are allowed."}
    forbidden = detect_forbidden(q)
    if forbidden:
        return {"ok": False, "reason": f"Forbidden SQL keywords detected: {', '.join(forbidden)}"}
    return {"ok": True, "reason": "Query is read-only and allowed."}


def ensure_limit(query: str, row_limit: int) -> str:
    q = query.strip().rstrip(";")
    if LIMIT_RE.search(q):
        return q
    return f"{q} LIMIT {int(row_limit)}"


def explain_common_fix(error_text: str) -> str | None:
    lowered = error_text.lower()
    if "does not exist" in lowered:
        return "Check table and column names with get_schema_summary() or describe_table()."
    if "syntax error" in lowered:
        return "Re-check commas, aliases, GROUP BY columns, and date expressions."
    if "must appear in the group by clause" in lowered:
        return "Every non-aggregated selected column must also appear in GROUP BY."
    return None


@mcp.tool()
def ping() -> dict[str, Any]:
    """Health check for the MCP server and PostgreSQL connectivity."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT current_database() AS db, current_user AS usr, version() AS version")
            row = cur.fetchone()
        return {"ok": True, "database": row["db"], "user": row["usr"], "version": row["version"]}
    except Exception as exc:
        logging.exception("Ping failed")
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def get_business_rules() -> dict[str, Any]:
    """Return business definitions that the agent should follow for analytics."""
    return BUSINESS_RULES


@mcp.tool()
def list_tables() -> dict[str, Any]:
    """List all tables and views in the configured schema."""
    query = """
    SELECT table_name, table_type
    FROM information_schema.tables
    WHERE table_schema = %s
    ORDER BY table_type, table_name;
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(query, (DEFAULT_SCHEMA,))
        rows = cur.fetchall()
    return {"database": PGDATABASE, "schema": DEFAULT_SCHEMA, "count": len(rows), "objects": rows}


@mcp.tool()
def describe_table(table_name: str) -> dict[str, Any]:
    """Describe columns for a specific table or view in the configured schema."""
    if not is_safe_identifier(table_name):
        return {"ok": False, "error": "Invalid table name."}

    query = """
    SELECT
        c.column_name,
        c.data_type,
        c.is_nullable,
        c.column_default,
        tc.constraint_type
    FROM information_schema.columns c
    LEFT JOIN information_schema.key_column_usage kcu
      ON c.table_schema = kcu.table_schema
     AND c.table_name = kcu.table_name
     AND c.column_name = kcu.column_name
    LEFT JOIN information_schema.table_constraints tc
      ON kcu.constraint_name = tc.constraint_name
     AND kcu.table_schema = tc.table_schema
    WHERE c.table_schema = %s
      AND c.table_name = %s
    ORDER BY c.ordinal_position;
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(query, (DEFAULT_SCHEMA, table_name))
        columns = cur.fetchall()

    if not columns:
        return {"ok": False, "error": f"Table or view '{table_name}' not found in schema '{DEFAULT_SCHEMA}'."}

    return {"ok": True, "schema": DEFAULT_SCHEMA, "table_name": table_name, "columns": columns}


@mcp.tool()
def sample_rows(table_name: str, limit: int = 5) -> dict[str, Any]:
    """Return sample rows from a configured-schema table or view."""
    if not is_safe_identifier(table_name):
        return {"ok": False, "error": "Invalid table name."}
    limit = max(1, min(limit, 20))

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
            LIMIT 1;
            """,
            (DEFAULT_SCHEMA, table_name),
        )
        exists = cur.fetchone()
        if not exists:
            return {"ok": False, "error": f"Table or view '{table_name}' not found in schema '{DEFAULT_SCHEMA}'."}

        q = sql.SQL("SELECT * FROM {}.{} LIMIT {}") .format(
            sql.Identifier(DEFAULT_SCHEMA),
            sql.Identifier(table_name),
            sql.Literal(limit),
        )
        cur.execute(q)
        rows = cur.fetchall()

    return {"ok": True, "schema": DEFAULT_SCHEMA, "table_name": table_name, "limit": limit, "rows": rows}


@mcp.tool()
def get_schema_summary() -> dict[str, Any]:
    """Return a concise schema summary for the configured schema."""
    object_query = """
    SELECT table_name, table_type
    FROM information_schema.tables
    WHERE table_schema = %s
    ORDER BY table_name;
    """
    column_query = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = %s
    ORDER BY table_name, ordinal_position;
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(object_query, (DEFAULT_SCHEMA,))
        objects = cur.fetchall()
        cur.execute(column_query, (DEFAULT_SCHEMA,))
        columns = cur.fetchall()

    grouped: dict[str, Any] = {}
    for obj in objects:
        grouped[obj["table_name"]] = {
            "table_type": obj["table_type"],
            "columns": [],
        }
    for col in columns:
        grouped.setdefault(col["table_name"], {"table_type": "UNKNOWN", "columns": []})
        grouped[col["table_name"]]["columns"].append(
            {"column_name": col["column_name"], "data_type": col["data_type"]}
        )

    return {"database": PGDATABASE, "schema": DEFAULT_SCHEMA, "objects": grouped}


@mcp.tool()
def validate_query(query: str) -> dict[str, Any]:
    """Validate whether a SQL query is safe and read-only."""
    result = validate_select_query(query)
    result["normalized_query"] = query.strip().rstrip(";") if result["ok"] else None
    return result


@mcp.tool()
def run_select_query(query: str, row_limit: int = 100) -> dict[str, Any]:
    """Run a read-only SQL query against PostgreSQL and return rows."""
    row_limit = max(1, min(row_limit, MAX_RETURN_ROWS))
    validation = validate_select_query(query)
    if not validation["ok"]:
        return {"ok": False, "error": validation["reason"]}

    safe_query = ensure_limit(query, row_limit)
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(safe_query)
            rows = cur.fetchall()
        return {
            "ok": True,
            "row_limit": row_limit,
            "executed_query": safe_query,
            "row_count": len(rows),
            "rows": rows,
        }
    except Exception as exc:
        logging.exception("Query execution failed")
        return {
            "ok": False,
            "error": str(exc),
            "suggestion": explain_common_fix(str(exc)),
            "executed_query": safe_query,
        }


@mcp.tool()
def revenue_by_month(year: int) -> dict[str, Any]:
    """Return monthly gross revenue from analytics.fct_orders for a given year."""
    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"
    query = """
    SELECT
        date_trunc('month', order_purchase_timestamp) AS month,
        SUM(order_gross_value) AS revenue,
        COUNT(DISTINCT order_id) AS order_count
    FROM analytics.fct_orders
    WHERE order_purchase_timestamp >= %s
      AND order_purchase_timestamp < %s
      AND order_status = ANY(%s)
    GROUP BY 1
    ORDER BY 1;
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(query, (start_date, end_date, list(VALID_REVENUE_STATUSES)))
            rows = cur.fetchall()
        return {"ok": True, "year": year, "rows": rows}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def top_categories(start_date: str, end_date: str, limit: int = 10) -> dict[str, Any]:
    """Return top product categories by gross revenue for a date range."""
    limit = max(1, min(limit, 50))
    query = """
    SELECT
        COALESCE(product_category_name_english, product_category_name, 'unknown') AS category,
        SUM(line_total) AS revenue,
        COUNT(DISTINCT order_id) AS order_count
    FROM analytics.fct_order_items
    WHERE order_purchase_timestamp::date BETWEEN %s::date AND %s::date
      AND order_status = ANY(%s)
    GROUP BY 1
    ORDER BY revenue DESC
    LIMIT %s;
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(query, (start_date, end_date, list(VALID_REVENUE_STATUSES), limit))
            rows = cur.fetchall()
        return {"ok": True, "start_date": start_date, "end_date": end_date, "limit": limit, "rows": rows}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def delivery_delay_summary(start_date: str, end_date: str) -> dict[str, Any]:
    """Return delivery delay metrics for delivered orders within a date range."""
    query = """
    SELECT
        COUNT(*) AS delivered_orders,
        COUNT(*) FILTER (WHERE is_delayed_delivery IS TRUE) AS delayed_orders,
        ROUND(
            100.0 * COUNT(*) FILTER (WHERE is_delayed_delivery IS TRUE) / NULLIF(COUNT(*), 0),
            2
        ) AS delayed_order_rate_pct,
        ROUND(AVG(delivery_days)::numeric, 2) AS avg_delivery_days,
        ROUND(AVG(review_score_avg)::numeric, 2) AS avg_review_score
    FROM analytics.fct_orders
    WHERE order_purchase_timestamp::date BETWEEN %s::date AND %s::date
      AND order_status = 'delivered';
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(query, (start_date, end_date))
            row = cur.fetchone()
        return {"ok": True, "start_date": start_date, "end_date": end_date, "summary": row}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def repeat_customer_rate(start_date: str, end_date: str) -> dict[str, Any]:
    """Return repeat customer rate using customer_unique_id and order grain."""
    query = """
    WITH customer_orders AS (
        SELECT
            customer_unique_id,
            COUNT(DISTINCT order_id) AS order_count
        FROM analytics.fct_orders
        WHERE order_purchase_timestamp::date BETWEEN %s::date AND %s::date
          AND order_status = ANY(%s)
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
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(query, (start_date, end_date, list(VALID_REVENUE_STATUSES)))
            row = cur.fetchone()
        return {"ok": True, "start_date": start_date, "end_date": end_date, "summary": row}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def suggest_analytics_queries() -> dict[str, Any]:
    """Return example analytics questions and SQL patterns for the Olist database."""
    return {
        "examples": [
            {
                "question": "Doanh thu theo tháng trong năm 2018",
                "sql_pattern": "SELECT date_trunc('month', order_purchase_timestamp) AS month, SUM(order_gross_value) AS revenue FROM analytics.fct_orders WHERE order_status IN ('delivered','shipped','invoiced','processing') AND order_purchase_timestamp >= '2018-01-01' AND order_purchase_timestamp < '2019-01-01' GROUP BY 1 ORDER BY 1"
            },
            {
                "question": "Top 10 danh mục sản phẩm theo doanh thu trong quý 1 năm 2018",
                "sql_pattern": "SELECT COALESCE(product_category_name_english, product_category_name, 'unknown') AS category, SUM(line_total) AS revenue FROM analytics.fct_order_items WHERE order_status IN ('delivered','shipped','invoiced','processing') AND order_purchase_timestamp >= '2018-01-01' AND order_purchase_timestamp < '2018-04-01' GROUP BY 1 ORDER BY revenue DESC LIMIT 10"
            },
            {
                "question": "Tỷ lệ khách hàng quay lại trong năm 2018",
                "sql_pattern": "WITH customer_orders AS (SELECT customer_unique_id, COUNT(DISTINCT order_id) AS order_count FROM analytics.fct_orders WHERE order_status IN ('delivered','shipped','invoiced','processing') AND order_purchase_timestamp >= '2018-01-01' AND order_purchase_timestamp < '2019-01-01' GROUP BY 1) SELECT COUNT(*) FILTER (WHERE order_count >= 2) * 100.0 / NULLIF(COUNT(*),0) AS repeat_rate_pct FROM customer_orders"
            }
        ]
    }

def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
