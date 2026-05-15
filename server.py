from __future__ import annotations

import logging
import os
import re
import sys
import json
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from psycopg import connect, sql
from psycopg.rows import dict_row

load_dotenv()
DEFAULT_SCHEMA = os.getenv("DEFAULT_SCHEMA", "analytics")
AGENT_API_URL = os.getenv("AGENT_API_URL", "http://127.0.0.1:8000/ask")
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

def make_agent():
    from agent.agent import OlistAgent

    return OlistAgent()


def load_system_prompt() -> str:
    return make_agent().load_system_prompt()


def call_agent_api(question: str, output_path: str = "") -> dict[str, Any]:
    payload = json.dumps({"question": question, "output_path": output_path}).encode("utf-8")
    request = Request(
        AGENT_API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=int(os.getenv("AGENT_API_TIMEOUT", "600"))) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def allow_agent_file_output(question: str, output_path: str) -> bool:
    if not output_path:
        return False
    lowered = question.lower()
    file_terms = [
        "tao file",
        "tạo file",
        "tao bao cao",
        "tạo báo cáo",
        "xuat bao cao",
        "xuất báo cáo",
        "luu bao cao",
        "lưu báo cáo",
        "export",
        "report file",
        "html",
    ]
    return any(term in lowered for term in file_terms)


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
    "Kiểm tra tình trạng hoạt động của máy chủ MCP và kết nối PostgreSQL."
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
    "Các định nghĩa nghiệp vụ"
    return make_agent().get_business_rules()


@mcp.tool()
def business_rules_agent(question: str) -> dict[str, Any]:
    " Trả lời câu hỏi về các quy tắc nghiệp vụ"
    try:
        return make_agent().answer_business_rule_question(question)
    except Exception as exc:
        logging.exception("business_rules_agent failed")
        return {"ok": False, "error": str(exc)}


@mcp.prompt(
    name="olist_data_analyst",
    title="Olist Data Analyst System Prompt",
    description="Instructions for using this MCP server as an Olist e-commerce data analyst.",
)
def olist_data_analyst_prompt() -> str:
    """Return the Olist data analyst prompt from agent/system_prompt.txt."""
    return load_system_prompt()


@mcp.resource(
    "prompt://olist/system",
    name="olist_system_prompt",
    title="Olist System Prompt",
    description="The system prompt used by the Olist MCP data analyst agent.",
    mime_type="text/markdown",
)
def olist_system_prompt_resource() -> str:
    return load_system_prompt()


@mcp.tool()
def get_system_prompt() -> dict[str, Any]:
    "Lời nhắc hệ thống cần thiết để hướng dẫn tác nhân phân tích"
    try:
        return make_agent().get_system_prompt_info()
    except Exception as exc:
        logging.exception("Unable to load system prompt")
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def list_tables() -> dict[str, Any]:
    "Liệt kê tất cả các bảng và chế độ xem trong lược đồ đã cấu hình."
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
    "Mô tả các cột cho một bảng hoặc chế độ xem cụ thể trong lược đồ đã cấu hình."
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
    "Trả về các hàng mẫu từ bảng hoặc chế độ xem có cấu hình lược đồ."
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
    "Trả về bản tóm tắt lược đồ ngắn gọn cho lược đồ đã được cấu hình"
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
    "Kiểm tra xem truy vấn SQL có an toàn và chỉ đọc hay không"
    result = validate_select_query(query)
    result["normalized_query"] = query.strip().rstrip(";") if result["ok"] else None
    return result


@mcp.tool()
def run_select_query(query: str, row_limit: int = 100) -> dict[str, Any]:
    "Chạy truy vấn SQL chỉ đọc trên PostgreSQL và trả về các hàng."
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
    "Doanh thu gộp hàng tháng từ analytics.fct_orders cho một năm cụ thể."
    try:
        agent = make_agent()
        df = agent.analyze_revenue_by_month(year)
        return {"ok": True, "year": year, "rows": agent.records(df)}
    except Exception as exc:
        logging.exception("revenue_by_month failed")
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def top_categories(start_date: str, end_date: str, limit: int = 10) -> dict[str, Any]:
    "Các danh mục sản phẩm hàng đầu theo doanh thu gộp trong một khoảng thời gian nhất định.s"
    try:
        agent = make_agent()
        limit = agent.clamp_limit(limit)
        df = agent.analyze_top_categories(start_date, end_date, limit)
        rows = agent.records(df)
        return {"ok": True, "start_date": start_date, "end_date": end_date, "limit": limit, "rows": rows}
    except Exception as exc:
        logging.exception("top_categories failed")
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def delivery_delay_summary(start_date: str, end_date: str) -> dict[str, Any]:
    "Số liệu thống kê về thời gian giao hàng trả lại đối với các đơn hàng đã giao trong một khoảng thời gian nhất định."
    try:
        agent = make_agent()
        df = agent.analyze_delivery_delay_summary(start_date, end_date)
        rows = agent.records(df)
        row = rows[0] if rows else {}
        return {"ok": True, "start_date": start_date, "end_date": end_date, "summary": row}
    except Exception as exc:
        logging.exception("delivery_delay_summary failed")
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def repeat_customer_rate(start_date: str, end_date: str) -> dict[str, Any]:
    "Tính toán tỷ lệ khách hàng quay lại dựa trên customer_unique_id và order grain."
    try:
        agent = make_agent()
        df = agent.analyze_repeat_customer_rate(start_date, end_date)
        rows = agent.records(df)
        row = rows[0] if rows else {}
        return {"ok": True, "start_date": start_date, "end_date": end_date, "summary": row}
    except Exception as exc:
        logging.exception("repeat_customer_rate failed")
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def gemma_runtime_status() -> dict[str, Any]:
    "Các thiết lập thời gian chạy cục bộ của Gemma hiển thị cho máy chủ MCP mà không cần tải mô hình."
    return make_agent().gemma_runtime_status()


@mcp.tool()
def agent(question: str, output_path: str = "") -> dict[str, Any]:
    "Trả lời các câu hỏi chính"
    try:
        requested_output_path = output_path
        effective_output_path = output_path if allow_agent_file_output(question, output_path) else ""
        result = call_agent_api(question=question, output_path=effective_output_path)
        if isinstance(result, dict):
            if requested_output_path and not effective_output_path:
                result["ignored_output_path"] = requested_output_path
                result["ignored_output_path_reason"] = (
                    "output_path was ignored because the question did not explicitly ask agent.py to create/export a report file."
                )
            result.setdefault(
                "workspace_policy",
                {
                    "handled_by": "agent.py",
                    "caller_should_create_files": False,
                    "caller_should_edit_files": False,
                    "chat_only": not bool(effective_output_path),
                    "file_output_rule": (
                        "Only agent.py may create a report file, and only when output_path is provided."
                    ),
                },
            )
        return result
    except URLError as exc:
        logging.exception("Gemma agent API is unavailable")
        return {
            "ok": False,
            "error": str(exc),
            "agent_api_url": AGENT_API_URL,
            "suggestion": "Start the agent API first with: python agent_api.py",
        }
    except Exception as exc:
        logging.exception("Gemma agent failed")
        return {"ok": False, "error": str(exc), "agent_api_url": AGENT_API_URL}
def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
