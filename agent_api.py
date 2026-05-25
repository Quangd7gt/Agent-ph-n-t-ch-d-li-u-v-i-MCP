from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sys
from threading import Lock
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import BaseModel, Field
from sqlalchemy import text

from agent.agent import Agent

load_dotenv()
WORKSPACE_ROOT = Path(__file__).resolve().parent
FRONTEND_DIST = WORKSPACE_ROOT / "web_dist"
LEGACY_UI_INDEX = WORKSPACE_ROOT / "web" / "index.html"
MCP_SERVER_SCRIPT = WORKSPACE_ROOT / "server.py"
MCP_TOOL_NAME = os.getenv("MCP_AGENT_TOOL", "gemma_agent")
MCP_SHARED_URL = os.getenv("MCP_SHARED_URL", "http://127.0.0.1:8010/mcp").strip()
CHAT_SCHEMA = os.getenv("CHAT_SCHEMA", "app")
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "30"))

agent = Agent()
agent_lock = Lock()
chat_lock = Lock()
startup_error: str | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    output_path: str = ""
    conversation_id: str | None = None


class RequestCancelled(Exception):
    pass


def sql_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid SQL identifier: {value}")
    return f'"{value}"'


CHAT_SCHEMA_SQL = sql_identifier(CHAT_SCHEMA)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def title_from_question(question: str) -> str:
    clean = " ".join(question.strip().split())
    if not clean:
        return "New chat"
    return f"{clean[:54]}..." if len(clean) > 54 else clean


def _looks_like_raw_json(text: str) -> bool:
    """Kiểm tra xem text có phải JSON thô (bắt đầu bằng [ hoặc {)."""
    if not text or not isinstance(text, str):
        return False
    trimmed = text.lstrip()
    return len(trimmed) > 20 and trimmed[0] in ("[", "{")


def _summarize_rows(result: dict[str, Any]) -> str:
    """Tạo tóm tắt tiếng Việt từ rows khi không có analysis."""
    rows = result.get("rows")
    if not isinstance(rows, list) or not rows:
        return ""
    first = rows[0] if isinstance(rows[0], dict) else {}
    if not first:
        return f"Đã truy vấn được {len(rows)} dòng dữ liệu."

    name = first.get("category") or first.get("state") or first.get("product_category_name") or first.get("month") or first.get("quarter") or ""
    parts: list[str] = []
    if name:
        parts.append(f"Đứng đầu: {name}")
    revenue = first.get("revenue")
    if revenue is not None:
        parts.append(f"doanh thu {revenue:,.2f}")
    orders = first.get("order_count") or first.get("total_orders")
    if orders is not None:
        parts.append(f"{orders:,} đơn hàng")
    review = first.get("avg_review_score")
    if review is not None:
        parts.append(f"review trung bình {review}")

    if not parts:
        return f"Đã truy vấn được {len(rows)} dòng dữ liệu."
    return f"{', '.join(parts)}. Tổng cộng {len(rows)} dòng dữ liệu — xem bảng chi tiết bên dưới."


def assistant_text_from_result(result: dict[str, Any]) -> str:
    result = sanitize_agent_result(result)
    if result.get("needs_clarification"):
        return result.get("clarifying_question") or result.get("reason") or "Cần bổ sung thông tin."
    for key in ("analysis", "answer", "safe_summary", "error", "text"):
        value = result.get(key)
        if value and not _looks_like_raw_json(str(value)):
            return str(value)
    summary = _summarize_rows(result)
    if summary:
        return summary
    if result.get("ok"):
        return "Đã nhận được kết quả từ agent."
    return "Không có dữ liệu trả lời."


def json_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def parse_jsonb(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


BAD_ANALYSIS_RE = re.compile(
    r"(?i)(?:"
    r"(?:\*\*)?\s*(?:c[ảa]nh|canh|k[ịi]ch\s*b[ảa]n|kich\s*ban)\s*\d+"
    r"|c[âa]u\s*h[ỏo]i\s*ph[âa]n\s*t[ií]ch\s*d[ữu]\s*li[ệe]u"
    r"|cau\s*hoi\s*phan\s*tich\s*du\s*lieu"
    r"|ph[âa]n\s*t[ií]ch\s*5\s*danh\s*m[ụu]c\s*s[ảa]n\s*ph[ẩa]m\s*t[ốo]t\s*nh[ấa]t"
    r"|phan\s*tich\s*5\s*danh\s*muc\s*san\s*pham\s*tot\s*nhat"
    r")"
)


def looks_like_bad_analysis(text_value: Any) -> bool:
    if not isinstance(text_value, str):
        return False
    return bool(BAD_ANALYSIS_RE.search(text_value))


def sanitize_agent_result(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        return result
    clean = dict(result)
    safe_summary = clean.get("safe_summary")
    if safe_summary and looks_like_bad_analysis(clean.get("analysis")):
        clean["analysis_rejected"] = clean.get("analysis")
        clean["analysis_rejected_reason"] = "Gemma output contained multiple scenes or unrelated analysis sections."
        clean["analysis"] = safe_summary
        clean["analysis_source"] = "safe_summary_after_bad_gemma_output"
    return clean


def ensure_chat_storage() -> None:
    with agent.engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {CHAT_SCHEMA_SQL}"))
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {CHAT_SCHEMA_SQL}.conversations (
                    id text PRIMARY KEY,
                    title text NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {CHAT_SCHEMA_SQL}.messages (
                    id text PRIMARY KEY,
                    conversation_id text NOT NULL
                        REFERENCES {CHAT_SCHEMA_SQL}.conversations(id)
                        ON DELETE CASCADE,
                    role text NOT NULL CHECK (role IN ('user', 'assistant')),
                    text text NOT NULL,
                    payload jsonb,
                    position integer NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_position
                ON {CHAT_SCHEMA_SQL}.messages(conversation_id, position)
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS idx_chat_conversations_updated_at
                ON {CHAT_SCHEMA_SQL}.conversations(updated_at DESC)
                """
            )
        )


def serialize_message(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "role": row["role"],
        "text": row["text"],
        "payload": parse_jsonb(row["payload"]),
        "position": row["position"],
        "createdAt": row["created_at"].isoformat() if row["created_at"] else None,
    }


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    with agent.engine.connect() as conn:
        conversation = conn.execute(
            text(
                f"""
                SELECT id, title, created_at, updated_at
                FROM {CHAT_SCHEMA_SQL}.conversations
                WHERE id = :conversation_id
                """
            ),
            {"conversation_id": conversation_id},
        ).mappings().first()
        if conversation is None:
            return None

        messages = conn.execute(
            text(
                f"""
                SELECT id, role, text, payload, position, created_at
                FROM {CHAT_SCHEMA_SQL}.messages
                WHERE conversation_id = :conversation_id
                ORDER BY position ASC
                """
            ),
            {"conversation_id": conversation_id},
        ).mappings().all()

    return {
        "id": conversation["id"],
        "title": conversation["title"],
        "createdAt": conversation["created_at"].isoformat() if conversation["created_at"] else None,
        "updatedAt": conversation["updated_at"].isoformat() if conversation["updated_at"] else None,
        "messages": [serialize_message(row) for row in messages],
    }


def list_conversations(limit: int = CHAT_HISTORY_LIMIT) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 100))
    with agent.engine.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT id
                FROM {CHAT_SCHEMA_SQL}.conversations
                ORDER BY updated_at DESC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        ).mappings().all()
    return [
        conversation
        for row in rows
        if (conversation := get_conversation(row["id"])) is not None
    ]


def delete_conversation(conversation_id: str) -> bool:
    with agent.engine.begin() as conn:
        result = conn.execute(
            text(
                f"""
                DELETE FROM {CHAT_SCHEMA_SQL}.conversations
                WHERE id = :conversation_id
                """
            ),
            {"conversation_id": conversation_id},
        )
    return result.rowcount > 0


def save_chat_exchange(
    conversation_id: str | None,
    question: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    result = sanitize_agent_result(result)
    current_time = utc_now()
    conversation_id = conversation_id or make_id("chat")
    title = title_from_question(question)
    user_message_id = make_id("msg")
    assistant_message_id = make_id("msg")
    assistant_text = assistant_text_from_result(result)

    with agent.engine.begin() as conn:
        conn.execute(
            text(
                f"""
                INSERT INTO {CHAT_SCHEMA_SQL}.conversations AS c
                    (id, title, created_at, updated_at)
                VALUES
                    (:id, :title, :created_at, :updated_at)
                ON CONFLICT (id) DO UPDATE SET
                    updated_at = EXCLUDED.updated_at,
                    title = CASE
                        WHEN c.title = 'New chat' THEN EXCLUDED.title
                        ELSE c.title
                    END
                """
            ),
            {
                "id": conversation_id,
                "title": title,
                "created_at": current_time,
                "updated_at": current_time,
            },
        )
        position = conn.execute(
            text(
                f"""
                SELECT COALESCE(MAX(position), -1) + 1 AS next_position
                FROM {CHAT_SCHEMA_SQL}.messages
                WHERE conversation_id = :conversation_id
                """
            ),
            {"conversation_id": conversation_id},
        ).scalar_one()
        conn.execute(
            text(
                f"""
                INSERT INTO {CHAT_SCHEMA_SQL}.messages
                    (id, conversation_id, role, text, payload, position, created_at)
                VALUES
                    (:id, :conversation_id, 'user', :text, NULL, :position, :created_at),
                    (:assistant_id, :conversation_id, 'assistant', :assistant_text,
                     CAST(:assistant_payload AS jsonb), :assistant_position, :created_at)
                """
            ),
            {
                "id": user_message_id,
                "assistant_id": assistant_message_id,
                "conversation_id": conversation_id,
                "text": question,
                "assistant_text": assistant_text,
                "assistant_payload": json_payload(result),
                "position": position,
                "assistant_position": position + 1,
                "created_at": current_time,
            },
        )

    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise RuntimeError("Conversation was not saved.")
    return conversation


def find_last_user_question_in_conversation(conversation_id: str | None, *, require_clarification: bool = False) -> str | None:
    """Tìm câu hỏi cuối cùng của user trong cuộc trò chuyện.

    Nếu require_clarification=True, chỉ trả về khi assistant yêu cầu clarification.
    Nếu require_clarification=False, trả về câu hỏi gần nhất mà assistant đã trả lời thành công.
    """
    if not conversation_id:
        return None
    conversation = get_conversation(conversation_id)
    if not conversation:
        return None
    messages = conversation.get("messages") or []
    if not messages:
        return None

    # Tìm cặp user→assistant cuối cùng
    last_user_text: str | None = None
    last_assistant_payload: dict | None = None

    for message in reversed(messages):
        role = message.get("role")
        if role == "assistant" and last_assistant_payload is None:
            last_assistant_payload = message.get("payload") or {}
        elif role == "user" and last_assistant_payload is not None:
            last_user_text = str(message.get("text") or "").strip()
            break

    if not last_user_text:
        return None

    if require_clarification:
        if isinstance(last_assistant_payload, dict) and last_assistant_payload.get("needs_clarification"):
            return last_user_text
        return None

    # Chỉ trả về nếu assistant đã trả lời thành công (ok=True)
    if isinstance(last_assistant_payload, dict) and last_assistant_payload.get("ok"):
        return last_user_text
    return None


# Các mẫu follow-up tiếng Việt
_FOLLOWUP_PATTERNS = [
    # "còn năm 2017 thì sao", "còn 2017", "thế năm 2017", "vậy năm 2017"
    r"(?:con|the|vay|nhu vay|nhu the|entao|roi)\s+(?:nam\s*)?20\d{2}",
    # "còn quý 1 thì sao", "thế quý 2 năm 2017"
    r"(?:con|the|vay|entao|roi)\s+(?:quy|qui|q)\s*[1-4]",
    # "năm 2017 thì sao", "2017 thì thế nào"
    r"(?:nam\s*)?20\d{2}\s+(?:thi\s+)?(?:sao|the\s*nao|nhu\s*the\s*nao)",
    # Thuần thời gian ngắn: "2017", "năm 2017", "quý 1 năm 2017"
    r"(?:nam\s*)?20\d{2}$",
    r"(?:quy|qui|q)\s*[1-4](?:\s*nam\s*20\d{2})?$",
    r"thang\s*(?:1[0-2]|0?[1-9])(?:\s*nam\s*20\d{2})?$",
    r"20\d{2}-\d{2}-\d{2}\s*(?:den|toi|-)\s*20\d{2}-\d{2}-\d{2}$",
]


def is_followup_question(question: str) -> bool:
    """Nhận diện câu hỏi follow-up mang ngữ cảnh thời gian."""
    normalized = normalize_text(question).strip()
    # Loại bỏ dấu ? cuối
    normalized = normalized.rstrip("?").strip()
    return any(re.search(pattern, normalized) for pattern in _FOLLOWUP_PATTERNS)


def _extract_time_from_followup(question: str) -> dict[str, str | None]:
    """Trích xuất thông tin thời gian từ câu follow-up."""
    normalized = normalize_text(question).strip().rstrip("?").strip()
    result: dict[str, str | None] = {"year": None, "quarter": None, "month": None}

    year_match = re.search(r"(20\d{2})", normalized)
    if year_match:
        result["year"] = year_match.group(1)

    quarter_match = re.search(r"(?:quy|qui|q)\s*([1-4])", normalized)
    if quarter_match:
        result["quarter"] = quarter_match.group(1)

    month_match = re.search(r"thang\s*(1[0-2]|0?[1-9])", normalized)
    if month_match:
        result["month"] = month_match.group(1)

    return result


def merge_followup_with_previous(previous_question: str, followup_question: str) -> str:
    """Ghép câu follow-up vào câu hỏi trước bằng cách thay thế thời gian."""
    time_info = _extract_time_from_followup(followup_question)

    result = previous_question

    if time_info["year"]:
        # Thay năm cũ bằng năm mới
        result = re.sub(r"(?:năm\s*)?20\d{2}", f"năm {time_info['year']}", result, count=1)
        # Nếu không tìm thấy năm trong câu cũ, thêm vào cuối
        if time_info["year"] not in result:
            result = f"{result} năm {time_info['year']}"

    if time_info["quarter"]:
        quarter_replaced = re.sub(r"(?:quý|quy|qui|q)\s*[1-4]", f"quý {time_info['quarter']}", result, count=1)
        if quarter_replaced != result:
            result = quarter_replaced
        else:
            result = f"{result} quý {time_info['quarter']}"

    if time_info["month"]:
        month_replaced = re.sub(r"tháng\s*(?:1[0-2]|0?[1-9])", f"tháng {time_info['month']}", result, count=1)
        if month_replaced != result:
            result = month_replaced
        else:
            result = f"{result} tháng {time_info['month']}"

    return result


def resolve_followup_question(conversation_id: str | None, question: str) -> tuple[str, dict[str, Any] | None]:
    """Giải quyết câu hỏi follow-up bằng cách ghép với câu hỏi trước.

    Ví dụ:
      Câu trước: "tỷ lệ khách hàng quay lại năm 2018 là bao nhiêu?"
      Follow-up: "còn năm 2017 thì sao"
      → Resolved: "tỷ lệ khách hàng quay lại năm 2017 là bao nhiêu?"
    """
    if not is_followup_question(question):
        return question, None

    # Ưu tiên tìm câu hỏi cần clarification trước
    previous = find_last_user_question_in_conversation(conversation_id, require_clarification=True)
    resolution_type = "merged_with_previous_question_after_clarification"

    # Nếu không có, tìm câu hỏi thành công gần nhất
    if not previous:
        previous = find_last_user_question_in_conversation(conversation_id, require_clarification=False)
        resolution_type = "merged_with_previous_successful_question"

    if not previous:
        return question, None

    resolved = merge_followup_with_previous(previous, question)
    return resolved, {
        "original_question": question,
        "previous_question": previous,
        "resolved_question": resolved,
        "followup_resolution": resolution_type,
    }



def mcp_result_payload(result: Any) -> Any:
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent

    content = getattr(result, "content", None) or []
    text_parts = []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            text_parts.append(text)
    text = "\n".join(text_parts).strip()
    if not text:
        return {"ok": False, "error": "MCP tool returned no content."}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"ok": True, "text": text}


def normalize_text(value: str) -> str:
    import unicodedata

    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("đ", "d").replace("Đ", "D")
    return value.lower()


def extract_table_name(question: str) -> str | None:
    normalized = normalize_text(question)
    table_aliases = {
        "orders": "fct_orders",
        "order_items": "fct_order_items",
        "products": "dim_products",
        "customers": "dim_customers",
        "sellers": "dim_sellers",
        "order_payments": "order_payments_summary",
        "order_reviews": "order_reviews_summary",
    }
    known_tables = [
        "fct_order_items",
        "fct_orders",
        "dim_customers",
        "dim_products",
        "dim_sellers",
        "order_payments_summary",
        "order_reviews_summary",
        "category_translation",
        "customers",
        "geolocation",
        "order_items",
        "order_payments",
        "order_reviews",
        "orders",
        "products",
        "sellers",
    ]
    for table in known_tables:
        if table in normalized:
            return table_aliases.get(table, table)

    match = re.search(r"(?:bang|table|view)\s+([A-Za-z_][A-Za-z0-9_]*)", normalized)
    if not match:
        return None
    table_name = match.group(1)
    return table_aliases.get(table_name, table_name)


def select_mcp_tool(question: str, output_path: str = "") -> tuple[str, dict[str, Any]]:
    normalized = normalize_text(question)
    table_name = extract_table_name(question)

    if "ping" in normalized or ("trang thai" in normalized and "mcp" in normalized):
        return "ping", {}

    if any(term in normalized for term in ["business rule", "quy tac", "quy dinh nghiep vu", "doanh thu tinh"]):
        return "business_rules_agent", {"question": question}

    wants_sample = any(term in normalized for term in ["du lieu mau", "hang mau", "sample", "xem mau"])
    if wants_sample and table_name:
        return "sample_rows", {"table_name": table_name, "limit": 5}

    wants_describe = any(
        term in normalized
        for term in ["mo ta bang", "mo ta table", "cot cua bang", "cac cot", "columns", "describe"]
    )
    if wants_describe and table_name:
        return "describe_table", {"table_name": table_name}

    wants_schema_summary = any(
        term in normalized
        for term in ["tom tat schema", "schema summary", "luoc do", "cau truc du lieu", "schema"]
    )
    if wants_schema_summary:
        return "get_schema_summary", {}

    wants_tables = (
        any(term in normalized for term in ["liet ke", "danh sach", "co nhung", "cac bang", "nhung bang"])
        and any(term in normalized for term in ["bang", "table", "schema", "du lieu", "database"])
    )
    if wants_tables:
        return "list_tables", {}

    # Business analysis questions should go through gemma_agent so agent.py both queries
    # data and returns an analysis, instead of exposing a raw helper-tool result.
    return MCP_TOOL_NAME, {"question": question, "output_path": output_path}


def normalize_direct_tool_result(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "list_tables" and isinstance(result.get("objects"), list):
        result.setdefault("rows", result["objects"])
        result.setdefault(
            "analysis",
            f"Schema {result.get('schema', '')} có {result.get('count', len(result['objects']))} bảng/view.",
        )
    elif tool_name == "describe_table" and isinstance(result.get("columns"), list):
        result.setdefault("rows", result["columns"])
        result.setdefault(
            "analysis",
            f"Bảng/view {result.get('schema', '')}.{result.get('table_name', '')} có {len(result['columns'])} cột.",
        )
    elif tool_name == "sample_rows" and isinstance(result.get("rows"), list):
        result.setdefault(
            "analysis",
            f"Dữ liệu mẫu từ {result.get('schema', '')}.{result.get('table_name', '')}.",
        )
    elif tool_name == "get_schema_summary" and isinstance(result.get("objects"), dict):
        rows = [
            {
                "table_name": table_name,
                "table_type": details.get("table_type"),
                "column_count": len(details.get("columns") or []),
                "columns": ", ".join(column.get("column_name", "") for column in (details.get("columns") or [])),
            }
            for table_name, details in result["objects"].items()
        ]
        result.setdefault("rows", rows)
        result.setdefault(
            "analysis",
            f"Schema {result.get('schema', '')} có {len(rows)} bảng/view.",
        )
    elif tool_name == "ping" and result.get("ok"):
        result.setdefault("analysis", "MCP server và PostgreSQL đang hoạt động.")
    elif tool_name == "business_rules_agent" and result.get("answer"):
        result.setdefault("analysis", result["answer"])
    elif tool_name == "top_categories" and isinstance(result.get("rows"), list):
        result.setdefault(
            "analysis",
            f"Đã tính doanh thu theo danh mục/sản phẩm từ {result.get('start_date')} đến {result.get('end_date')}.",
        )
    elif tool_name == "revenue_by_month" and isinstance(result.get("rows"), list):
        result.setdefault("analysis", f"Đã tính doanh thu theo tháng cho năm {result.get('year')}.")
    return result


def result_has_analysis(result: dict[str, Any]) -> bool:
    return any(result.get(key) for key in ("safe_summary", "analysis", "answer", "text"))


def ensure_result_analysis(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("ok") or result.get("needs_clarification") or result_has_analysis(result):
        return result

    rows = result.get("rows")
    summary = result.get("summary")
    if isinstance(rows, list):
        result["analysis"] = (
            f"Tool {tool_name} đã trả {len(rows)} dòng dữ liệu phù hợp với câu hỏi. "
            "Các dòng này là dữ liệu nền để đọc insight chính trong bảng kết quả."
        )
    elif isinstance(summary, dict) and summary:
        result["analysis"] = (
            f"Tool {tool_name} đã trả một bản tóm tắt dữ liệu. "
            "Các chỉ số trong phần tóm tắt là cơ sở để diễn giải câu trả lời."
        )
    else:
        result["analysis"] = (
            f"Tool {tool_name} đã chạy thành công và trả về dữ liệu có cấu trúc. "
            "Kết quả này cần được đọc trực tiếp từ payload kèm theo."
        )
    result.setdefault("analysis_source", "fallback_tool_summary")
    return result


async def call_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    timeout = timedelta(seconds=int(os.getenv("MCP_BRIDGE_TIMEOUT", "900")))
    if MCP_SHARED_URL:
        async with streamablehttp_client(
            MCP_SHARED_URL,
            timeout=timeout,
            sse_read_timeout=timeout,
        ) as (read, write, _get_session_id):
            async with ClientSession(read, write, read_timeout_seconds=timeout) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments, read_timeout_seconds=timeout)

        payload = mcp_result_payload(result)
        if isinstance(payload, dict):
            payload.setdefault("via", "shared_mcp")
            payload.setdefault("mcp_tool", tool_name)
            payload.setdefault("mcp_server_url", MCP_SHARED_URL)
            return payload
        return {"ok": True, "via": "shared_mcp", "mcp_tool": tool_name, "result": payload}

    env = os.environ.copy()
    env.setdefault("AGENT_API_URL", os.getenv("AGENT_API_URL", "http://127.0.0.1:8000/ask"))
    server = StdioServerParameters(
        command=os.getenv("MCP_PYTHON", sys.executable),
        args=[str(MCP_SERVER_SCRIPT)],
        cwd=str(WORKSPACE_ROOT),
        env=env,
        encoding="utf-8",
        encoding_error_handler="replace",
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timeout) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments, read_timeout_seconds=timeout)

    payload = mcp_result_payload(result)
    if isinstance(payload, dict):
        payload.setdefault("via", "mcp")
        payload.setdefault("mcp_tool", tool_name)
        payload.setdefault("mcp_server", str(MCP_SERVER_SCRIPT))
        return payload
    return {"ok": True, "via": "mcp", "mcp_tool": tool_name, "result": payload}


async def call_mcp_tool_until_disconnect(
    request: Request,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    task = asyncio.create_task(call_mcp_tool(tool_name, arguments))
    try:
        while not task.done():
            if await request.is_disconnected():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                raise RequestCancelled()
            await asyncio.sleep(0.25)
        return await task
    except asyncio.CancelledError:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        raise RequestCancelled() from None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_error
    try:
        ensure_chat_storage()
    except Exception as exc:
        startup_error = str(exc)
    if os.getenv("AGENT_API_PRELOAD_GEMMA", "true").lower() in {"1", "true", "yes", "on"}:
        try:
            agent.ensure_gemma()
        except Exception as exc:
            startup_error = str(exc)
    yield


app = FastAPI(
    title="Olist Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount(
    "/assets",
    StaticFiles(directory=FRONTEND_DIST / "assets", check_dir=False),
    name="frontend-assets",
)


def ui_index_path() -> Path:
    built_index = FRONTEND_DIST / "index.html"
    return built_index if built_index.exists() else LEGACY_UI_INDEX


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ui_index_path())


@app.get("/app", include_in_schema=False)
def customer_app() -> FileResponse:
    return FileResponse(ui_index_path())


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "olist-gemma-agent-api"}


@app.get("/status")
def status() -> dict[str, Any]:
    status_result = agent.gemma_runtime_status()
    status_result["preload_error"] = startup_error
    status_result["model_loaded"] = agent.gemma is not None
    return status_result


@app.get("/conversations")
def conversations() -> dict[str, Any]:
    with chat_lock:
        return {"ok": True, "conversations": list_conversations()}


@app.get("/conversations/{conversation_id}")
def conversation_detail(conversation_id: str) -> dict[str, Any]:
    with chat_lock:
        conversation = get_conversation(conversation_id)
    if conversation is None:
        return {"ok": False, "error": "Conversation not found."}
    return {"ok": True, "conversation": conversation}


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str) -> dict[str, Any]:
    with chat_lock:
        deleted = delete_conversation(conversation_id)
    return {"ok": True, "deleted": deleted}


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    with agent_lock:
        return agent.answer_question(question=request.question, output_path=request.output_path)


@app.post("/ask-via-mcp")
async def ask_via_mcp(payload: AskRequest, request: Request) -> dict[str, Any]:
    with chat_lock:
        effective_question, followup_context = resolve_followup_question(payload.conversation_id, payload.question)
    tool_name, tool_arguments = select_mcp_tool(effective_question, payload.output_path)
    try:
        result = await call_mcp_tool_until_disconnect(
            request,
            tool_name,
            tool_arguments,
        )
        if tool_name != MCP_TOOL_NAME:
            result = normalize_direct_tool_result(tool_name, result)
        if followup_context:
            result.update(followup_context)
        result = sanitize_agent_result(result)
        result = ensure_result_analysis(tool_name, result)
    except RequestCancelled:
        return {
            "ok": False,
            "cancelled": True,
            "error": "Đã tạm dừng phân tích.",
            "mcp_tool": tool_name,
        }
    except Exception as exc:
        result = {
            "ok": False,
            "error": f"Không gọi được MCP server: {exc}",
            "mcp_tool": tool_name,
        }

    with chat_lock:
        conversation = save_chat_exchange(payload.conversation_id, payload.question, result)
    result["conversation_id"] = conversation["id"]
    result["conversation"] = conversation
    return result


@app.get("/mcp-status")
async def mcp_status() -> dict[str, Any]:
    return await call_mcp_tool("ping", {})


class VerifyRequest(BaseModel):
    question: str = Field(..., min_length=1)
    result: dict[str, Any]


@app.post("/verify")
def verify(request: VerifyRequest) -> dict[str, Any]:
    """Kiểm tra kết quả phân tích của Agent qua AI Verifier."""
    from agent.verifier_integration import verify_agent_result

    enriched = verify_agent_result(agent.engine, request.question, request.result)
    return {
        "ok": True,
        "verification": enriched.get("verification", {}),
    }


@app.get("/verification-status")
def verification_status() -> dict[str, Any]:
    """Trạng thái bật/tắt AI Verifier."""
    from agent.verifier_integration import ENABLE_VERIFICATION

    return {
        "ok": True,
        "enabled": ENABLE_VERIFICATION,
        "cross_validate": os.getenv("VERIFIER_CROSS_VALIDATE", "true").lower()
        in {"1", "true", "yes", "on"},
        "tolerance": float(os.getenv("VERIFIER_CV_TOLERANCE", "0.01")),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agent_api:app",
        host=os.getenv("AGENT_API_HOST", "127.0.0.1"),
        port=int(os.getenv("AGENT_API_PORT", "8000")),
        reload=False,
    )
