from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
import unicodedata
from typing import Any, Callable


SUPPORTED_INTENTS = {
    "schema_tables",
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
    "order_payment",
    "order_shipping",
    "order_products",
    "order_sellers",
    "order_review",
    "order_customer",
    "order_detail",
    "unknown",
}

TIME_SENSITIVE_INTENTS = {
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


@dataclass
class QuestionUnderstanding:
    original_question: str
    normalized_question: str
    intent: str = "unknown"
    entity: str | None = None
    metric: str | None = None
    time_grain: str | None = None
    year: int | None = None
    start_date: str | None = None
    end_date: str | None = None
    quarters: list[int] = field(default_factory=list)
    top_n: int = 10
    order_id: str | None = None
    needs_clarification: bool = False
    clarifying_question: str | None = None
    reason: str | None = None
    confidence: float = 0.0
    source: str = "rule_structured"
    evidence: list[str] = field(default_factory=list)
    llm_raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_order_detail(self) -> bool:
        return self.intent.startswith("order_") or bool(self.order_id)


class QuestionUnderstandingEngine:
    def __init__(
        self,
        llm_generate: Callable[[str], str] | None = None,
        enable_llm: bool = True,
    ):
        self.llm_generate = llm_generate
        self.enable_llm = enable_llm

    def understand(self, question: str) -> QuestionUnderstanding:
        structured = self.rule_understand(question)
        if self.enable_llm and self.llm_generate is not None:
            llm_result = self.llm_understand(question, structured)
            if llm_result is not None:
                structured = self.merge_llm_result(structured, llm_result)
        self.apply_clarification_policy(structured)
        return structured

    def rule_understand(self, question: str) -> QuestionUnderstanding:
        q = normalize_text(question)
        order_id = extract_order_id(question)
        top_n = extract_top_n(question)
        year = extract_year(question)
        start_date, end_date = extract_date_range(question, year) if year else (None, None)
        quarters = extract_quarters(question)

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

        intent = "unknown"
        entity: str | None = None
        metric: str | None = None
        evidence: list[str] = []
        confidence = 0.55

        if order_id:
            intent = self.order_detail_intent(q)
            entity = "order"
            metric = intent.removeprefix("order_")
            evidence.append("found_order_id")
            confidence = 0.95
        elif self.looks_like_order_detail(q):
            intent = self.order_detail_intent(q)
            entity = "order"
            metric = intent.removeprefix("order_")
            evidence.append("order_detail_without_id")
            confidence = 0.75
        elif (
            any(term in q for term in ["liet ke", "danh sach", "co nhung", "cac bang", "nhung bang"])
            and any(term in q for term in ["bang", "table", "schema", "du lieu", "database"])
        ):
            intent = "schema_tables"
            entity = "schema"
            metric = "metadata"
            evidence.append("schema_terms")
            confidence = 0.9
        elif has_favorite and has_revenue and any(
            term in q for term in ["dong thoi", "co phai", "cao khong", "doanh thu cao"]
        ):
            intent = "favorite_vs_revenue"
            entity = "category"
            metric = "review_vs_revenue"
            evidence.extend(["favorite_terms", "revenue_terms"])
            confidence = 0.86
        elif wants_comparison and has_order_volume and has_review:
            intent = "orders_vs_review"
            entity = "category"
            metric = "order_count_vs_review"
            evidence.extend(["comparison_terms", "order_volume_terms", "review_terms"])
            confidence = 0.86
        elif has_revenue and is_quarter_question(q) and any(
            term in q for term in ["so sanh", "khac nhau", "chenh lech", "cao nhat", "thap nhat", "noi bat"]
        ):
            intent = "quarterly_revenue_comparison"
            entity = "order"
            metric = "revenue"
            evidence.extend(["quarter_terms", "revenue_terms"])
            confidence = 0.88
        elif asks_category and has_revenue and any(term in q for term in ["review", "danh gia", "so don", "don hang"]):
            intent = "category_performance"
            entity = "category"
            metric = "performance"
            evidence.extend(["category_terms", "revenue_terms", "multi_metric_terms"])
            confidence = 0.86
        elif has_revenue and "thang" in q and any(
            term in q for term in ["cao nhat", "thap nhat", "bat thuong", "khac nhau", "so sanh"]
        ):
            intent = "monthly_revenue_extremes"
            entity = "order"
            metric = "revenue"
            evidence.extend(["month_terms", "revenue_terms", "extreme_terms"])
            confidence = 0.9
        elif "giao hang" in q and any(term in q for term in ["tre", "cham", "delay"]) and has_review:
            intent = "delivery_review_impact"
            entity = "delivery"
            metric = "review"
            evidence.extend(["delivery_delay_terms", "review_terms"])
            confidence = 0.9
        elif ("bang" in q or "state" in q) and any(term in q for term in ["thi truong", "market", "quan trong"]) and has_revenue:
            intent = "state_market_importance"
            entity = "state"
            metric = "market_importance"
            evidence.extend(["state_terms", "market_terms", "revenue_terms"])
            confidence = 0.86
        elif ("bang" in q or "state" in q or "khu vuc" in q) and has_revenue and has_review:
            intent = "state_revenue_low_review"
            entity = "state"
            metric = "revenue_review_priority"
            evidence.extend(["state_terms", "revenue_terms", "review_terms"])
            confidence = 0.82
        elif any(term in q for term in ["cai thien trai nghiem", "trai nghiem khach hang", "uu tien"]):
            intent = "customer_experience_priority"
            entity = "category_and_state"
            metric = "customer_experience_priority"
            evidence.append("customer_experience_terms")
            confidence = 0.86
        elif any(term in q for term in ["giao cham", "giao chậm", "tre", "trễ", "delay"]):
            intent = "delivery_delay"
            entity = "delivery"
            metric = "delay_rate"
            evidence.append("delivery_delay_terms")
            confidence = 0.8
        elif any(term in q for term in ["khach quay lai", "khach hang quay lai", "repeat"]):
            intent = "repeat_customer_rate"
            entity = "customer"
            metric = "repeat_customer_rate"
            evidence.append("repeat_customer_terms")
            confidence = 0.86
        elif any(term in q for term in ["theo thang", "theo tháng", "monthly", "month"]):
            intent = "revenue_by_month"
            entity = "order"
            metric = "revenue"
            evidence.append("monthly_terms")
            confidence = 0.75
        elif has_revenue and asks_product and not asks_category:
            intent = "top_revenue_products"
            entity = "product"
            metric = "revenue"
            evidence.extend(["product_terms", "revenue_terms"])
            confidence = 0.86
        elif has_revenue:
            intent = "top_revenue_categories"
            entity = "category"
            metric = "revenue"
            evidence.append("revenue_terms")
            confidence = 0.7
        elif has_review:
            intent = "favorite_products"
            entity = "category"
            metric = "review"
            evidence.append("review_terms")
            confidence = 0.72
        elif asks_category and has_order_volume:
            intent = "top_categories_by_orders"
            entity = "category"
            metric = "order_count"
            evidence.extend(["category_terms", "order_volume_terms"])
            confidence = 0.82
        elif asks_product and has_order_volume:
            intent = "top_products_by_orders"
            entity = "product"
            metric = "order_count"
            evidence.extend(["product_terms", "order_volume_terms"])
            confidence = 0.82
        elif has_order_volume and any(term in q for term in ["don hang", "order"]):
            intent = "revenue_by_month" if any(term in q for term in ["thang", "month"]) else "top_categories_by_orders"
            entity = "order" if intent == "revenue_by_month" else "category"
            metric = "order_count"
            evidence.append("order_volume_terms")
            confidence = 0.68

        time_grain = infer_time_grain(q, quarters)
        return QuestionUnderstanding(
            original_question=question,
            normalized_question=q,
            intent=intent,
            entity=entity,
            metric=metric,
            time_grain=time_grain,
            year=year,
            start_date=start_date,
            end_date=end_date,
            quarters=quarters,
            top_n=top_n,
            order_id=order_id,
            confidence=confidence if intent != "unknown" else 0.25,
            evidence=evidence,
        )

    def llm_understand(
        self,
        question: str,
        fallback: QuestionUnderstanding,
    ) -> dict[str, Any] | None:
        prompt = self.build_llm_prompt(question, fallback)
        try:
            raw = self.llm_generate(prompt) if self.llm_generate is not None else ""
        except Exception:
            return None
        parsed = parse_json_object(raw)
        if parsed is None:
            return None
        parsed["_llm_raw"] = raw
        return parsed

    def merge_llm_result(
        self,
        fallback: QuestionUnderstanding,
        llm_result: dict[str, Any],
    ) -> QuestionUnderstanding:
        intent = normalize_intent(llm_result.get("intent"))
        if intent is not None:
            if fallback.intent == "unknown" or fallback.confidence < 0.8:
                fallback.intent = intent
                fallback.source = "gemma_structured"
            elif intent == fallback.intent:
                fallback.source = "hybrid_gemma_rule"
            else:
                fallback.source = "rule_structured_with_gemma_check"

        for field_name in ("entity", "metric", "time_grain", "reason", "clarifying_question"):
            value = llm_result.get(field_name)
            if isinstance(value, str) and value.strip():
                current = getattr(fallback, field_name)
                if not current or fallback.source == "gemma_structured":
                    setattr(fallback, field_name, value.strip())

        if isinstance(llm_result.get("top_n"), int):
            fallback.top_n = clamp_limit(llm_result["top_n"])
        if isinstance(llm_result.get("year"), int) and fallback.year is None:
            fallback.year = llm_result["year"]
            if fallback.start_date is None or fallback.end_date is None:
                fallback.start_date, fallback.end_date = extract_date_range(fallback.original_question, fallback.year)
        if isinstance(llm_result.get("quarters"), list) and not fallback.quarters:
            fallback.quarters = [int(q) for q in llm_result["quarters"] if str(q).isdigit() and 1 <= int(q) <= 4]
        for field_name in ("start_date", "end_date", "order_id"):
            value = llm_result.get(field_name)
            if isinstance(value, str) and value.strip() and getattr(fallback, field_name) is None:
                setattr(fallback, field_name, value.strip())

        needs_clarification = llm_result.get("needs_clarification")
        if isinstance(needs_clarification, bool):
            fallback.needs_clarification = needs_clarification
        if isinstance(llm_result.get("confidence"), (int, float)):
            fallback.confidence = max(fallback.confidence, min(float(llm_result["confidence"]), 1.0))
        fallback.llm_raw = llm_result.get("_llm_raw")
        return fallback

    def apply_clarification_policy(self, result: QuestionUnderstanding) -> None:
        if result.intent in TIME_SENSITIVE_INTENTS and not has_explicit_time_scope(result.original_question):
            result.needs_clarification = True
            result.reason = "Câu hỏi phân tích cần có phạm vi thời gian rõ ràng trước khi truy vấn dữ liệu."
            result.clarifying_question = (
                "Bạn muốn phân tích trong khoảng thời gian nào? "
                "Ví dụ: năm 2018, quý 1 năm 2018, tháng 11 năm 2017, "
                "hoặc từ 2017-01-01 đến 2017-12-31."
            )
        if result.intent.startswith("order_") and result.order_id is None:
            result.needs_clarification = True
            result.reason = "Câu hỏi đang hỏi một đơn hàng cụ thể nhưng chưa có order_id."
            result.clarifying_question = "Bạn vui lòng cung cấp order_id của đơn hàng cần tra cứu."

    @staticmethod
    def order_detail_intent(q: str) -> str:
        if "thanh toan" in q or "payment" in q:
            return "order_payment"
        if "van chuyen" in q or "giao hang" in q or "phi ship" in q or "freight" in q:
            return "order_shipping"
        if "san pham" in q or "product" in q:
            return "order_products"
        if "nguoi ban" in q or "seller" in q:
            return "order_sellers"
        if "danh gia" in q or "review" in q or "sao" in q:
            return "order_review"
        if "khach hang" in q or "customer" in q:
            return "order_customer"
        return "order_detail"

    @staticmethod
    def looks_like_order_detail(q: str) -> bool:
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
        )

    @staticmethod
    def build_llm_prompt(question: str, fallback: QuestionUnderstanding) -> str:
        supported = ", ".join(sorted(SUPPORTED_INTENTS))
        return (
            "Bạn là bộ phân tích câu hỏi cho agent dữ liệu Olist. "
            "Chỉ trả về một JSON object hợp lệ, không markdown, không giải thích ngoài JSON.\n\n"
            f"Các intent hợp lệ: {supported}\n"
            "Schema JSON bắt buộc:\n"
            "{\n"
            '  "intent": "one_supported_intent",\n'
            '  "entity": "product|category|state|order|customer|delivery|schema|category_and_state|null",\n'
            '  "metric": "revenue|order_count|review|delivery_delay|repeat_customer_rate|payment|performance|unknown",\n'
            '  "time_grain": "year|quarter|month|date_range|null",\n'
            '  "year": 2018,\n'
            '  "start_date": "YYYY-MM-DD",\n'
            '  "end_date": "YYYY-MM-DD",\n'
            '  "quarters": [1,2],\n'
            '  "top_n": 10,\n'
            '  "order_id": null,\n'
            '  "needs_clarification": false,\n'
            '  "clarifying_question": null,\n'
            '  "reason": null,\n'
            '  "confidence": 0.0\n'
            "}\n\n"
            "Nếu câu hỏi phân tích dữ liệu thiếu năm/quý/tháng/khoảng ngày, đặt needs_clarification=true. "
            "Nếu hỏi đơn hàng cụ thể nhưng thiếu order_id, đặt intent phù hợp dạng order_* và needs_clarification=true. "
            "Phân biệt product_id/sản phẩm với category/danh mục.\n\n"
            f"Câu hỏi: {question}\n"
            f"Fallback hiện tại: {json.dumps(fallback.to_dict(), ensure_ascii=False)}\n"
            "JSON:"
        )


def normalize_text(text_value: str) -> str:
    text_value = unicodedata.normalize("NFKD", text_value)
    text_value = "".join(ch for ch in text_value if not unicodedata.combining(ch))
    text_value = text_value.replace("đ", "d").replace("Đ", "D")
    return text_value.lower()


def extract_order_id(question: str) -> str | None:
    match = re.search(r"\b[a-f0-9]{32}\b", question.lower())
    return match.group(0) if match else None


def extract_year(question: str, default: int | None = None) -> int | None:
    match = re.search(r"\b(20\d{2})\b", question)
    return int(match.group(1)) if match else default


def has_explicit_time_scope(question: str) -> bool:
    q = normalize_text(question)
    return bool(
        re.search(r"\b20\d{2}\b", question)
        or re.search(r"\b\d{4}-\d{2}-\d{2}\b", question)
        or re.search(r"\b(?:q|quy|qui)\s*[1-4]\b", q)
        or re.search(r"\bthang\s*(?:1[0-2]|0?[1-9])\b", q)
    )


def extract_date_range(question: str, year: int) -> tuple[str, str]:
    q = normalize_text(question)
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


def extract_quarters(question: str) -> list[int]:
    q = normalize_text(question)
    quarters = {int(match.group(1)) for match in re.finditer(r"\b(?:quy|qui|q)\s*([1-4])\b", q)}
    return sorted(quarters)


def extract_top_n(question: str, default: int = 10) -> int:
    q = normalize_text(question)
    match = re.search(r"\btop\s*(\d{1,2})\b", q)
    if not match:
        match = re.search(r"\b(\d{1,2})\s+(?:danh muc|san pham|bang|khu vuc|state)", q)
    return clamp_limit(int(match.group(1)) if match else default)


def clamp_limit(limit: int, maximum: int = 50) -> int:
    return max(1, min(int(limit), maximum))


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


def infer_time_grain(q: str, quarters: list[int]) -> str | None:
    if re.search(r"\b20\d{2}-\d{2}-\d{2}\b", q):
        return "date_range"
    if quarters or is_quarter_question(q):
        return "quarter"
    if "thang" in q or "month" in q:
        return "month"
    if re.search(r"\b20\d{2}\b", q):
        return "year"
    return None


def normalize_intent(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value in SUPPORTED_INTENTS else None


def parse_json_object(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None
