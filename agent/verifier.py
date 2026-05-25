"""
AI Result Verifier cho Olist MCP Agent.

Kiểm tra kết quả phân tích của Agent qua 5 lớp:
1. Structure      — Cấu trúc kết quả JSON hợp lệ
2. Data Integrity — Tính toàn vẹn dữ liệu (không trùng, không âm, ...)
3. Business Rules — Tuân thủ quy tắc nghiệp vụ Olist
4. Analysis Quality — Chất lượng phân tích Gemma (không bịa số, tiếng Việt có dấu, ...)
5. Cross-Validation — Đối chiếu chéo với PostgreSQL

Sử dụng:
    from agent.verifier import ResultVerifier

    verifier = ResultVerifier(engine)
    report = verifier.verify(question, intent, agent_result)
    print(report.summary())
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

VALID_REVENUE_STATUSES = ("delivered", "shipped", "invoiced", "processing")
ANALYTICS_SCHEMA = os.getenv("DEFAULT_SCHEMA", "analytics")

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

REVENUE_INTENTS = {
    "category_performance",
    "monthly_revenue_extremes",
    "quarterly_revenue_comparison",
    "revenue_by_month",
    "top_revenue_categories",
    "top_revenue_products",
    "state_market_importance",
}

# Cross-validation SQL tolerance
CV_TOLERANCE = float(os.getenv("VERIFIER_CV_TOLERANCE", "0.01"))  # 1%


# ============================================================
# Data Classes
# ============================================================


@dataclass
class VerificationCheck:
    """Một bước kiểm tra cụ thể."""

    layer: str
    check_name: str
    passed: bool
    expected: Any = None
    actual: Any = None
    message: str = ""
    severity: str = "warning"  # "error" | "warning" | "info"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationReport:
    """Báo cáo kiểm tra tổng hợp."""

    question: str
    intent: str
    checks: list[VerificationCheck] = field(default_factory=list)
    overall_passed: bool = True
    confidence_score: float = 1.0

    def add_check(self, check: VerificationCheck) -> None:
        self.checks.append(check)
        if not check.passed and check.severity == "error":
            self.overall_passed = False
        if not check.passed:
            penalty = 0.15 if check.severity == "error" else 0.05
            self.confidence_score = max(0.0, self.confidence_score - penalty)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.intent,
            "overall_passed": self.overall_passed,
            "confidence_score": round(self.confidence_score, 2),
            "total_checks": len(self.checks),
            "passed_checks": sum(1 for c in self.checks if c.passed),
            "failed_checks": sum(1 for c in self.checks if not c.passed),
            "checks": [c.to_dict() for c in self.checks],
        }

    def summary(self) -> str:
        status = "✅ PASSED" if self.overall_passed else "❌ FAILED"
        passed = sum(1 for c in self.checks if c.passed)
        return (
            f"{status} | Confidence: {self.confidence_score:.0%} | "
            f"Checks: {passed}/{len(self.checks)} passed"
        )


# ============================================================
# Verifier
# ============================================================


class ResultVerifier:
    """Kiểm tra kết quả phân tích của Agent."""

    def __init__(self, engine):
        self.engine = engine

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def verify(
        self,
        question: str,
        intent: str,
        result: dict[str, Any],
    ) -> VerificationReport:
        """Chạy toàn bộ quy trình kiểm tra 5 lớp."""
        report = VerificationReport(question=question, intent=intent)

        self._check_result_structure(result, report)
        self._check_data_integrity(intent, result, report)
        self._check_business_rules(intent, result, report)
        self._check_analysis_quality(question, result, report)

        if os.getenv("VERIFIER_CROSS_VALIDATE", "true").lower() in {"1", "true", "yes", "on"}:
            self._cross_validate(intent, result, report)

        return report

    # --------------------------------------------------
    # Layer 1: Result Structure
    # --------------------------------------------------

    def _check_result_structure(
        self,
        result: dict[str, Any],
        report: VerificationReport,
    ) -> None:
        # 1.1 — Phải có trường "ok"
        report.add_check(VerificationCheck(
            layer="structure",
            check_name="has_ok_field",
            passed="ok" in result,
            message="Kết quả phải có trường 'ok'.",
            severity="error",
        ))

        # 1.2 — ok phải là True
        report.add_check(VerificationCheck(
            layer="structure",
            check_name="ok_is_true",
            passed=result.get("ok") is True,
            expected=True,
            actual=result.get("ok"),
            message="Trường 'ok' phải là True cho kết quả thành công.",
            severity="error",
        ))

        # 1.3 — Phải có nội dung (analysis, rows, answer, summary, data)
        has_content = any(
            result.get(k)
            for k in ("analysis", "rows", "answer", "summary", "data", "safe_summary")
        )
        report.add_check(VerificationCheck(
            layer="structure",
            check_name="has_content",
            passed=has_content,
            message="Kết quả phải có analysis, rows, answer, summary hoặc data.",
            severity="error",
        ))

        # 1.4 — Nếu có rows, phải là list
        rows = result.get("rows")
        if rows is not None:
            report.add_check(VerificationCheck(
                layer="structure",
                check_name="rows_is_list",
                passed=isinstance(rows, list),
                expected="list",
                actual=type(rows).__name__,
                message="Trường 'rows' phải là danh sách.",
                severity="error",
            ))

        # 1.5 — Phải có intent
        report.add_check(VerificationCheck(
            layer="structure",
            check_name="has_intent",
            passed=bool(result.get("intent")),
            actual=result.get("intent"),
            message="Kết quả phải gắn intent đã nhận diện.",
            severity="warning",
        ))

    # --------------------------------------------------
    # Layer 2: Data Integrity
    # --------------------------------------------------

    def _check_data_integrity(
        self,
        intent: str,
        result: dict[str, Any],
        report: VerificationReport,
    ) -> None:
        rows = result.get("rows")
        if not isinstance(rows, list) or not rows:
            return

        # 2.1 — Không có dòng trùng lặp
        if rows and isinstance(rows[0], dict):
            row_strs = [json.dumps(r, sort_keys=True, default=str) for r in rows]
            unique_count = len(set(row_strs))
            dup_count = len(rows) - unique_count
            report.add_check(VerificationCheck(
                layer="data_integrity",
                check_name="no_duplicate_rows",
                passed=unique_count == len(rows),
                expected=len(rows),
                actual=unique_count,
                message=f"Phát hiện {dup_count} dòng trùng lặp." if dup_count else "Không có dòng trùng lặp.",
                severity="warning" if dup_count else "info",
            ))

        # 2.2 — Doanh thu không âm
        revenue_keys = ("revenue", "line_total", "order_gross_value", "total_revenue")
        found_negative = False
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in revenue_keys:
                val = row.get(key)
                if val is not None:
                    try:
                        if float(val) < 0:
                            found_negative = True
                            break
                    except (ValueError, TypeError):
                        pass
            if found_negative:
                break
        report.add_check(VerificationCheck(
            layer="data_integrity",
            check_name="non_negative_revenue",
            passed=not found_negative,
            message="Phát hiện doanh thu âm." if found_negative else "Doanh thu đều không âm.",
            severity="error" if found_negative else "info",
        ))

        # 2.3 — row_count nhất quán
        reported_count = result.get("row_count")
        if reported_count is not None:
            report.add_check(VerificationCheck(
                layer="data_integrity",
                check_name="row_count_consistent",
                passed=int(reported_count) == len(rows),
                expected=int(reported_count),
                actual=len(rows),
                message="row_count phải khớp với số lượng rows thực tế.",
                severity="warning",
            ))

        # 2.4 — Kiểm tra ranking hợp lệ (nếu có)
        if rows and isinstance(rows[0], dict):
            rank_cols = [k for k in rows[0] if k.endswith("_rank")]
            for col in rank_cols:
                ranks = []
                for row in rows:
                    val = row.get(col)
                    if val is not None:
                        try:
                            ranks.append(int(val))
                        except (ValueError, TypeError):
                            pass
                if ranks:
                    is_sorted = all(ranks[i] <= ranks[i + 1] for i in range(len(ranks) - 1))
                    report.add_check(VerificationCheck(
                        layer="data_integrity",
                        check_name=f"ranking_order_{col}",
                        passed=is_sorted,
                        message=f"Cột {col} {'đúng thứ tự' if is_sorted else 'không đúng thứ tự'}.",
                        severity="warning" if not is_sorted else "info",
                    ))

    # --------------------------------------------------
    # Layer 3: Business Rules
    # --------------------------------------------------

    def _check_business_rules(
        self,
        intent: str,
        result: dict[str, Any],
        report: VerificationReport,
    ) -> None:
        # 3.1 — Intent hợp lệ
        report.add_check(VerificationCheck(
            layer="business_rules",
            check_name="valid_intent",
            passed=intent in SUPPORTED_INTENTS,
            expected="supported intent",
            actual=intent,
            message=f"Intent '{intent}' {'hợp lệ' if intent in SUPPORTED_INTENTS else 'không hợp lệ'}.",
            severity="warning" if intent not in SUPPORTED_INTENTS else "info",
        ))

        # 3.2 — Kiểm tra valid_statuses cho intent doanh thu
        if intent in REVENUE_INTENTS:
            valid_statuses = result.get("valid_statuses")
            if valid_statuses and isinstance(valid_statuses, list):
                expected_set = set(VALID_REVENUE_STATUSES)
                actual_set = set(valid_statuses)
                report.add_check(VerificationCheck(
                    layer="business_rules",
                    check_name="revenue_status_filter",
                    passed=actual_set == expected_set,
                    expected=sorted(expected_set),
                    actual=sorted(actual_set),
                    message="Trạng thái đơn hàng hợp lệ cho doanh thu phải đúng.",
                    severity="error" if actual_set != expected_set else "info",
                ))

        # 3.3 — Không dùng payment_value_total cho doanh thu
        rows = result.get("rows", [])
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            uses_payment_total = any("payment_value_total" in r for r in rows)
            if intent in REVENUE_INTENTS and uses_payment_total:
                report.add_check(VerificationCheck(
                    layer="business_rules",
                    check_name="no_payment_total_for_revenue",
                    passed=False,
                    message="Không được dùng payment_value_total để tính doanh thu chính.",
                    severity="error",
                ))

        # 3.4 — Delivery delay: phải dùng order_status = 'delivered'
        if intent in ("delivery_delay", "delivery_review_impact"):
            # Kiểm tra gián tiếp qua kết quả
            summary = result.get("summary", {})
            if isinstance(summary, dict) and summary.get("delivered_orders") is not None:
                delivered = int(summary.get("delivered_orders", 0))
                report.add_check(VerificationCheck(
                    layer="business_rules",
                    check_name="delivery_uses_delivered_status",
                    passed=delivered > 0,
                    expected="> 0",
                    actual=delivered,
                    message="Phân tích giao hàng trễ phải dựa trên đơn đã giao (delivered).",
                    severity="error" if delivered == 0 else "info",
                ))

    # --------------------------------------------------
    # Layer 4: Analysis Quality
    # --------------------------------------------------

    def _check_analysis_quality(
        self,
        question: str,
        result: dict[str, Any],
        report: VerificationReport,
    ) -> None:
        analysis = result.get("analysis") or result.get("safe_summary") or ""
        if not analysis:
            return

        rows = result.get("rows", [])

        # 4.1 — Phân tích không quá ngắn
        report.add_check(VerificationCheck(
            layer="analysis_quality",
            check_name="analysis_not_too_short",
            passed=len(analysis) >= 20,
            expected=">= 20 ký tự",
            actual=len(analysis),
            message="Phân tích quá ngắn." if len(analysis) < 20 else "Độ dài phân tích đạt yêu cầu.",
            severity="warning" if len(analysis) < 20 else "info",
        ))

        # 4.2 — Không copy prompt / tạo Cảnh-Kịch bản
        bad_patterns = [
            r"(?i)(?:\*\*)?\s*(?:c[ảa]nh|canh|k[ịi]ch\s*b[ảa]n|kich\s*ban)\s*\d+",
            r"(?i)c[âa]u\s*h[ỏo]i\s*ph[âa]n\s*t[ií]ch\s*d[ữu]\s*li[ệe]u",
            r"(?i)cau\s*hoi\s*phan\s*tich\s*du\s*lieu",
            r"(?i)ph[âa]n\s*t[ií]ch\s*5\s*danh\s*m[ụu]c\s*s[ảa]n\s*ph[ẩa]m\s*t[ốo]t\s*nh[ấa]t",
        ]
        has_bad = any(re.search(p, analysis) for p in bad_patterns)
        report.add_check(VerificationCheck(
            layer="analysis_quality",
            check_name="no_prompt_copying",
            passed=not has_bad,
            message="Phân tích có dấu hiệu copy prompt." if has_bad else "Không phát hiện copy prompt.",
            severity="error" if has_bad else "info",
        ))

        # 4.3 — Tiếng Việt có dấu
        vn_pattern = re.compile(
            r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]"
        )
        has_vn = bool(vn_pattern.search(analysis))
        report.add_check(VerificationCheck(
            layer="analysis_quality",
            check_name="vietnamese_with_diacritics",
            passed=has_vn,
            message="Phân tích phải viết tiếng Việt có dấu." if not has_vn else "Tiếng Việt có dấu.",
            severity="warning" if not has_vn else "info",
        ))

        # 4.4 — Kiểm tra số liệu bịa
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            self._verify_numbers_in_analysis(analysis, rows, report)

        # 4.5 — Kiểm tra kết luận "cao nhất / thấp nhất" có đúng không
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            self._verify_superlative_claims(analysis, rows, report)

    def _verify_numbers_in_analysis(
        self,
        analysis: str,
        rows: list[dict],
        report: VerificationReport,
    ) -> None:
        """Kiểm tra các con số trong analysis có tồn tại trong dữ liệu."""
        numbers_in_analysis = set()
        for match in re.finditer(r"[\d,.]+", analysis):
            try:
                num_str = match.group().replace(",", ".").rstrip(".")
                if num_str:
                    numbers_in_analysis.add(float(num_str))
            except ValueError:
                pass

        numbers_in_data = set()
        for row in rows:
            for val in row.values():
                try:
                    numbers_in_data.add(float(val))
                except (ValueError, TypeError):
                    pass

        fabricated = []
        for num in numbers_in_analysis:
            if num < 10:
                continue
            found = any(
                abs(num - d) / max(abs(d), 1) < 0.02
                for d in numbers_in_data
                if d != 0
            )
            if not found and num > 100:
                fabricated.append(num)

        if fabricated:
            report.add_check(VerificationCheck(
                layer="analysis_quality",
                check_name="no_fabricated_numbers",
                passed=False,
                expected="Chỉ dùng số từ dữ liệu",
                actual=fabricated[:5],
                message=f"Phát hiện {len(fabricated)} con số nghi bịa trong phân tích.",
                severity="warning",
            ))
        else:
            report.add_check(VerificationCheck(
                layer="analysis_quality",
                check_name="no_fabricated_numbers",
                passed=True,
                message="Các con số trong phân tích đều khớp với dữ liệu.",
                severity="info",
            ))

    def _verify_superlative_claims(
        self,
        analysis: str,
        rows: list[dict],
        report: VerificationReport,
    ) -> None:
        """Kiểm tra kết luận cao nhất / thấp nhất có đúng với dữ liệu."""
        if not rows or not isinstance(rows[0], dict):
            return

        # Tìm cột doanh thu / số đơn
        metric_cols = [
            k for k in rows[0]
            if k in ("revenue", "order_count", "total_orders", "avg_review_score")
        ]
        if not metric_cols:
            return

        # Kiểm tra nếu analysis nhắc đến tên entity nào đứng đầu
        name_cols = [
            k for k in rows[0]
            if k in ("category", "state", "month", "quarter", "product_category_name")
        ]
        if not name_cols:
            return

        name_col = name_cols[0]
        for metric_col in metric_cols:
            try:
                vals = [(row.get(name_col, ""), float(row.get(metric_col, 0))) for row in rows]
                vals_sorted = sorted(vals, key=lambda x: x[1], reverse=True)
                actual_top = vals_sorted[0][0] if vals_sorted else ""
            except (ValueError, TypeError):
                continue

            if not actual_top:
                continue

            # Kiểm tra nếu analysis nhắc đến entity "cao nhất" mà không phải actual_top
            cao_nhat_match = re.search(
                r"(cao\s*nh[ấa]t|d[ẫa]n\s*d[ầa]u|đ[ứu]ng\s*đ[ầa]u|l[ớo]n\s*nh[ấa]t)"
                r".{0,80}?"
                r"([A-Za-z_]+(?:\s+[A-Za-z_]+){0,3})",
                analysis,
                re.IGNORECASE,
            )
            if cao_nhat_match:
                mentioned = cao_nhat_match.group(2).strip().lower()
                actual_lower = str(actual_top).lower()
                if mentioned and actual_lower and mentioned not in actual_lower and actual_lower not in mentioned:
                    report.add_check(VerificationCheck(
                        layer="analysis_quality",
                        check_name=f"superlative_correct_{metric_col}",
                        passed=False,
                        expected=actual_top,
                        actual=mentioned,
                        message=(
                            f"Phân tích nhắc '{mentioned}' là cao nhất về {metric_col}, "
                            f"nhưng dữ liệu cho thấy '{actual_top}' mới đứng đầu."
                        ),
                        severity="warning",
                    ))

    # --------------------------------------------------
    # Layer 5: Cross-Validation
    # --------------------------------------------------

    def _cross_validate(
        self,
        intent: str,
        result: dict[str, Any],
        report: VerificationReport,
    ) -> None:
        validators = {
            "revenue_by_month": self._cv_revenue_by_month,
            "monthly_revenue_extremes": self._cv_revenue_by_month,
            "category_performance": self._cv_category_performance,
            "top_revenue_categories": self._cv_category_performance,
            "delivery_delay": self._cv_delivery_delay,
            "repeat_customer_rate": self._cv_repeat_customer_rate,
        }
        validator = validators.get(intent)
        if validator:
            try:
                validator(result, report)
            except Exception as exc:
                logger.warning("Cross-validation error for intent=%s: %s", intent, exc)
                report.add_check(VerificationCheck(
                    layer="cross_validation",
                    check_name=f"cv_{intent}",
                    passed=False,
                    message=f"Cross-validation lỗi: {exc}",
                    severity="warning",
                ))

    def _cv_revenue_by_month(
        self,
        result: dict[str, Any],
        report: VerificationReport,
    ) -> None:
        year = result.get("year")
        rows = result.get("rows", [])
        if not year or not rows:
            return

        agent_total = sum(float(r.get("revenue", 0)) for r in rows if isinstance(r, dict))

        query = text(f"""
            SELECT ROUND(SUM(order_gross_value)::numeric, 2) AS total_revenue
            FROM {ANALYTICS_SCHEMA}.fct_orders
            WHERE order_purchase_timestamp >= :start_date
              AND order_purchase_timestamp < :end_date
              AND order_status IN ('delivered', 'shipped', 'invoiced', 'processing')
        """)
        with self.engine.connect() as conn:
            db_result = conn.execute(query, {
                "start_date": f"{year}-01-01",
                "end_date": f"{year + 1}-01-01",
            }).mappings().first()

        if db_result and db_result["total_revenue"] is not None:
            db_total = float(db_result["total_revenue"])
            diff_pct = abs(agent_total - db_total) / max(db_total, 1)
            report.add_check(VerificationCheck(
                layer="cross_validation",
                check_name="revenue_total_match",
                passed=diff_pct <= CV_TOLERANCE,
                expected=db_total,
                actual=round(agent_total, 2),
                message=(
                    f"Tổng doanh thu Agent: {agent_total:,.2f} vs "
                    f"DB: {db_total:,.2f} (sai lệch: {diff_pct:.2%})"
                ),
                severity="error" if diff_pct > CV_TOLERANCE else "info",
            ))

    def _cv_category_performance(
        self,
        result: dict[str, Any],
        report: VerificationReport,
    ) -> None:
        rows = result.get("rows", [])
        if not rows or not isinstance(rows[0], dict):
            return

        year = result.get("year")
        start_date = result.get("start_date")
        end_date = result.get("end_date")

        if not year and not start_date:
            qu = result.get("question_understanding", {})
            year = qu.get("year")
            start_date = qu.get("start_date")
            end_date = qu.get("end_date")

        if not start_date:
            if not year:
                return
            start_date = f"{year}-01-01"
            end_date = f"{year + 1}-01-01"

        top_category = rows[0].get("category", "")
        if not top_category:
            return

        query = text(f"""
            SELECT
                COALESCE(product_category_name_english, product_category_name, 'unknown') AS category,
                ROUND(SUM(line_total)::numeric, 2) AS revenue
            FROM {ANALYTICS_SCHEMA}.fct_order_items
            WHERE order_purchase_timestamp >= CAST(:start_date AS timestamp)
              AND order_purchase_timestamp < CAST(:end_date AS timestamp)
              AND order_status IN ('delivered', 'shipped', 'invoiced', 'processing')
            GROUP BY 1
            ORDER BY revenue DESC
            LIMIT 1
        """)
        with self.engine.connect() as conn:
            db_result = conn.execute(query, {
                "start_date": start_date,
                "end_date": end_date,
            }).mappings().first()

        if db_result:
            db_top = db_result["category"]
            report.add_check(VerificationCheck(
                layer="cross_validation",
                check_name="top_category_match",
                passed=top_category == db_top,
                expected=db_top,
                actual=top_category,
                message=(
                    f"Top category Agent: '{top_category}' vs "
                    f"DB: '{db_top}'"
                ),
                severity="warning" if top_category != db_top else "info",
            ))

    def _cv_delivery_delay(
        self,
        result: dict[str, Any],
        report: VerificationReport,
    ) -> None:
        summary = result.get("summary", {})
        if not isinstance(summary, dict):
            rows = result.get("rows", [])
            summary = rows[0] if rows and isinstance(rows[0], dict) else {}

        reported_rate = summary.get("delayed_order_rate_pct")
        if reported_rate is None:
            return

        start_date = result.get("start_date")
        end_date = result.get("end_date")
        if not start_date or not end_date:
            return

        query = text(f"""
            SELECT
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE is_delayed_delivery IS TRUE)
                    / NULLIF(COUNT(*), 0), 2
                ) AS delayed_order_rate_pct
            FROM {ANALYTICS_SCHEMA}.fct_orders
            WHERE order_purchase_timestamp::date
                  BETWEEN CAST(:start_date AS date) AND CAST(:end_date AS date)
              AND order_status = 'delivered'
        """)
        with self.engine.connect() as conn:
            db_result = conn.execute(query, {
                "start_date": start_date,
                "end_date": end_date,
            }).mappings().first()

        if db_result and db_result["delayed_order_rate_pct"] is not None:
            db_rate = float(db_result["delayed_order_rate_pct"])
            reported = float(reported_rate)
            diff = abs(reported - db_rate)
            report.add_check(VerificationCheck(
                layer="cross_validation",
                check_name="delay_rate_match",
                passed=diff <= 0.1,
                expected=db_rate,
                actual=reported,
                message=(
                    f"Tỷ lệ giao trễ Agent: {reported}% vs "
                    f"DB: {db_rate}% (sai lệch: {diff:.2f}%)"
                ),
                severity="error" if diff > 0.1 else "info",
            ))

    def _cv_repeat_customer_rate(
        self,
        result: dict[str, Any],
        report: VerificationReport,
    ) -> None:
        summary = result.get("summary", {})
        if not isinstance(summary, dict):
            rows = result.get("rows", [])
            summary = rows[0] if rows and isinstance(rows[0], dict) else {}

        reported_rate = summary.get("repeat_customer_rate_pct")
        if reported_rate is None:
            return

        start_date = result.get("start_date")
        end_date = result.get("end_date")
        if not start_date or not end_date:
            return

        query = text(f"""
            WITH customer_orders AS (
                SELECT
                    customer_unique_id,
                    COUNT(DISTINCT order_id) AS order_count
                FROM {ANALYTICS_SCHEMA}.fct_orders
                WHERE order_purchase_timestamp::date
                      BETWEEN CAST(:start_date AS date) AND CAST(:end_date AS date)
                  AND order_status IN ('delivered', 'shipped', 'invoiced', 'processing')
                  AND customer_unique_id IS NOT NULL
                GROUP BY 1
            )
            SELECT
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE order_count >= 2)
                    / NULLIF(COUNT(*), 0), 2
                ) AS repeat_customer_rate_pct
            FROM customer_orders
        """)
        with self.engine.connect() as conn:
            db_result = conn.execute(query, {
                "start_date": start_date,
                "end_date": end_date,
            }).mappings().first()

        if db_result and db_result["repeat_customer_rate_pct"] is not None:
            db_rate = float(db_result["repeat_customer_rate_pct"])
            reported = float(reported_rate)
            diff = abs(reported - db_rate)
            report.add_check(VerificationCheck(
                layer="cross_validation",
                check_name="repeat_rate_match",
                passed=diff <= 0.1,
                expected=db_rate,
                actual=reported,
                message=(
                    f"Tỷ lệ khách quay lại Agent: {reported}% vs "
                    f"DB: {db_rate}% (sai lệch: {diff:.2f}%)"
                ),
                severity="error" if diff > 0.1 else "info",
            ))
