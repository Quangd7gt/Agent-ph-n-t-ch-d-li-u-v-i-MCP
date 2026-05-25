
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

ENABLE_VERIFICATION = os.getenv("ENABLE_VERIFICATION", "true").lower() in {
    "1", "true", "yes", "on",
}


def verify_agent_result(
    engine,
    question: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Kiểm tra kết quả Agent và gắn báo cáo verification vào result.

    Args:
        engine: SQLAlchemy engine kết nối PostgreSQL.
        question: Câu hỏi gốc của người dùng.
        result: Kết quả JSON mà Agent trả về.

    Returns:
        result đã được bổ sung trường "verification".
    """
    if not ENABLE_VERIFICATION:
        return result

    # Không verify kết quả lỗi hoặc cần clarification
    if not result.get("ok") or result.get("needs_clarification"):
        return result

    try:
        from agent.verifier import ResultVerifier

        understanding = result.get("question_understanding", {})
        intent = understanding.get("intent") or result.get("intent", "unknown")

        verifier = ResultVerifier(engine)
        report = verifier.verify(question, intent, result)

        result["verification"] = {
            "passed": report.overall_passed,
            "confidence": report.confidence_score,
            "summary": report.summary(),
            "total_checks": len(report.checks),
            "passed_checks": sum(1 for c in report.checks if c.passed),
            "failed_checks": sum(1 for c in report.checks if not c.passed),
            "checks": [c.to_dict() for c in report.checks],
        }
    except Exception as exc:
        logger.warning("Verification failed: %s", exc, exc_info=True)
        result["verification"] = {
            "passed": None,
            "confidence": None,
            "summary": f"Verification error: {exc}",
            "error": str(exc),
        }

    return result
