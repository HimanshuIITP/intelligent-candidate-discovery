

import logging
import math
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Reference date for recency calculations
REFERENCE_DATE = datetime(2026, 6, 15)

def _recruiter_response_multiplier(rate: float) -> float:
    
    return 0.60 + 0.50 * min(max(rate, 0.0), 1.0)

def _recency_multiplier(last_active_str: str) -> float:
    
    if not last_active_str:
        return 0.70  # No data → assume somewhat stale

    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return 0.70

    days_since = (REFERENCE_DATE - last_active).days
    if days_since < 0:
        days_since = 0  # Future date, treat as very recent

    if days_since <= 30:
        return 1.0
    elif days_since <= 90:
        return 1.0 - 0.05 * (days_since - 30) / 60  # 1.0 → 0.95
    elif days_since <= 180:
        return 0.95 - 0.25 * (days_since - 90) / 90  # 0.95 → 0.70
    elif days_since <= 365:
        return 0.70 - 0.20 * (days_since - 180) / 185  # 0.70 → 0.50
    else:
        return 0.50

def _open_to_work_multiplier(flag: bool) -> float:
    
    return 1.05 if flag else 0.95

def _interview_completion_multiplier(rate: float) -> float:
    
    return 0.80 + 0.25 * min(max(rate, 0.0), 1.0)

def _profile_completeness_multiplier(score: float) -> float:
    
    return 0.90 + 0.15 * min(max(score, 0.0), 100.0) / 100.0

def compute_activity_score(candidate: dict) -> float:
    
    signals = candidate.get("redrob_signals", {})

    # Individual multipliers
    m_response = _recruiter_response_multiplier(
        signals.get("recruiter_response_rate", 0.0)
    )
    m_recency = _recency_multiplier(
        signals.get("last_active_date", "")
    )
    m_open = _open_to_work_multiplier(
        signals.get("open_to_work_flag", False)
    )
    m_interview = _interview_completion_multiplier(
        signals.get("interview_completion_rate", 0.0)
    )
    m_completeness = _profile_completeness_multiplier(
        signals.get("profile_completeness_score", 0.0)
    )

    # Multiplicative combination
    combined = m_response * m_recency * m_open * m_interview * m_completeness

    # Clamp to [0.5, 1.15]
    return max(0.5, min(1.15, combined))

def get_activity_details(candidate: dict) -> Dict[str, float]:
    
    signals = candidate.get("redrob_signals", {})

    return {
        "recruiter_response_multiplier": _recruiter_response_multiplier(
            signals.get("recruiter_response_rate", 0.0)
        ),
        "recency_multiplier": _recency_multiplier(
            signals.get("last_active_date", "")
        ),
        "open_to_work_multiplier": _open_to_work_multiplier(
            signals.get("open_to_work_flag", False)
        ),
        "interview_completion_multiplier": _interview_completion_multiplier(
            signals.get("interview_completion_rate", 0.0)
        ),
        "profile_completeness_multiplier": _profile_completeness_multiplier(
            signals.get("profile_completeness_score", 0.0)
        ),
    }
