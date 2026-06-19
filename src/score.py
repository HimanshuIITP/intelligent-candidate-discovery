

import logging
from typing import Dict, Any, List, Tuple

from src.structured_features import extract_features
from src.activity import compute_activity_score
from src.gates import compute_fake_check_score

logger = logging.getLogger(__name__)

# Scoring logic

# Weight rationale — each weight maps to a JD requirement with priority
FEATURE_WEIGHTS = {
    "retrieval_score": 0.25,

    # This is the JD's "decisive signal" — second-heaviest weight
    "title_trajectory_score": 0.18,

    # built a real recommendation/search/ranking system at a product company"
    "feature_combination_score": 0.17,

    # Important but explicitly the "weakest, most gameable signal" per JD
    "skill_depth_score": 0.10,

    # deployed to real users"
    "production_deployment_evidence": 0.08,

    # systems — NDCG, MRR, MAP, offline-to-online correlation"
    "evaluation_framework_evidence": 0.06,

    "vector_db_experience": 0.05,

    "product_company_ratio": 0.04,

    "experience_band_fit": 0.04,

    "location_fit": 0.02,

    "notice_period_score": 0.01,
}

def compute_final_score(
    candidate: dict,
    retrieval_score: float,
    fake_check_score: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    
    # Extract structured features
    features = extract_features(candidate)
    features["retrieval_score"] = retrieval_score

    # Compute weighted sum
    weighted_sum = 0.0
    for feature_name, weight in FEATURE_WEIGHTS.items():
        value = features.get(feature_name, 0.0)
        weighted_sum += weight * value

    # Behavioral multiplier (multiplicative, bounded [0.5, 1.15])
    activity = compute_activity_score(candidate)

    # Career evidence graph boost — already included in weighted sum via its weight,
    # but we also use it as a mild multiplicative boost for candidates with strong
    # corroborating evidence chains
    comb_score = features.get("feature_combination_score", 0.0)
    comb_boost = 1.0 + 0.15 * comb_score  # 1.0-1.15× boost

    # FakeCheck penalty
    hp_penalty = 1.0 - 0.5 * fake_check_score

    final = round(weighted_sum * activity * comb_boost * hp_penalty, 4)

    # Build breakdown for debugging/reasoning
    breakdown = {
        **features,
        "activity_score": activity,
        "comb_boost": comb_boost,
        "fake_check_penalty": hp_penalty,
        "weighted_sum": weighted_sum,
        "final_score": final,
    }

    return final, breakdown

def rank_candidates(
    scored_candidates: List[Tuple[str, float, dict, Dict[str, float]]],
    top_k: int = 100,
) -> List[Tuple[str, int, float, dict, Dict[str, float]]]:
    
    # Sort by score descending, then by candidate_id ascending for ties
    sorted_list = sorted(
        scored_candidates,
        key=lambda x: (-x[1], x[0]),  score for descending, id for ascending tie-break
    )

    # Take top k and assign ranks
    result = []
    for i, (cid, score, candidate, breakdown) in enumerate(sorted_list[:top_k]):
        rank = i + 1
        result.append((cid, rank, score, candidate, breakdown))

    return result
