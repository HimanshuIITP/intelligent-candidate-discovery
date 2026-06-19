

import logging
import math
import re
from typing import Dict, Any, List, Tuple

from src.jd_config import (
    EXPERIENCE_CENTER,
    EXPERIENCE_SIGMA,
    MUST_HAVE_SKILLS,
    SERVICES_COMPANIES,
    PRODUCTION_KEYWORDS,
    EVAL_FRAMEWORK_KEYWORDS,
    NON_TECHNICAL_TITLES,
    PREFERRED_LOCATIONS_INDIA,
    NOTICE_PERIOD_SCORES,
    PROFICIENCY_WEIGHTS,
    CV_SPEECH_ROBOTICS_SKILLS,
)
from src.gates import _is_services_company, _career_text, _has_keyword_overlap

logger = logging.getLogger(__name__)

# Individual Feature Extractors

def experience_band_fit(candidate: dict) -> float:
    
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    return math.exp(-0.5 * ((yoe - EXPERIENCE_CENTER) / EXPERIENCE_SIGMA) ** 2)

def title_trajectory_score(candidate: dict) -> float:
    
    ai_ml_titles = {
        "ai engineer", "ml engineer", "machine learning engineer",
        "data scientist", "nlp engineer", "search engineer",
        "recommendation systems engineer", "recommendation engineer",
        "applied ml engineer", "applied scientist", "research engineer",
        "ranking engineer", "retrieval engineer", "senior ai engineer",
        "staff ml engineer", "principal ml engineer",
        "senior machine learning engineer", "junior ml engineer",
    }

    adjacent_tech_titles = {
        "software engineer", "backend engineer", "data engineer",
        "full stack developer", "frontend engineer", "devops engineer",
        "platform engineer", "cloud engineer", "site reliability",
        "sre", "python developer", "java developer", ".net developer",
        "qa engineer", "mobile developer",
    }

    career = candidate.get("career_history", [])
    if not career:
        return 0.0

    # Score each role
    role_scores = []
    for ch in career:
        title = ch.get("title", "").lower().strip()
        duration = ch.get("duration_months", 0)
        is_current = ch.get("is_current", False)

        if any(ai in title for ai in ai_ml_titles):
            base = 1.0
        elif any(adj in title for adj in adjacent_tech_titles):
            base = 0.45
        elif title in NON_TECHNICAL_TITLES or any(nt in title for nt in NON_TECHNICAL_TITLES):
            base = 0.05
        else:
            base = 0.2  # Unknown title, slight benefit of doubt

        # Weight by duration and recency (current role worth more)
        weight = min(duration / 36.0, 1.0)  # Cap at 3 years
        if is_current:
            weight *= 1.3

        role_scores.append(base * weight)

    if not role_scores:
        return 0.0

    # Trajectory bonus: did titles improve toward AI/ML over time?
    # Career history is ordered newest-first in our data
    trajectory_bonus = 0.0
    if len(role_scores) >= 2:
        # If latest role scores higher than earlier ones → positive trajectory
        latest = role_scores[0]
        earliest = role_scores[-1]
        if latest > earliest:
            trajectory_bonus = 0.15  # Bonus for growth

    # Weighted average of role scores + trajectory
    total_weight = sum(min(ch.get("duration_months", 1) / 36.0, 1.0) *
                       (1.3 if ch.get("is_current") else 1.0)
                       for ch in career)
    if total_weight > 0:
        weighted_avg = sum(role_scores) / len(role_scores)
    else:
        weighted_avg = 0.0

    return min(weighted_avg + trajectory_bonus, 1.0)

def product_company_ratio(candidate: dict) -> float:
    
    career = candidate.get("career_history", [])
    if not career:
        return 0.5  # No data, neutral

    total_months = 0
    product_months = 0

    for ch in career:
        dur = ch.get("duration_months", 0)
        total_months += dur
        if not _is_services_company(ch.get("company", "")):
            product_months += dur

    if total_months == 0:
        return 0.5

    ratio = product_months / total_months
    return ratio

def skill_depth_score(candidate: dict) -> float:
    
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    total_score = 0.0
    max_possible = 0.0

    for skill in skills:
        name = skill.get("name", "").lower().strip()
        proficiency = skill.get("proficiency", "beginner")
        duration = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)

        # Find the best matching JD skill weight
        jd_weight = 0.0
        for jd_skill, weight in MUST_HAVE_SKILLS.items():
            if jd_skill in name or name in jd_skill:
                jd_weight = max(jd_weight, weight)
                break

        if jd_weight == 0:
            continue  # Not a JD-relevant skill

        # Proficiency weight
        prof_weight = PROFICIENCY_WEIGHTS.get(proficiency, 0.1)

        # Duration weight — logarithmic (diminishing returns past ~36 months)
        dur_weight = math.log(1 + duration) / math.log(1 + 60)  # Normalize to ~1.0 at 60mo

        # Endorsement credibility — minimum threshold to be believable
        endorse_weight = min(1.0, endorsements / 10.0)
        # But don't zero out just because endorsements are 0
        endorse_weight = max(endorse_weight, 0.3)

        skill_score = jd_weight * prof_weight * dur_weight * endorse_weight
        total_score += skill_score
        max_possible += jd_weight  # Theoretical max if expert, 60mo, 10+ endorsements

    if max_possible == 0:
        return 0.0

    # Normalize: a perfect candidate might score ~5-8 total; cap at 1.0
    return min(total_score / 3.0, 1.0)

def evaluation_framework_evidence(candidate: dict) -> float:
    
    career = _career_text(candidate)
    if not career:
        return 0.0

    hits = sum(1 for kw in EVAL_FRAMEWORK_KEYWORDS if kw in career)

    # Also check skill_assessment_scores existence as a meta-signal
    assessment_count = len(
        candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    )

    # Normalize: 3+ keyword hits is strong evidence
    keyword_score = min(hits / 3.0, 1.0)

    return keyword_score

def production_deployment_evidence(candidate: dict) -> float:
    
    career = _career_text(candidate)
    if not career:
        return 0.0

    hits = sum(1 for kw in PRODUCTION_KEYWORDS if kw in career)

    # Normalize: 4+ hits = strong production evidence
    return min(hits / 4.0, 1.0)

def vector_db_experience(candidate: dict) -> float:
    
    vector_db_names = {
        "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "opensearch", "elasticsearch", "chroma", "chromadb",
        "pgvector", "annoy", "scann",
    }

    # Check skills
    skill_score = 0.0
    for skill in candidate.get("skills", []):
        name = skill.get("name", "").lower()
        if any(vdb in name for vdb in vector_db_names):
            prof_w = PROFICIENCY_WEIGHTS.get(skill.get("proficiency"), 0.1)
            dur_w = min(skill.get("duration_months", 0) / 24.0, 1.0)
            skill_score = max(skill_score, prof_w * dur_w)

    # Check career descriptions
    career = _career_text(candidate)
    career_score = 0.0
    vector_career_kw = {
        "vector database", "vector search", "vector db", "hybrid search",
        "approximate nearest neighbor", "ann", "knn search",
        "similarity search", "embedding index",
    }
    career_score = min(
        sum(1 for kw in vector_career_kw if kw in career) / 2.0, 1.0
    )
    # Also check for specific DB names in career text
    for vdb in vector_db_names:
        if vdb in career:
            career_score = max(career_score, 0.5)

    return max(skill_score, career_score)

def location_fit(candidate: dict) -> float:
    
    profile = candidate.get("profile", {})
    country = profile.get("country", "").lower().strip()
    location = profile.get("location", "").lower().strip()
    signals = candidate.get("redrob_signals", {})
    willing = signals.get("willing_to_relocate", False)
    work_mode = signals.get("preferred_work_mode", "")

    if country == "india":
        # In India — check preferred cities
        for city in PREFERRED_LOCATIONS_INDIA:
            if city in location:
                return 1.0
        return 0.85  # India but not preferred city

    # Outside India
    if willing:
        return 0.5  # Willing to relocate — possible but uncertain
    return 0.15  # Outside India, won't relocate

def notice_period_score_fn(candidate: dict) -> float:
    
    signals = candidate.get("redrob_signals", {})
    notice = signals.get("notice_period_days", 90)

    for (lo, hi), score in NOTICE_PERIOD_SCORES.items():
        if lo <= notice <= hi:
            return score

    return 0.2  # Outside all ranges (> 180)

# Career Evidence Graph — novel differentiator

def feature_combination_score(candidate: dict) -> float:
    
    career = _career_text(candidate)
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "").lower()
    skills = candidate.get("skills", [])

    # Detect evidence nodes
    ai_titles = {
        "ai engineer", "ml engineer", "machine learning", "data scientist",
        "nlp engineer", "search engineer", "recommendation", "ranking",
        "applied ml", "applied scientist", "research engineer",
    }
    has_ai_title = any(at in title for at in ai_titles)

    has_production = _has_keyword_overlap(career, PRODUCTION_KEYWORDS)
    has_eval = _has_keyword_overlap(career, EVAL_FRAMEWORK_KEYWORDS)

    vdb_names = {"faiss", "pinecone", "weaviate", "qdrant", "milvus",
                 "opensearch", "elasticsearch"}
    has_vector_db = any(
        any(vdb in s.get("name", "").lower() for vdb in vdb_names)
        for s in skills
    ) or any(vdb in career for vdb in vdb_names)

    has_product = any(
        not _is_services_company(ch.get("company", ""))
        for ch in candidate.get("career_history", [])
    )

    ranking_kw = {
        "ranking", "search system", "recommendation system",
        "retrieval", "relevance", "search quality", "learning to rank",
        "learning-to-rank", "l2r", "re-ranking", "reranking",
    }
    has_ranking = _has_keyword_overlap(career, ranking_kw)

    # Count evidence nodes
    evidence = [
        has_ai_title, has_production, has_eval,
        has_vector_db, has_product, has_ranking,
    ]
    evidence_count = sum(evidence)

    if evidence_count == 0:
        return 0.0

    # Base score from node count
    base = evidence_count / 6.0

    # Edge boosts — specific combinations that corroborate
    boost = 1.0

    # AI title + production narrative = they actually build AI in production
    if has_ai_title and has_production:
        boost *= 1.25

    # Production + eval framework = they measure what they ship
    if has_production and has_eval:
        boost *= 1.15

    # AI title + product company = real product-company AI work
    if has_ai_title and has_product:
        boost *= 1.15

    # Vector DB + ranking experience = the exact JD must-have combination
    if has_vector_db and has_ranking:
        boost *= 1.20

    # Full chain: AI title + production + ranking + product company
    if has_ai_title and has_production and has_ranking and has_product:
        boost *= 1.15  # Extra for the full chain

    return min(base * boost, 1.0)

# Feature Extraction Entry Point

def extract_features(candidate: dict) -> Dict[str, float]:
    
    return {
        "experience_band_fit": experience_band_fit(candidate),
        "title_trajectory_score": title_trajectory_score(candidate),
        "product_company_ratio": product_company_ratio(candidate),
        "skill_depth_score": skill_depth_score(candidate),
        "evaluation_framework_evidence": evaluation_framework_evidence(candidate),
        "production_deployment_evidence": production_deployment_evidence(candidate),
        "vector_db_experience": vector_db_experience(candidate),
        "location_fit": location_fit(candidate),
        "notice_period_score": notice_period_score_fn(candidate),
        "feature_combination_score": feature_combination_score(candidate),
    }
