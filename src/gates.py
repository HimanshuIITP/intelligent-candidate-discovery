

import logging
import re
from datetime import datetime, date
from typing import Tuple, List, Dict, Any

from src.jd_config import (
    SERVICES_COMPANIES,
    CV_SPEECH_ROBOTICS_SKILLS,
    MUST_HAVE_SKILLS,
    NON_TECHNICAL_TITLES,
    PRODUCTION_KEYWORDS,
    RESEARCH_ONLY_KEYWORDS,
    PROFICIENCY_WEIGHTS,
)

logger = logging.getLogger(__name__)

# Helper utilities

def _lower_set(items):
    
    return {s.lower().strip() for s in items if isinstance(s, str)}

def _career_text(candidate: dict) -> str:
    
    parts = []
    for ch in candidate.get("career_history", []):
        desc = ch.get("description", "")
        title = ch.get("title", "")
        parts.append(f"{title} {desc}")
    return " ".join(parts).lower()

def _has_keyword_overlap(text: str, keywords: set) -> bool:
    
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)

def _count_keyword_hits(text: str, keywords: set) -> int:
    
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)

def _parse_date_safe(date_str: str) -> datetime | None:
    
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None

def _is_services_company(company_name: str) -> bool:
    
    name = company_name.lower().strip()
    return any(svc in name or name in svc for svc in SERVICES_COMPANIES)

# Hard Disqualifiers — boolean gates from the JD

def _gate_g1_pure_research(candidate: dict) -> str | None:
    
    career = _career_text(candidate)
    if not career:
        return None

    has_production = _has_keyword_overlap(career, PRODUCTION_KEYWORDS)
    has_research = _has_keyword_overlap(career, RESEARCH_ONLY_KEYWORDS)

    # Also check titles for research-only indicators
    titles = [ch.get("title", "").lower() for ch in candidate.get("career_history", [])]
    research_titles = any(
        "research" in t and "engineer" not in t
        for t in titles
    )

    if has_research and not has_production and research_titles:
        return "G1: Career appears to be pure research/academic with no production deployment evidence"
    return None

def _gate_g2_recent_llm_only(candidate: dict) -> str | None:
    
    career = candidate.get("career_history", [])
    if not career:
        return None

    total_months = sum(ch.get("duration_months", 0) for ch in career)
    if total_months > 24:
        # More than 2 years of career — probably not just recent LLM work
        return None

    career_text = _career_text(candidate)

    # Check if career mentions only LLM/LangChain without deeper ML terms
    llm_only_keywords = {"langchain", "chatgpt", "openai api", "gpt-4", "claude api"}
    deeper_ml_keywords = {
        "training", "model training", "feature engineering", "sklearn",
        "xgboost", "pytorch", "tensorflow", "embeddings", "retrieval",
        "ranking model", "recommendation", "classifier", "regression",
    }

    has_llm = _has_keyword_overlap(career_text, llm_only_keywords)
    has_deep_ml = _has_keyword_overlap(career_text, deeper_ml_keywords)

    if has_llm and not has_deep_ml and total_months < 18:
        return "G2: AI experience appears limited to recent LLM/LangChain work (<18 months) with no pre-LLM ML evidence"
    return None

def _gate_g3_no_recent_code(candidate: dict) -> str | None:
    
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "").lower()

    management_titles = {
        "vp", "vice president", "director", "cto", "cio", "chief",
        "head of", "architect",  # "tech lead" is borderline — keep
    }

    is_management = any(mt in title for mt in management_titles)
    if not is_management:
        return None

    # Check current role description for code/build language
    career = candidate.get("career_history", [])
    if not career:
        return None

    current_role = career[0] if career[0].get("is_current", False) else None
    if not current_role:
        return None

    desc = current_role.get("description", "").lower()
    code_keywords = {
        "code", "coding", "implemented", "built", "developed", "shipped",
        "python", "java", "wrote", "designed and built", "engineered",
    }

    if not _has_keyword_overlap(desc, code_keywords):
        return "G3: Senior/management title with no evidence of recent hands-on coding in current role"
    return None

def _gate_g4_pure_services(candidate: dict) -> str | None:
    
    career = candidate.get("career_history", [])
    if not career:
        return None

    all_services = all(
        _is_services_company(ch.get("company", ""))
        for ch in career
    )

    if all_services:
        companies = [ch.get("company", "") for ch in career]
        return f"G4: Entire career at IT-services firms: {', '.join(companies)}"
    return None

def _gate_g5_cv_speech_only(candidate: dict) -> str | None:
    
    skills = candidate.get("skills", [])
    if not skills:
        return None

    # Count advanced/expert skills in CV/speech vs NLP/IR domains
    cv_speech_count = 0
    nlp_ir_count = 0

    nlp_ir_skills = {
        "nlp", "natural language processing", "information retrieval",
        "search", "ranking", "recommendation", "embeddings", "bert",
        "transformers", "text classification", "sentiment analysis",
        "named entity recognition", "ner", "text mining",
        "sentence transformers", "bm25",
    }

    for skill in skills:
        name = skill.get("name", "").lower()
        prof = skill.get("proficiency", "")
        if prof in ("advanced", "expert"):
            if name in CV_SPEECH_ROBOTICS_SKILLS or any(
                cv in name for cv in CV_SPEECH_ROBOTICS_SKILLS
            ):
                cv_speech_count += 1
            if name in nlp_ir_skills or any(n in name for n in nlp_ir_skills):
                nlp_ir_count += 1

    # Also check career text for NLP/IR terms
    career = _career_text(candidate)
    has_nlp_career = _has_keyword_overlap(
        career,
        {"nlp", "information retrieval", "search", "ranking", "recommendation",
         "text", "language model", "embeddings", "retrieval"},
    )

    if cv_speech_count >= 3 and nlp_ir_count == 0 and not has_nlp_career:
        return "G5: Primary expertise appears to be CV/speech/robotics with no NLP/IR exposure"
    return None

def _gate_g6_outside_india_no_relocate(candidate: dict) -> str | None:
    
    profile = candidate.get("profile", {})
    country = profile.get("country", "").strip()
    signals = candidate.get("redrob_signals", {})
    willing = signals.get("willing_to_relocate", False)

    if country.lower() != "india" and not willing:
        return f"G6: Located in {country} with willing_to_relocate=false (no visa sponsorship)"
    return None

def _gate_g7_non_technical_no_ai(candidate: dict) -> str | None:
    
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "").lower()

    # Check if title is non-technical
    is_non_tech = any(nt in title for nt in NON_TECHNICAL_TITLES)
    if not is_non_tech:
        return None

    # Check for ANY AI/ML evidence in skills
    skills = candidate.get("skills", [])
    ai_skill_count = 0
    for skill in skills:
        name = skill.get("name", "").lower()
        prof = skill.get("proficiency", "")
        if prof in ("advanced", "expert"):
            if name in MUST_HAVE_SKILLS or any(
                ms in name for ms in MUST_HAVE_SKILLS
            ):
                ai_skill_count += 1

    # Check career descriptions for AI evidence
    career = _career_text(candidate)
    ai_career_keywords = {
        "machine learning", "ml", "deep learning", "neural",
        "embeddings", "nlp", "ranking model", "recommendation",
        "search system", "retrieval", "classifier", "training",
    }
    has_ai_career = _has_keyword_overlap(career, ai_career_keywords)

    if ai_skill_count == 0 and not has_ai_career:
        return f"G7: Non-technical title '{profile.get('current_title', '')}' with no AI/ML skill or career evidence"
    return None

def apply_hard_disqualifiers(
    candidate: dict,
) -> Tuple[bool, List[str]]:
    
    gates = [
        _gate_g1_pure_research,
        _gate_g2_recent_llm_only,
        _gate_g3_no_recent_code,
        _gate_g4_pure_services,
        _gate_g5_cv_speech_only,
        _gate_g6_outside_india_no_relocate,
        _gate_g7_non_technical_no_ai,
    ]

    reasons = []
    for gate_fn in gates:
        result = gate_fn(candidate)
        if result:
            reasons.append(result)

    passed = len(reasons) == 0
    return passed, reasons

# FakeCheck Detection — internal data consistency checks

def _fake_check_h1_expert_zero_duration(candidate: dict) -> Tuple[float, str | None]:
    
    score = 0.0
    triggers = []
    for skill in candidate.get("skills", []):
        if skill.get("proficiency") == "expert" and skill.get("duration_months", 99) <= 3:
            triggers.append(
                f"expert '{skill['name']}' with only {skill.get('duration_months', 0)} months"
            )
            score += 0.3

    if triggers:
        return min(score, 1.0), f"H1: {'; '.join(triggers)}"
    return 0.0, None

def _fake_check_h2_career_duration_exceeds_experience(
    candidate: dict,
) -> Tuple[float, str | None]:
    
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    if yoe <= 0:
        return 0.0, None

    total_months = sum(
        ch.get("duration_months", 0) for ch in candidate.get("career_history", [])
    )
    expected_months = yoe * 12
    ratio = total_months / max(expected_months, 1)

    if ratio > 1.5:
        excess = total_months - expected_months
        severity = min((ratio - 1.5) / 1.0, 1.0)  # Linear ramp from 1.5x to 2.5x
        return 0.3 + 0.7 * severity, (
            f"H2: Career duration {total_months}mo vs stated {yoe}yr "
            f"({expected_months}mo) — ratio {ratio:.1f}x"
        )
    return 0.0, None

def _fake_check_h3_end_before_start(candidate: dict) -> Tuple[float, str | None]:
    
    for ch in candidate.get("career_history", []):
        start = _parse_date_safe(ch.get("start_date"))
        end = _parse_date_safe(ch.get("end_date"))
        if start and end and end < start:
            return 1.0, (
                f"H3: end_date ({ch['end_date']}) before start_date "
                f"({ch['start_date']}) at {ch.get('company', '?')}"
            )
    return 0.0, None

def _fake_check_h4_duration_date_mismatch(candidate: dict) -> Tuple[float, str | None]:
    
    triggers = []
    for ch in candidate.get("career_history", []):
        start = _parse_date_safe(ch.get("start_date"))
        end = _parse_date_safe(ch.get("end_date"))
        stated = ch.get("duration_months", 0)

        if start and end and stated:
            actual = (end.year - start.year) * 12 + (end.month - start.month)
            if abs(actual - stated) > 6:
                triggers.append(
                    f"'{ch.get('company', '?')}': actual={actual}mo vs stated={stated}mo"
                )

    if triggers:
        score = min(0.5 * len(triggers), 1.0)
        return score, f"H4: Duration mismatches — {'; '.join(triggers)}"
    return 0.0, None

def _fake_check_h5_education_end_before_start(
    candidate: dict,
) -> Tuple[float, str | None]:
    
    for edu in candidate.get("education", []):
        start_yr = edu.get("start_year", 0)
        end_yr = edu.get("end_year", 9999)
        if end_yr < start_yr:
            return 1.0, (
                f"H5: Education end_year ({end_yr}) before start_year ({start_yr}) "
                f"at {edu.get('institution', '?')}"
            )
    return 0.0, None

def _fake_check_h6_mass_expert_zero_endorsements(
    candidate: dict,
) -> Tuple[float, str | None]:
    
    stuffed = []
    for skill in candidate.get("skills", []):
        if (
            skill.get("proficiency") in ("expert", "advanced")
            and skill.get("endorsements", 0) == 0
            and skill.get("duration_months", 0) <= 6
        ):
            stuffed.append(skill.get("name", "?"))

    if len(stuffed) >= 4:
        return 1.0, f"H6: {len(stuffed)} expert/advanced skills with 0 endorsements and ≤6mo: {', '.join(stuffed[:5])}"
    if len(stuffed) >= 2:
        return 0.4, f"H6: {len(stuffed)} expert/advanced skills with 0 endorsements and ≤6mo: {', '.join(stuffed)}"
    return 0.0, None

def _fake_check_h7_salary_inversion(candidate: dict) -> Tuple[float, str | None]:
    
    signals = candidate.get("redrob_signals", {})
    salary = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = salary.get("min", 0)
    sal_max = salary.get("max", float("inf"))

    if sal_min > sal_max:
        return 0.8, f"H7: Salary min ({sal_min}) > max ({sal_max}) LPA"
    return 0.0, None

def _fake_check_h8_signup_after_last_active(
    candidate: dict,
) -> Tuple[float, str | None]:
    
    signals = candidate.get("redrob_signals", {})
    signup = _parse_date_safe(signals.get("signup_date"))
    last_active = _parse_date_safe(signals.get("last_active_date"))

    if signup and last_active and signup > last_active:
        return 0.7, (
            f"H8: signup_date ({signals['signup_date']}) after "
            f"last_active_date ({signals['last_active_date']})"
        )
    return 0.0, None

def _fake_check_h9_offer_without_interview(
    candidate: dict,
) -> Tuple[float, str | None]:
    
    signals = candidate.get("redrob_signals", {})
    offer_rate = signals.get("offer_acceptance_rate", -1)
    interview_rate = signals.get("interview_completion_rate", 0)

    if offer_rate > 0 and interview_rate == 0:
        return 0.9, (
            f"H9: offer_acceptance_rate={offer_rate} but "
            f"interview_completion_rate=0"
        )
    return 0.0, None

def _fake_check_h10_mass_zero_duration_skills(
    candidate: dict,
) -> Tuple[float, str | None]:
    
    zero_dur = [
        s.get("name", "?")
        for s in candidate.get("skills", [])
        if s.get("duration_months", 0) == 0
    ]

    if len(zero_dur) >= 8:
        return 0.9, f"H10: {len(zero_dur)} skills with 0 duration_months"
    return 0.0, None

def _fake_check_h11_title_description_mismatch(
    candidate: dict,
) -> Tuple[float, str | None]:
    
    for ch in candidate.get("career_history", []):
        title = ch.get("title", "").lower()
        desc = ch.get("description", "").lower()
        if not title or not desc:
            continue

        # Map title domains to expected description keywords
        title_domain_keywords = {
            "marketing": {"marketing", "campaign", "brand", "seo", "content", "demand"},
            "accountant": {"accounting", "financial", "gaap", "tax", "audit", "ledger"},
            "hr": {"recruitment", "hiring", "employee", "onboarding", "talent"},
            "sales": {"quota", "revenue", "pipeline", "deal", "prospect", "closing"},
            "mechanical": {"cad", "manufacturing", "design", "assembly", "tolerance"},
            "civil": {"construction", "structural", "site", "building", "concrete"},
            "support": {"support", "ticket", "customer", "escalation", "helpdesk"},
        }

        for domain, expected_kw in title_domain_keywords.items():
            if domain in title:
                # Title is in this domain — check if description matches
                desc_has_domain = any(kw in desc for kw in expected_kw)
                if not desc_has_domain:
                    # Description doesn't mention anything from this domain
                    # This alone isn't definitive but contributes
                    pass  # We'll handle this differently — just a soft signal

    # For now, this rule is a no-op in the fast gates phase.
    # The real title-description coherence check happens in structured_features
    # where we have access to embeddings.
    return 0.0, None

def _fake_check_h12_impossible_timeline(candidate: dict) -> Tuple[float, str | None]:
    
    education = candidate.get("education", [])
    career = candidate.get("career_history", [])

    if not education or not career:
        return 0.0, None

    # Find earliest education start (proxy for birth year + ~18)
    earliest_edu_start = min(
        (edu.get("start_year", 9999) for edu in education), default=9999
    )
    if earliest_edu_start == 9999:
        return 0.0, None

    # Approximate birth year: education start - 18 (typical age for undergrad)
    # For masters/PhD, adjust
    min_degree_age = {
        "b.tech": 17, "b.e.": 17, "b.sc": 17, "b.s.": 17, "ba": 17,
        "m.tech": 21, "m.e.": 21, "m.sc": 21, "m.s.": 21, "mba": 21,
        "ph.d": 24, "phd": 24,
    }

    estimated_birth_year = earliest_edu_start - 18  # Default assumption
    for edu in education:
        degree = edu.get("degree", "").lower().strip()
        start_yr = edu.get("start_year", 9999)
        age_at_start = min_degree_age.get(degree, 17)
        birth_est = start_yr - age_at_start
        estimated_birth_year = min(estimated_birth_year, birth_est)

    # Find earliest career start
    for ch in career:
        start = _parse_date_safe(ch.get("start_date"))
        if start:
            career_start_year = start.year
            age_at_career_start = career_start_year - estimated_birth_year
            if age_at_career_start < 14:
                return 1.0, (
                    f"H12: Career started in {career_start_year} but education "
                    f"timeline implies birth ~{estimated_birth_year} "
                    f"(age {age_at_career_start} at career start)"
                )

    return 0.0, None

# All fake_check rule functions
HONEYPOT_RULES = [
    _fake_check_h1_expert_zero_duration,
    _fake_check_h2_career_duration_exceeds_experience,
    _fake_check_h3_end_before_start,
    _fake_check_h4_duration_date_mismatch,
    _fake_check_h5_education_end_before_start,
    _fake_check_h6_mass_expert_zero_endorsements,
    _fake_check_h7_salary_inversion,
    _fake_check_h8_signup_after_last_active,
    _fake_check_h9_offer_without_interview,
    _fake_check_h10_mass_zero_duration_skills,
    _fake_check_h11_title_description_mismatch,
    _fake_check_h12_impossible_timeline,
]

def compute_fake_check_score(
    candidate: dict,
) -> Tuple[float, List[str]]:
    
    total_score = 0.0
    triggered = []

    for rule_fn in HONEYPOT_RULES:
        rule_score, rule_msg = rule_fn(candidate)
        if rule_score > 0 and rule_msg:
            total_score += rule_score
            triggered.append(rule_msg)

    # Cap at 1.0
    return min(total_score, 1.0), triggered

def apply_gates(
    candidate: dict,
) -> Tuple[bool, float, List[str]]:
    
    # Basic checks to drop bad profiles
    gate_passed, gate_reasons = apply_hard_disqualifiers(candidate)

    # FakeCheck detection
    hp_score, hp_reasons = compute_fake_check_score(candidate)

    all_reasons = gate_reasons + hp_reasons

    # Candidate passes if: no hard disqualifiers AND fake_check score < 0.5
    passed = gate_passed and hp_score < 0.5

    return passed, hp_score, all_reasons
