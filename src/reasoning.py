

import random
from typing import Dict, Any, List, Tuple

random.seed(42)  # Reproducible reasoning variation

def _get_top_jd_skills(candidate: dict, n: int = 2) -> List[Tuple[str, str, int]]:
    
    from src.jd_config import MUST_HAVE_SKILLS, PROFICIENCY_WEIGHTS

    skills = candidate.get("skills", [])
    scored = []
    for skill in skills:
        name = skill.get("name", "")
        prof = skill.get("proficiency", "beginner")
        dur = skill.get("duration_months", 0)

        # Check if JD-relevant
        name_lower = name.lower()
        jd_weight = 0.0
        for jd_skill, weight in MUST_HAVE_SKILLS.items():
            if jd_skill in name_lower or name_lower in jd_skill:
                jd_weight = weight
                break

        if jd_weight > 0:
            score = jd_weight * PROFICIENCY_WEIGHTS.get(prof, 0.1) * (dur + 1)
            scored.append((score, name, prof, dur))

    scored.sort(reverse=True)
    return [(name, prof, dur) for _, name, prof, dur in scored[:n]]

def _get_concern(candidate: dict, breakdown: Dict[str, float]) -> str | None:
    
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    concerns = []

    # Notice period
    notice = signals.get("notice_period_days", 0)
    if notice > 60:
        concerns.append(f"notice period is {notice} days (JD prefers sub-30d)")

    # Response rate
    rr = signals.get("recruiter_response_rate", 0)
    if rr < 0.3:
        concerns.append(f"recruiter response rate is only {rr:.0%}")

    # Recency
    last_active = signals.get("last_active_date", "")
    if last_active and last_active < "2026-01-01":
        concerns.append(f"last active {last_active} (potentially stale)")

    # Experience outside band
    yoe = profile.get("years_of_experience", 0)
    if yoe < 4:
        concerns.append(f"only {yoe:.1f} years experience (JD targets 5-9)")
    elif yoe > 12:
        concerns.append(f"{yoe:.1f} years experience may be overqualified for band")

    # Location
    country = profile.get("country", "")
    if country.lower() != "india":
        concerns.append(f"located in {country} (JD prefers India-based)")

    # Low production evidence
    if breakdown.get("production_deployment_evidence", 0) < 0.3:
        concerns.append("limited production deployment evidence in career history")

    # Low eval framework evidence
    if breakdown.get("evaluation_framework_evidence", 0) < 0.2:
        concerns.append("no explicit evaluation framework experience mentioned")

    if concerns:
        return concerns[0]  # Return the most relevant concern
    return None

def generate_reasoning(
    rank: int,
    candidate: dict,
    breakdown: Dict[str, float],
) -> str:
    
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    country = profile.get("country", "")
    location = profile.get("location", "")

    top_skills = _get_top_jd_skills(candidate, n=2)
    concern = _get_concern(candidate, breakdown)

    rr = signals.get("recruiter_response_rate", 0)
    open_flag = signals.get("open_to_work_flag", False)
    github = signals.get("github_activity_score", -1)

    # Build skill mention string
    skill_str = ""
    if top_skills:
        skill_parts = []
        for name, prof, dur in top_skills:
            skill_parts.append(f"{name} ({prof}, {dur}mo)")
        skill_str = "; ".join(skill_parts)

    # Determine engagement summary
    engagement_parts = []
    if rr >= 0.5:
        engagement_parts.append(f"response rate {rr:.0%}")
    if open_flag:
        engagement_parts.append("actively seeking")
    if github > 20:
        engagement_parts.append(f"GitHub activity {github:.0f}")
    engagement_str = ", ".join(engagement_parts) if engagement_parts else ""

    #  Top 10: Confident tone 
    if rank <= 10:
        templates = [
            lambda: (
                f"{title} at {company} with {yoe:.1f} years"
                + (f"; strong JD-aligned skills: {skill_str}" if skill_str else "")
                + (f"; career history shows production search/ranking system experience" if breakdown.get("feature_combination_score", 0) > 0.5 else "")
                + (f". {engagement_str}." if engagement_str else ".")
                + (f" Minor concern: {concern}." if concern else "")
            ),
            lambda: (
                f"Strong fit: {yoe:.1f}yr {title} at {company}"
                + (f" with depth in {skill_str}" if skill_str else "")
                + (f"; evidence of production deployment and evaluation framework experience" if breakdown.get("production_deployment_evidence", 0) > 0.5 and breakdown.get("evaluation_framework_evidence", 0) > 0.3 else "; career narrative aligns with JD core requirements")
                + (f". {engagement_str}." if engagement_str else ".")
                + (f" Note: {concern}." if concern else "")
            ),
            lambda: (
                f"{title} ({yoe:.1f}yr) at {company} — "
                + ("career trajectory shows clear production AI/ML focus" if breakdown.get("title_trajectory_score", 0) > 0.6 else "relevant technical background")
                + (f"; key skills: {skill_str}" if skill_str else "")
                + (f". Behavioral signals positive ({engagement_str})." if engagement_str else ".")
                + (f" Caveat: {concern}." if concern else "")
            ),
        ]

    #  11-40: Measured tone 
    elif rank <= 40:
        templates = [
            lambda: (
                f"{title} at {company} with {yoe:.1f} years"
                + (f"; relevant skills include {skill_str}" if skill_str else "")
                + ". "
                + (f"Career shows some alignment with JD requirements" if breakdown.get("feature_combination_score", 0) > 0.3 else "Profile shows adjacent technical capabilities")
                + (f". {concern}." if concern else ".")
            ),
            lambda: (
                f"{yoe:.1f}yr {title} ({company})"
                + (f" with {skill_str}" if skill_str else "")
                + (f"; career evidence suggests {('production ML/search experience' if breakdown.get('production_deployment_evidence', 0) > 0.3 else 'technical depth with some JD overlap')}")
                + (f". However, {concern}." if concern else ".")
            ),
        ]

    #  41-80: Balanced with caveats 
    elif rank <= 80:
        templates = [
            lambda: (
                f"{title} ({yoe:.1f}yr, {company})"
                + (f"; some JD-relevant skills: {skill_str}" if skill_str else "; limited direct skill match to JD")
                + (f". {concern}." if concern else ". Included based on partial technical alignment and activity signals.")
            ),
            lambda: (
                f"Partial match: {title} at {company} with {yoe:.1f} years"
                + (f". Has {skill_str}" if skill_str else "")
                + (f", but {concern}" if concern else "; career narrative shows some adjacency to JD requirements")
                + "."
            ),
        ]

    #  81-100: Hedged, explicit borderline 
    else:
        templates = [
            lambda: (
                f"Borderline inclusion: {title} ({yoe:.1f}yr, {company})"
                + (f"; {skill_str}" if skill_str else "")
                + (f". {concern}." if concern else ". Limited direct JD alignment but included as filler given available pool.")
                + " Ranked at bottom of shortlist due to thin evidence of core JD requirements."
            ),
            lambda: (
                f"Included as tail-end candidate: {title} at {company} ({yoe:.1f}yr)"
                + (f". Has {top_skills[0][0]} ({top_skills[0][1]})" if top_skills else "")
                + (f", however {concern}" if concern else "")
                + ". Below ideal threshold but included to fill top-100 given pool composition."
            ),
        ]

    # Pick a template (deterministic per candidate for reproducibility)
    idx = hash(candidate.get("candidate_id", "")) % len(templates)
    reasoning = templates[idx]()

    # Clean up double periods, extra spaces
    reasoning = reasoning.replace("..", ".").replace("  ", " ").strip()

    # Ensure it's 1-2 sentences (truncate if somehow too long)
    if len(reasoning) > 500:
        reasoning = reasoning[:497] + "..."

    return reasoning
