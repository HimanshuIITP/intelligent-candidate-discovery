

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gates import (
    compute_fake_check_score,
    apply_hard_disqualifiers,
    apply_gates,
)

# Fixture: Clean candidate (should pass all rules)

CLEAN_CANDIDATE = {
    "candidate_id": "CAND_9999999",
    "profile": {
        "anonymized_name": "Clean Candidate",
        "headline": "ML Engineer | Search & Ranking",
        "summary": "ML engineer with 6 years of production experience.",
        "location": "Pune, Maharashtra",
        "country": "India",
        "years_of_experience": 6.0,
        "current_title": "ML Engineer",
        "current_company": "Swiggy",
        "current_company_size": "5001-10000",
        "current_industry": "Food Delivery",
    },
    "career_history": [
        {
            "company": "Swiggy",
            "title": "ML Engineer",
            "start_date": "2023-01-15",
            "end_date": None,
            "duration_months": 41,
            "is_current": True,
            "industry": "Food Delivery",
            "company_size": "5001-10000",
            "description": "Built and shipped ranking models for production search system using XGBoost. Designed evaluation framework with NDCG metrics.",
        },
        {
            "company": "Flipkart",
            "title": "Data Scientist",
            "start_date": "2020-06-01",
            "end_date": "2022-12-31",
            "duration_months": 31,
            "is_current": False,
            "industry": "E-commerce",
            "company_size": "10001+",
            "description": "Deployed embeddings-based retrieval system for product search. Built A/B testing framework.",
        },
    ],
    "education": [
        {
            "institution": "IIT Delhi",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2014,
            "end_year": 2018,
            "grade": "8.5 CGPA",
            "tier": "tier_1",
        }
    ],
    "skills": [
        {"name": "Python", "proficiency": "expert", "endorsements": 40, "duration_months": 60},
        {"name": "PyTorch", "proficiency": "advanced", "endorsements": 25, "duration_months": 36},
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 15, "duration_months": 24},
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": {
        "profile_completeness_score": 85.0,
        "signup_date": "2025-01-15",
        "last_active_date": "2026-05-20",
        "open_to_work_flag": True,
        "profile_views_received_30d": 50,
        "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.75,
        "avg_response_time_hours": 12.0,
        "skill_assessment_scores": {"Python": 85.0},
        "connection_count": 500,
        "endorsements_received": 80,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 25.0, "max": 45.0},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 55.0,
        "search_appearance_30d": 200,
        "saved_by_recruiters_30d": 10,
        "interview_completion_rate": 0.85,
        "offer_acceptance_rate": 0.6,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    },
}

def _make_candidate(**overrides):
    
    import copy
    c = copy.deepcopy(CLEAN_CANDIDATE)
    for key, value in overrides.items():
        if "." in key:
            # Handle nested keys like "profile.country"
            parts = key.split(".")
            obj = c
            for part in parts[:-1]:
                obj = obj[part]
            obj[parts[-1]] = value
        else:
            c[key] = value
    return c

# Tests: Clean candidate should pass everything

class TestCleanCandidate:
    def test_clean_passes_gates(self):
        passed, reasons = apply_hard_disqualifiers(CLEAN_CANDIDATE)
        assert passed, f"Clean candidate should pass gates but got: {reasons}"

    def test_clean_zero_fake_check(self):
        score, triggered = compute_fake_check_score(CLEAN_CANDIDATE)
        assert score == 0.0, f"Clean candidate got fake_check score {score}: {triggered}"

    def test_clean_passes_combined(self):
        passed, hp_score, reasons = apply_gates(CLEAN_CANDIDATE)
        assert passed, f"Clean candidate should pass combined gates: {reasons}"

# Tests: Each fake_check rule should trigger on crafted input

class TestFakeCheckH1ExpertZeroDuration:
    def test_triggers_on_expert_with_zero_months(self):
        c = _make_candidate(
            skills=[
                {"name": "Python", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                {"name": "PyTorch", "proficiency": "expert", "endorsements": 0, "duration_months": 2},
            ]
        )
        score, triggered = compute_fake_check_score(c)
        assert score > 0, "H1 should trigger for expert skills with 0 duration"
        assert any("H1" in t for t in triggered)

    def test_does_not_trigger_on_expert_with_long_duration(self):
        c = _make_candidate(
            skills=[
                {"name": "Python", "proficiency": "expert", "endorsements": 40, "duration_months": 60},
            ]
        )
        score, triggered = compute_fake_check_score(c)
        assert not any("H1" in t for t in triggered), "H1 should not trigger for expert with 60mo"

class TestFakeCheckH2CareerDuration:
    def test_triggers_on_excess_career_duration(self):
        c = _make_candidate()
        c["profile"]["years_of_experience"] = 3.0
        c["career_history"] = [
            {
                "company": "A", "title": "Eng", "start_date": "2020-01-01",
                "end_date": None, "duration_months": 80, "is_current": True,
                "industry": "Tech", "company_size": "201-500",
                "description": "Built production ML systems.",
            }
        ]
        score, triggered = compute_fake_check_score(c)
        assert score > 0, "H2 should trigger when career months >> YOE months"
        assert any("H2" in t for t in triggered)

class TestFakeCheckH3EndBeforeStart:
    def test_triggers_on_reversed_dates(self):
        c = _make_candidate()
        c["career_history"][1]["start_date"] = "2023-01-01"
        c["career_history"][1]["end_date"] = "2020-01-01"
        score, triggered = compute_fake_check_score(c)
        assert score >= 1.0, "H3 should give max score for end < start"
        assert any("H3" in t for t in triggered)

class TestFakeCheckH4DurationMismatch:
    def test_triggers_on_mismatched_duration(self):
        c = _make_candidate()
        c["career_history"][1]["start_date"] = "2021-01-01"
        c["career_history"][1]["end_date"] = "2022-01-01"
        c["career_history"][1]["duration_months"] = 36  # Should be ~12
        score, triggered = compute_fake_check_score(c)
        assert score > 0, "H4 should trigger for duration mismatch"
        assert any("H4" in t for t in triggered)

class TestFakeCheckH5EducationEndBeforeStart:
    def test_triggers_on_reversed_education_years(self):
        c = _make_candidate()
        c["education"][0]["start_year"] = 2020
        c["education"][0]["end_year"] = 2016
        score, triggered = compute_fake_check_score(c)
        assert score >= 1.0, "H5 should give max score"
        assert any("H5" in t for t in triggered)

class TestFakeCheckH6MassExpertZeroEndorsements:
    def test_triggers_on_many_unendorsed_expert_skills(self):
        c = _make_candidate(
            skills=[
                {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 0, "duration_months": 3}
                for i in range(6)
            ]
        )
        score, triggered = compute_fake_check_score(c)
        assert score > 0, "H6 should trigger for mass expert skills with 0 endorsements"
        assert any("H6" in t for t in triggered)

class TestFakeCheckH7SalaryInversion:
    def test_triggers_on_min_gt_max(self):
        c = _make_candidate()
        c["redrob_signals"]["expected_salary_range_inr_lpa"] = {"min": 30.0, "max": 15.0}
        score, triggered = compute_fake_check_score(c)
        assert score >= 0.8, "H7 should trigger for salary min > max"
        assert any("H7" in t for t in triggered)

    def test_does_not_trigger_on_valid_salary(self):
        c = _make_candidate()
        c["redrob_signals"]["expected_salary_range_inr_lpa"] = {"min": 15.0, "max": 30.0}
        score, triggered = compute_fake_check_score(c)
        assert not any("H7" in t for t in triggered)

class TestFakeCheckH8SignupAfterLastActive:
    def test_triggers_on_future_signup(self):
        c = _make_candidate()
        c["redrob_signals"]["signup_date"] = "2026-06-01"
        c["redrob_signals"]["last_active_date"] = "2025-12-01"
        score, triggered = compute_fake_check_score(c)
        assert score >= 0.7, "H8 should trigger when signup > last_active"
        assert any("H8" in t for t in triggered)

class TestFakeCheckH9OfferWithoutInterview:
    def test_triggers_on_offer_without_interviews(self):
        c = _make_candidate()
        c["redrob_signals"]["offer_acceptance_rate"] = 0.8
        c["redrob_signals"]["interview_completion_rate"] = 0.0
        score, triggered = compute_fake_check_score(c)
        assert score >= 0.9, "H9 should trigger for positive offers with 0 interviews"
        assert any("H9" in t for t in triggered)

class TestFakeCheckH10MassZeroDurationSkills:
    def test_triggers_on_many_zero_duration_skills(self):
        c = _make_candidate(
            skills=[
                {"name": f"Skill{i}", "proficiency": "intermediate", "endorsements": 5, "duration_months": 0}
                for i in range(10)
            ]
        )
        score, triggered = compute_fake_check_score(c)
        assert score >= 0.9, "H10 should trigger for 10 skills with 0 duration"
        assert any("H10" in t for t in triggered)

# Tests: Hard disqualifier gates

class TestGateG4PureServices:
    def test_triggers_on_all_services_career(self):
        c = _make_candidate()
        c["career_history"] = [
            {
                "company": "TCS", "title": "Software Engineer",
                "start_date": "2020-01-01", "end_date": None,
                "duration_months": 48, "is_current": True,
                "industry": "IT Services", "company_size": "10001+",
                "description": "Worked on various client projects.",
            },
            {
                "company": "Infosys", "title": "Software Engineer",
                "start_date": "2018-01-01", "end_date": "2019-12-31",
                "duration_months": 24, "is_current": False,
                "industry": "IT Services", "company_size": "10001+",
                "description": "Worked on various client projects.",
            },
        ]
        passed, reasons = apply_hard_disqualifiers(c)
        assert not passed, "G4 should gate all-services career"
        assert any("G4" in r for r in reasons)

    def test_passes_with_one_product_company(self):
        c = _make_candidate()
        c["career_history"] = [
            {
                "company": "TCS", "title": "Software Engineer",
                "start_date": "2022-01-01", "end_date": None,
                "duration_months": 24, "is_current": True,
                "industry": "IT Services", "company_size": "10001+",
                "description": "Worked on ML pipeline.",
            },
            {
                "company": "Flipkart", "title": "Data Engineer",
                "start_date": "2020-01-01", "end_date": "2021-12-31",
                "duration_months": 24, "is_current": False,
                "industry": "E-commerce", "company_size": "10001+",
                "description": "Built search system in production.",
            },
        ]
        passed, reasons = apply_hard_disqualifiers(c)
        assert not any("G4" in r for r in reasons), "G4 should not trigger with product company in history"

class TestGateG6OutsideIndia:
    def test_triggers_outside_india_no_relocate(self):
        c = _make_candidate()
        c["profile"]["country"] = "USA"
        c["redrob_signals"]["willing_to_relocate"] = False
        passed, reasons = apply_hard_disqualifiers(c)
        assert not passed
        assert any("G6" in r for r in reasons)

    def test_passes_outside_india_with_relocate(self):
        c = _make_candidate()
        c["profile"]["country"] = "USA"
        c["redrob_signals"]["willing_to_relocate"] = True
        passed, reasons = apply_hard_disqualifiers(c)
        assert not any("G6" in r for r in reasons)

class TestGateG7NonTechnical:
    def test_triggers_on_marketing_manager_no_ai(self):
        c = _make_candidate()
        c["profile"]["current_title"] = "Marketing Manager"
        c["skills"] = [
            {"name": "SEO", "proficiency": "advanced", "endorsements": 10, "duration_months": 36},
            {"name": "Content Writing", "proficiency": "expert", "endorsements": 20, "duration_months": 48},
        ]
        c["career_history"][0]["description"] = "Led marketing campaigns for B2B SaaS company."
        c["career_history"][0]["title"] = "Marketing Manager"
        c["career_history"][1]["description"] = "Managed digital marketing budget."
        c["career_history"][1]["title"] = "Marketing Assistant"
        passed, reasons = apply_hard_disqualifiers(c)
        assert any("G7" in r for r in reasons), "G7 should gate Marketing Manager with no AI evidence"

    def test_passes_marketing_manager_with_ai_skills(self):
        c = _make_candidate()
        c["profile"]["current_title"] = "Marketing Manager"
        c["skills"] = [
            {"name": "Machine Learning", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},
            {"name": "NLP", "proficiency": "expert", "endorsements": 15, "duration_months": 24},
        ]
        passed, reasons = apply_hard_disqualifiers(c)
        assert not any("G7" in r for r in reasons), "G7 should not trigger with AI skills"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
