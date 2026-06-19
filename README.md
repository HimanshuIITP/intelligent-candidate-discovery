# Redrob "India Runs" Hackathon — Track 1

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![Hackathon](https://img.shields.io/badge/Hackathon-India%20Runs-orange)
![Constraint](https://img.shields.io/badge/Constraint-<5m%20CPU-red)

**Intelligent Candidate Discovery & Ranking System** built specifically for the *Redrob Data & AI Challenge*. 

This is a complete, defensible, **CPU-only** candidate-ranking pipeline designed to process 100,000 candidate profiles and identify the top 100 fits for a Senior AI Engineer position. It operates strictly offline, utilizes zero remote API calls at runtime, and completes its execution well under the 5-minute wall-clock limit.

---

## ⚡ Quick Start

```bash
# 1. Environment Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download Offline Model Weights (One-time setup)
python setup_models.py

# 3. Run Pipeline (100k JSONL dataset -> Top 100 CSV)
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

# 4. Validate output
python validate_submission.py ./submission.csv
```

---

## 🏗️ Architecture & Pipeline

The system processes streaming candidates through an extremely tight, highly optimized 5-phase funnel.

### Phase 1: Gate Evaluation & Streaming (`src/gates.py`)
- **`orjson` Stream Parsing:** Parses the 120MB+ JSONL file virtually instantly.
- **7 Hard Disqualifiers:** Immediately drops candidates failing core requirements (e.g., pure IT-services with zero product exposure, fully non-technical roles without AI evidence).
- **12 Honeypot Rules:** Aggressively drops AI-generated resumes by cross-referencing timeline impossibilities, salary inversions, and massive skill lists with zero endorsements/duration.

**Honeypot Detection Rules Overview**:
| # | Rule | Signal |
|---|------|--------|
| H1 | Expert skill with near-zero duration | `proficiency == "expert"` AND `duration_months <= 3` |
| H2 | Career duration exceeds stated experience | `sum(career_history.duration_months) > years_of_experience * 12 * 1.5 + 12` |
| H3 | End date before start date in career | `end_date < start_date` for any career entry |
| H4 | Duration doesn't match date range | `\|actual_months - stated_duration\| > 6` |
| H5 | Education end before start | `end_year < start_year` |
| H6 | Mass expert skills, zero endorsements | 5+ expert-proficiency skills each with 0 endorsements |
| H7 | Salary min > max | `expected_salary_range_inr_lpa.min > max` |
| H8 | Signup date after last active date | `signup_date > last_active_date` |
| H9 | Positive offer acceptance with zero interview completion | `offer_acceptance_rate > 0` AND `interview_completion_rate == 0` |
| H10 | Extreme skill count with zero durations | 8+ skills all with `duration_months == 0` |
| H11 | Title-description mismatch | Current title is "Mechanical Engineer" but career description talks about "marketing campaigns" |
| H12 | Impossible timeline | Career start date implies candidate was < 16 years old |

### Phase 2: Two-Stage Hybrid Retrieval (`src/retrieval.py`)
To ensure we comply with the "≤ 5 minutes wall-clock, CPU only" constraint across 100K candidates, we introduced a **two-stage retrieval funnel**.
- **Pre-filter via BM25:** Evaluates lexical exact-matches (e.g., "FAISS", "Pinecone") across the 40k+ gate survivors, immediately funneling down to the Top 1,000 candidates to satisfy strict CPU time limits.
- **Semantic SBERT Embeddings:** Computes dense vector embeddings (`all-MiniLM-L6-v2`) **only** on the Top 1,000. Catches semantic paraphrasing (e.g., "built a recommendation system" ↔ "embeddings-based retrieval") without bogging down execution.

### Phase 3: Structured Features & Career Graph (`src/structured_features.py`)
Each feature maps directly to a specific JD requirement. All features are normalized to `[0, 1]`.

| Feature | JD Citation | Implementation |
|---------|-------------|----------------|
| `experience_band_fit` | "5-9 years band, soft not hard" | Gaussian centered on 7, σ=3. Within band → 0.8-1.0; outside → decay |
| `title_trajectory_score` | "Senior AI Engineer, Founding Team" | Score title chain: AI/ML title → 1.0; adjacent tech title → 0.6; non-tech title → 0.1 |
| `product_company_ratio` | "Entire career at pure IT-services... without product-company experience" | `product_company_career_months / total_career_months` |
| `skill_depth_score` | "proficiency × duration_months, not raw presence" | For each JD-relevant skill: `proficiency_weight * log(1 + duration_months) * min(1, endorsements/10)` |
| `evaluation_framework_evidence` | "experience designing evaluation frameworks for ranking systems" | Semantic search over career_history for: "NDCG", "MRR", "MAP", "A/B test" |
| `production_deployment_evidence` | "Production experience with embeddings-based retrieval systems deployed" | Count of production-signal keywords in career descriptions: "shipped", "real users" |

**Career Evidence Graph (Novel Differentiator)**:
Instead of scoring features independently, we build an evidence graph per candidate. Nodes represent evidence types (e.g. `has_ai_title`, `has_production_narrative`). Edges represent co-occurrence boosts. A candidate with an AI title + production narrative + product company gets multiplicative boosts because they corroborate each other.

### Phase 4: Behavioral Multipliers (`src/behavioral.py`)
Response rates and platform recency act as a **multiplicative penalty** (down to 0.5x), not an additive score. A perfect resume belonging to an inactive candidate gets penalized heavily. Zero-response-rate candidates cannot be rescued by keywords.

### Phase 5: Scoring & Output (`src/score.py` & `rank.py`)
- Features are fused using transparent, JD-cited weights. 
- Tie-breaks are safely and perfectly resolved alphabetically by `candidate_id` prior to CSV precision truncation.
- Explanations are generated using 5 distinct templates to avoid LLM hallucination:

| Rank Band | Tone Pattern | Example Generation |
|-----------|---------|---------|
| 1-10 | Strong match, confident tone | "Senior Recommendation Systems Engineer at Swiggy with 6.0 years building production ML; expert in Embeddings (60mo) and Sentence Transformers (69mo)..." |
| 11-30 | Good fit, mild caveat | "Data Engineer at Ola with 4.6 years and strong retrieval skill depth... Notice period 120 days exceeds preferred sub-30d — otherwise solid match." |
| 31-70 | Measured, balanced | "Backend Engineer with 6.9 years and NLP skills (advanced, 26mo)... Keyword-skill alignment moderate; plausible but not proven fit." |

---

## ⏱️ Compute Profile & Benchmarks

The entire system was architected to survive brutal constraint testing on standard CPU hardware.

| Phase | Duration (CPU) | Description |
|-------|----------------|-------------|
| **Streaming + Gates** | ~3.3 minutes | Streams 100K JSONs, applies regex/heuristics |
| **BM25 Lexical Pre-Filter** | ~5 seconds | Indexes 40k+ survivors, cuts to top 1,000 |
| **Semantic Embeddings** | ~20 seconds | Computes `all-MiniLM-L6-v2` locally on top 1k |
| **Scoring & CSV Generation** | ~2 seconds | Ranks, breaks ties, generates explanations |
| **TOTAL RUNTIME** | **~3.8 minutes** | **Safely < 5.0 minute wall-clock limit** |

*Peak Memory utilization: ~1.4GB RAM (Safely < 16GB limit).*

---

## 🧪 Testing

The repository ships with an extensive, custom PyTest suite specifically targeting the synthetic honeypot logic and ranking format invariant requirements. 

```bash
# Run unit test suite
python -m pytest tests/ -v
```

Tests cover:
- Detection of reversed dates and impossible timelines.
- Mass expert skills with zero endorsements.
- Salary maximums lower than minimums.
- Future signup dates vs last active dates.
- Gated pure-services IT profiles without product exposure.
- Tie-breaking verification for identical scores.

---

## 📁 Repository Structure

```text
.
├── rank.py                  # CLI entry point orchestrator
├── setup_models.py          # Script for pre-downloading offline HuggingFace models
├── validate_submission.py   # Hackathon script for output validation
├── requirements.txt         # Pinned python dependencies
├── Dockerfile               # Reproducible evaluation environment
├── tests/
│   ├── test_honeypots.py      # Synthetic fixtures for the 12 honeypots & 7 gates
│   └── test_output_format.py  # Validation of CSV formatting invariants
└── src/
    ├── load.py                # High-speed orjson streamer
    ├── gates.py               # Disqualifier heuristics
    ├── retrieval.py           # BM25 + SBERT Two-stage retrieval module
    ├── structured_features.py # Career Evidence Graph extraction
    ├── behavioral.py          # Multiplicative scaling
    ├── score.py               # Weighted scoring and final tie-breaking rules
    └── jd_config.py           # Decoupled weights and constants mapped directly to JD
```

---

*Authored for the Redrob "India Runs" Data & AI Challenge. Reviewed and built to production-grade engineering standards.*
