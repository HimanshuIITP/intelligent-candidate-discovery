# Idea Submission Presentation Content
Copy and paste the text below into your Redrob Idea Submission Template PPT.

---

### Slide 1: Title Slide
**Team Name:** [Your Team Name]
**Team Leader Name:** [Your Name]
**Problem Statement:** Redrob India Runs - Data & AI Challenge (Track 1)
Our goal is to build a scalable, highly optimized CPU-only ranking system capable of parsing 100,000 JSON candidate profiles to identify the top 100 matches for a Senior AI Engineer position. The challenge demands executing complex retrieval and scoring within a strict 5-minute wall-clock limit, completely offline, and without relying on any external APIs or GPUs.

---

### Slide 2: Solution Overview
**What is your proposed solution?**
We engineered a robust, completely offline ranking pipeline that utilizes a "Two-Stage Hybrid Search Funnel" and a mathematical "Career Evidence Graph". By separating fast keyword filtering from heavy semantic processing, the system processes 120MB+ of data in under 4 minutes. It ranks candidates based on verifiable production experience while actively filtering out fabricated AI-generated profiles and penalizing inactive users.

**What differentiates your approach from traditional candidate matching systems?**
1. **Mathematical Anti-Keyword Stuffing:** Traditional systems just count keywords. We evaluate actual skill depth using a decay formula: `Proficiency Weight × Log(Duration) × Endorsements`. A candidate who lists 20 expert skills with zero months of experience scores a zero.
2. **Career Evidence Linking:** We don't score signals in isolation. If a candidate has an "AI title", plus a "production-deployment narrative", plus "product company experience", they receive a multiplicative boost because these signals strongly corroborate each other.
3. **Multiplicative Behavioral Penalties:** A perfect resume is useless if the candidate ghosted recruiters. We use platform activity signals (response rates, recency) as a multiplier. Poor activity literally cuts their score in half.
4. **Honeypot Traps:** We deployed 12 specific detection rules to instantly penalize fabricated resumes (e.g., timeline impossibilities, massive unendorsed skill lists, reversed dates).

---

### Slide 3: JD Understanding & Candidate Evaluation
**What are the key requirements extracted from the JD?**
- **Core Technical Must-haves:** Production experience with embeddings-based retrieval, hands-on work with vector databases (FAISS, Pinecone, Qdrant), and deep knowledge of evaluation frameworks (NDCG, MRR, MAP).
- **Red Flags & Disqualifiers:** Pure IT-services backgrounds lacking product exposure, AI leadership titles with no recent coding evidence, and purely academic/research backgrounds with zero production deployment.
- **Logistical Preferences:** India-based (specifically Pune/Noida preferred), sub-30-day notice periods, and a soft 5-9 years experience band.

**How does your solution evaluate candidate fit beyond keyword matching?**
We use local Semantic Embeddings (`all-MiniLM-L6-v2`) to capture intent and paraphrasing. For example, if a candidate's career history states "migrated our search to dense vector retrieval", the semantic engine recognizes this as highly relevant to "recommendation systems" and "embeddings" without needing the exact keywords to be present. 

---

### Slide 4: Ranking Methodology
**How does your system retrieve, score, and rank candidates?**
1. **Gate & Stream:** We parse the dataset instantly, dropping candidates who fail hard requirements (like pure IT services or completely non-technical roles without any AI evidence).
2. **Retrieve (Stage 1):** We use fast lexical text search (BM25) to narrow the 40,000+ gate survivors down to a focused pool of the top 1,000 candidates.
3. **Score (Stage 2):** We run heavy Semantic Embeddings strictly on those 1,000, and fuse that semantic score with our extracted structured features (like the Career Evidence Graph).
4. **Rank:** We sort the top 100 based on the final weighted sum, intentionally breaking any identical score ties alphabetically by Candidate ID to guarantee deterministic, flawless output.

**What models, algorithms, or heuristics are used?**
- `orjson`: For ultra-fast, memory-efficient file streaming.
- `BM25`: A fast lexical ranking function used for initial keyword pre-filtering.
- `SentenceTransformers (all-MiniLM-L6-v2)`: A lightweight language model used for dense semantic vector similarity scoring.
- `Bounded Multipliers`: Mathematical scalars used to heavily penalize poor behavioral signals.

---

### Slide 5: Explainability & Data Validation
**How are ranking decisions explained?**
Our system dynamically generates a 1-2 sentence explanation for every single ranked candidate in the final output. The tone of the explanation adjusts based on their rank band (e.g., Confident for Top 10, Measured/Hedged for Bottom 30).

**How do you prevent hallucinations or unsupported justifications?**
We intentionally avoid using LLMs for reasoning generation to ensure 100% factual accuracy. Explanations are built using strict, logic-driven string templates that only inject verifiable data points extracted directly from the candidate's JSON profile (e.g., specific skill duration, company name, response rate).

**How does your solution handle inconsistent, low-quality, or suspicious profiles?**
We built a strict 12-rule Honeypot Detection engine that roots out AI-generated resumes. It detects anomalies like:
- Massive expert skill lists with 0 months duration or zero peer endorsements.
- Education dates ending before they even start.
- Total career durations wildly exceeding the stated years of experience.
- Salary minimums that are higher than salary maximums.
Suspicious profiles are hit with severe percentage penalties or gated out entirely.

---

### Slide 6: End-to-End Workflow
**What is the complete workflow from JD input to ranked candidate output?**
1. **Streaming Data Ingestion (`src/load.py`)**: Streams the massive 100K `candidates.jsonl` into memory one profile at a time to keep RAM usage low.
2. **Disqualification Phase (`src/gates.py`)**: Evaluates 7 hard disqualifiers and 12 honeypot checks per profile in real-time.
3. **Two-Stage Retrieval (`src/retrieval.py`)**: Indexes the survivors into BM25, filters to the top 1,000, then computes Semantic Embeddings locally.
4. **Feature Extraction (`src/structured_features.py`)**: Extracts and normalizes features like notice period, experience band, and skill depth.
5. **Activity Penalties (`src/behavioral.py`)**: Applies the [0.5x - 1.15x] multiplier based on responsiveness and platform recency.
6. **Final Ranking (`rank.py`)**: Sorts the scores, generates the factual explanation strings, and writes to `submission.csv`.

---

### Slide 7: System Architecture
*(You can just list this or draw a quick diagram in the PPT based on this)*
- **Data Layer**: Streaming JSONL Parser (`orjson`).
- **Filtering Layer**: Hard Gates & Honeypot Anomaly Detector.
- **Retrieval Funnel (Two-Stage Funnel)**: 
  - Stage 1: Lexical BM25 (Fast processing, 40k -> 1,000)
  - Stage 2: Dense SBERT (Heavy Semantic, 1,000 -> 100)
- **Feature Layer**: Career Evidence Combinator & Behavioral Multiplier.
- **Output Layer**: Templated Explainer & CSV Writer.

---

### Slide 8: Results & Performance
**What results or insights demonstrate ranking quality?**
Our pipeline passes 100% of our custom synthetic unit test suite. These tests actively verify that our system correctly isolates and drops fabricated resumes (like those with impossible timelines or salary inversions), while perfectly surfacing candidates with authentic, deployed recommendation system experience.

**How does your solution meet the challenge’s runtime and compute constraints?**
- **Total Runtime:** Processes 100,000 candidates in ~3.8 minutes (safely under the 5-minute wall-clock CPU limit).
- **Peak Memory:** Only requires ~1.4 GB RAM (safely under the 16 GB memory limit).
- **Optimization Strategy:** By aggressively funneling the candidates via BM25 first, we completely bypass the need to run computationally expensive Transformer embeddings on all 40,000 gate survivors, completely eliminating the CPU bottleneck.

---

### Slide 9: Technologies Used
**What technologies, frameworks, and tools were used and why were they selected?**
- **Python 3.10+**: Core language for robust data processing.
- **Sentence-Transformers**: Selected to generate dense semantic vectors entirely offline without requiring any GPUs or remote APIs.
- **Rank-BM25**: Chosen for its ability to perform ultra-fast lexical pre-filtering on tens of thousands of documents in mere seconds.
- **Orjson**: Implemented for its blazing fast C-backend to stream the massive 120MB JSON file instantly without blowing up the memory budget.
- **PyTest**: Used to build a reliable test suite for validating our honeypots and formatting invariants.

---

### Slide 10: Submission Assets
- **Github Repo**: [Insert Link]
- **Video Walkthrough**: [Insert Link]
- **CSV Output**: `submission.csv` included in repo.
