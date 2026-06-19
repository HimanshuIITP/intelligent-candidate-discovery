# JD Query Text — used for text search and semantic embedding retrieval
# Distilled from the "must-haves" and "ideal candidate" sections of the JD.

JD_QUERY_TEXT = """
Senior AI Engineer with production experience building embeddings-based retrieval
systems, recommendation systems, ranking and search systems deployed to real users.
Production experience with vector databases and hybrid search infrastructure
including Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
Strong Python, hands-on experience designing evaluation frameworks for ranking
systems including NDCG, MRR, MAP, offline-to-online correlation, A/B test
interpretation. Experience with sentence-transformers, OpenAI embeddings, BGE,
E5 or similar embedding models handling embedding drift, index refresh, and
retrieval quality regression in production. Learning-to-rank models using XGBoost
or neural approaches. LLM fine-tuning with LoRA, QLoRA, PEFT. NLP and information
retrieval background. Building ML-powered features in production, feature
engineering, model training, deployment pipeline. MLOps, experiment tracking,
model serving. Applied ML at product companies, not consulting. Shipped end-to-end
ranking, search, or recommendation system to real users at meaningful scale.
"""

EXPERIENCE_IDEAL_MIN = 5.0
EXPERIENCE_IDEAL_MAX = 9.0
EXPERIENCE_CENTER = 7.0  # Gaussian center
EXPERIENCE_SIGMA = 3.0   # Wide tolerance per "soft not hard"

PREFERRED_LOCATIONS_INDIA = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore",
    "bengaluru", "gurgaon", "gurugram", "chennai",
]

# Accenture, Cognizant, Capgemini, etc.) in their entire career"

SERVICES_COMPANIES = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcl technologies", "tech mahindra",
    "mindtree", "mphasis", "ltimindtree", "lti", "dxc", "dxc technology",
    "hexaware", "cyient", "zensar", "persistent systems", "l&t infotech",
    "birlasoft", "niit technologies", "coforge", "sonata software",
}

# JD Must-have skills — used for skill_depth_score
# Weighted by how central they are to the JD requirements

MUST_HAVE_SKILLS = {
    # Tier 1: Explicitly required in JD "Things you absolutely need"
    "embeddings": 1.0,
    "sentence transformers": 1.0,
    "sentence-transformers": 1.0,
    "information retrieval": 1.0,
    "search": 0.9,
    "ranking": 0.9,
    "recommendation systems": 0.9,
    "python": 0.8,
    "ndcg": 0.8,
    "evaluation": 0.8,

    # Tier 2: Vector DBs and hybrid search — JD "Things you absolutely need"
    "faiss": 0.9,
    "pinecone": 0.9,
    "weaviate": 0.9,
    "qdrant": 0.9,
    "milvus": 0.9,
    "opensearch": 0.9,
    "elasticsearch": 0.85,

    # Tier 3: ML/AI core — JD "ideal candidate" and "what you'd actually be doing"
    "nlp": 0.85,
    "natural language processing": 0.85,
    "machine learning": 0.8,
    "deep learning": 0.8,
    "pytorch": 0.75,
    "tensorflow": 0.75,
    "transformers": 0.8,
    "hugging face": 0.8,
    "hugging face transformers": 0.8,
    "bert": 0.75,
    "scikit-learn": 0.7,
    "xgboost": 0.7,
    "lightgbm": 0.7,

    # Tier 4: Nice-to-haves from JD
    "lora": 0.6,
    "qlora": 0.6,
    "peft": 0.6,
    "fine-tuning llms": 0.65,
    "llm": 0.6,
    "langchain": 0.4,  # JD explicitly de-weights "Framework enthusiasts"
    "rag": 0.65,
    "prompt engineering": 0.5,
    "mlops": 0.65,
    "mlflow": 0.6,
    "weights & biases": 0.55,
    "bentoml": 0.5,
    "kubeflow": 0.55,
    "feature engineering": 0.7,
    "bm25": 0.8,

    # Tier 5: Adjacent strong signals
    "data pipelines": 0.5,
    "airflow": 0.45,
    "spark": 0.45,
    "sql": 0.4,
    "docker": 0.35,
    "kubernetes": 0.35,
    "aws": 0.3,
    "gcp": 0.3,
    "azure": 0.3,
}

# Skills that indicate a CV/Speech/Robotics focus (JD disqualifier if dominant)

CV_SPEECH_ROBOTICS_SKILLS = {
    "computer vision", "image classification", "object detection",
    "image segmentation", "opencv", "yolo", "image processing",
    "speech recognition", "tts", "text-to-speech", "speech synthesis",
    "asr", "robotics", "ros", "slam", "autonomous driving",
    "3d reconstruction", "point cloud", "lidar",
}

# Production-evidence keywords mined from career_history descriptions

PRODUCTION_KEYWORDS = {
    "production", "deployed", "shipped", "launched", "live", "real users",
    "scale", "latency", "throughput", "sla", "uptime", "monitoring",
    "a/b test", "a/b testing", "ab test", "experiment", "canary",
    "rollout", "pipeline", "ci/cd", "mlops", "model serving",
    "inference", "batch processing", "real-time", "api",
}

# Research-only keywords (for gate G1 — pure research career detection)

RESEARCH_ONLY_KEYWORDS = {
    "published", "publication", "paper", "thesis", "dissertation",
    "postdoc", "post-doc", "research lab", "academic", "journal",
    "conference paper", "arxiv", "peer-reviewed", "phd research",
}

# Evaluation framework keywords

EVAL_FRAMEWORK_KEYWORDS = {
    "ndcg", "mrr", "map", "mean average precision", "mean reciprocal rank",
    "normalized discounted cumulative gain", "precision@k", "recall@k",
    "f1", "auc", "roc", "a/b test", "a/b testing", "offline-online",
    "evaluation framework", "evaluation pipeline", "evaluation metric",
    "benchmark", "ground truth", "relevance label", "annotation",
    "inter-annotator", "click-through", "click-through rate", "ctr",
    "conversion rate", "engagement metric", "dwell time",
}

# Non-technical titles — gate G7 (non-tech with zero AI evidence)

NON_TECHNICAL_TITLES = {
    "marketing manager", "accountant", "hr manager", "human resources",
    "sales executive", "sales manager", "operations manager",
    "graphic designer", "content writer", "civil engineer",
    "mechanical engineer", "customer support", "customer service",
    "brand manager", "project manager", "business analyst",
    "financial analyst", "supply chain", "procurement",
    "administrative", "executive assistant", "receptionist",
}

# Proficiency weights for skill_depth_score
# Used as: weight * log(1 + duration_months) * min(1, endorsements/10)

PROFICIENCY_WEIGHTS = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.4,
    "beginner": 0.15,
}

# but the bar gets higher"

NOTICE_PERIOD_SCORES = {
    (0, 30): 1.0,
    (31, 60): 0.85,
    (61, 90): 0.70,
    (91, 120): 0.50,
    (121, 180): 0.30,
}
