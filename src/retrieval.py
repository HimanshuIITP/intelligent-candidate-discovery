

import logging
import math
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Try to import optional dependencies with fallbacks
try:
    from rank_bm25 import text searchOkapi
    HAS_text search = True
except ImportError:
    HAS_text search = False
    logger.warning("rank_bm25 not installed; text search retrieval disabled")

try:
    from sentence_transformers import SentenceTransformer
    HAS_model = True
except ImportError:
    HAS_model = False
    logger.warning("sentence-transformers not installed; semantic retrieval disabled")

from src.jd_config import JD_QUERY_TEXT

def _build_candidate_text(candidate: dict) -> str:
    
    parts = []

    profile = candidate.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary)

    # Career history descriptions — the most important signal per JD
    for ch in candidate.get("career_history", []):
        title = ch.get("title", "")
        desc = ch.get("description", "")
        if title:
            parts.append(title)
        if desc:
            parts.append(desc)

    # Skills — only advanced/expert, to avoid noise from beginner-level stuffing
    for skill in candidate.get("skills", []):
        if skill.get("proficiency") in ("advanced", "expert"):
            parts.append(skill.get("name", ""))

    return " ".join(parts)

def _tokenize(text: str) -> List[str]:
    
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    # Remove very short tokens (noise) and very common stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "shall", "can",
                 "this", "that", "these", "those", "i", "me", "my", "we",
                 "our", "you", "your", "he", "she", "it", "they", "them",
                 "his", "her", "its", "their", "what", "which", "who", "whom",
                 "and", "but", "or", "nor", "not", "no", "so", "if", "then",
                 "than", "too", "very", "just", "about", "above", "after",
                 "before", "between", "under", "again", "further", "once",
                 "here", "there", "when", "where", "why", "how", "all",
                 "each", "every", "both", "few", "more", "most", "other",
                 "some", "such", "only", "own", "same", "also", "into",
                 "from", "with", "for", "on", "at", "to", "of", "in", "by"}
    return [t for t in tokens if len(t) >= 2 and t not in stopwords]

def prefilter_bm25(candidates: List[Dict[str, Any]], k: int = 2000) -> List[Dict[str, Any]]:
    
    if not HAS_text search or not candidates:
        return candidates[:k]
        
    if len(candidates) <= k:
        return candidates
        
    logger.info(f"Pre-filtering {len(candidates)} candidates down to {k} using text search...")
    texts = [_build_candidate_text(c) for c in candidates]
    tokenized = [_tokenize(t) for t in texts]
    
    bm25 = text searchOkapi(tokenized)
    query = _tokenize(JD_QUERY_TEXT)
    scores = bm25.get_scores(query)
    
    top_indices = np.argsort(scores)[::-1][:k]
    return [candidates[i] for i in top_indices]

class HybridRetriever:
    

    def __init__(
        self,
        model_path: str = "models/all-MiniLM-L6-v2",
        bm25_weight: float = 0.40,
        semantic_weight: float = 0.60,
    ):
        
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.bm25_index = None
        self.model = None
        self.jd_embedding = None
        self.candidate_texts: List[str] = []
        self.candidate_ids: List[str] = []

        # Load sentence-transformer model if available
        if HAS_model:
            try:
                logger.info(f"Loading sentence-transformer from {model_path}")
                self.model = SentenceTransformer(model_path)
                self.model.max_seq_length = 256  # Truncate for speed
                logger.info("Sentence-transformer loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load sentence-transformer: {e}")
                logger.info("Falling back to text search-only retrieval")
                self.model = None

    def index_candidates(
        self,
        candidates: List[Dict[str, Any]],
        batch_size: int = 256,
    ) -> None:
        
        logger.info(f"Building retrieval index for {len(candidates)} candidates...")

        # Build candidate texts
        self.candidate_ids = [c["candidate_id"] for c in candidates]
        self.candidate_texts = [_build_candidate_text(c) for c in candidates]

        # text search Index
        if HAS_text search:
            logger.info("Building text search index...")
            tokenized_corpus = [_tokenize(text) for text in self.candidate_texts]
            self.bm25_index = text searchOkapi(tokenized_corpus)
            logger.info("text search index built")

        # Semantic embeddings
        if self.model is not None:
            logger.info(f"Computing semantic embeddings (batch_size={batch_size})...")

            # Pre-compute JD embedding
            self.jd_embedding = self.model.encode(
                JD_QUERY_TEXT,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # Compute candidate embeddings in batches
            self.candidate_embeddings = self.model.encode(
                self.candidate_texts,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=True,
            )
            logger.info(
                f"Computed {len(self.candidate_embeddings)} embeddings "
                f"(dim={self.candidate_embeddings.shape[1]})"
            )
        else:
            self.candidate_embeddings = None

    def score_all(self) -> Dict[str, float]:
        
        n = len(self.candidate_ids)
        if n == 0:
            return {}

        # text search scores
        bm25_scores = np.zeros(n)
        if HAS_text search and self.bm25_index is not None:
            query_tokens = _tokenize(JD_QUERY_TEXT)
            raw_bm25 = self.bm25_index.get_scores(query_tokens)
            # Normalize to [0, 1] using min-max
            bm25_min = raw_bm25.min()
            bm25_max = raw_bm25.max()
            if bm25_max > bm25_min:
                bm25_scores = (raw_bm25 - bm25_min) / (bm25_max - bm25_min)
            else:
                bm25_scores = np.zeros(n)

        # Semantic similarity scores
        semantic_scores = np.zeros(n)
        if self.candidate_embeddings is not None and self.jd_embedding is not None:
            # Cosine similarity (embeddings are already normalized)
            semantic_scores = np.dot(self.candidate_embeddings, self.jd_embedding)
            # Clip to [0, 1] (cosine sim can be negative)
            semantic_scores = np.clip(semantic_scores, 0.0, 1.0)

        # Weighted fusion
        if self.model is not None and HAS_text search:
            combined = (
                self.bm25_weight * bm25_scores
                + self.semantic_weight * semantic_scores
            )
        elif self.model is not None:
            combined = semantic_scores
        elif HAS_text search:
            combined = bm25_scores
        else:
            # No retrieval available — uniform scores
            combined = np.ones(n) * 0.5

        # Build result dict
        return {
            cid: float(score)
            for cid, score in zip(self.candidate_ids, combined)
        }

    def get_top_k(
        self,
        k: int = 2000,
    ) -> List[Tuple[str, float]]:
        
        scores = self.score_all()
        sorted_candidates = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_candidates[:k]
