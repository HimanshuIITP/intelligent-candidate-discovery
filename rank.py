#!/usr/bin/env python3

import argparse
import csv
import logging
import sys
import time
import tracemalloc
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rank")

# Import pipeline modules
from src.load import stream_jsonl
from src.gates import apply_gates
from src.retrieval import HybridRetriever, prefilter_bm25
from src.structured_features import extract_features
from src.activity import compute_activity_score
from src.score import compute_final_score, rank_candidates
from src.reasoning import generate_reasoning

def run_pipeline(
    candidates_path: str,
    output_path: str,
    top_k: int = 100,
    retrieval_cutoff: int = 1000,
    model_path: str = "models/all-MiniLM-L6-v2",
) -> None:
    
    # Start memory tracking
    tracemalloc.start()
    pipeline_start = time.time()
    timings = {}

    # Phase 1: Stream and Gate
    logger.info("Phase 1: Streaming candidates through gates")
    phase_start = time.time()

    gate_survivors = []
    total_count = 0
    gated_count = 0
    fake_check_count = 0
    gate_reasons_summary: Dict[str, int] = {}

    # Store fake_check scores for survivors too (for scoring penalty)
    survivor_hp_scores: Dict[str, float] = {}

    for candidate in stream_jsonl(candidates_path):
        total_count += 1
        passed, hp_score, reasons = apply_gates(candidate)

        if passed:
            gate_survivors.append(candidate)
            survivor_hp_scores[candidate["candidate_id"]] = hp_score
        else:
            gated_count += 1
            if hp_score >= 0.5:
                fake_check_count += 1
            for r in reasons:
                gate_key = r.split(":")[0]
                gate_reasons_summary[gate_key] = gate_reasons_summary.get(gate_key, 0) + 1

    timings["gate"] = time.time() - phase_start
    logger.info(
        f"Gate results: {len(gate_survivors)}/{total_count} passed "
        f"({gated_count} gated, {fake_check_count} fake_checks)"
    )
    for gate, count in sorted(gate_reasons_summary.items()):
        logger.info(f"  {gate}: {count}")

    current, peak = tracemalloc.get_traced_memory()
    logger.info(f"Memory after gates: {current / 1e6:.1f}MB (peak: {peak / 1e6:.1f}MB)")

    # Phase 2: Hybrid Retrieval
    logger.info("Phase 2: Hybrid retrieval (text search + semantic embedding)")
    phase_start = time.time()

    retrieval_candidates = prefilter_bm25(gate_survivors, k=retrieval_cutoff)
    
    retriever = HybridRetriever(model_path=model_path)
    retriever.index_candidates(retrieval_candidates)
    retrieval_scores = retriever.score_all()

    # Get top candidates by retrieval score (already limited to cutoff)
    top_retrieval = retriever.get_top_k(k=retrieval_cutoff)
    top_cids = {cid for cid, _ in top_retrieval}

    # Build lookup for retrieval scores
    retrieval_score_map = dict(top_retrieval)

    timings["retrieval"] = time.time() - phase_start
    logger.info(f"Retrieval: kept top {len(top_cids)} of {len(gate_survivors)}")

    current, peak = tracemalloc.get_traced_memory()
    logger.info(f"Memory after retrieval: {current / 1e6:.1f}MB (peak: {peak / 1e6:.1f}MB)")

    # Phase 3: Detailed Scoring
    logger.info("Phase 3: Structured features + activity + final scoring")
    phase_start = time.time()

    # Build candidate lookup for the top retrieval candidates
    candidate_lookup = {c["candidate_id"]: c for c in gate_survivors if c["candidate_id"] in top_cids}

    scored_candidates = []
    for cid, ret_score in top_retrieval:
        candidate = candidate_lookup.get(cid)
        if candidate is None:
            continue

        hp_score = survivor_hp_scores.get(cid, 0.0)
        final_score, breakdown = compute_final_score(
            candidate,
            retrieval_score=ret_score,
            fake_check_score=hp_score,
        )

        scored_candidates.append((cid, final_score, candidate, breakdown))

    timings["scoring"] = time.time() - phase_start
    logger.info(f"Scored {len(scored_candidates)} candidates")

    # Phase 4: Ranking & Reasoning
    logger.info("Phase 4: Ranking and reasoning generation")
    phase_start = time.time()

    ranked = rank_candidates(scored_candidates, top_k=top_k)

    # Generate reasoning for each ranked candidate
    output_rows = []
    for cid, rank, score, candidate, breakdown in ranked:
        reasoning = generate_reasoning(rank, candidate, breakdown)
        output_rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 4),
            "reasoning": reasoning,
        })

    timings["ranking"] = time.time() - phase_start
    logger.info(f"Ranked top {len(output_rows)} candidates")

    # Phase 5: Output Generation
    logger.info("Phase 5: Writing output CSV")
    phase_start = time.time()

    output_file = Path(output_path)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    timings["output"] = time.time() - phase_start
    logger.info(f"Written to {output_file}")

    #  Summary 
    total_time = time.time() - pipeline_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logger.info("")
    logger.info("Pipeline complete")
    logger.info(f"{'Phase':<20} {'Time':>10}")
    logger.info("-" * 32)
    for phase, t in timings.items():
        logger.info(f"{phase:<20} {t:>9.1f}s")
    logger.info("-" * 32)
    logger.info(f"{'TOTAL':<20} {total_time:>9.1f}s")
    logger.info(f"Peak memory: {peak / 1e6:.1f} MB")
    logger.info(f"Output: {output_file} ({len(output_rows)} rows)")

    # Warn if over budget
    if total_time > 300:
        logger.warning(f"⚠ OVER TIME BUDGET: {total_time:.1f}s > 300s limit")
    if peak > 16e9:
        logger.warning(f"⚠ OVER MEMORY BUDGET: {peak / 1e9:.1f}GB > 16GB limit")

    # Print top-5 preview
    logger.info("")
    logger.info("Top-5 Preview:")
    for row in output_rows[:5]:
        logger.info(
            f"  #{row['rank']} {row['candidate_id']} "
            f"(score={row['score']}) — {row['reasoning'][:120]}..."
        )

def main():
    parser = argparse.ArgumentParser(
        description="Redrob Candidate Ranking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=,
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates JSONL/JSON file",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write output CSV",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of candidates to rank (default: 100)",
    )
    parser.add_argument(
        "--retrieval-cutoff",
        type=int,
        default=2000,
        help="Number of candidates to pass to detailed scoring (default: 2000)",
    )
    parser.add_argument(
        "--model-path",
        default="models/all-MiniLM-L6-v2",
        help="Path to sentence-transformer model (default: models/all-MiniLM-L6-v2)",
    )

    args = parser.parse_args()

    run_pipeline(
        candidates_path=args.candidates,
        output_path=args.out,
        top_k=args.top_k,
        retrieval_cutoff=args.retrieval_cutoff,
        model_path=args.model_path,
    )

if __name__ == "__main__":
    main()
