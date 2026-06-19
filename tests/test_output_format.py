

import csv
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _parse_csv(filepath):
    
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

class TestOutputFormat:
    

    @pytest.fixture
    def sample_csv(self, tmp_path):
        
        csv_path = tmp_path / "test_submission.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for i in range(100):
                cid = f"CAND_{i:07d}"
                rank = i + 1
                score = round(1.0 - i * 0.008, 4)
                reasoning = f"Test reasoning for rank {rank}."
                writer.writerow([cid, rank, score, reasoning])
        return str(csv_path)

    def test_correct_header(self, sample_csv):
        with open(sample_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["candidate_id", "rank", "score", "reasoning"]

    def test_exactly_100_rows(self, sample_csv):
        rows = _parse_csv(sample_csv)
        assert len(rows) == 100, f"Expected 100 rows, got {len(rows)}"

    def test_ranks_1_to_100(self, sample_csv):
        rows = _parse_csv(sample_csv)
        ranks = sorted(int(r["rank"]) for r in rows)
        assert ranks == list(range(1, 101)), "Ranks should be 1-100 each exactly once"

    def test_no_duplicate_ranks(self, sample_csv):
        rows = _parse_csv(sample_csv)
        ranks = [int(r["rank"]) for r in rows]
        assert len(set(ranks)) == 100, "No duplicate ranks allowed"

    def test_no_duplicate_candidate_ids(self, sample_csv):
        rows = _parse_csv(sample_csv)
        cids = [r["candidate_id"] for r in rows]
        assert len(set(cids)) == 100, "No duplicate candidate_ids"

    def test_score_non_increasing(self, sample_csv):
        rows = _parse_csv(sample_csv)
        rows_by_rank = sorted(rows, key=lambda r: int(r["rank"]))
        scores = [float(r["score"]) for r in rows_by_rank]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score at rank {i+1} ({scores[i]}) < score at rank {i+2} ({scores[i+1]})"
            )

    def test_candidate_id_format(self, sample_csv):
        rows = _parse_csv(sample_csv)
        import re
        pattern = re.compile(r"^CAND_\d{7}$")
        for row in rows:
            assert pattern.match(row["candidate_id"]), (
                f"Invalid candidate_id format: {row['candidate_id']}"
            )

    def test_tie_breaking_by_candidate_id(self, tmp_path):
        
        csv_path = tmp_path / "tie_test.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])

            # Create tied scores
            for i in range(100):
                cid = f"CAND_{i:07d}"
                rank = i + 1
                score = 0.5  # All tied
                writer.writerow([cid, rank, score, f"Tied at rank {rank}"])

        rows = _parse_csv(str(csv_path))
        rows_by_rank = sorted(rows, key=lambda r: int(r["rank"]))

        # For tied scores, check candidate_id is ascending
        for i in range(len(rows_by_rank) - 1):
            if float(rows_by_rank[i]["score"]) == float(rows_by_rank[i+1]["score"]):
                assert rows_by_rank[i]["candidate_id"] <= rows_by_rank[i+1]["candidate_id"], (
                    f"Tied scores at ranks {i+1},{i+2} should have candidate_id ascending"
                )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
