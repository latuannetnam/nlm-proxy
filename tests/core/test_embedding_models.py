"""Test embedding model performance for Vietnamese and multilingual queries."""

import pytest

# Skip if sentence-transformers not installed
st = pytest.importorskip("sentence_transformers")

from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import time


# Use MiniLM (the target model for optimization)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@pytest.fixture(scope="module")
def model():
    """Load embedding model once for all tests."""
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


def cosine_sim(model: HuggingFaceEmbeddings, text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts."""
    embeddings = model.embed_documents([text_a, text_b])
    a, b = np.array(embeddings[0]), np.array(embeddings[1])
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class TestSameIntent:
    """Category 1: Same intent queries should be similar."""

    def test_hr_policy_vietnamese(self, model):
        sim = cosine_sim(model, "Chính sách nhân sự của công ty là gì?", "Cho tôi biết về chính sách nhân sự")
        assert sim > 0.5, f"Same intent Vietnamese: sim={sim:.4f}, expected >0.5"

    def test_department_staff_vietnamese(self, model):
        sim = cosine_sim(model, "Phòng TSD có những ai", "Danh sách nhân sự phòng TSD")
        assert sim > 0.5, f"Same intent Vietnamese: sim={sim:.4f}, expected >0.5"

    def test_greeting_with_context(self, model):
        sim = cosine_sim(model, "Xin chào", "Xin chào, tôi muốn tìm hiểu thông tin")
        assert sim > 0.3, f"Greeting + context: sim={sim:.4f}, expected >0.3"


class TestDifferentIntent:
    """Category 2: Different intent queries should be dissimilar."""

    def test_hr_vs_weather(self, model):
        sim = cosine_sim(model, "Chính sách nhân sự", "Thời tiết hôm nay")
        assert sim < 0.3, f"Different intent: sim={sim:.4f}, expected <0.3"

    def test_staff_vs_revenue(self, model):
        sim = cosine_sim(model, "Phòng TSD có những ai", "Doanh thu quý 3")
        assert sim < 0.3, f"Different intent: sim={sim:.4f}, expected <0.3"


class TestCrossLingual:
    """Category 3: Cross-lingual Vietnamese <-> English."""

    def test_greeting_cross_lingual(self, model):
        sim = cosine_sim(model, "Xin chào, tôi muốn biết thông tin", "Hello, I want to know information")
        assert sim > 0.7, f"Cross-lingual: sim={sim:.4f}, expected >0.7"


class TestRewriteVariants:
    """Category 4: Query rewrite with context enrichment."""

    def test_short_to_contextual(self, model):
        sim = cosine_sim(model, "Có những ai", "Phòng TSD có những ai")
        assert sim > 0.3, f"Rewrite variant: sim={sim:.4f}, expected >0.3"

    def test_salary_rewrite(self, model):
        sim = cosine_sim(model, "Lương bao nhiêu", "Mức lương trung bình của nhân viên NetNam")
        assert sim > 0.3, f"Rewrite variant: sim={sim:.4f}, expected >0.3"


class TestPerformance:
    """Embedding performance benchmarks.

    Note: Design target is <50ms single query, but CI environments are slower.
    We use relaxed thresholds here (200ms/100ms) to avoid flaky CI failures.
    """

    def test_single_query_latency(self, model):
        text = "Chính sách nhân sự của công ty là gì?"
        start = time.perf_counter()
        model.embed_query(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Design target: <50ms. CI tolerance: <200ms.
        assert elapsed_ms < 200, f"Single query latency: {elapsed_ms:.1f}ms, expected <200ms"

    def test_batch_latency(self, model):
        texts = [f"Query number {i} about company policy" for i in range(10)]
        start = time.perf_counter()
        model.embed_documents(texts)
        elapsed_ms = (time.perf_counter() - start) * 1000
        per_query = elapsed_ms / len(texts)
        assert per_query < 100, f"Batch per-query: {per_query:.1f}ms, expected <100ms"
