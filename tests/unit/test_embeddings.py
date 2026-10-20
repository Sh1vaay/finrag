"""Unit tests for embedding service with BGE-M3 local embedding provider."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.config.settings import settings
except ImportError:
    class _MockSettings:
        EMBEDDING_PROVIDER = "bge-m3"
        EMBEDDING_MODEL = "BAAI/bge-m3"
        EMBEDDING_DIMENSIONS = 1024
    settings = _MockSettings()

import src.services.embeddings as emb_mod
from src.services.embeddings import embed_text, embed_texts, _bgem3_embed


class TestEmbeddingsService(unittest.TestCase):
    def test_bgem3_vector_dimension(self):
        """BGE-M3 outputs 1024-dimensional dense float vectors."""
        texts = ["Apple Q4 revenue was $89.5 billion.", "Microsoft cloud segment growth."]
        vectors = _bgem3_embed(texts, input_type="document")
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 1024)
        self.assertEqual(len(vectors[1]), 1024)
        self.assertIsInstance(vectors[0][0], float)

    def test_bgem3_dispatch_in_embed_text(self):
        orig = getattr(emb_mod.settings, "EMBEDDING_PROVIDER", "openai") if emb_mod.settings else None
        try:
            if emb_mod.settings is None:
                emb_mod.settings = settings
            emb_mod.settings.EMBEDDING_PROVIDER = "bge-m3"
            vec = embed_text("What was Apple's operating margin in 2023?", input_type="query")
            self.assertEqual(len(vec), 1024)
            self.assertIsInstance(vec[0], float)
        finally:
            if emb_mod.settings and orig:
                emb_mod.settings.EMBEDDING_PROVIDER = orig

    def test_bgem3_dispatch_in_embed_texts(self):
        orig = getattr(emb_mod.settings, "EMBEDDING_PROVIDER", "openai") if emb_mod.settings else None
        try:
            if emb_mod.settings is None:
                emb_mod.settings = settings
            emb_mod.settings.EMBEDDING_PROVIDER = "bge-m3"
            vecs = embed_texts(["Passage 1", "Passage 2"], input_type="document")
            self.assertEqual(len(vecs), 2)
            self.assertEqual(len(vecs[0]), 1024)
        finally:
            if emb_mod.settings and orig:
                emb_mod.settings.EMBEDDING_PROVIDER = orig


if __name__ == "__main__":
    unittest.main()
