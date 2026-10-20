"""Unit tests for tracing & observability service (Arize Phoenix / OpenInference)."""

import os
import unittest
from unittest.mock import MagicMock, patch

from src.services.tracing_service import init_tracing, trace_span, get_tracer
import src.services.tracing_service as tracing_mod


class TestTracingService(unittest.TestCase):
    def setUp(self):
        tracing_mod._is_tracing_initialized = False
        tracing_mod._tracer = None

    def tearDown(self):
        tracing_mod._is_tracing_initialized = False
        tracing_mod._tracer = None

    def test_tracing_none_backend(self):
        with patch.dict(os.environ, {"TRACING_BACKEND": "none"}):
            init_tracing()
            self.assertTrue(tracing_mod._is_tracing_initialized)
            self.assertIsNone(get_tracer())

            with trace_span("test_span", {"key": "val"}) as span:
                self.assertIsNone(span)

    def test_tracing_langfuse_backend(self):
        with patch.dict(os.environ, {"TRACING_BACKEND": "langfuse"}):
            init_tracing()
            self.assertTrue(tracing_mod._is_tracing_initialized)

    def test_tracing_phoenix_backend_with_mocked_otel(self):
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        tracing_mod._tracer = mock_tracer
        tracing_mod._is_tracing_initialized = True

        with trace_span("retrieval.hybrid", {"company": "Apple", "top_k": 50}) as span:
            self.assertEqual(span, mock_span)

        mock_tracer.start_as_current_span.assert_called_once_with("retrieval.hybrid")
        mock_span.set_attribute.assert_any_call("company", "Apple")
        mock_span.set_attribute.assert_any_call("top_k", 50)


if __name__ == "__main__":
    unittest.main()
