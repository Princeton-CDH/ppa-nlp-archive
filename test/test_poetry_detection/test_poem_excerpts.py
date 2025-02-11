import pytest

from corppa.poetry_detection.poem_excerpts import PoemExcerpt


class TestPoemExcerpt:
    def test_to_json(self):
        # No optional fields
        poem_excerpt = PoemExcerpt(
            page_id="page_id",
            ppa_span_start=0,
            ppa_span_end=1,
            ppa_span_text="page_text",
            poem_id="poem_id",
            ref_corpus="corpus_id",
            detection_methods={"detect"},
            identification_methods={"id"},
        )
        expected_result = {
            "page_id": "page_id",
            "ppa_span_start": 0,
            "ppa_span_end": 1,
            "ppa_span_text": "page_text",
            "poem_id": "poem_id",
            "ref_corpus": "corpus_id",
            "ref_span_start": None,
            "ref_span_end": None,
            "ref_span_text": None,
            "detection_methods": ["detect"],
            "identification_methods": ["id"],
            "notes": None,
        }

        result = poem_excerpt.to_json()
        assert result == expected_result

        # With optional fields
        poem_excerpt.ref_span_start = 2
        poem_excerpt.ref_span_end = 3
        poem_excerpt.ref_span_text = "ref_text"
        poem_excerpt.notes = "notes"

        expected_result |= {
            "ref_span_start": 2,
            "ref_span_end": 3,
            "ref_span_text": "ref_text",
            "notes": "notes",
        }

        result = poem_excerpt.to_json()
        assert result == expected_result
