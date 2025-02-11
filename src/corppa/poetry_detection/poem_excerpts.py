"""
Custom data type for poetry excerpts identified with the text of PPA pages.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(kw_only=True)
class PoemExcerpt:
    """
    A poem excerpt data type representing an identified poem excerpt within a
    PPA page
    """

    # PPA page related
    page_id: str
    ppa_span_start: int
    ppa_span_end: int
    ppa_span_text: str
    # Reference poem related
    poem_id: str
    ref_corpus: str
    ref_span_start: Optional[int] = None
    ref_span_end: Optional[int] = None
    ref_span_text: Optional[str] = None
    # Methods & additional notes
    detection_methods: set[str]
    identification_methods: set[str]
    notes: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        """
        Returns a JSON-style dict of the poem excerpt
        """
        json_dict = {
            "page_id": self.page_id,
            "poem_id": self.poem_id,
            "ppa_span_start": self.ppa_span_start,
            "ppa_span_end": self.ppa_span_end,
            "ppa_span_text": self.ppa_span_text,
            "poem_id": self.poem_id,
            "ref_corpus": self.ref_corpus,
            "ref_span_start": self.ref_span_start,
            "ref_span_end": self.ref_span_end,
            "ref_span_text": self.ref_span_text,
            "detection_methods": list(self.detection_methods),
            "identification_methods": list(self.identification_methods),
            "notes": self.notes,
        }
        return json_dict
