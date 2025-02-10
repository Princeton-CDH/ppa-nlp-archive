"""
Custom data type for poetry excerpts identified with the text of PPA pages.
"""

from dataclasses import dataclass


@dataclass
class PoemExcerpt:
    """
    A poem excerpt data type representing an identified poem excerpt within a
    PPA page
    """

    # PPA page related
    ppa_page_id: str
    ppa_span_start: int
    ppa_span_end: int
    ppa_span_text: str
    # Reference poem related
    poem_id: str
    ref_corpus: str
    ref_span_start: Optional[int]
    ref_span_end: Optional[int]
    ref_span_text: Optional[str]
    # Methods & additional notes
    detection_methods: set[str]
    identification_methods: set[str]
    notes: Optional[str]
