from dataclasses import replace

import polars as pl
import pytest

from corppa.poetry_detection.core import Excerpt, LabeledExcerpt
from corppa.poetry_detection.merge_excerpts import combine_excerpts, merge_duplicate_ids

excerpt1 = Excerpt(
    page_id="p.1",
    ppa_span_start=10,
    ppa_span_end=20,
    ppa_span_text="some text",
    detection_methods={"manual"},
)
excerpt2 = Excerpt(
    page_id="p.23",
    ppa_span_start=5,
    ppa_span_end=22,
    ppa_span_text="other text",
    detection_methods={"xml"},
)

excerpt1_label1 = LabeledExcerpt.from_excerpt(
    excerpt1,
    poem_id="poem-01",
    ref_corpus="test",
    ref_span_start=22,
    ref_span_end=35,
    ref_span_text="similar text",
    notes="extra info",
    identification_methods={"manual"},
)

excerpt1_label2 = LabeledExcerpt.from_excerpt(
    excerpt1,
    poem_id="poem-02",
    ref_corpus="test",
    ref_span_start=22,
    ref_span_end=35,
    ref_span_text="similar text",
    notes="id info",
    identification_methods={"refmatcha"},
)

excerpt2_label1 = LabeledExcerpt.from_excerpt(
    excerpt2,
    poem_id="poem-32",
    ref_corpus="test",
    ref_span_start=32,
    ref_span_end=54,
    ref_span_text="more text",
    identification_methods={"test"},
)


def test_combine_excerpts_1ex_1label():
    # excerpt + labeled excerpt (same id)
    df = pl.from_dicts([excerpt1.to_dict()])
    other_df = pl.from_dicts([excerpt1_label1.to_dict()])
    merged = combine_excerpts(df, other_df)
    # expect one row
    assert len(merged) == 1
    # should have all columns for labeled excerpt (order-agnostic)
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    row = merged.row(0, named=True)
    merged_excerpt = LabeledExcerpt.from_dict(row)
    # result should exactly match the labeled excerpt since all other fields are same
    assert merged_excerpt == excerpt1_label1


def test_combine_excerpts_1ex_2labels():
    # excerpt + two labeled excerpt (same excerpt id, two different ref ids)
    df = pl.from_dicts([excerpt1.to_dict()])
    other_df = pl.from_dicts([excerpt1_label1.to_dict(), excerpt1_label2.to_dict()])
    merged = combine_excerpts(df, other_df)
    # expect two rows with  two different labels
    assert len(merged) == 2
    # polars preserves order, so we can check that they match what was sent in
    # results should exactly match the labeled excerpts since all other fields are same
    assert LabeledExcerpt.from_dict(merged.row(0, named=True)) == excerpt1_label1
    assert LabeledExcerpt.from_dict(merged.row(1, named=True)) == excerpt1_label2


def test_combine_excerpts_1ex_note_1label():
    # excerpt with note + labeled excerpt (same id)
    ex1_notes = replace(excerpt1, notes="detection information")
    df = pl.from_dicts([ex1_notes.to_dict()])
    other_df = pl.from_dicts([excerpt1_label1.to_dict()])
    merged = combine_excerpts(df, other_df)
    # expect one row
    assert len(merged) == 1
    # should have all columns for labeled excerpt - no extra notes field
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    merged_excerpt = LabeledExcerpt.from_dict(merged.row(0, named=True))
    # result should match the labeled excerpt except for the updated notes field
    assert merged_excerpt != excerpt1_label1
    expected_notes = "\n".join([ex1_notes.notes, excerpt1_label1.notes])
    excerpt_with_notes = replace(excerpt1_label1, notes=expected_notes)
    assert merged_excerpt == excerpt_with_notes


def test_combine_excerpts_1ex_different_label():
    # excerpt 2 + labeled excerpt 1 - should preserve both
    df = pl.from_dicts([excerpt2.to_dict()])
    other_df = pl.from_dicts([excerpt1_label1.to_dict()])
    merged = combine_excerpts(df, other_df)
    # expect two rows
    assert len(merged) == 2
    # should have all columns for labeled excerpt (order-agnostic)
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    # the first row is our unlabeled excerpt
    row = merged.row(0, named=True)
    # filter out null values (unset labeled excerpt fields) and init as Excerpt
    row_subset = {k: v for k, v in row.items() if v is not None}
    merged_excerpt2 = Excerpt.from_dict(row_subset)
    assert merged_excerpt2 == excerpt2
    # second row should be our labeled excerpt
    merged_excerpt1_label1 = LabeledExcerpt.from_dict(merged.row(1, named=True))
    assert merged_excerpt1_label1 == excerpt1_label1


def test_combine_excerpts_two_different_labels():
    assert excerpt1_label1.excerpt_id != excerpt2_label1.excerpt_id
    df = pl.from_dicts([excerpt1_label1.to_dict()])
    other_df = pl.from_dicts([excerpt2_label1.to_dict()])
    merged = combine_excerpts(df, other_df)
    # expect two rows
    assert len(merged) == 2
    # should have all columns for labeled excerpt (order-agnostic)
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    # order should match input labeled excerpts
    assert LabeledExcerpt.from_dict(merged.row(0, named=True)) == excerpt1_label1
    assert LabeledExcerpt.from_dict(merged.row(1, named=True)) == excerpt2_label1


def test_combine_excerpts_1ex_2labels_diffmethod():
    # excerpt + two matching labeled excerpts
    # - same excerpt id, two labels with same ref ids but different method
    # combine method does not merge these
    df = pl.from_dicts([excerpt1.to_dict()])
    # everything the same except for the method (unlikely!)
    excerpt1_label1_method2 = replace(
        excerpt1_label1, identification_methods={"refmatcha"}
    )
    other_df = pl.from_dicts(
        [excerpt1_label1.to_dict(), excerpt1_label1_method2.to_dict()]
    )
    merged = combine_excerpts(df, other_df)
    assert len(merged) == 2


def test_merge_duplicate_ids():
    # excerpt + two matching labeled excerpts
    # - same excerpt id, two labels with same ref ids but different method

    # everything the same except for the method (unlikely!)
    excerpt1_label1_method2 = replace(
        excerpt1_label1, identification_methods={"refmatcha"}
    )
    df = pl.from_dicts([excerpt1_label1.to_dict(), excerpt1_label1_method2.to_dict()])
    merged = merge_duplicate_ids(df)
    assert len(merged) == 1
    # should have all columns for labeled excerpt (order-agnostic)
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    excerpt = LabeledExcerpt.from_dict(merged.row(0, named=True))
    # should have both methods
    assert excerpt.identification_methods == {"manual", "refmatcha"}

    # more likely scenario: manual label with no ref span, system label with more details
    excerpt1_label1_other = replace(
        excerpt1_label1,
        ref_span_start=None,
        ref_span_end=None,
        ref_span_text=None,
        identification_methods={"other"},
    )
    df = pl.from_dicts([excerpt1_label1.to_dict(), excerpt1_label1_other.to_dict()])
    merged = merge_duplicate_ids(df)
    assert len(merged) == 1
    # should have all columns for labeled excerpt (order-agnostic)
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    # should have both methods
    assert merged.row(0, named=True)["identification_methods"] == ["manual", "other"]
    excerpt = LabeledExcerpt.from_dict(merged.row(0, named=True))
    assert excerpt.identification_methods == {"manual", "other"}

    # order should not matter
    df = pl.from_dicts([excerpt1_label1_other.to_dict(), excerpt1_label1.to_dict()])
    merged = merge_duplicate_ids(df)
    assert len(merged) == 1
    excerpt = LabeledExcerpt.from_dict(merged.row(0, named=True))
    assert excerpt.identification_methods == {"manual", "other"}
    # should have the non-null ref values
    assert excerpt.ref_span_start == excerpt1_label1.ref_span_start
    assert excerpt.ref_span_end == excerpt1_label1.ref_span_end
    assert excerpt.ref_span_text == excerpt1_label1.ref_span_text
