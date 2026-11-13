"""OCR metric contracts using real jiwer and independently known edit counts."""

import json
from copy import deepcopy
from dataclasses import replace
from typing import Literal

import pytest

from parking_monitoring.ocr_evaluation import evaluate_ocr
from parking_monitoring.schemas import OCRResult


def reading(
    text: str,
    *,
    status: Literal["read", "uncertain"] = "read",
    score: float | None = 0.9,
) -> OCRResult:
    """Build text-bearing results; preserve supplied text to test normalization."""
    return OCRResult(text, text, score, status)


def test_perfect_readings_return_exactly_the_agreed_metrics() -> None:
    metrics = evaluate_ocr(
        ["YGN AB12", "NY CD34"],
        [reading("ygn\tab12"), reading("ny cd34")],
    )
    # Seven and six normalized reference characters, two fully correct reads.
    assert metrics == {
        "sample_count": 2,
        "reference_characters": 13,
        "character_errors": 0,
        "exact_matches": 2,
        "exact_match_rate": 1.0,
        "character_error_rate": 0.0,
        "accepted_reads": 2,
        "correct_accepted_reads": 2,
        "accepted_read_accuracy": 1.0,
        "accepted_read_coverage": 1.0,
    }


@pytest.mark.parametrize(
    ("reference", "prediction", "edits", "characters"),
    [
        pytest.param("AB12", "AB12", 0, 4, id="exact"),
        pytest.param("AB12", "AB13", 1, 4, id="substitution"),
        pytest.param("AB12", "AB1", 1, 4, id="deletion"),
        pytest.param("AB12", "AB123", 1, 4, id="insertion"),
        pytest.param("AB12", "", 4, 4, id="empty-prediction"),
        pytest.param("AB12", "BA12", 2, 4, id="transposition-is-two-edits"),
        pytest.param("A", "AAAA", 3, 1, id="cer-can-exceed-one"),
    ],
)
def test_independently_known_character_edits(
    reference, prediction, edits, characters
) -> None:
    result = (
        reading(prediction) if prediction else OCRResult(None, None, None, "unreadable")
    )
    metrics = evaluate_ocr([reference], [result])
    assert metrics["character_errors"] == edits
    assert metrics["reference_characters"] == characters
    assert metrics["character_error_rate"] == pytest.approx(edits / characters)
    assert metrics["exact_matches"] == int(edits == 0)


def test_interview_example_uses_corpus_counts_and_retains_empty_plate() -> None:
    metrics = evaluate_ocr(
        ["YGN3L-5479", "NYAB12"],
        [reading("YGM3L-5479"), OCRResult(None, None, None, "unreadable")],
    )
    # A: one substitution / ten characters; B: six deletions / six characters.
    # Corpus CER is 7/16, not the unweighted average (0.1 + 1.0) / 2.
    assert metrics == {
        "sample_count": 2,
        "reference_characters": 16,
        "character_errors": 7,
        "exact_matches": 0,
        "exact_match_rate": 0.0,
        "character_error_rate": 0.4375,
        "accepted_reads": 1,
        "correct_accepted_reads": 0,
        "accepted_read_accuracy": 0.0,
        "accepted_read_coverage": 0.5,
    }


def test_alignment_cannot_transfer_characters_between_plates() -> None:
    metrics = evaluate_ocr(["AB", "CD"], [reading("ABC"), reading("D")])
    # Concatenating both lists would falsely look perfect: ABCD == ABCD.
    # Separately, the first plate inserts C and the second deletes C.
    assert metrics["character_errors"] == 2
    assert metrics["reference_characters"] == 4
    assert metrics["character_error_rate"] == 0.5
    assert metrics["exact_matches"] == 0


def test_mixed_statuses_use_distinct_accuracy_and_coverage_denominators() -> None:
    metrics = evaluate_ocr(
        ["YGNAB12", "NYCD34", "AB12", "XY99"],
        [
            reading("YGNAB13"),
            reading("NYCD34", status="uncertain", score=0.4),
            OCRResult(None, None, None, "unreadable"),
            OCRResult("XY99", "XY99", 0.8, "error", "failure after partial output"),
        ],
    )
    # One wrong accepted read, one correct uncertain read, two failed readings.
    assert metrics["sample_count"] == 4
    assert metrics["exact_matches"] == 1
    assert metrics["exact_match_rate"] == 0.25
    assert metrics["reference_characters"] == 21
    assert metrics["character_errors"] == 9  # 1 substitution + 4 + 4 deletions.
    assert metrics["character_error_rate"] == pytest.approx(9 / 21)
    assert metrics["accepted_reads"] == 1
    assert metrics["correct_accepted_reads"] == 0
    assert metrics["accepted_read_accuracy"] == 0.0
    assert metrics["accepted_read_coverage"] == 0.25


def test_acceptance_uses_supplied_status_without_guessing_a_threshold() -> None:
    # These statuses could come from different caller-chosen thresholds.
    # The evaluator has no confidence-threshold parameter and must trust them.
    metrics = evaluate_ocr(
        ["AB12", "CD34"],
        [reading("AB12", score=0.01), reading("CD34", status="uncertain", score=0.99)],
    )
    assert metrics["exact_matches"] == 2
    assert metrics["accepted_reads"] == 1
    assert metrics["correct_accepted_reads"] == 1
    assert metrics["accepted_read_accuracy"] == 1.0
    assert metrics["accepted_read_coverage"] == 0.5


def test_correct_uncertain_text_counts_toward_exact_match_without_acceptance() -> None:
    metrics = evaluate_ocr(
        ["AB12", "CD34"],
        [
            reading("AB12", status="uncertain", score=None),
            reading("CD34", status="uncertain"),
        ],
    )
    assert metrics["exact_match_rate"] == 1.0
    assert metrics["character_error_rate"] == 0.0
    assert metrics["accepted_reads"] == metrics["correct_accepted_reads"] == 0
    assert metrics["accepted_read_accuracy"] is None
    assert metrics["accepted_read_coverage"] == 0.0


@pytest.mark.parametrize("status", ["unreadable", "error"])
@pytest.mark.parametrize("partial_text", [None, "AB12"])
def test_failed_status_is_empty_even_when_partial_text_is_present(
    status, partial_text
) -> None:
    metrics = evaluate_ocr(
        ["AB12"],
        [OCRResult(partial_text, partial_text, 0.8, status, "recorded failure")],
    )
    assert metrics["sample_count"] == 1
    assert metrics["reference_characters"] == metrics["character_errors"] == 4
    assert metrics["character_error_rate"] == 1.0
    assert metrics["exact_matches"] == metrics["accepted_reads"] == 0
    assert metrics["accepted_read_accuracy"] is None


def test_all_failed_readings_keep_the_complete_reference_denominator() -> None:
    metrics = evaluate_ocr(
        ["AB12", "CD3456"],
        [
            OCRResult(None, None, None, "error"),
            OCRResult(None, None, None, "unreadable"),
        ],
    )
    assert metrics["sample_count"] == 2
    assert metrics["reference_characters"] == metrics["character_errors"] == 10
    assert metrics["character_error_rate"] == 1.0
    assert metrics["exact_match_rate"] == metrics["accepted_read_coverage"] == 0.0
    assert metrics["accepted_read_accuracy"] is None


def test_normalization_handles_case_and_unicode_whitespace() -> None:
    metrics = evaluate_ocr([" ygn\u00a0 3l-5479 "], [reading("YgN\t3L-\n5479")])
    assert metrics["reference_characters"] == 10
    assert metrics["exact_match_rate"] == 1.0
    assert metrics["character_errors"] == 0


@pytest.mark.parametrize(
    ("reference", "prediction", "characters", "edits"),
    [
        pytest.param("YGN AB12", "AB12", 7, 3, id="region-required"),
        pytest.param("NEW YORK AB12", "AB12", 11, 7, id="state-required"),
        pytest.param("YGN3L-5479", "YGN3L5479", 10, 1, id="punctuation-preserved"),
        pytest.param("OI-B8", "01-B8", 5, 2, id="no-letter-digit-substitution"),
        pytest.param(
            "NEW YORK AB12",
            "BRAND NEW YORK AB12",
            11,
            5,
            id="extra-branding-is-an-error",
        ),
    ],
)
def test_meaningful_target_text_is_not_filtered_or_silently_corrected(
    reference, prediction, characters, edits
) -> None:
    metrics = evaluate_ocr([reference], [reading(prediction)])
    assert metrics["reference_characters"] == characters
    assert metrics["character_errors"] == edits
    assert metrics["character_error_rate"] == pytest.approx(edits / characters)
    assert metrics["exact_matches"] == 0


def test_inputs_and_shared_result_objects_are_preserved() -> None:
    shared = reading(" yGn\tab12 ", status="uncertain", score=None)
    references = [" ygn AB12 ", "YGN\nAB12"]
    results = [shared, shared]
    original = deepcopy((references, results))
    first = evaluate_ocr(references, results)
    second = evaluate_ocr(references, results)
    assert (references, results) == original
    assert results[0] is results[1] is shared
    assert first == second
    assert first["sample_count"] == 2
    assert first["reference_characters"] == 14
    assert first["exact_matches"] == 2


def test_output_values_are_native_json_numbers_or_none() -> None:
    metrics = evaluate_ocr(["AB12"], [reading("AB12", status="uncertain", score=None)])
    count_keys = {
        "sample_count",
        "reference_characters",
        "character_errors",
        "exact_matches",
        "accepted_reads",
        "correct_accepted_reads",
    }
    rate_keys = {"exact_match_rate", "character_error_rate", "accepted_read_coverage"}
    assert all(type(metrics[key]) is int for key in count_keys)
    assert all(type(metrics[key]) is float for key in rate_keys)
    assert metrics["accepted_read_accuracy"] is None
    assert json.loads(json.dumps(metrics, allow_nan=False)) == metrics


@pytest.mark.parametrize(
    ("references", "results"),
    [
        (None, [reading("AB")]),
        ("AB", [reading("AB")]),
        (("AB",), [reading("AB")]),
        (["AB"], None),
        (["AB"], {}),
        (["AB"], (reading("AB"),)),
    ],
)
def test_outer_inputs_must_be_lists(references, results) -> None:
    with pytest.raises(ValueError, match="must be lists"):
        evaluate_ocr(references, results)


@pytest.mark.parametrize(
    ("references", "results"),
    [
        ([], []),
        ([], [reading("AB")]),
        (["AB"], []),
        (["AB", "CD"], [reading("AB")]),
        (["AB"], [reading("AB"), reading("CD")]),
    ],
)
def test_empty_or_unequal_inputs_are_rejected_instead_of_zipped_away(
    references, results
) -> None:
    with pytest.raises(ValueError, match="nonempty lists with equal lengths"):
        evaluate_ocr(references, results)


@pytest.mark.parametrize("reference", [None, 17, False, "", " \t\n", "\u00a0"])
def test_references_must_be_usable_strings(reference) -> None:
    with pytest.raises(ValueError, match=r"expected_texts\[0\]"):
        evaluate_ocr([reference], [reading("AB")])


@pytest.mark.parametrize("result", [None, {}, "AB"])
def test_result_records_must_be_ocr_results(result) -> None:
    with pytest.raises(ValueError, match="must be an OCRResult"):
        evaluate_ocr(["AB"], [result])


@pytest.mark.parametrize("status", [None, "success", "", True, [], 0])
def test_unknown_or_nonstrings_statuses_are_rejected(status) -> None:
    with pytest.raises(ValueError, match=r"results\[0\].status"):
        evaluate_ocr(["AB"], [replace(reading("AB"), status=status)])


@pytest.mark.parametrize("field", ["raw_text", "normalized_text", "error_message"])
@pytest.mark.parametrize("value", [7, True, []])
def test_text_metadata_requires_strings_or_none(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        evaluate_ocr(["AB"], [replace(reading("AB"), **{field: value})])


@pytest.mark.parametrize(
    "score", [True, -0.1, 1.1, float("nan"), float("inf"), -float("inf"), "0.9", [], 1j]
)
def test_invalid_scores_are_rejected(score) -> None:
    with pytest.raises(ValueError, match=r"results\[0\].score"):
        evaluate_ocr(["AB"], [replace(reading("AB"), score=score)])


@pytest.mark.parametrize("score", [0, 1, 0.0, 1.0])
def test_native_numeric_score_boundaries_are_valid(score) -> None:
    assert evaluate_ocr(["AB"], [reading("AB", score=score)])["accepted_reads"] == 1


@pytest.mark.parametrize("status", ["read", "uncertain"])
@pytest.mark.parametrize("text", [None, "", " \t\n"])
def test_text_bearing_statuses_require_usable_normalized_text(status, text) -> None:
    with pytest.raises(ValueError, match="requires usable normalized_text"):
        evaluate_ocr(
            ["AB"], [replace(reading("AB"), status=status, normalized_text=text)]
        )


def test_read_status_requires_a_score() -> None:
    with pytest.raises(ValueError, match="requires a score for read status"):
        evaluate_ocr(["AB"], [reading("AB", score=None)])


def test_invalid_later_record_does_not_mutate_earlier_inputs() -> None:
    references = [" ab ", "cd"]
    results = [reading(" AB "), reading("CD", score=None)]
    original = deepcopy((references, results))
    with pytest.raises(ValueError, match=r"results\[1\]"):
        evaluate_ocr(references, results)
    assert (references, results) == original
