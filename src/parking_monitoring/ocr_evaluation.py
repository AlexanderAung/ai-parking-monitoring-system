"""Pure OCR benchmark metrics."""

import jiwer

from parking_monitoring.schemas import OCRResult


def evaluate_ocr(
    expected_texts: list[str],
    results: list[OCRResult],
) -> dict[str, int | float | None]:
    """Compare aligned reference readings and OCR results.

    Args:
        expected_texts: Verified readable references, including region/state
            text, in the same order as results.
        results: One OCRResult per reference, retaining empty/error outcomes.
            Text-bearing read/uncertain results require usable normalized_text;
            read results also require a score. Acceptance uses the supplied
            status without reapplying a confidence threshold.

    Returns:
        Counts and rates under these keys:
        sample_count, reference_characters, character_errors,
        exact_matches, exact_match_rate, character_error_rate,
        accepted_reads, correct_accepted_reads,
        accepted_read_accuracy, accepted_read_coverage.

        Rates are fractions. accepted_read_accuracy is None when
        there are no accepted reads. Acceptance uses status="read".
        Character error rate can exceed 1 when there are many insertions.
        Unreadable/error results are scored as empty predictions. Inputs
        are preserved, and correct uncertain text counts as an exact match.

    Raises:
        ValueError: Inputs are empty, lengths differ, references normalize
            to empty text, or result data is invalid.
    """
    # 1. Validate inputs: require nonempty, equal-length lists, usable
    # references and valid OCRResult data before calculating any metrics.
    if not isinstance(expected_texts, list) or not isinstance(results, list):
        # Preserve the reviewed ValueError contract for invalid input types.
        raise ValueError("expected_texts and results must be lists")  # noqa: TRY004
    if not expected_texts or len(expected_texts) != len(results):
        raise ValueError("Inputs must be nonempty lists with equal lengths")
    for index, reference in enumerate(expected_texts):
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(
                f"expected_texts[{index}] must contain readable reference text"
            )
    for index, result in enumerate(results):
        if not isinstance(result, OCRResult):
            raise ValueError(f"results[{index}] must be an OCRResult")  # noqa: TRY004
        if not isinstance(result.status, str) or result.status not in (
            "read",
            "uncertain",
            "unreadable",
            "error",
        ):
            raise ValueError(f"results[{index}].status is invalid")
        for field in ("raw_text", "normalized_text", "error_message"):
            value = getattr(result, field)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"results[{index}].{field} must be a string or None")
        if result.score is not None and (
            type(result.score) not in (int, float) or not 0 <= result.score <= 1
        ):
            raise ValueError(
                f"results[{index}].score must be a finite Python number in [0, 1] or None"
            )
        if result.status in ("read", "uncertain") and (
            result.normalized_text is None or not result.normalized_text.strip()
        ):
            raise ValueError(
                f"results[{index}] requires usable normalized_text for {result.status}"
            )
        if result.status == "read" and result.score is None:
            raise ValueError(f"results[{index}] requires a score for read status")

    # 2. Normalize references and supplied readings: uppercase and remove
    # whitespace only, preserving region/state text, punctuation and characters.
    references = ["".join(text.upper().split()) for text in expected_texts]
    predictions = [
        "".join(result.normalized_text.upper().split())
        if result.normalized_text is not None
        else ""
        for result in results
    ]

    # 3. Handle unsuccessful readings: use an empty prediction for unreadable
    # or error outcomes while keeping their sample and reference characters.
    predictions = [
        "" if result.status in ("unreadable", "error") else prediction
        for result, prediction in zip(results, predictions, strict=True)
    ]

    # 4. Count full normalized exact matches across all samples, including
    # correct uncertain readings and failures (whose predictions are empty).
    sample_count = len(references)
    matches = [
        reference == prediction
        for reference, prediction in zip(references, predictions, strict=True)
    ]
    exact_matches = sum(matches)

    # 5. Calculate character errors with jiwer, aligning each plate separately.
    # Sum substitutions/deletions/insertions over the corpus and divide by the
    # total reference characters, rather than averaging per-plate error rates.
    alignment = jiwer.process_characters(references, predictions)
    character_errors = int(
        alignment.substitutions + alignment.deletions + alignment.insertions
    )
    reference_characters = sum(len(reference) for reference in references)

    # 6. Calculate acceptance counts from the existing read status. No second
    # confidence threshold is introduced; accepted accuracy is undefined when
    # there are no accepted readings, so return None for that rate.
    accepted = [result.status == "read" for result in results]
    accepted_reads = sum(accepted)
    correct_accepted_reads = sum(
        is_match and is_accepted
        for is_match, is_accepted in zip(matches, accepted, strict=True)
    )
    accepted_read_accuracy = (
        correct_accepted_reads / accepted_reads if accepted_reads else None
    )

    # 7. Return exactly the ten agreed Python-valued counts/rates. Input
    # validation guarantees positive sample and reference-character counts.
    return {
        "sample_count": sample_count,
        "reference_characters": reference_characters,
        "character_errors": character_errors,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / sample_count,
        "character_error_rate": character_errors / reference_characters,
        "accepted_reads": accepted_reads,
        "correct_accepted_reads": correct_accepted_reads,
        "accepted_read_accuracy": accepted_read_accuracy,
        "accepted_read_coverage": accepted_reads / sample_count,
    }
