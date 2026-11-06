from __future__ import annotations

from learn_from.scripture import find_explicit_scripture_references, parse_scripture_reference


def test_parse_bible_chapter_and_verse() -> None:
    locator = parse_scripture_reference("Luke chapter 10 verse 18")

    assert locator is not None
    assert locator.canon == "bible"
    assert locator.collection == "new_testament"
    assert locator.book == "Luke"
    assert locator.chapter_start == 10
    assert locator.verse_start == 18
    assert locator.canonical_label == "Luke 10:18"
    assert locator.canonical_locator == "scripture:bible:luke:10:18"


def test_parse_latter_day_saint_section_reference() -> None:
    locator = parse_scripture_reference("D&C 76:22-24")

    assert locator is not None
    assert locator.canon == "latter_day_saint"
    assert locator.collection == "doctrine_and_covenants"
    assert locator.section_start == 76
    assert locator.verse_start == 22
    assert locator.verse_end == 24
    assert locator.canonical_label == "Doctrine and Covenants 76:22-24"


def test_find_explicit_scripture_references_is_full_text_and_evidence_bearing() -> None:
    results = find_explicit_scripture_references(
        "The speaker first mentions Isaiah 14 and later compares it with Luke 10:18."
    )

    assert [item["locator"]["canonical_label"] for item in results] == ["Isaiah 14", "Luke 10:18"]
    assert all(item["evidence"] for item in results)


def test_single_chapter_book_uses_verse_semantics() -> None:
    locator = parse_scripture_reference("Jude 6")

    assert locator is not None
    assert locator.chapter_start == 1
    assert locator.verse_start == 6
    assert locator.canonical_label == "Jude 6"


def test_numbered_book_boundary_does_not_create_false_reference() -> None:
    assert parse_scripture_reference("Revelation 2 Peter") is None


def test_parse_other_scripture_chapter_range() -> None:
    locator = parse_scripture_reference("1 Enoch 6-11")

    assert locator is not None
    assert locator.canon == "second_temple_jewish"
    assert locator.chapter_start == 6
    assert locator.chapter_end == 11
    assert locator.canonical_label == "1 Enoch 6-11"
