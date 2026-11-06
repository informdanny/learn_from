from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ScriptureLocator:
    canon: str
    collection: str
    work: str
    book: str | None
    chapter_start: int | None
    chapter_end: int | None
    verse_start: int | None
    verse_end: int | None
    section_start: int | None
    section_end: int | None
    canonical_label: str
    canonical_locator: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_OLD_TESTAMENT = (
    ("Genesis", ("gen",)),
    ("Exodus", ("exod", "ex")),
    ("Leviticus", ("lev",)),
    ("Numbers", ("num",)),
    ("Deuteronomy", ("deut",)),
    ("Joshua", ("josh",)),
    ("Judges", ("judg",)),
    ("Ruth", ()),
    ("1 Samuel", ("1 sam", "first samuel")),
    ("2 Samuel", ("2 sam", "second samuel")),
    ("1 Kings", ("1 kgs", "first kings")),
    ("2 Kings", ("2 kgs", "second kings")),
    ("1 Chronicles", ("1 chr", "first chronicles")),
    ("2 Chronicles", ("2 chr", "second chronicles")),
    ("Ezra", ()),
    ("Nehemiah", ("neh",)),
    ("Esther", ("esth",)),
    ("Job", ()),
    ("Psalms", ("psalm", "ps")),
    ("Proverbs", ("prov",)),
    ("Ecclesiastes", ("eccl",)),
    ("Song of Solomon", ("song of songs", "song")),
    ("Isaiah", ("isa",)),
    ("Jeremiah", ("jer",)),
    ("Lamentations", ("lam",)),
    ("Ezekiel", ("ezek",)),
    ("Daniel", ("dan",)),
    ("Hosea", ("hos",)),
    ("Joel", ()),
    ("Amos", ()),
    ("Obadiah", ("obad",)),
    ("Jonah", ()),
    ("Micah", ()),
    ("Nahum", ()),
    ("Habakkuk", ("hab",)),
    ("Zephaniah", ("zeph",)),
    ("Haggai", ("hag",)),
    ("Zechariah", ("zech",)),
    ("Malachi", ("mal",)),
)

_NEW_TESTAMENT = (
    ("Matthew", ("matt",)),
    ("Mark", ("mk",)),
    ("Luke", ("lk",)),
    ("John", ("jn",)),
    ("Acts", ()),
    ("Romans", ("rom",)),
    ("1 Corinthians", ("1 cor", "first corinthians")),
    ("2 Corinthians", ("2 cor", "second corinthians")),
    ("Galatians", ("gal",)),
    ("Ephesians", ("eph",)),
    ("Philippians", ("phil",)),
    ("Colossians", ("col",)),
    ("1 Thessalonians", ("1 thess", "first thessalonians")),
    ("2 Thessalonians", ("2 thess", "second thessalonians")),
    ("1 Timothy", ("1 tim", "first timothy")),
    ("2 Timothy", ("2 tim", "second timothy")),
    ("Titus", ()),
    ("Philemon", ("philem",)),
    ("Hebrews", ("heb",)),
    ("James", ("jas",)),
    ("1 Peter", ("1 pet", "first peter")),
    ("2 Peter", ("2 pet", "second peter")),
    ("1 John", ("1 jn", "first john")),
    ("2 John", ("2 jn", "second john")),
    ("3 John", ("3 jn", "third john")),
    ("Jude", ()),
    ("Revelation", ("book of revelation", "rev")),
)

_BOOK_OF_MORMON = (
    ("1 Nephi", ("first nephi",)),
    ("2 Nephi", ("second nephi",)),
    ("Jacob", ()),
    ("Enos", ()),
    ("Jarom", ()),
    ("Omni", ()),
    ("Words of Mormon", ()),
    ("Mosiah", ()),
    ("Alma", ()),
    ("Helaman", ()),
    ("3 Nephi", ("third nephi",)),
    ("4 Nephi", ("fourth nephi",)),
    ("Mormon", ()),
    ("Ether", ()),
    ("Moroni", ()),
)

_PEARL_OF_GREAT_PRICE = (
    ("Moses", ()),
    ("Abraham", ()),
    ("Joseph Smith-Matthew", ("joseph smith matthew", "js-m")),
    ("Joseph Smith-History", ("joseph smith history", "js-h")),
    ("Articles of Faith", ()),
)

_OTHER_SCRIPTURE = (
    ("1 Enoch", ("first enoch", "book of enoch")),
    ("Jubilees", ("book of jubilees",)),
)

_SINGLE_CHAPTER_BIBLE_BOOKS = {"Obadiah", "Philemon", "2 John", "3 John", "Jude"}
_NUMBERED_BOOK_TAILS = {
    "samuel",
    "kings",
    "chronicles",
    "corinthians",
    "thessalonians",
    "timothy",
    "peter",
    "john",
    "nephi",
}


def _normalize_reference_input(value: str) -> str:
    value = re.sub(r"\bfirst\b", "1", value, flags=re.IGNORECASE)
    value = re.sub(r"\bsecond\b", "2", value, flags=re.IGNORECASE)
    value = re.sub(r"\bthird\b", "3", value, flags=re.IGNORECASE)
    value = re.sub(r"\bfourth\b", "4", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value)


def _normalize_book_key(value: str) -> str:
    value = _normalize_reference_input(value)
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _book_alias_names() -> set[str]:
    names: set[str] = set()
    for books in (
        _OLD_TESTAMENT,
        _NEW_TESTAMENT,
        _BOOK_OF_MORMON,
        _PEARL_OF_GREAT_PRICE,
        _OTHER_SCRIPTURE,
    ):
        for canonical, aliases in books:
            names.update((canonical, *aliases))
    names.update(("Doctrine and Covenants", "D&C", "DC"))
    return names


def _build_book_index() -> dict[str, tuple[str, str, str, str]]:
    index: dict[str, tuple[str, str, str, str]] = {}
    groups = (
        ("bible", "old_testament", "Bible", _OLD_TESTAMENT),
        ("bible", "new_testament", "Bible", _NEW_TESTAMENT),
        ("latter_day_saint", "book_of_mormon", "Book of Mormon", _BOOK_OF_MORMON),
        (
            "latter_day_saint",
            "pearl_of_great_price",
            "Pearl of Great Price",
            _PEARL_OF_GREAT_PRICE,
        ),
        ("second_temple_jewish", "other_scripture", "Other Scripture", _OTHER_SCRIPTURE),
    )
    for canon, collection, work, books in groups:
        for canonical, aliases in books:
            for name in (canonical, *aliases):
                index[_normalize_book_key(name)] = (canonical, canon, collection, work)

    for alias in ("doctrine and covenants", "d&c", "dc"):
        index[_normalize_book_key(alias)] = (
            "Doctrine and Covenants",
            "latter_day_saint",
            "doctrine_and_covenants",
            "Doctrine and Covenants",
        )
    return index


_BOOK_INDEX = _build_book_index()
_BOOK_ALTERNATION = "|".join(
    re.escape(name)
    for name in sorted(
        {name for name in _book_alias_names() if len(name) > 1},
        key=len,
        reverse=True,
    )
)
_REFERENCE_RE = re.compile(
    rf"\b(?P<book>{_BOOK_ALTERNATION})\s+"
    r"(?:(?:chapter|chap\.?|section)\s+)?"
    r"(?P<start>\d{1,3})"
    r"(?:\s*(?::|verse\s+)\s*(?P<verse>\d{1,3}))?"
    r"(?:\s*[-–]\s*(?P<end>\d{1,3}))?",
    flags=re.IGNORECASE,
)
_BOOK_ONLY_RE = re.compile(
    rf"\bbook\s+of\s+(?P<book>{_BOOK_ALTERNATION})\b",
    flags=re.IGNORECASE,
)


def parse_scripture_reference(value: str) -> ScriptureLocator | None:
    compact = re.sub(r"\s+", " ", value).strip(" ,.;")
    normalized_input = _normalize_reference_input(compact)
    match = _REFERENCE_RE.search(normalized_input)
    if match and not _is_numbered_book_list_boundary(match, normalized_input):
        return _locator_from_match(match)

    book_only = _BOOK_ONLY_RE.search(_normalize_reference_input(compact))
    if not book_only:
        return None
    book_record = _BOOK_INDEX.get(_normalize_book_key(book_only.group("book")))
    if not book_record:
        return None
    book, canon, collection, work = book_record
    locator = _base_locator(canon=canon, work=work, book=book)
    return ScriptureLocator(
        canon=canon,
        collection=collection,
        work=work,
        book=book,
        chapter_start=None,
        chapter_end=None,
        verse_start=None,
        verse_end=None,
        section_start=None,
        section_end=None,
        canonical_label=book if work == "Bible" else f"{work}: {book}",
        canonical_locator=locator,
    )


def find_explicit_scripture_references(text: str) -> list[dict[str, Any]]:
    normalized_input = _normalize_reference_input(text)
    results: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for match in _REFERENCE_RE.finditer(normalized_input):
        if _is_numbered_book_list_boundary(match, normalized_input):
            continue
        locator = _locator_from_match(match)
        if locator is None:
            continue
        key = (match.start(), match.end(), locator.canonical_locator)
        if key in seen:
            continue
        seen.add(key)
        start = max(0, match.start() - 90)
        end = min(len(normalized_input), match.end() + 90)
        results.append(
            {
                "raw_mention": normalized_input[match.start() : match.end()],
                "evidence": normalized_input[start:end].strip(),
                "character_start": match.start(),
                "character_end": match.end(),
                "locator": locator.to_dict(),
            }
        )
    return results


def _locator_from_match(match: re.Match[str]) -> ScriptureLocator | None:
    book_record = _BOOK_INDEX.get(_normalize_book_key(match.group("book")))
    if not book_record:
        return None
    book, canon, collection, work = book_record
    start = int(match.group("start"))
    verse = int(match.group("verse")) if match.group("verse") else None
    end = int(match.group("end")) if match.group("end") else None

    if work == "Doctrine and Covenants":
        section_start = start
        section_end = end if verse is None and end is not None else start
        chapter_start = chapter_end = None
        verse_start = verse
        verse_end = end if verse is not None else None
        label = f"Doctrine and Covenants {start}"
        if verse is not None:
            label += f":{verse}"
            if end is not None:
                label += f"-{end}"
        elif section_end != section_start:
            label += f"-{section_end}"
        locator = f"scripture:{canon}:doctrine_and_covenants:{start}"
        if verse is not None:
            locator += f":{verse}"
            if end is not None:
                locator += f"-{end}"
        elif section_end != section_start:
            locator += f"-{section_end}"
    else:
        section_start = section_end = None
        single_chapter = canon == "bible" and book in _SINGLE_CHAPTER_BIBLE_BOOKS
        if single_chapter and verse is None:
            chapter_start = chapter_end = 1
            verse_start = start
            verse_end = end
            label = f"{book} {start}"
            locator = f"{_base_locator(canon=canon, work=work, book=book)}:{start}"
            if end is not None:
                label += f"-{end}"
                locator += f"-{end}"
        else:
            chapter_start = start
            chapter_end = end if verse is None and end is not None else start
            verse_start = verse
            verse_end = end if verse is not None else None
            label = f"{book} {start}"
            locator = f"{_base_locator(canon=canon, work=work, book=book)}:{start}"
            if verse is not None:
                label += f":{verse}"
                locator += f":{verse}"
                if end is not None:
                    label += f"-{end}"
                    locator += f"-{end}"
            elif chapter_end != chapter_start:
                label += f"-{chapter_end}"
                locator += f"-{chapter_end}"

    return ScriptureLocator(
        canon=canon,
        collection=collection,
        work=work,
        book=book,
        chapter_start=chapter_start,
        chapter_end=chapter_end,
        verse_start=verse_start,
        verse_end=verse_end,
        section_start=section_start,
        section_end=section_end,
        canonical_label=label,
        canonical_locator=locator,
    )


def _base_locator(*, canon: str, work: str, book: str) -> str:
    if work == "Bible":
        return f"scripture:{canon}:{_slug(book)}"
    if work == "Other Scripture":
        return f"scripture:{canon}:{_slug(book)}"
    return f"scripture:{canon}:{_slug(work)}:{_slug(book)}"


def _is_numbered_book_list_boundary(match: re.Match[str], text: str) -> bool:
    if match.group("verse") is not None:
        return False
    trailing = text[match.end() :]
    next_word = re.match(r"\s+([A-Za-z]+)", trailing)
    return bool(next_word and next_word.group(1).lower() in _NUMBERED_BOOK_TAILS)
