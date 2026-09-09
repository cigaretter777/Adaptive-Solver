import hashlib

from adaptive_math.core.hashing import canonical_text, make_source_hash, make_task_id


def test_task_id_is_content_addressed() -> None:
    first = make_task_id("aime_2024", "What is 1+1?")
    second = make_task_id("aime_2024", "  What   is 1+1? ")
    assert first == second
    assert first.startswith("aime_2024:")


def test_task_id_has_20_hex_digits() -> None:
    task_id = make_task_id("aime_2024", "What is 1+1?")
    digest_part = task_id.split(":", 1)[1]
    assert len(digest_part) == 20
    assert all(c in "0123456789abcdef" for c in digest_part)


def test_different_datasets_get_different_ids() -> None:
    assert make_task_id("aime_2024", "What is 1+1?") != make_task_id("math", "What is 1+1?")


def test_different_expressions_get_different_ids() -> None:
    # canonicalization is surface-only; it must not simplify algebra
    assert make_task_id("aime_2024", "x + 1") != make_task_id("aime_2024", "x - 1")


def test_unicode_equivalent_text_normalizes_equal() -> None:
    assert canonical_text("café") == canonical_text("café")
    assert canonical_text("２＋２") == canonical_text("2+2")
    assert make_task_id("d", "２＋２") == make_task_id("d", "2+2")


def test_spacing_around_operators_is_preserved() -> None:
    # surface canonicalization must not rewrite math notation
    assert canonical_text("2 + 2") != canonical_text("2+2")


def test_line_endings_are_normalized() -> None:
    assert canonical_text("a\r\nb\rc\nd") == canonical_text("a b c d")


def test_whitespace_collapse_and_trim() -> None:
    assert canonical_text("  a\t\t b \n c ") == "a b c"


def test_canonical_text_is_stable() -> None:
    once = canonical_text("  What \r\n is\t1+1? ")
    assert canonical_text(once) == once


def test_source_hash_is_sha256_of_exact_bytes() -> None:
    digest = make_source_hash(b'{"answer": "2"}')
    assert digest == hashlib.sha256(b'{"answer": "2"}').hexdigest()
    assert len(digest) == 64


def test_source_hash_uses_unmodified_bytes() -> None:
    assert make_source_hash(b"2") != make_source_hash(b" 2 ")
