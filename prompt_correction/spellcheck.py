"""
Spell correction, token-by-token, against the curated vocabulary.
Uses difflib (stdlib, no external dependency) for fuzzy matching.
Only touches tokens that look like plausible typos of a known domain/common
word — leaves proper nouns, numbers, and unrecognized-but-plausible words
(e.g. party names) alone, since over-correcting names would corrupt the query.
"""
import difflib
import re

from prompt_correction.vocabulary import VOCAB

CONTRACTION_FIXES = {
    "didnt": "didn't", "dont": "don't", "wasnt": "wasn't", "isnt": "isn't",
    "wont": "won't", "cant": "can't", "havent": "haven't", "hasnt": "hasn't",
    "wouldnt": "wouldn't", "shouldnt": "shouldn't", "couldnt": "couldn't",
}

TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[^\w\s]")


def correct_token(token: str, cutoff: float = 0.78) -> tuple[str, bool]:
    """Returns (corrected_token, was_changed)."""
    lower = token.lower()
    if not token.isalpha() or len(token) < 3:
        return token, False
    if lower in VOCAB:
        return token, False
    # simple plural check — don't flag "cases" as a typo just because
    # only the singular "case" is in the vocab
    if lower.endswith("s") and lower[:-1] in VOCAB:
        return token, False
    matches = difflib.get_close_matches(lower, VOCAB, n=1, cutoff=cutoff)
    if not matches:
        return token, False
    corrected = matches[0]
    # preserve original capitalization style
    if token[0].isupper():
        corrected = corrected.capitalize()
    return corrected, True


def correct_spelling(text: str) -> tuple[str, list[dict]]:
    tokens = TOKEN_RE.findall(text)
    corrections = []
    out_tokens = []
    for tok in tokens:
        lower = tok.lower()
        if lower in CONTRACTION_FIXES:
            fixed = CONTRACTION_FIXES[lower]
            corrections.append({"original": tok, "corrected": fixed})
            out_tokens.append(fixed)
            continue
        corrected, changed = correct_token(tok)
        if changed:
            corrections.append({"original": tok, "corrected": corrected})
        out_tokens.append(corrected)

    # reassemble with sensible spacing (no space before punctuation,
    # and no space after an apostrophe — keeps contractions/possessives intact)
    result = ""
    for i, tok in enumerate(out_tokens):
        is_punct = bool(re.match(r"^[^\w\s]+$", tok))
        prev_is_apostrophe = i > 0 and out_tokens[i - 1] == "'"
        if i > 0 and not is_punct and not prev_is_apostrophe:
            result += " "
        result += tok
    return result.strip(), corrections
