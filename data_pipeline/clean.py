"""
Stage 1b: Cleaning.
Normalizes whitespace, strips page headers/footers/watermark noise commonly
found in NCDRC/State Commission judgment exports.
"""
import re

NOISE_PATTERNS = [
    r"^\s*Page \d+ of \d+\s*$",
    r"^\s*\d+\s*$",  # bare page numbers on their own line
]
# NOTE: forum header lines ("NATIONAL CONSUMER DISPUTES REDRESSAL COMMISSION" /
# "STATE CONSUMER DISPUTES REDRESSAL COMMISSION") are intentionally kept —
# structuring.segment.detect_forum() depends on them.
NOISE_RE = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)


def clean_text(raw: str) -> str:
    lines = raw.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if NOISE_RE.match(stripped):
            continue
        kept.append(stripped)
    text = "\n".join(kept)
    # collapse repeated whitespace / hyphenation artifacts from OCR
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate OCR line-wraps
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


if __name__ == "__main__":
    import sys
    from extract import extract
    result = extract(sys.argv[1])
    cleaned = clean_text(result["text"])
    print(cleaned[:1000])
