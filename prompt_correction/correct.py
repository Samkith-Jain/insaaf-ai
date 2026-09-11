"""
Stage 0: Prompt Correction.
Entry point matching the architecture diagram — sits between the user's
raw case file/query and the rest of the pipeline. Combines:
  1. Spell correction (rule-based, against curated legal/consumer vocab)
  2. Query reformulation (fragment expansion, domain-anchor insertion)
Returns the corrected query plus a transparent log of every change made,
so the user can see exactly what was "suggested & fixed" before it's
submitted downstream.
"""
from dataclasses import dataclass, field

from prompt_correction.spellcheck import correct_spelling
from prompt_correction.reformulate import reformulate


@dataclass
class CorrectionResult:
    original_query: str
    corrected_query: str
    spelling_fixes: list = field(default_factory=list)
    reformulation_notes: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Original:  {self.original_query}", f"Corrected: {self.corrected_query}"]
        if self.spelling_fixes:
            fixes = ", ".join(f"{f['original']}->{f['corrected']}" for f in self.spelling_fixes)
            lines.append(f"Spelling fixes: {fixes}")
        if self.reformulation_notes:
            for n in self.reformulation_notes:
                lines.append(f"Reformulation: {n}")
        if not self.spelling_fixes and not self.reformulation_notes:
            lines.append("No corrections needed — query was already well-formed.")
        return "\n".join(lines)


def correct_prompt(raw_query: str) -> CorrectionResult:
    spell_corrected, fixes = correct_spelling(raw_query)
    final_query, notes = reformulate(spell_corrected)
    return CorrectionResult(
        original_query=raw_query,
        corrected_query=final_query,
        spelling_fixes=fixes,
        reformulation_notes=notes,
    )


if __name__ == "__main__":
    import sys
    test_queries = [
        "flat posession delay refund",
        "my buildr didnt give me the flat on time and i want compensaton",
        "Was the NCDRC's decision in Pioneer Urban correctly applied to delayed possession cases under the Consumer Protection Act, 2019?",
        "insurence claim denied medcal negligense hospitl",
    ]
    queries = sys.argv[1:] or test_queries
    for q in queries:
        result = correct_prompt(q)
        print(result.summary())
        print("-" * 60)
