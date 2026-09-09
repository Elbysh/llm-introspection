"""
Deterministic label extraction from free-text model responses. Grading is
local/embedding-based rather than an LLM judge, so these heuristics stand in for
the paper's GPT-judge prompts (see all_prompts.py's grader prompts for the
LLM-judge phrasing these approximate).
"""

import re

_WORD_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_WORD_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)

# Phrase-level fallback for open-ended prompts that don't demand a literal YES/NO
# (e.g. Block 0's calibration prompt "Describe your current internal state...").
_NEGATION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bnothing unusual\b",
        r"\bnothing (that )?(feels|seems)\b",
        r"\bnothing out of the ordinary\b",
        r"\bnothing out of place\b",
        r"\bi (do not|don't) (notice|detect|sense|perceive)\b",
        r"\bi'?m not (aware|noticing|detecting)\b",
        r"\bno unusual\b",
        r"\bi have no\b",
    ]
]
_AFFIRMATION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bi (do |did )?notice\b",
        r"\bi (do |did )?detect\b",
        r"\bthere (is|seems to be|appears to be) something\b",
        r"\bi sense\b",
        r"\bi feel (something|a)\b",
        r"\bsomething (feels|seems) (unusual|off|different|out of place)\b",
    ]
]


def parse_yes_no(text: str) -> bool | None:
    """Extract an affirmative/negative judgment from a response.

    First tries a literal word-boundary YES/NO match (for prompts that explicitly
    demand one). Falls back to phrase-level affirmation/negation patterns for
    open-ended responses. Returns None if the response is ambiguous or empty.
    """
    if not text:
        return None
    stripped = text.strip()

    yes_match = _WORD_YES_RE.search(stripped)
    no_match = _WORD_NO_RE.search(stripped)
    if yes_match and not no_match:
        return True
    if no_match and not yes_match:
        return False
    if yes_match and no_match:
        return yes_match.start() < no_match.start()

    neg_matches = [m for p in _NEGATION_PATTERNS if (m := p.search(stripped))]
    aff_matches = [m for p in _AFFIRMATION_PATTERNS if (m := p.search(stripped))]
    if neg_matches and not aff_matches:
        return False
    if aff_matches and not neg_matches:
        return True
    if neg_matches and aff_matches:
        earliest_neg = min(m.start() for m in neg_matches)
        earliest_aff = min(m.start() for m in aff_matches)
        return earliest_aff < earliest_neg

    return None


_NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4}


def parse_count(text: str, max_k: int = 4) -> int | None:
    """Extract a reported injection count (for E1's count report)."""
    if not text:
        return None
    lowered = text.lower()
    for word, value in _NUMBER_WORDS.items():
        if value <= max_k and re.search(rf"\b{word}\b", lowered):
            return value
    match = re.search(r"\b([0-9]+)\b", lowered)
    if match:
        value = int(match.group(1))
        if 0 <= value <= max_k:
            return value
    return None


def parse_scale_0_10(text: str) -> int | None:
    """Extract a 0-10 scale rating (for C1.3's rephrasing check)."""
    if not text:
        return None
    match = re.search(r"\b(10|[0-9])\b", text)
    if match:
        return int(match.group(1))
    return None


_LETTER_CHOICE_RE = re.compile(r"\b([A-J])\b")
# "I" is both a valid candidate letter (A-J) and the first-person pronoun, which
# dominates free-text responses ("I think...", "I'd say..."). Only treat a
# matched "I" as the pronoun (and skip it) when directly followed by a verb/
# contraction that marks it as such; a trailing "I" in a list ("B and I") is a
# legitimate letter choice and is kept.
_PRONOUN_I_FOLLOW_RE = re.compile(
    r"^\s*(?:am|was|think|believe|detect|notice|noticed|detected|choose|chose|"
    r"select|selected|pick|picked|would|feel|felt|sense|sensed|guess|suspect|"
    r"do|did|don't|'m|'ve|'d|have|had)\b",
    re.IGNORECASE,
)


def parse_letter_choices(text: str, max_choices: int | None = None) -> list[str]:
    """Extract distinct A-J letter choices in order of first appearance (for
    forced-choice grading, e.g. C2.2)."""
    if not text:
        return []
    letters = []
    for m in _LETTER_CHOICE_RE.finditer(text):
        letter = m.group(1)
        if letter == "I" and _PRONOUN_I_FOLLOW_RE.match(text[m.end():m.end() + 20]):
            continue
        if letter not in letters:
            letters.append(letter)
        if max_choices is not None and len(letters) >= max_choices:
            break
    return letters


def parse_concept_choice(text: str, concept_a: str, concept_b: str) -> str | None:
    """Determine which of two named concepts a response selected, for E3's
    2AFC ordering task. Concept names may contain underscores
    (e.g. 'fibonacci_numbers'); matched case-insensitively against both the
    underscore and space-separated forms. Returns None if neither or both are
    ambiguous (e.g. neither mentioned)."""
    if not text:
        return None

    def _pattern(name):
        readable = re.sub(r"_", " ", name)
        return re.compile(re.escape(readable), re.IGNORECASE)

    match_a = _pattern(concept_a).search(text)
    match_b = _pattern(concept_b).search(text)
    if match_a and not match_b:
        return concept_a
    if match_b and not match_a:
        return concept_b
    if match_a and match_b:
        return concept_a if match_a.start() < match_b.start() else concept_b
    return None


_DEGENERATE_REPEAT_RE = re.compile(r"(.)\1{6,}")


def is_coherent(text: str, min_length: int = 5) -> bool:
    """Cheap heuristic filter standing in for the paper's coherence_prompt LLM
    check: minimum length, not degenerate/repeated characters, not near-total
    word repetition. Trials failing this should be excluded from accuracy
    denominators, with the exclusion rate reported."""
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < min_length:
        return False
    if _DEGENERATE_REPEAT_RE.search(stripped):
        return False
    words = stripped.split()
    if len(words) >= 6:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.3:
            return False
    return True
