"""
Deterministic label extraction from free-text model responses. Grading is
local/embedding-based rather than an LLM judge, so these heuristics stand in for
the paper's GPT-judge prompts (see all_prompts.py's grader prompts for the
LLM-judge phrasing these approximate).
"""

import re

_WORD_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_WORD_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)

# Explicit trailing tag some prompts now force (e.g. get_calibration_messages),
# checked before any other heuristic since it's unambiguous by construction.
_ANSWER_TAG_RE = re.compile(r"\banswer:\s*(yes|no)\b", re.IGNORECASE)

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


def strip_answer_tag(text: str) -> str:
    """Drop a trailing 'ANSWER: YES/NO' tag (see get_calibration_messages) so
    downstream embedding similarity isn't diluted by boilerplate constant
    across every trial."""
    if not text:
        return text
    return _ANSWER_TAG_RE.split(text, maxsplit=1)[0].strip()


def parse_yes_no(text: str) -> bool | None:
    """Extract an affirmative/negative judgment from a response.

    First tries a literal word-boundary YES/NO match (for prompts that explicitly
    demand one). Falls back to phrase-level affirmation/negation patterns for
    open-ended responses. Returns None if the response is ambiguous or empty.
    """
    if not text:
        return None
    stripped = text.strip()

    tag_matches = list(_ANSWER_TAG_RE.finditer(stripped))
    if tag_matches:
        return tag_matches[-1].group(1).lower() == "yes"

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

_AB_CHOICE_RE = re.compile(r"\b([AB])\b")


def parse_ab_choice(text: str) -> str | None:
    """Extract a single A/B choice (Exp 11's ordering 2AFC, section 16.2,
    where the two concepts are presented as lettered options rather than
    named directly in the question). Returns the first unambiguous A or B
    token, or None if neither appears."""
    if not text:
        return None
    match = _AB_CHOICE_RE.search(text)
    return match.group(1) if match else None


def parse_two_label_choice(text: str, valid_letters: list[str], max_choices: int = 2) -> list[str]:
    """Extract up to max_choices labels for Exp 10's forced-choice
    identification (section 15.3), where each slot's correct label is either
    a candidate letter or NONE. Candidate letters are matched case-sensitively
    like parse_letter_choices (to avoid stray lowercase words such as the
    article "a"); NONE is matched case-insensitively. Returns labels in order
    of first appearance -- not deduplicated, since a trial can legitimately
    need NONE in both slots (e.g. the sham condition)."""
    if not text:
        return []
    letters_pattern = "|".join(re.escape(letter) for letter in valid_letters)
    pattern = re.compile(rf"\b((?i:NONE)|{letters_pattern})\b")
    found = []
    for m in pattern.finditer(text):
        token = m.group(1)
        label = "NONE" if token.upper() == "NONE" else token
        if label == "I" and _PRONOUN_I_FOLLOW_RE.match(text[m.end():m.end() + 20]):
            continue
        found.append(label)
        if len(found) >= max_choices:
            break
    return found


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
