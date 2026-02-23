"""Text preprocessing utilities for YouTube comment analysis.

Strips noise (URLs, emojis, extra whitespace) while preserving original
wording for transformer models like DistilBERT that have their own tokenizers.
"""

import logging
import re

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002700-\U000027bf"  # dingbats
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0001f900-\U0001f9ff"  # supplemental symbols
    "\U0001fa00-\U0001fa6f"  # chess symbols
    "\U0001fa70-\U0001faff"  # symbols extended-A
    "\U00002702-\U000027b0"
    "]+",
    flags=re.UNICODE,
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean(text: str) -> str:
    """Remove noise but preserve original wording.

    Strips URLs, emojis, and excess whitespace. Does NOT lowercase,
    lemmatize, or remove punctuation -- leaves text in its natural form
    so transformer models (DistilBERT) can leverage word forms and casing.

    Args:
        text: A single raw comment string.

    Returns:
        The cleaned text with original wording intact.
    """
    if not text or not isinstance(text, str):
        return ""
    text = _URL_PATTERN.sub("", text)
    text = _EMOJI_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def clean_batch(comments: list[str]) -> list[str]:
    """Clean a batch of comments.

    Args:
        comments: List of raw comment strings.

    Returns:
        A list of cleaned strings in the same order as input.
    """
    if not comments:
        return []
    return [clean(c) for c in comments]
