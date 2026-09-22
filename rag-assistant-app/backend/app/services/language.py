"""Language helpers: Arabic detection and normalization for search, prompts and refusals."""
import re

REFUSAL_EN = "I don't have enough information in the provided documents to answer that."
REFUSAL_AR = "لا تتوفر لدي معلومات كافية في المستندات المقدمة للإجابة عن هذا السؤال."

_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")  # tashkeel, superscript alef, tatweel
_CHAR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})
_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _is_arabic_letter(ch: str) -> bool:
    o = ord(ch)
    return ch.isalpha() and (
        0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF
        or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF
    )


def has_arabic(text: str) -> bool:
    return any(_is_arabic_letter(c) for c in text)


def is_arabic(text: str, threshold: float = 0.3) -> bool:
    """True when at least `threshold` of the letters are Arabic (so mixed text like 'ما هو ReLU؟' counts)."""
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(_is_arabic_letter(c) for c in letters) / len(letters) >= threshold


def normalize_arabic(text: str) -> str:
    """Search-friendly Arabic: no diacritics/tatweel, unified alef/yaa/taa-marbuta, ASCII digits."""
    return _DIACRITICS.sub("", text).translate(_CHAR_MAP).translate(_DIGIT_MAP)


_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")


def stem_arabic(token: str) -> str:
    """Light stemming: drop the definite article (and its common prefixes) when enough letters remain."""
    for prefix in _PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= 3:
            return token[len(prefix):]
    return token


AR_STOPWORDS = frozenset(
    normalize_arabic(w)
    for w in (
        "في من على إلى الى عن ما ماذا لماذا كيف هو هي هذا هذه ذلك تلك هل أو او و ثم كان كانت يكون "
        "أن ان أنا انا أنت نحن هم مع بين الذي التي الذين إذا اذا كل بعض قد لا لم لن أيضا ايضا حتى عند أي اي كم"
    ).split()
)


def refusal_for(question: str) -> str:
    return REFUSAL_AR if is_arabic(question) else REFUSAL_EN


def is_refusal(answer: str) -> bool:
    return REFUSAL_EN in answer or REFUSAL_AR in answer
