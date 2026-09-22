"""Small dependency-free BM25 index plus Reciprocal Rank Fusion for hybrid retrieval."""
import math
import re
from collections import Counter

from app.services.language import AR_STOPWORDS, has_arabic, normalize_arabic, stem_arabic

STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it of on or that the this to was "
    "what when where which who why with can i you do does".split()
)
_TOKEN = re.compile(r"[^\W_]+")  # letters and digits in any script


def tokenize(text: str) -> list[str]:
    """Lowercase, normalize Arabic, drop stop words (English and Arabic) and lightly stem Arabic."""
    tokens: list[str] = []
    for token in _TOKEN.findall(normalize_arabic(text).lower()):
        if token in STOPWORDS or token in AR_STOPWORDS:
            continue
        tokens.append(stem_arabic(token) if has_arabic(token) else token)
    return tokens


class BM25Index:
    """Okapi BM25 over a fixed set of documents (Lucene-style non-negative IDF)."""

    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        self.ids = list(ids)
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(t)) for t in texts]
        self._doc_len = [sum(tf.values()) for tf in self._tf]
        self._avgdl = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0
        doc_freq: Counter = Counter()
        for tf in self._tf:
            doc_freq.update(tf.keys())
        n = len(self.ids)
        self._idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def scores(self, query: str) -> dict[str, float]:
        """BM25 score for every document that shares at least one term with the query."""
        terms = set(tokenize(query))
        result: dict[str, float] = {}
        if not terms or not self.ids:
            return result
        for doc_id, tf, doc_len in zip(self.ids, self._tf, self._doc_len):
            norm = self.k1 * (1 - self.b + self.b * doc_len / self._avgdl) if self._avgdl else self.k1
            score = 0.0
            for term in terms:
                freq = tf.get(term)
                if freq:
                    score += self._idf[term] * freq * (self.k1 + 1) / (freq + norm)
            if score > 0:
                result[doc_id] = score
        return result


def fuse_rankings(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d))."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused
