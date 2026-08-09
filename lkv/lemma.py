"""WordNet lemmatization, matching stock-kvcomm/utils/metric_tools.py:188-210.

Isolated in its own module because it is the codebase's only nltk dependency and
it needs corpora that must be present before the first call. It raises rather
than degrading to unlemmatized text: a silent fallback would change the metric
while still reporting it as KVComm's metric.
"""
from __future__ import annotations

import functools

_REQUIRED = ("punkt", "punkt_tab", "wordnet", "omw-1.4",
             "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng")


@functools.lru_cache(maxsize=1)
def _setup():
    import nltk
    from nltk.stem import WordNetLemmatizer
    for pkg in _REQUIRED:
        nltk.download(pkg, quiet=True)
    return nltk, WordNetLemmatizer()


@functools.lru_cache(maxsize=100_000)
def _lemma_one(word: str) -> str:
    nltk, lemmatizer = _setup()
    from nltk.corpus import wordnet
    tag = nltk.pos_tag([word])[0][1][0].upper()
    pos = {"J": wordnet.ADJ, "N": wordnet.NOUN,
           "V": wordnet.VERB, "R": wordnet.ADV}.get(tag, wordnet.NOUN)
    return lemmatizer.lemmatize(word, pos)


def lemmatize_text(text: str) -> str:
    nltk, _ = _setup()
    from nltk.tokenize import word_tokenize
    return " ".join(_lemma_one(w) for w in word_tokenize(text))
