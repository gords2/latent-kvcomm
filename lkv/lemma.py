"""WordNet lemmatization, matching stock-kvcomm/utils/metric_tools.py:188-210.

Isolated in its own module because it is the codebase's only nltk dependency and
it needs corpora that must be present before the first call. It raises rather
than degrading to unlemmatized text: a silent fallback would change the metric
while still reporting it as KVComm's metric.

Downloads are attempted only when a capability is genuinely missing. Upstream
calls `nltk.download()` unconditionally at import (`metric_tools.py:18-23`), which
re-attempts a network index fetch in every fresh process -- measured at 1250
bytes of SSL error output per run on a host whose Python has no linked CA trust
store, even with every corpus already cached. Probing *functionally* rather
than by corpus path also sidesteps guessing whether a corpus is stored unzipped
(`corpora/wordnet`) or zipped (`corpora/wordnet.zip`), which varies by install --
on this machine the path probe reports wordnet missing while lemmatization works.
"""
from __future__ import annotations

import functools

_PACKAGES = ("punkt", "punkt_tab", "wordnet", "omw-1.4",
             "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng")


def _missing_capabilities() -> list[str]:
    """Exercise each capability; return the names that raise LookupError."""
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    probes = (
        ("tokenize", lambda: word_tokenize("a b")),
        ("pos_tag", lambda: nltk.pos_tag(["cats"])),
        ("wordnet", lambda: WordNetLemmatizer().lemmatize("cats", "n")),
    )
    missing = []
    for name, probe in probes:
        try:
            probe()
        except LookupError:
            missing.append(name)
    return missing


@functools.lru_cache(maxsize=1)
def _setup():
    import nltk
    from nltk.stem import WordNetLemmatizer

    if _missing_capabilities():
        for pkg in _PACKAGES:
            nltk.download(pkg, quiet=True)
        still_missing = _missing_capabilities()
        if still_missing:
            raise RuntimeError(
                "nltk corpora unavailable for: " + ", ".join(still_missing)
                + ". KVComm's F1 lemmatizes both sides "
                "(stock-kvcomm/utils/f1.py:12-13), so scoring without it would "
                "report a different metric under the same name. Install them: "
                "python -m nltk.downloader punkt punkt_tab wordnet omw-1.4 "
                "averaged_perceptron_tagger averaged_perceptron_tagger_eng"
            )
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
