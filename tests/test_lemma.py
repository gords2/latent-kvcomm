"""Round 2 fix (D13/D14 follow-up): _setup() only calls nltk.download() when a
capability probe finds something genuinely missing, instead of unconditionally
re-attempting a network index fetch every process (which cost 1250 bytes of SSL
error output per run on a host with no linked CA trust store). This test pins
down that every probe -- tokenize, pos_tag, wordnet -- succeeds in this CI/dev
environment, so a future environment regression (a corpus disappearing, a
Python upgrade breaking a resource path) surfaces as a named test failure here
rather than as a silent, systemic drop in every F1-based score."""
from lkv.lemma import _missing_capabilities


def test_no_missing_capabilities_in_this_environment():
    assert _missing_capabilities() == []
