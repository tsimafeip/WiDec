#!/usr/bin/env python
# Simple translation model and language model data structures
import sys
from collections import namedtuple

from typing import Dict, Tuple, List

phrase = namedtuple("phrase", "english, logprob")
"""
A translation model is a dictionary where keys are tuples of French words
and values are lists of (english, logprob) named tuples. For instance,
the French phrase "que se est" has two translations, represented like so:

tm[('que', 'se', 'est')] = [
  phrase(english='what has', logprob=-0.301030009985),
  phrase(english='what has been', logprob=-0.301030009985)]

:param filename: 
:param k: 
:return:
    dict with f as key and top-k translations as values
"""


def TM(filename: str, k: int) -> Dict[tuple, List[phrase]]:
    """

    Parameters
    ----------
    filename: str
        Source filename for translation model.
    k: int
        Pruning parameter to keep only the top k translations for each f.

    Returns
    -------
    Dict[tuple, List[phrase]]
        Translation model with french ngrams as key and top-k english translations as values.

    """
    sys.stderr.write("Reading translation model from %s...\n" % (filename,))
    tm = {}
    for line in open(filename).readlines():
        (f, e, logprob) = line.strip().split(" ||| ")
        # sets default list of translations for the french phrase
        # key - tuple of french words
        # values - translations with logprob (lower - better)
        tm.setdefault(tuple(f.split()), []).append(phrase(e, float(logprob)))
    for f in tm:  # prune all but top k translations
        tm[f].sort(key=lambda x: -x.logprob)
        del tm[f][k:]
    return tm


ngram_stats = namedtuple("ngram_stats", "logprob, backoff")


class LM:
    """
    A language model scores sequences of English words and accounts for both beginning and end of each sequence.

    Example API usage:
        lm = models.LM(filename)
        sentence = "This is a test ."
        lm_state = lm.begin() # initial state is always <s>
        logprob = 0.0
        for word in sentence.split():
            (lm_state, word_logprob) = lm.score(lm_state, word)
            logprob += word_logprob
        logprob += lm.end(lm_state) # transition to </s>, can also use lm.score(lm_state, "</s>")[1]
    """

    def __init__(self, filename: str):
        """
        Initializes language model using data from source file.

        Parameters
        ----------
        filename: str
            Name of source file for the language model.
        """
        sys.stderr.write("Reading language model from %s...\n" % (filename,))
        self.table = {}
        for line in open(filename, 'r'):
            entry = line.strip().split("\t")
            if len(entry) > 1 and entry[0] != "ngram":
                (logprob, ngram, backoff) = (
                    float(entry[0]),
                    tuple(entry[1].split()),
                    float(entry[2] if len(entry) == 3 else 0.0),
                )
                self.table[ngram] = ngram_stats(logprob, backoff)

    def begin(self) -> Tuple[str]:
        """
        Creates initial state of the LM represented by special start token.

        Returns
        -------
        Tuple[str]
            Initial LM state as tuple from single start token.
        """
        return "<s>",

    def score(self, state: Tuple, word: str) -> Tuple[tuple, float]:
        """

        Parameters
        ----------
        state: Tuple
            Current model state as tuple of string values.
        word: str
            Next word.

        Returns
        -------
        Tuple[tuple, float]
            Next state of the model and current logprob score of the sentence.
        """
        ngram = state + (word,)
        score = 0.0
        while len(ngram) > 0:
            if ngram in self.table:
                # return only two last word as next language state, since we are operating max by trigrams.
                return ngram[-2:], score + self.table[ngram].logprob
            else:
                # Backoff mechanism applied when we cannot estimate probability of the full ngram.
                # We are doing two steps
                #   - adding to the score backoff value for the current ngram without last word
                #   - shift ngram to remove first (oldest) word
                # details here: https://cmusphinx.github.io/wiki/arpaformat/

                score += self.table[ngram[:-1]].backoff if len(ngram) > 1 else 0.0
                ngram = ngram[1:]

        # return empty LM state and add probability of unknown word to final score
        return (), score + self.table[("<unk>",)].logprob

    def end(self, state: Tuple) -> float:
        """
        Returns logprob of the finishing sentence in current state.

        Parameters
        ----------
        state: Tuple
            Final state of the model.

        Returns
        -------
        float
            Logprob value of the finishing sentence in current state.

        """
        return self.score(state, "</s>")[1]
