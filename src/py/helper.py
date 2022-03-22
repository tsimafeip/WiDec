import math
import sys

from timeit import default_timer as timer

from typing import Tuple, List, Union, Dict, Optional

import numpy as np

import models
from collections import namedtuple, defaultdict
from functools import reduce
from typing import Sequence

# Helper types and structures declaration
Sentence = Union[Tuple[str], str]
hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, fi, fj")
# we add only french_bitmap_coverage, since we must avoid overlapping
widec_hypothesis = namedtuple("widec_hypothesis",
                              "logprob, lm_state, predecessor, phrase, french_bitmap_coverage, fi, fj")


# Set of utility functions
def extract_english(h: Union[hypothesis, widec_hypothesis]) -> str:
    """Extracts English phrase from hypothesis."""
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)


def str_to_tokens_tuple(sentence: str) -> Tuple[str]:
    """

    Parameters
    ----------
    sentence

    Returns
    -------

    """
    return tuple(sentence.split())


def extract_tm_logprob(h: hypothesis) -> float:
    """Recursively extracts logprob of the resulting phrase."""
    return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)


def bitmap(sequence: Sequence[int]) -> int:
    """ Generate a coverage bitmap for a sequence of indexes """
    return reduce(lambda x, y: x | y, map(lambda i: int('1' + '0' * i, 2), sequence), 0)


def bitmap2str(b, n, on='o', off='.') -> str:
    """ Generate a length-n string representation of bitmap b """
    return '' if n == 0 else (on if b & 1 == 1 else off) + bitmap2str(b >> 1, n - 1, on, off)


def logadd10(x: float, y: float) -> float:
    """ Addition in logspace (base 10): if x=log(a) and y=log(b), returns log(a+b) """
    return x + math.log10(1 + pow(10, y - x))


# Decoding functions: monotone and beam

def monotone_decoding_translate(french_sentence: Tuple[str], lm: models.LM, tm: models.TM,
                                stack_size=10000, verbose: bool = False) -> hypothesis:
    """

    Parameters
    ----------
    french_sentence
    lm
    tm
    stack_size
    verbose

    Returns
    -------

    """
    # First, it seeks the Viterbi approximation to the most probable translation.
    # Instead of computing the intractable sum over all alignments for each sentence,
    # we simply find the best single alignment and use its translation.
    # Stacks in an analogue for Viterbi matrix.
    initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, -1, -1)
    stacks = [{} for _ in french_sentence] + [{}]
    stacks[0][lm.begin()] = initial_hypothesis

    start = timer()
    # iterates by all stacks except the last one which contains the result
    for fi, stack in enumerate(stacks[:-1]):
        # sort all hypothesis ending in current i position and select top-s
        for h in sorted(stack.values(), key=lambda h: -h.logprob)[:stack_size]:  # prune
            # all possible next spans starting in i
            for fj in range(fi + 1, len(french_sentence) + 1):
                french_span = french_sentence[fi:fj]
                english_translations = tm.get(french_span, [])
                # Iterates by all possible translations of french phrase.
                # By default, here is only the most probable translation.
                for phrase in english_translations:
                    # accumulate TM-logprob with prev hypothesis
                    logprob = h.logprob + phrase.logprob
                    lm_state = h.lm_state
                    # add LM-logprob
                    for word in phrase.english.split():
                        (lm_state, word_logprob) = lm.score(lm_state, word)
                        logprob += word_logprob
                    logprob += lm.end(lm_state) if fj == len(french_sentence) else 0.0
                    new_hypothesis = hypothesis(logprob, lm_state, h, phrase, fi, fj)
                    # second case is recombination, keep in mind that logprobs are negative
                    if lm_state not in stacks[fj] or stacks[fj][lm_state].logprob < logprob:
                        stacks[fj][lm_state] = new_hypothesis

    winner = max(stacks[-1].values(), key=lambda h: h.logprob)
    end = timer()

    if verbose:
        print_verbose_statistics(start, end, winner)

    return winner


def beam_decoding(f: Tuple[str], lm: models.LM, tm: models.TM, stack_size: int = 10000,
                  stack_threshold_alpha: float = 0.0, reordering_limit: int = sys.maxsize,
                  reordering_base: float = 0.75, disable_reordering_penalty: bool = False,
                  verbose: bool = False) -> widec_hypothesis:
    """

    Parameters
    ----------
    f
    lm
    tm
    stack_size
    stack_threshold_alpha
    reordering_limit
    reordering_base
    disable_reordering_penalty
    verbose

    Returns
    -------

    """

    def _threshold_pruning(hypotheses: List[widec_hypothesis], alpha: float) -> List[widec_hypothesis]:
        """

        Parameters
        ----------
        hypotheses
        alpha

        Returns
        -------

        """
        # best logprob is a small negative number
        best_h_prob = hypotheses[0].logprob

        # threshold value is a positive number
        threshold_value = -1 * (1 / alpha) * best_h_prob

        # we keep only hypotheses, which have -logprob less than the threshold, defined above
        pruned_hypotheses = [h for h in hypotheses if -h.logprob <= threshold_value]

        return pruned_hypotheses

    initial_hypothesis = widec_hypothesis(0.0, lm.begin(), None, None, 0, -1, -1)

    all_phrases_translations = defaultdict(list)
    # 1) Generate all possible french spans
    # 2) Collect all possible translations
    for fi in range(len(f)):
        for fj in range(fi + 1, len(f) + 1):
            english_translations = tm.get(f[fi:fj], [])
            for phrase_obj in english_translations:
                all_phrases_translations[(fi, fj)].append(phrase_obj)

    stacks = [{} for _ in f] + [{}]
    # 0 French words translated - empty hypothesis for English side
    stacks[0][lm.begin()] = initial_hypothesis

    '''
    place empty hypothesis into stack 0 for all stacks 0...n − 1 do
    for all hypotheses in stack do 
        for all translation options do
            if applicable then
            create new hypothesis
            place in stack
            recombine with existing hypothesis if possible 
            prune stack if too big
    '''

    # iterates by all stacks except the last one which contains the result
    local_start = timer()
    for i, stack in enumerate(stacks[:-1]):
        # iterate by all hypotheses covering i French words

        # histogram pruning
        pruned_hypotheses = sorted(stack.values(), key=lambda h: -h.logprob)[:stack_size]

        # if threshold pruning is enabled
        if stack_threshold_alpha:
            pruned_hypotheses = _threshold_pruning(pruned_hypotheses, stack_threshold_alpha)

        for h in pruned_hypotheses:  # prune
            # find all possible next spans of length [1, len(f) + 1 - i]
            for span_len in range(1, len(f) + 1 - i):
                for fi in range(len(f)):
                    fj = min(fi + span_len, len(f) + 1)
                    # if selected French words are not translated
                    if bitmap(range(fi, fj)) & h.french_bitmap_coverage == 0:
                        reordering_distance = fi - h.fj - 1

                        if reordering_distance > reordering_limit:
                            continue

                        f_words_translated = i + (fj - fi)

                        # prune
                        if len(stacks[f_words_translated]) > stack_size:
                            continue

                        new_french_bitmap_coverage = bitmap(range(fi, fj)) | h.french_bitmap_coverage
                        english_translations = all_phrases_translations.get((fi, fj), [])
                        for phrase in english_translations:
                            # accumulate TM-logprob with prev hypothesis
                            logprob = h.logprob + phrase.logprob
                            lm_state = h.lm_state
                            # add LM-logprob
                            for word in phrase.english.split():
                                (lm_state, word_logprob) = lm.score(lm_state, word)
                                # by summing logprobs we are basically multiplying absolute values
                                logprob += word_logprob
                            logprob += lm.end(lm_state) if f_words_translated == len(f) else 0.0

                            if not disable_reordering_penalty:
                                # calculate reordering distance: start_i - end_(i-1) - 1
                                reordering_score = np.log10(reordering_base ** abs(reordering_distance))
                                # add reordering score
                                logprob += reordering_score

                            new_hypothesis = \
                                widec_hypothesis(logprob, lm_state, h, phrase, new_french_bitmap_coverage, fi, fj)

                            # second case is recombination, keep in mind that logprobs are negative
                            if lm_state not in stacks[f_words_translated] \
                                    or stacks[f_words_translated][lm_state].logprob < logprob:
                                stacks[f_words_translated][lm_state] = new_hypothesis

    winner = max(stacks[-1].values(), key=lambda h: h.logprob)
    local_end = timer()

    if verbose:
        print_verbose_statistics(local_start, local_end, winner)

    return winner


# Scoring functions

def score_lm(e: Sentence, lm: models.LM, configured_verbosity: int = 1) -> float:
    """

    Parameters
    ----------
    e
    lm
    configured_verbosity

    Returns
    -------

    """
    lm_state = lm.begin()
    lm_logprob = 0.0
    for word in e + ("</s>",):
        maybe_write("%s: " % " ".join(lm_state + (word,)), 1, configured_verbosity)
        (lm_state, word_logprob) = lm.score(lm_state, word)
        lm_logprob += word_logprob
        maybe_write("%f\n" % (word_logprob,), 1, configured_verbosity)

    return lm_logprob


def generate_all_alignments(f: Sentence, e: Sentence, tm: models.TM, configured_verbosity: int = 1) -> List[
    List[Tuple]]:
    alignments = [[] for _ in e]
    # 1) Generate all possible french spans
    # 2) Append all alignments that exist in target phrase
    for fi in range(len(f)):
        for fj in range(fi + 1, len(f) + 1):
            french_span = f[fi:fj]
            english_translations = tm.get(french_span, [])
            for phrase_obj in english_translations:
                e_phrase = tuple(phrase_obj.english.split())
                for ei in range(len(e) + 1 - len(e_phrase)):
                    ej = ei + len(e_phrase)
                    if e_phrase == e[ei:ej]:
                        maybe_write("%s ||| %d, %d : %d, %d ||| %s ||| %f\n" %
                                    (" ".join(f[fi:fj]), fi, fj, ei, ej, " ".join(e_phrase),
                                     phrase_obj.logprob), 1, configured_verbosity)
                        alignments[ei].append((ej, phrase_obj.logprob, fi, fj))
    return alignments


def calculate_chart_over_alignments(
        e: Sentence, f: Sentence, alignments: List[List[Tuple]], configured_verbosity: int = 1,
) -> List[Dict[int, float]]:
    """

    Parameters
    ----------
    e
    f
    alignments
    configured_verbosity

    Returns
    -------

    """
    chart = [{} for _ in e] + [{}]
    chart[0][0] = 0.0
    for ei, sums in enumerate(chart[:-1]):
        # v is a coverage mask represented by int. it is a key in dictionary.
        # value is a cost of this coverage.
        # index is a supposed english sentence end.
        for v in sums:
            for ej, logprob, fi, fj in alignments[ei]:
                # if selected French words are not translated
                if bitmap(range(fi, fj)) & v == 0:
                    # update mask to add newly translated words
                    new_v = bitmap(range(fi, fj)) | v
                    maybe_write("(%d, %s): %f + (%d, %d, %s): %f -> (%d, %s): %f\n" %
                                (ei, bitmap2str(v, len(f)), sums[v],
                                 ei, ej, bitmap2str(bitmap(range(fi, fj)), len(f)), logprob,
                                 ej, bitmap2str(new_v, len(f)), sums[v] + logprob), 2, configured_verbosity)
                    # coverage until here
                    if new_v in chart[ej]:
                        # the same coverage of French words with the same English words met previously.
                        # the only difference is a way of split.
                        # For example ['honourable', 'senators', ','] or ['honourable', 'senators ,'].
                        # This implementation sum up all possible segmentations as valid translations.
                        chart[ej][new_v] = logadd10(chart[ej][new_v], sums[v] + logprob)
                    else:
                        # add coverage until here + coverage here
                        chart[ej][new_v] = sums[v] + logprob
        maybe_write(".", 1, configured_verbosity)
        maybe_write("\n", 2, configured_verbosity)

    return chart


def score_translation(
        e: Tuple[str], f: Tuple[str], lm: models.LM, tm: models.TM, configured_verbosity: int = 1,
) -> Optional[float]:
    """
    Basic scoring function reflecting logic from compute-model-score script without additional logging messages.

    Parameters
    ----------
    e
    f
    lm
    tm
    configured_verbosity

    Returns
    -------

    """
    total_logprob = 0
    lm_logprob = score_lm(e, lm, configured_verbosity)
    total_logprob += lm_logprob
    alignments = generate_all_alignments(f, e, tm, configured_verbosity)
    chart = calculate_chart_over_alignments(e, f, alignments, configured_verbosity)

    goal = bitmap(range(len(f)))
    if goal in chart[len(e)]:
        total_logprob += chart[len(e)][goal]
        return total_logprob

    return None


# Logging helper functions

def print_verbose_statistics(local_start: float, local_end: float,
                             winner_hypothesis: Union[widec_hypothesis, hypothesis]):
    """

    Parameters
    ----------
    local_start
    local_end
    winner_hypothesis

    Returns
    -------

    """
    sys.stderr.write(f'%f seconds elapsed: ' % (local_end - local_start))
    tm_logprob = extract_tm_logprob(winner_hypothesis)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
                     (winner_hypothesis.logprob - tm_logprob, tm_logprob, winner_hypothesis.logprob))


def maybe_write(s: str, threshold_verbosity: int, configured_verbosity: int):
    if configured_verbosity > threshold_verbosity:
        sys.stdout.write(s)
        sys.stdout.flush()
