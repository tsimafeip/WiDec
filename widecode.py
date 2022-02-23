#!/usr/bin/env python
import optparse
import sys
from hw3 import models
from collections import namedtuple, defaultdict

'''
Algorithm

1. Find all possible translations from TM for all possible phrases.
 Each translation phrase can be a start, the goal is to cover all input words.
2. Introduce reordering penalty to estimation process.
3. Model beam search with optimizations.

'''

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="hw3/data/input",
                     help="File containing sentences to translate (default=hw3/data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="hw3/data/tm",
                     help="File containing translation model (default=hw3/data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="hw3/data/lm",
                     help="File containing ARPA-format language model (default=hw3/data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxsize, type="int",
                     help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=sys.maxsize, type="int",
                     help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,
                     help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french, ())):
    if (word,) not in tm:
        tm[(word,)] = [models.phrase(word, 0.0)]

hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase")
initial_hypothesis = hypothesis(0.0, lm.begin(), None, None)

for f in french:
    all_phrases_translations = defaultdict(list)
    # 1) Generate all possible french spans
    # 2) Collect all possible translations
    for fi in range(len(f)):
        for fj in range(fi + 1, len(f) + 1):
            french_span = f[fi:fj]
            english_translations = tm.get(french_span, [])
            for phrase_obj in english_translations:
                all_phrases_translations[french_span].append(phrase_obj)

    stacks = [{} for _ in f] + [{}]
    stacks[0][lm.begin()] = initial_hypothesis

    # iterates by all stacks except the last one which contains the result
    for i, stack in enumerate(stacks[:-1]):
        # iterate by all hypotheses covering i-words
        for h in stack.values():
            pass
