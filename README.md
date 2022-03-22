#  WiDec (Wizard Decoder) :mage_man:

This project aims to implement an effective decoding step for Statistical Machine Translation.

It is based on HW3 ([description](http://mt-class.org/jhu/hw2.html), [github](https://github.com/xutaima/jhu-mt-hw/tree/master/hw3)) from [JHU Machine Translation class](http://mt-class.org/jhu/).

Structure of folders is the following:
- ***'data'***
  - input French sentences
  - language model in [ARPA format](https://cmusphinx.github.io/wiki/arpaformat/)
  - translation model
- ***'meta'*** - meta-information, currently here is only report file with full description of the project
- ***'model_translations'*** - translations produced by different decoding algorithms
- ***'src/cpp'*** - initial version of cpp code, currently only translation model is implemented
- ***'src/py'*** - main code repository in Python
  There are several python programs here (-h for usage):
  - `decode` translates input sentences from French to English using **monotone** decoding.
  - `widecode` translates input sentences from French to English using beam **search** decoding.
  - `widecode_greedy` translates input sentences from French to English using **greedy** decoding.
  - `compute-model-score` computes the model score of a translated sentence.
  - `helper.py` holds common functions for different models in one place.
  - `models.py` implements very simple interfaces for language models and translation models.

  These commands work in a pipeline or via files. For example:
    > python3 decode | python3 compute-model-score <br>
    > python3 decode > output.txt <br>
    > python3 compute-model-score < output.txt <br>


