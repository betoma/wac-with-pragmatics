import itertools
from collections import defaultdict

import numpy as np

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize

# from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))


with open("prepositions.txt", "r") as f:
    prepositions = [x.strip() for x in f.readlines()]


def generate_pos_replacements(vocabulary) -> dict[list[str]]:
    pos_replacements = defaultdict(set)
    for word in vocabulary:
        if word in prepositions:
            pos_replacements["IN"].add(word)
        if lex_list := wn.synsets(word):
            pos_set = set([s.pos() for s in lex_list])
            if "a" in pos_set or "s" in pos_set:
                pos_replacements["JJ"].add(word)
            if "n" in pos_set:
                pos_replacements["NN"].add(word)
    return {k: list(v) for k, v in pos_replacements.items()}


def generate_baseline_utterances(pos_replacements: dict[list[str]]) -> list[tuple[str]]:
    baseline_utterances = set()
    for pair in itertools.product(pos_replacements["JJ"], pos_replacements["NN"]):
        baseline_utterances.add(frozenset(pair))
    return baseline_utterances


def generate_specific_utterances(
    refexp: str, pos_replacements: dict[list[str]], vocabulary
):
    text = pos_tag(word_tokenize(refexp))
    output_text = [(x, y) for (x, y) in text if x in vocabulary and x not in stop_words]
    if len(output_text) > 2:
        pattern = []
        for word, tag in output_text:
            if tag in {"NN", "NNS", "NNP", "NNPS"}:
                pattern.append("NN")
            elif tag in {"JJ", "JJR", "JJS"}:
                pattern.append("JJ")
            else:
                pattern.append(word)
        lists_for_product = [
            pos_replacements[x] if x in {"NN", "JJ"} else [x] for x in pattern
        ]
        return itertools.product(*lists_for_product)
    else:
        return None
