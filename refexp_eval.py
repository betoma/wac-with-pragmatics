from itertools import combinations
from typing import Callable

import numpy as np
import spacy
from sklearn.preprocessing import normalize

import corpus

# This file is going to contain the code that takes in the classifier outputs and the refexp(s) and does the actual stuff to turn that into the appropriate numbers

# Datasets:
# both refcoco and refcocoplus
# maybe the normal referit and grex or whatever too? check sc

# refexp Preprocessing Options:
# with/without relational expressions (is this the same as above?)
# with/without stopwords

# Approach differences:
# include random selection and selection of largest bb automatically as control (as in Schlangen 2016)
# Using product vs. using geometric mean
# More complex parsing/weighting vs. just using all the words
# ofc, inclusion of RSA or not

# Will I include the model that can actually handle relational expressions? Not sure, will have to ask Schlangen if it's included.


def geo_mean_overflow(iterable):
    a = np.log(iterable)
    return np.exp(a.mean())


def remove_wrong_words(
    exp,
    not_in_vocab: bool = True,
    wordlist: list = [],
    stopwords: bool = False,
    model: spacy.lang = None,
    extra_stopwords: set = None,
    removed_stopwords: set = None,
) -> list[str]:
    if stopwords:
        if not model:
            model = spacy.load("en_core_web_sm")
        all_stopwords = model.Defaults.stop_words
        if extra_stopwords:
            all_stopwords |= extra_stopwords
        if removed_stopwords:
            all_stopwords -= removed_stopwords
    if isinstance(exp, str):
        exp = exp.split()
    return [
        x
        for x in exp
        if (x in wordlist or not not_in_vocab)
        and (x not in all_stopwords or not stopwords)
    ]


def make_classic_res_arr(
    exp: str,
    ic: int,
    ii: int,
    applied: dict,
    dataset: str,
    region_data: dict,
):
    all_applied, word2ind, X_idx, _ = applied[dataset]
    reg_ids, region_rows = corpus.imageid2rows(region_data, X_idx, ic, ii)
    return (
        {i: r for i, r in enumerate(reg_ids)},
        all_applied[region_rows, :][:, corpus.exp2indseq(word2ind, exp)],
    )


def rsa_first_arr(
    exp: str,
    ic: int,
    ii: int,
    applied: dict,
    dataset: str,
    region_data: dict,
    put_it_together: Callable,
    no_stopwords: bool = False,
    stopwords_edits: tuple[set] = ({}, {}),
):
    all_applied, word2ind, X_idx, wordlist = applied[dataset]
    reg_ids, region_rows = corpus.imageid2rows(region_data, X_idx, ic, ii)
    if no_stopwords:
        model = spacy.load("en_core_web_sm")
    else:
        model = None
    exp = remove_wrong_words(
        exp,
        wordlist=wordlist,
        stopwords=no_stopwords,
        model=model,
        extra_stopwords=stopwords_edits[0],
        removed_stopwords=stopwords_edits[1],
    )
    n = len(exp)
    if no_stopwords:
        wordlist = remove_wrong_words(
            wordlist,
            not_in_vocab=False,
            stopwords=True,
            model=model,
            extra_stopwords=stopwords_edits[0],
            removed_stopwords=stopwords_edits[1],
        )
    possible_exp = []
    for i in range(n):
        possible_exp += combinations(wordlist, i + 1)
    matrix_list = []
    for r in region_rows:
        this_row = []
        for x in possible_exp:
            if len(x) == 1:
                result = all_applied[r, word2ind[x[0]]]
            else:
                results_per_word = all_applied[r, :][
                    corpus.exp2indseq(word2ind, " ".join(x))
                ]
                result = put_it_together(results_per_word)
            this_row.append(result)
        matrix_list.append(this_row)
    return (
        {i: r for i, r in enumerate(reg_ids)},
        np.array(matrix_list, dtype=np.float64),
        {r: i for i, r in enumerate([frozenset(x) for x in possible_exp])},
    )


def RSA_func(
    prior_arr: np.ndarray,
    possible_exp: dict,
    cost_func: Callable,
    alpha: float = 1.0,
    prag_priors: np.array = None,
):
    lit_list_arr = normalize(prior_arr, axis=1, norm="l1")

    n_rows = lit_list_arr.shape[0]
    cost_list = []
    for _ in range(n_rows):
        this_row = []
        for exp in possible_exp.values():
            this_row.append(cost_func(exp))
    cost_matrix = np.array(cost_list, dtype=np.float64)

    speak_vals = np.exp(alpha * (np.log(lit_list_arr) - cost_matrix))
    speak_arr = normalize(speak_vals, axis=0, norm="l1")

    if prag_priors is None:
        prag_priors = np.ones(speak_arr.shape)
    return normalize(speak_arr * prag_priors, axis=1, norm="l1")
