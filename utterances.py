import itertools
from collections import defaultdict
import random

from lemminflect import getInflection

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize


stop_words = set(stopwords.words("english"))

colors = {
    "black",
    "white",
    "brown",
    "gray",
    "pink",
    "red",
    "blue",
    "purple",
    "yellow",
    "green",
    "orange",
}
small_sizes = {"short", "little", "small", "smaller"}
large_sizes = {"large", "bigger", "tall", "taller", "big"}
locations = {"right", "front", "top", "middle", "bottom", "left", "back", "center"}
central_locations = {"middle", "center"}
non_central_locations = {"right", "front", "top", "bottom", "left", "back"}
patterns = {"striped", "checkered", "plaid"}


def vocab_synsets(
    vocab, noun: bool = True
) -> tuple[dict[wn.synset : set[str]], dict[str : wn.synset]]:
    syns_dict = defaultdict(set)
    syns_key = {}
    if noun:
        pos = wn.NOUN
    else:
        pos = wn.VERB
    for w in vocab:
        if w not in stop_words:
            word = wn.morphy(w, pos)
            if word is None:
                continue
            synset = None
            if w_set := wn.synsets(word, pos=pos):
                synset = w_set[0]
                syns_key[w] = synset
            syns_dict[synset].add(word)
    return syns_dict, syns_key


def produce_sim_scores(
    word: wn.synset, syns_dict: dict[wn.synset : set[str]]
) -> list[tuple[str, float]]:
    return sorted(
        [
            (syns_dict[sense], word.wup_similarity(sense))
            for sense in syns_dict
            if sense != word and sense not in word.hyponyms()
        ],
        key=lambda tup: tup[1],
        reverse=True,
    )


def produce_noun_alt_dict(
    syns_key: dict[str : wn.synset], syns_dict: dict[wn.synset : set[str]]
) -> dict[str : set[str]]:
    alt_dict = {}
    for word, synset in syns_key.items():
        alternatives = set()
        scores = produce_sim_scores(synset, syns_dict)
        if len(scores) > 20:
            alt_list = scores[:20]
        else:
            alt_list = scores
        for item in alt_list:
            alternatives.update(item[0])
        alt_dict[word] = alternatives
    return alt_dict


def produce_verb_alt_dict(
    syns_key: dict[str : wn.synset], syns_dict: dict[wn.synset : set[str]], vocab
) -> dict[str : list[str]]:
    alt_dict = {}
    for word, synset in syns_key.items():
        alternatives = set()
        ing = word.endswith("ing") and word not in {"sing", "bring", "cling"}
        for word_set in syns_dict.values():
            replace_list = []
            for w in word_set:
                if w not in synset.lemma_names() and w not in stop_words:
                    if ing:
                        if (progressive := getInflection(w, tag="VBG")[0]) in vocab:
                            replace_list.append(progressive)
                    elif w in vocab:
                        replace_list.append(w)
            alternatives.update(replace_list)
        alt_dict[word] = list(alternatives)
    return alt_dict


def generate_patterns(
    exp,
    vocab,
    noun_alt_dict: dict[str : set[str]],
    verb_alt_dict: dict[str : list[str]],
):
    if type(exp) == str:
        exp = word_tokenize(exp)
    elif type(exp) != list:
        raise TypeError("Expression must be provided as a string or a list")
    words_in_exp = set(exp)
    tagged = pos_tag(exp)
    filtered = [
        pair for pair in tagged if pair[0] in vocab and pair[0] not in stop_words
    ]
    split_exp = tuple([pair[0] for pair in filtered])
    pattern_list = []
    for word, tag in filtered:
        alt_set = set()
        if word in colors:
            alt_set.update(colors)
        elif word in small_sizes:
            alt_set.update(large_sizes)
        elif word in large_sizes:
            alt_set.update(small_sizes)
        elif word in central_locations:
            alt_set.update(non_central_locations)
        elif word in locations:
            alt_set.update(locations)
        elif word in patterns:
            alt_set.update(patterns)
        elif word in verb_alt_dict and tag in {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}:
            alt_set.update(random.sample(verb_alt_dict[word], 20))
        elif word in noun_alt_dict and tag in {"NN", "NNS", "NNP", "NNPS"}:
            alt_set.update(noun_alt_dict[word])
        else:
            pattern_list.append([word, ""])
            continue
        alt_set = alt_set.difference(words_in_exp)
        alt_set.add("")
        pattern_list.append(list(alt_set))
    return split_exp, pattern_list


def generate_utterances(pattern_list: list[list]):
    return itertools.product(*pattern_list)
