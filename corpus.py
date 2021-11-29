import json
import sys

# import numpy as np
import pandas as pd

import codecs
import dask.array as da
import h5py

from config import load_config_paths

corpora_base, preproc_path, dsgv_home = load_config_paths()
sys.path.append(dsgv_home + "/Utils")
from utils import query_by_id
from data_utils import load_dfs
from apply_utils import apply_wac_set_matrix, logreg

sys.path.append(dsgv_home + "/WACs/WAC_Utils")
from wac_utils import (
    create_word2den,
    is_relational,
    filter_refdf_by_filelist,
    filter_relational_expr,
)
from wac_utils import filter_X_by_filelist, make_mask_matrix, make_X_id_index


MODEL_DFS = {
    "referit": {"refdf": "saiapr_refdf", "bbdf": "saiapr_bbdf"},
    "refcoco": {"refdf": "refcoco_refdf", "bbdf": "mscoco_bbdf"},
    "refcocoplus": {"refdf": "refcocoplus_refdf", "bbdf": "mscoco_bbdf"},
    "grex": {"refdf": "grex_refdf", "bbdf": "mscoco_bbdf"},
}


def get_corpus_dfs(*args):
    datasets = args
    df_names = list(set([x for d in datasets for x in MODEL_DFS[d].values()]))
    df = load_dfs(preproc_path, df_names)
    if "mscoco_catsdf" in args:
        cococat = dict(
            zip(df["mscoco_catsdf"].index, df["mscoco_catsdf"]["cat"].values)
        )
        return df, cococat
    else:
        return df


def get_model_idx(
    X: da.core.Array,
    dataset_name: str,
    big_dfs: dict,
    *splits_filename,
    which_split="all",
):
    if len(splits_filename) > 1:
        raise ValueError("get_model_idx can only accept one splits file")
    elif len(splits_filename) == 1:
        with open(preproc_path + "/" + splits_filename + ".json", "r") as f:
            splits = json.load(f)
    else:
        correspondence = {
            "referit": "saiapr_90-10_splits.json",
            "refcoco": "refcoco_splits.json",
            "refcocoplus": "refcoco_splits.json",
            "grex": "google_refexp_rexsplits.json",
        }
        if dataset_name not in correspondence:
            raise ValueError(
                "non-standard dataset - please explicitly pass splits filename"
            )
        with open(preproc_path + "/" + correspondence[dataset_name], "r") as f:
            splits = json.load(f)
    if which_split == "all":
        right_splits = [item for k in splits for item in splits[k]]
    elif which_split == "train":
        right_splits = [
            item for k in splits for item in splits[k] if k.startswith("train")
        ]
    elif which_split == "test":
        right_splits = [
            item for k in splits for item in splits[k] if k.startswith("test")
        ]
    elif which_split == "val":
        right_splits = [
            item for k in splits for item in splits[k] if k.startswith("val")
        ]
    else:
        raise ValueError("which_split must be either train, test, val, or all")
    if not right_splits:
        raise ValueError("chosen split does not exist in this dataset")
    X_ts = filter_X_by_filelist(X, right_splits)
    refdf = filter_refdf_by_filelist(
        big_dfs[MODEL_DFS[dataset_name]["refdf"]], right_splits
    )

    word2den_ts = create_word2den(refdf)
    X_idx_ts = make_X_id_index(X_ts)
    mask_matrix_ts = make_mask_matrix(X_ts, X_idx_ts, word2den_ts, word2den_ts.keys())
    return X_ts, X_idx_ts, mask_matrix_ts


def get_image_feats():
    mscoco_bbdf_pattern = (
        "../_data/Models/ForBToma/mscoco_bbdf_rsn50-max/mscoco_bbdf_rsn50-max_%d.hdf5"
    )
    model_path_prefix = "../_data/Models/ForBToma/01_refcoco_rsn"
    das = []
    fhs = []
    for n in range(1, 8):
        f = h5py.File(mscoco_bbdf_pattern % (n), "r")
        fhs.append(f)
        das.append(da.from_array(f["img_feats"], chunks=(1000, 4106)))
    X = da.concatenate(das)

    with h5py.File(model_path_prefix + ".hdf5", "r") as f:
        wacs = f["wac_weights"][:]  # slice, to actually read into memory (as ndarray)

    with codecs.open(model_path_prefix + ".json", "r") as f:
        modelpars, wordlist = json.load(f)

    return X, wacs, modelpars, wordlist


def get_classifications(big_dfs, *datasets, ID_FEATS=3, which_split="all"):
    X, wacs, _, wordlist = get_image_feats()
    applied = {}
    for data in datasets:
        X_ts, X_idx_ts, _ = get_model_idx(X, data, big_dfs, which_split=which_split)
        all_applied = apply_wac_set_matrix(
            X_ts[:, ID_FEATS:], wacs.T, net=logreg
        )  # applies all wac and returns a matrix of results, rows are entities and columns are words
        word2ind = {w[0]: n for n, w in enumerate(wordlist)}
        applied[data] = (all_applied, word2ind, X_idx_ts, wordlist)
    return applied


def get_region_info(df: dict, dataset_name: str):
    refdf = MODEL_DFS[dataset_name]["refdf"]
    bbdf = MODEL_DFS[dataset_name]["bbdf"]
    corp_im = (
        df[refdf]
        .drop_duplicates(subset=["i_corpus", "image_id"], ignore_index=True)
        .loc[:, ["i_corpus", "image_id"]]
    )
    combos = list(zip(corp_im["i_corpus"], corp_im["image_id"]))
    region_data = {
        (ic, ii): {
            r[1]: {
                "bb": r[0],
                "refexp": list(query_by_id(df[refdf], (ic, ii, r[1]), "refexp")),
            }
            for r in query_by_id(df[bbdf], (ic, ii), ["bb", "region_id"]).values
        }
        for ic, ii in combos
    }
    all_refexp = {
        ii: {refexp: ri for (ri, d) in v.items() for refexp in d["refexp"]}
        for (ii, v) in region_data.items()
    }
    return region_data, all_refexp


def exp2indseq(word2ind, exp):
    if isinstance(exp, str):
        return [word2ind[w] for w in exp.split() if w in word2ind]
    elif isinstance(exp, list):
        return [word2ind[w] for w in exp if w in word2ind]
    else:
        raise TypeError(
            f"exp can only take list or string, but provided exp is type {type(exp)}"
        )
    # returns the list of indices for each "word" in exp


def imageid2rows(region_data, idx, ic, ii):
    """return all regions that belong to an image, as indices into X (via idx)"""
    # or should this be a separate dictionary?
    rows = [
        (ri, idx[(ic, ii, ri)]) for ri in region_data[(ic, ii)] if (ic, ii, ri) in idx
    ]
    return map(list, zip(*rows))


def regionid2row(idx, ic, ii, ri):
    return idx[(ic, ii, ri)]
