from datasets import load_dataset, load_from_disk, Dataset
import torch
import transformers
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Tuple, Callable
import logging
import plotly.express as px
import plotly.graph_objects as go
import json
import pickle
import re
import os
from random import shuffle

from utils.sae_loading import get_model
from sklearn.metrics import accuracy_score, f1_score
from sklearn import tree
import matplotlib.pyplot as plt
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import TransformersNlpEngine

from utils.tree_loader import get_root_node, get_tree_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = "YOUR_HF_TOKEN"

PII_LABELS_TO_ID = {
    "TIME": 0,
    "USERNAME": 1,
    "EMAIL": 2,
    "IDCARD": 3,
    "SOCIALNUMBER": 4,
    "LASTNAME": 5,
    "PASSPORT": 6,
    "DRIVERLICENSE": 7,
    "BOD": 8,
    "IP": 9,
    "GIVENNAME": 10,
    "CITY": 11,
    "STATE": 12,
    "TITLE": 13,
    "SEX": 14,
    "POSTCODE": 15,
    "BUILDING": 16,
    "STREET": 17,
    "TEL": 18,
    "DATE": 19,
    "COUNTRY": 20,
    "PASS": 21,
    "SECADDRESS": 22,
    "GEOCOORD": 23,
    "CARDISSUER": 24,
    "O": 25,
}


def PII_tree(
    save_path: str,
    create_latents: Callable = None,
) -> None:
    assert create_latents is not None, "create_latents function must be provided"
    root_nodes = PII_LABELS_TO_ID.copy()

    for concept in tqdm(PII_LABELS_TO_ID):
        if concept == "O" or os.path.exists(f"{save_path}trees/pii_tree_{concept}.pkl"):
            continue

        ls, labels = create_latents(concept)

        class_weight = False
        clf = tree.DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            class_weight="balanced" if class_weight else None,
        )
        clf = clf.fit(ls, labels)
        root_nodes[concept] = get_root_node(tree_model=clf)
        tree_stats = get_tree_stats(clf=clf)
        tree_stats.to_csv(f"{save_path}trees/pii_tree_{concept}.csv")
        s = pickle.dumps(clf)
        with open(
            f"{save_path}trees/pii_tree_{concept}pkl",
            "wb",
        ) as f:
            f.write(s)
        logger.info(
            f"{concept}, {PII_LABELS_TO_ID[concept]}, F1 for latent feature {root_nodes[concept]['feature_index']}: {tree_stats['F1_micro'][1].item()}"
        )

    with open(
        f"{save_path}root_nodes/root_nodes.json",
        "w",
    ) as f:
        json.dump(root_nodes, f)


def cut_PII_tree(
    save_path: str,
    label_shuffle: bool = False,
    create_latents: Callable = None,
) -> None:
    assert create_latents is not None, "create_latents function must be provided"

    for concept in tqdm(PII_LABELS_TO_ID):
        if concept == "O" or os.path.exists(
            f"{save_path}trees/pii_tree_{concept}{'_r' if label_shuffle else ''}_cut.csv"
        ):
            continue

        ls, labels = create_latents(concept)
        if label_shuffle:
            shuffle(labels)

        res = pd.DataFrame()
        root_node = None
        root_nodes = []
        for i in tqdm(range(10)):
            ls_new = []
            if root_node is not None:
                for l in ls:
                    l[root_node] = 0
                    ls_new.append(l)
                ls = ls_new

            clf = tree.DecisionTreeClassifier(
                criterion="gini",
                max_depth=3,
            )
            clf = clf.fit(ls, labels)

            tree_stats = get_tree_stats(clf=clf)
            root_node = get_root_node(tree_model=clf)["feature_index"]

            tree_stats["num_cuts"] = len(root_nodes)
            res = pd.concat([res, tree_stats])

            root_nodes.append(root_node)

        res.to_csv(
            f"{save_path}trees/pii_tree_{concept}{'_r' if label_shuffle else ''}_cut.csv"
        )


def RTP_tree(
    file,
    all: bool = True,
    class_weight: bool = False,
    tox_levels: tuple = (0.5, 0.5),
    create_latents: Callable = None,
) -> None:
    assert create_latents is not None, "create_latents function must be provided"
    if os.path.exists(f"{file}_statsV2.csv"):
        return

    ls, labels = create_latents(all, tox_levels)
    clf = tree.DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        class_weight="balanced" if class_weight else None,
    )
    clf = clf.fit(ls, labels)
    print(clf.get_depth())

    res = {"depth": []}
    res["depth"].append(clf.get_depth())

    plt.figure(figsize=(40, 20))
    tree.plot_tree(
        clf,
        proportion=False,
        class_names=["non toxic", "toxic"],
        filled=True,
        max_depth=3,
    )
    plt.savefig(file)
    s = pickle.dumps(clf)
    with open(
        f"{file}.pkl",
        "wb",
    ) as f:
        f.write(s)

    df = pd.DataFrame.from_dict(res)
    df.to_csv(f"{file}.csv")

    df = get_tree_stats(clf=clf)
    df.to_csv(f"{file}_statsV2.csv")


def cut_RTP_tree(
    file,
    all: bool = True,
    class_weight: bool = False,
    tox_levels: tuple = (0.5, 0.5),
    label_shuffle: bool = False,
    create_latents: Callable = None,
) -> None:
    assert create_latents is not None, "create_latents function must be provided"
    if os.path.exists(f"{file}{'_r' if label_shuffle else ''}_cut.csv"):
        return

    ls, labels = create_latents(all, tox_levels)

    if label_shuffle:
        shuffle(labels)

    res = pd.DataFrame()
    root_node = None
    root_nodes = []
    for i in tqdm(range(10)):
        ls_new = []
        if root_node is not None:
            for l in ls:
                l[root_node] = 0
                ls_new.append(l)
            ls = ls_new

        clf = tree.DecisionTreeClassifier(
            criterion="gini",
            max_depth=3,
            class_weight="balanced" if class_weight else None,
        )
        clf = clf.fit(ls, labels)

        tree_stats = get_tree_stats(clf=clf)
        root_node = get_root_node(tree_model=clf)["feature_index"]

        tree_stats["num_cuts"] = len(root_nodes)
        res = pd.concat([res, tree_stats])

        root_nodes.append(root_node)

    res.to_csv(f"{file}{'_r' if label_shuffle else ''}_cut.csv")


def SP_tree(
    file,
    valid: bool = True,
    create_latents: Callable = None,
) -> None:
    assert create_latents is not None, "create_latents function must be provided"
    if os.path.exists(f"{file}_statsV2.csv"):
        return
    ls, labels = create_latents(valid)

    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=None)
    clf = clf.fit(ls, labels)
    print(clf.get_depth())

    res = {"depth": []}
    res["depth"].append(clf.get_depth())

    plt.figure(figsize=(40, 20))
    tree.plot_tree(
        clf,
        proportion=False,
        class_names=["modern", "original"],
        filled=True,
        max_depth=3,
    )
    plt.savefig(file)
    s = pickle.dumps(clf)
    with open(
        f"{file}.pkl",
        "wb",
    ) as f:
        f.write(s)

    df = pd.DataFrame.from_dict(res)
    df.to_csv(f"{file}.csv")

    df = get_tree_stats(clf=clf)
    df.to_csv(f"{file}_statsV2.csv")


def cut_SP_tree(
    file,
    valid: bool = True,
    label_shuffle: bool = False,
    create_latents: Callable = None,
) -> None:
    assert create_latents is not None, "create_latents function must be provided"
    if os.path.exists(f"{file}{'_r' if label_shuffle else ''}_cut.csv"):
        return

    ls, labels = create_latents(valid)

    if label_shuffle:
        shuffle(labels)

    res = pd.DataFrame()
    root_node = None
    root_nodes = []
    for i in tqdm(range(10)):
        ls_new = []
        if root_node is not None:
            for l in ls:
                l[root_node] = 0
                ls_new.append(l)
            ls = ls_new

        clf = tree.DecisionTreeClassifier(
            criterion="gini",
            max_depth=3,
        )
        clf = clf.fit(ls, labels)

        tree_stats = get_tree_stats(clf=clf)
        root_node = get_root_node(tree_model=clf)["feature_index"]

        tree_stats["num_cuts"] = len(root_nodes)
        res = pd.concat([res, tree_stats])

        root_nodes.append(root_node)

    res.to_csv(f"{file}{'_r' if label_shuffle else ''}_cut.csv")
