from datasets import load_dataset, load_from_disk, Dataset
import torch
import transformers
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Tuple
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
test_dataset = load_from_disk(
    "./datasets_v2/llama3-PII_300k-B11-block",
)["test"].with_format("torch")
test_dataset = test_dataset.map(
    lambda x: {
        "label_num": x["label"].nonzero()[0, 0].item() if x["label"].sum() > 0 else 25
    },
    num_proc=32,
)

load_from_ckpt = True
ckpt_dirs = {
    "Checkpoint Path": "Result Dir",
}
ckpt_dir, save_dir = list(ckpt_dirs.items())[-1]
save_path = (
    f"./llama3_SAE/SAE_eval/{save_dir}/"
)


def PII_latents(
    llama3_sae, tokenizer, concept: str, dataset: Dataset
) -> Tuple[list, list]:
    pos_dataset = dataset.filter(
        lambda x: x["label_num"] == PII_LABELS_TO_ID[concept], num_proc=1
    )
    len_pos = len(pos_dataset)
    neg_dataset = (
        dataset.filter(
            lambda x: x["label_num"] != PII_LABELS_TO_ID[concept], num_proc=1
        )
        .shuffle(42)
        .select(range(len_pos))
    )

    pos_latents = []
    for line in pos_dataset:
        acts = line["acts"].to(dtype=torch.float16)
        latents = llama3_sae.SAE(acts)[1]
        pos_latents.append(latents.detach().cpu())

    neg_latents = []
    for line in neg_dataset:
        acts = line["acts"].to(dtype=torch.float16)
        latents = llama3_sae.SAE(acts)[1]
        neg_latents.append(latents.detach().cpu())

    ls = pos_latents + neg_latents
    labels = [1 for _ in range(len(pos_latents))] + [0 for _ in range(len(neg_latents))]

    return ls, labels


def PII_tree(
    llama3_sae, tokenizer, model_name: str, save_path: str, test_dataset: Dataset
) -> None:
    for concept in tqdm(PII_LABELS_TO_ID):
        if concept == "O" or os.path.exists(
            f"{save_path}trees/pii_tree_{concept}_{model_name}.pkl"
        ):
            continue

        ls, labels = PII_latents(llama3_sae, tokenizer, concept, test_dataset)

        class_weight = False
        clf = tree.DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            class_weight="balanced" if class_weight else None,
        )
        clf = clf.fit(ls, labels)
        root_nodes[concept] = get_root_node(tree_model=clf)
        tree_stats = get_tree_stats(clf=clf)
        tree_stats.to_csv(f"{save_path}trees/pii_tree_{concept}_{model_name}.csv")
        s = pickle.dumps(clf)
        with open(
            f"{save_path}trees/pii_tree_{concept}_{model_name}.pkl",
            "wb",
        ) as f:
            f.write(s)
        logger.info(
            f"{concept}, {PII_LABELS_TO_ID[concept]}, F1 for latent feature {root_nodes[concept]['feature_index']}: {tree_stats['F1_micro'][1].item()}"
        )

    with open(
        f"{save_path}root_nodes/root_nodes-{model_name}.json",
        "w",
    ) as f:
        json.dump(root_nodes, f)


def cut_PII_tree(
    llama3_sae,
    tokenizer,
    model_name: str,
    save_path: str,
    test_dataset: Dataset,
    label_shuffle: bool = False,
) -> None:
    for concept in tqdm(PII_LABELS_TO_ID):
        if concept == "O" or os.path.exists(
            f"{save_path}trees/pii_tree_{concept}_{model_name}{'r' if label_shuffle else ''}_cut.csv"
        ):
            continue

        ls, labels = PII_latents(llama3_sae, tokenizer, concept, test_dataset)
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
            f"{save_path}trees/pii_tree_{concept}_{model_name}{'r' if label_shuffle else ''}_cut.csv"
        )


if __name__ == "__main__":
    for model_name in [
        f"llama3-l{str(j)}-b{str(k)}-k{str(i)}{l}"
        for i in [192]
        for j in [131072]
        for k in [11]
        for l in ["_s0"]
    ]:
        scaling = (re.findall(r"_s\d+", model_name) or ["_s1"])[0]
        model_name = model_name.replace(scaling, "")
        llama3_sae, tokenizer = None, None
        llama3_sae, tokenizer = get_model(
            model_name,
            load_from_ckpt,
            scaling,
            "block",
            ckpt_dir,
            act="topk-sigmoid" if save_dir != "pretrained" else "topk",
            epoch=100,
        )

        model_name += scaling
        root_nodes = PII_LABELS_TO_ID.copy()

        PII_tree(llama3_sae, tokenizer, model_name, save_path, test_dataset)
        cut_PII_tree(llama3_sae, tokenizer, model_name, save_path, test_dataset)
        cut_PII_tree(
            llama3_sae,
            tokenizer,
            model_name,
            save_path,
            test_dataset,
            label_shuffle=True,
        )
