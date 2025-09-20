from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
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
import gc

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


def PII_latents(
    concept: str, dataset: Dataset, extract_func: Callable = None
) -> Tuple[list, list]:
    assert extract_func is not None, "extract_func must be provided"

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
        latents = extract_func(line)
        pos_latents.append(latents.detach().cpu())

    neg_latents = []
    for line in neg_dataset:
        latents = extract_func(line)
        neg_latents.append(latents.detach().cpu())

    ls = pos_latents + neg_latents
    labels = [1 for _ in range(len(pos_latents))] + [0 for _ in range(len(neg_latents))]

    return ls, labels


def RTP_latents(
    all: bool = True,
    tox_levels: tuple = (0.5, 0.5),
    extract_func: Callable = None,
) -> Tuple[list, list]:
    assert extract_func is not None, "extract_func must be provided"

    test = load_from_disk("./datasets/RTP")[
        "test"
    ]
    if all:
        dataset_toxic = test.filter(
            lambda x: x["prompt"]["toxicity"] is not None
            and x["prompt"]["toxicity"] >= tox_levels[1]
        )
        dataset_non_toxic = test.filter(
            lambda x: x["prompt"]["toxicity"] is not None
            and x["prompt"]["toxicity"] < tox_levels[0]
        )
    else:
        dataset_toxic = test.filter(
            lambda x: x["prompt"]["toxicity"] is not None
            and x["prompt"]["toxicity"] >= 0.75
        )
        dataset_non_toxic = test.filter(
            lambda x: x["prompt"]["toxicity"] is not None
            and x["prompt"]["toxicity"] <= 0.0456735
        )

    def preprocess(example):
        is_toxic = example["prompt"]["toxicity"] >= 0.75
        return {
            "text": example["prompt"]["text"],
            "is_toxic": is_toxic,
            "toxcitity_level": example["prompt"]["toxicity"],
        }

    dataset_toxic = dataset_toxic.map(preprocess, num_proc=4)
    dataset_toxic = dataset_toxic.remove_columns(
        [
            "filename",
            "begin",
            "end",
            "challenging",
            "prompt",
            "continuation",
        ]
    )
    dataset_non_toxic = dataset_non_toxic.map(preprocess, num_proc=4)
    dataset_non_toxic = dataset_non_toxic.remove_columns(
        [
            "filename",
            "begin",
            "end",
            "challenging",
            "prompt",
            "continuation",
        ]
    )

    toxic_latents = []
    with torch.no_grad():
        for line in tqdm(dataset_toxic):
            latents = extract_func(line["text"])
            toxic_latents.append(latents.detach().cpu())

    torch.cuda.empty_cache()
    gc.collect()

    non_toxic_latents = []
    with torch.no_grad():
        for line in tqdm(dataset_non_toxic):
            latents = extract_func(line["text"])
            non_toxic_latents.append(latents.detach().cpu())

    torch.cuda.empty_cache()
    gc.collect()

    toxic_l_mean = []
    for latents in tqdm(toxic_latents):
        l_mean = latents.mean(1)
        toxic_l_mean.append(l_mean.detach().cpu().numpy()[0])

    non_toxic_l_mean = []
    for latents in tqdm(non_toxic_latents):
        l_mean = latents.mean(1)
        non_toxic_l_mean.append(l_mean.detach().cpu().numpy()[0])

    ls = toxic_l_mean + non_toxic_l_mean
    labels = [1 for _ in range(len(toxic_l_mean))] + [
        0 for _ in range(len(non_toxic_l_mean))
    ]

    return ls, labels


def SP_latents(valid: bool = True, extract_func: Callable = None) -> Tuple[list, list]:
    assert extract_func is not None, "extract_func must be provided"
    test = load_from_disk(
        "./datasets/Shakespeare"
    )[f"{'valid' if valid else 'test'}"]

    test = test.map(lambda x: {"label": 0 if x["label"] == "modern" else 1})
    dataset_modern = test.filter(lambda x: x["label"] == 0)
    dataset_original = test.filter(lambda x: x["label"] == 1)

    original_latents = []
    with torch.no_grad():
        for line in tqdm(dataset_original):
            latents = extract_func(line["text"])
            original_latents.append(latents.detach().cpu())

    modern_latents = []
    with torch.no_grad():
        for line in tqdm(dataset_modern):
            latents = extract_func(line["text"])
            modern_latents.append(latents.detach().cpu())

    modern_l_mean = []
    for latents in tqdm(modern_latents):
        l_mean = latents.mean(1)
        modern_l_mean.append(l_mean.detach().cpu().numpy()[0])

    original_l_mean = []
    for latents in tqdm(original_latents):
        l_mean = latents.mean(1)
        original_l_mean.append(l_mean.detach().cpu().numpy()[0])

    ls = modern_l_mean + original_l_mean
    labels = [0 for _ in range(len(modern_l_mean))] + [
        1 for _ in range(len(original_l_mean))
    ]

    return ls, labels
