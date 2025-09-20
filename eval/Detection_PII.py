from datasets import (
    load_dataset,
    load_from_disk,
)
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

from utils.sae_loading import get_model
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
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


def get_input_ids_and_labels_PII(text: str, spans: list) -> Tuple[torch.Tensor, list]:
    all_input_ids = torch.tensor([[tokenizer.bos_token_id]])
    labels = ["O"]
    for span in spans:
        input_ids = tokenizer(
            text[span[0] : span[1]],
            return_tensors="pt",
            truncation=True,
            padding=True,
            add_special_tokens=False,
        ).input_ids
        all_input_ids = torch.concat([all_input_ids, input_ids], dim=1)
        labels += [span[2] for _ in range(len(input_ids[0]))]

    return all_input_ids, labels


def convert_to_sequence(input_list, total_length):
    """
    Convert a list of labeled segments into a sequential list with gaps labeled as 'O'.

    :param input_list: List of dictionaries with 'value', 'start', 'end', and 'label' keys.
    :param total_length: Total length of the sequence.
    :return: Transformed list with gaps labeled as 'O'.
    """
    result = []
    previous_end = 0

    for entry in input_list:
        start = entry["start"]
        end = entry["end"]
        label = entry["label"]

        # Remove numbers from the label
        label = "".join(filter(lambda x: not x.isdigit(), entry["label"]))

        # Add gap with label 'O' if there is a gap between entries
        if previous_end < start:
            result.append([previous_end, start, "O"])

        # Add the current entry
        result.append([start, end, label])

        # Update previous_end to the end of the current segment
        previous_end = end

    # Add trailing gap with label 'O' if needed
    if previous_end < total_length:
        result.append([previous_end, total_length, "O"])

    return result


dataset = load_dataset(
    "ai4privacy/pii-masking-300k", split="validation", token=HF_TOKEN
)
dataset = dataset.filter(lambda x: x["language"] == "en" or x["language"] == "English")

load_from_ckpt = True
ckpt_dirs = {
    "Checkpoint Path": "Result Dir",
}
ckpt_dir, save_dir = list(ckpt_dirs.items())[-1]
save_path = (
    f"./llama3_SAE/SAE_eval/{save_dir}/"
)
if __name__ == "__main__":
    for model_name in [
        f"llama3-l{str(j)}-b{str(k)}-k{str(i)}{l}"
        for i in [2048]
        for j in [24576]
        for k in [11]
        for l in ["_s0", "_s1"]
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
            act="topk-sigmoid",
            epoch=100,
        )
        llama3_sae.eval()
        model_name += scaling
        with open(
            f"{save_path}root_nodes/root_nodes-{model_name}.json",
            "r",
        ) as f:
            root_nodes = json.load(f)

        latent_features = [
            j["feature_index"] for i, j in root_nodes.items() if i != "O"
        ]

        res = defaultdict(list)
        preds_list = []
        label_ids_list = []

        for sent in tqdm(dataset):
            text = sent["source_text"]
            spans = convert_to_sequence(sent["privacy_mask"], len(text))
            input_ids, labels = get_input_ids_and_labels_PII(text, spans)
            label_ids = [PII_LABELS_TO_ID[label] for label in labels]
            llama3_sae(input_ids.cuda())
            acts = llama3_sae.hook.pop()
            pre_act, latents, recons = llama3_sae.SAE(acts)

            values, preds = latents[0, :, latent_features].max(1)
            filter_for_class_O = (latents[0, :, latent_features] == 0).all(1)
            preds[filter_for_class_O] = 25
            values[filter_for_class_O] = 0.0

            f1_micro_pre_sentense = f1_score(
                label_ids,
                preds.cpu().numpy(),
                average="micro",
                labels=[i for i in range(26)],
            ).item()

            label_ids_list += label_ids
            preds_list += preds.cpu().numpy().tolist()
            res["text"].append(text)
            res["f1_micro"].append(f1_micro_pre_sentense)
            res["preds"].append(preds.cpu().detach().numpy().tolist())
            res["labels"].append(label_ids)
            res["values"].append(values.cpu().detach().numpy().tolist())

        df = pd.DataFrame(res)
        df.to_csv(f"{save_path}pii_detection-{model_name}.csv")

        F1_micro = f1_score(
            label_ids_list,
            preds_list,
            average="micro",
            labels=[i for i in range(26)],
        ).item()

        F1_macro = f1_score(
            label_ids_list,
            preds_list,
            average="macro",
            labels=[i for i in range(26)],
        ).item()

        F1_micro_classes = f1_score(
            label_ids_list,
            preds_list,
            average=None,
            labels=[i for i in range(26)],
        )

        acc = accuracy_score(label_ids_list, preds_list)
        acc_balanced = balanced_accuracy_score(label_ids_list, preds_list)

        f1s = root_nodes.copy()
        for concept, value in zip(f1s.keys(), F1_micro_classes.tolist()):
            f1s[concept] = value
        f1s["F1_micro"] = F1_micro
        f1s["F1_macro"] = F1_macro
        f1s["acc"] = acc
        f1s["acc_balanced"] = acc_balanced

        with open(
            f"{save_path}pii_detection_scores-{model_name}.json",
            "w",
        ) as f:
            json.dump(f1s, f)
