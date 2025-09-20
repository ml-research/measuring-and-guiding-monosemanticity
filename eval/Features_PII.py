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
from torch.utils.data import DataLoader

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

test_dataset = load_from_disk(
    "./datasets_v2/llama3-PII_300k-B11-block",
)["test"].with_format("torch")
test_dataset = test_dataset.map(
    lambda x: {
        "label_num": x["label"].nonzero()[0, 0].item() if x["label"].sum() > 0 else 25
    },
    num_proc=32,
)
dataset = DataLoader(test_dataset, batch_size=2048, num_workers=16)
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

        for line in tqdm(dataset):
            acts, labels = line["acts"].cuda().to(dtype=torch.float16), line["label"]
            label_names = [
                list(PII_LABELS_TO_ID.keys())[i.argmax()] if i.sum() > 0 else "O"
                for i in labels
            ]
            pre_act, latents, recons = llama3_sae.SAE(acts)
            res["label"] += label_names
            for concept, feat in PII_LABELS_TO_ID.items():
                if concept == "O":
                    continue

                res[f"{concept}_feature_value"] += (
                    latents[:, feat].cpu().detach().numpy().tolist()
                )
                res[f"{concept}_pre_feature_value"] += (
                    pre_act[:, feat].cpu().detach().numpy().tolist()
                )

        df = pd.DataFrame(res)
        df.to_csv(f"{save_path}pii_features-{model_name}.csv")
