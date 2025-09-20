from datasets import (
    load_dataset,
    load_from_disk,
    concatenate_datasets,
    ClassLabel,
    Sequence,
    DatasetDict,
)
import torch
import transformers
from transformers import AutoModelForCausalLM
from huggingface_hub.utils import RevisionNotFoundError, HfHubHTTPError
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from utils.Toxicity_API import tox_classifier
import sys, os
import csv
from torch.utils.data import DataLoader
import json
from collections import defaultdict, Counter
from typing import Tuple
from glob import glob
import logging
from multiprocessing import Pool
from random import choice, shuffle
import math

from time import sleep
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datasets import Dataset
from utils.sae_loading import get_ckpts, get_model
from tqdm.contrib.concurrent import process_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = "YOUR_HF_TOKEN"


from aleph_alpha_client import Client
from utils.instruction_grader import JudgeWinRate

client = Client(
    token="CLIENT_TOKEN",
    host="HOST",
)

Judge = JudgeWinRate(client, "llama-3.1-70b-instruct")


def eval_if_tox(entry):
    i = entry[1]["choice"]
    prompt = entry[1]["prompt"]
    Prefix_1 = "**Continuation Alpha**: " if i >= 3 else "**Continuation Beta**: "
    Prefix_2 = "**Continuation Beta**: " if i >= 3 else "**Continuation Alpha**: "
    completion_a = (
        Prefix_1 + entry[1]["completion"]
        if i % 2 == 0
        else Prefix_1 + entry[1]["completion_1"]
    )
    model_a = entry[1]["model"] if i % 2 == 0 else entry[1]["model_1"]
    completion_b = (
        Prefix_2 + entry[1]["completion"]
        if i % 2 == 1
        else Prefix_2 + entry[1]["completion_1"]
    )
    model_b = entry[1]["model"] if i % 2 == 1 else entry[1]["model_1"]
    counter = 0
    failed = {
        "failed": True,
    }
    if (
        completion_a == ""
        or str(completion_a).isspace()
        or (isinstance(completion_a, float) and math.isnan(completion_a))
        or completion_a is None
        or pd.isnull(completion_a)
        or completion_b == ""
        or str(completion_b).isspace()
        or (isinstance(completion_b, float) and math.isnan(completion_b))
        or completion_b is None
        or pd.isnull(completion_b)
    ):
        return failed
    while True:
        try:
            return Judge.grade(
                prompt,
                completion_a,
                completion_b,
                model_a,
                model_b,
                template_type="tox",
            ).__dict__
        except Exception as e:
            counter += 1
            if counter >= 5:
                print(
                    f"Stuck in a loop with error {e}"
                )
                return failed
            print(f"Waiting because of error {e}")
            sleep(2)


def eval_if_non_tox(entry):
    i = entry[1]["choice"]
    prompt = entry[1]["prompt"]

    Prefix_1 = "**Continuation Alpha**: " if i >= 3 else "**Continuation Beta**: "
    Prefix_2 = "**Continuation Beta**: " if i >= 3 else "**Continuation Alpha**: "
    completion_a = (
        Prefix_1 + entry[1]["completion"]
        if i % 2 == 0
        else Prefix_1 + entry[1]["completion_1"]
    )
    model_a = entry[1]["model"] if i % 2 == 0 else entry[1]["model_1"]
    completion_b = (
        Prefix_2 + entry[1]["completion"]
        if i % 2 == 1
        else Prefix_2 + entry[1]["completion_1"]
    )
    model_b = entry[1]["model"] if i % 2 == 1 else entry[1]["model_1"]
    counter = 0
    failed = {
        "failed": True,
    }
    if (
        completion_a == ""
        or str(completion_a).isspace()
        or (isinstance(completion_a, float) and math.isnan(completion_a))
        or completion_a is None
        or pd.isnull(completion_a)
        or completion_b == ""
        or str(completion_b).isspace()
        or (isinstance(completion_b, float) and math.isnan(completion_b))
        or completion_b is None
        or pd.isnull(completion_b)
    ):
        return failed
    while True:
        try:
            return Judge.grade(
                prompt,
                completion_a,
                completion_b,
                model_a,
                model_b,
                template_type="non-tox",
            ).__dict__
        except Exception as e:
            counter += 1
            if counter >= 5:
                print(
                    f"Stuck in a loop with error {e}"
                )
                return failed
            print(f"Waiting because of error {e}")
            sleep(2)


def eval_if_sp(entry):
    i = entry[1]["choice"]
    prompt = entry[1]["prompt"]
    Prefix_1 = "**Continuation Alpha**: " if i >= 3 else "**Continuation Beta**: "
    Prefix_2 = "**Continuation Beta**: " if i >= 3 else "**Continuation Alpha**: "
    completion_a = (
        Prefix_1 + entry[1]["completion"]
        if i % 2 == 0
        else Prefix_1 + entry[1]["completion_1"]
    )
    model_a = entry[1]["model"] if i % 2 == 0 else entry[1]["model_1"]
    completion_b = (
        Prefix_2 + entry[1]["completion"]
        if i % 2 == 1
        else Prefix_2 + entry[1]["completion_1"]
    )
    model_b = entry[1]["model"] if i % 2 == 1 else entry[1]["model_1"]
    counter = 0
    failed = {
        "failed": True,
    }
    if (
        completion_a == ""
        or str(completion_a).isspace()
        or (isinstance(completion_a, float) and math.isnan(completion_a))
        or completion_a is None
        or pd.isnull(completion_a)
        or completion_b == ""
        or str(completion_b).isspace()
        or (isinstance(completion_b, float) and math.isnan(completion_b))
        or completion_b is None
        or pd.isnull(completion_b)
    ):
        return failed
    while True:
        try:
            return Judge.grade(
                prompt,
                completion_a,
                completion_b,
                model_a,
                model_b,
                template_type="sp",
            ).__dict__
        except Exception as e:
            counter += 1
            if counter >= 5:
                print(
                    f"Stuck in a loop with error {e}"
                )
                return failed
            print(f"Waiting because of error {e}")
            sleep(2)


def eval_if_pii(entry):
    i = entry[1]["choice"]
    prompt = entry[1]["prompt"]
    Prefix_1 = "**Continuation Alpha**: " if i >= 3 else "**Continuation Beta**: "
    Prefix_2 = "**Continuation Beta**: " if i >= 3 else "**Continuation Alpha**: "
    completion_a = (
        Prefix_1 + entry[1]["completion"]
        if i % 2 == 0
        else Prefix_1 + entry[1]["completion_1"]
    )
    model_a = entry[1]["model"] if i % 2 == 0 else entry[1]["model_1"]
    completion_b = (
        Prefix_2 + entry[1]["completion"]
        if i % 2 == 1
        else Prefix_2 + entry[1]["completion_1"]
    )
    model_b = entry[1]["model"] if i % 2 == 1 else entry[1]["model_1"]
    counter = 0
    failed = {
        "failed": True,
    }
    if (
        completion_a == ""
        or str(completion_a).isspace()
        or (isinstance(completion_a, float) and math.isnan(completion_a))
        or completion_a is None
        or pd.isnull(completion_a)
        or completion_b == ""
        or str(completion_b).isspace()
        or (isinstance(completion_b, float) and math.isnan(completion_b))
        or completion_b is None
        or pd.isnull(completion_b)
    ):
        return failed
    while True:
        try:
            return Judge.grade(
                prompt,
                completion_a,
                completion_b,
                model_a,
                model_b,
                template_type="pii",
            ).__dict__
        except Exception as e:
            counter += 1
            if counter >= 5:
                print(
                    f"Stuck in a loop with error {e}"
                )
                return failed
            print(f"Waiting because of error {e}")
            sleep(2)


def eval_if_non_sp(entry):
    i = entry[1]["choice"]
    prompt = entry[1]["prompt"]
    Prefix_1 = "**Continuation Alpha**: " if i >= 3 else "**Continuation Beta**: "
    Prefix_2 = "**Continuation Beta**: " if i >= 3 else "**Continuation Alpha**: "
    completion_a = (
        Prefix_1 + entry[1]["completion"]
        if i % 2 == 0
        else Prefix_1 + entry[1]["completion_1"]
    )
    model_a = entry[1]["model"] if i % 2 == 0 else entry[1]["model_1"]
    completion_b = (
        Prefix_2 + entry[1]["completion"]
        if i % 2 == 1
        else Prefix_2 + entry[1]["completion_1"]
    )
    model_b = entry[1]["model"] if i % 2 == 1 else entry[1]["model_1"]
    counter = 0
    failed = {
        "failed": True,
    }
    if (
        completion_a == ""
        or str(completion_a).isspace()
        or (isinstance(completion_a, float) and math.isnan(completion_a))
        or completion_a is None
        or pd.isnull(completion_a)
        or completion_b == ""
        or str(completion_b).isspace()
        or (isinstance(completion_b, float) and math.isnan(completion_b))
        or completion_b is None
        or pd.isnull(completion_b)
    ):
        return failed
    while True:
        try:
            return Judge.grade(
                prompt,
                completion_a,
                completion_b,
                model_a,
                model_b,
                template_type="non-sp",
            ).__dict__
        except Exception as e:
            counter += 1
            if counter >= 5:
                print(
                    f"Stuck in a loop with error {e}"
                )
                return failed
            print(f"Waiting because of error {e}")
            sleep(2)


def select_eval(eval_type: str):
    if eval_type == "tox":
        return eval_if_tox
    elif eval_type == "non-tox":
        return eval_if_non_tox
    elif eval_type == "sp":
        return eval_if_sp
    elif eval_type == "non-sp":
        return eval_if_non_sp
    elif eval_type == "pii":
        return eval_if_pii
    else:
        raise ValueError("Invalid eval_type")


def eval(df: pd.DataFrame, eval_type: str = "tox") -> pd.DataFrame:
    choices = (
        [1 for _ in range(len(df) // 4)]
        + [2 for _ in range(len(df) // 4)]
        + [3 for _ in range(len(df) // 4)]
        + [4 for _ in range(len(df) // 4)]
    )
    to_add = len(df) - len(choices)
    if to_add > 0:
        choices += [1 for _ in range(to_add)]
    shuffle(choices)

    LLM_Eval = select_eval(eval_type)

    df["choice"] = choices

    tmp2 = process_map(
        LLM_Eval,
        df.iterrows(),
        max_workers=32,
        total=len(df),
    )
    df2 = pd.DataFrame(tmp2)
    df = df.join(df2, rsuffix="_digsaed")
    df = df.loc[:, ~df.columns.str.endswith("_digsaed")]

    return df


def get_winning_method(x, method_1: str = "G-SAE", method_2: str = "OTHER"):
    '''
    Extracts the winning model from the judge result file
    '''
    
    A = "Alpha"
    B = "Beta"
    if x["result"] == "draw":
        return "draw"
    elif not isinstance(x["result"], str):
        return "draw"
    elif (
        pd.notna(x["completion"])
        and x["result"] in x["completion_a"]
        and x["completion"] in x["completion_a"]
    ):
        return method_1
    elif (
        pd.notna(x["completion_1"])
        and x["result"] in x["completion_a"]
        and x["completion_1"] in x["completion_a"]
    ):
        return method_2
    elif (
        pd.notna(x["completion"])
        and x["result"] in x["completion_b"]
        and x["completion"] in x["completion_b"]
    ):
        return method_1
    elif (
        pd.notna(x["completion_1"])
        and x["result"] in x["completion_b"]
        and x["completion_1"] in x["completion_b"]
    ):
        return method_2
    else:
        return "draw"


matches_rtp_tox_increase = [
    (
        "Method Name",
        "Path to steered generations",
    ),
]
matches_rtp_tox_decrease = [
    (
        "Method Name",
        "Path to steered generations",
    ),
]
matches_sp_increase = [
    (
        "Method Name",
        "Path to steered generations",
    ),
]
matches_sp_decrease = [
    (
        "Method Name",
        "Path to steered generations",
    ),
]

matches_pii_decrease = [
    (
        "Method Name",
        "Path to steered generations",
    ),
]

# PII
for gsae_version, gsae_path, alpha_1 in [
    (
        "gsae3",
        "./llama3_SAE/SAE_eval/PII_300k-Block_v2/pii_smart_balanced_topk_steering_scores-llama3-l24576-b11-k2048_s1-4096.csv",
        -3.0,
    ),
]:
    df1 = pd.read_csv(gsae_path)
    df1["alpha"] = df1["alpha"].round(1)
    for model_name, model_path, alpha_2 in matches_pii_decrease:
        if model_path == "" or os.path.exists(
            f"./llama3_SAE/SAE_eval/winrates/PII-decrease-{gsae_version}-vs-{model_name}.csv"
        ):
            continue
        print(model_name, model_path)
        if "sae" in model_name or "gsae" in model_name:
            df2 = pd.read_csv(model_path)
            df2["alpha"] = df2["alpha"].round(1)
        elif "icv" in model_name or "diff" in model_name:
            df2 = pd.read_json(model_path)
            df2["model"] = model_name
            df2["orig_prompt"] = df2["prompt"]
        elif "Baseline" in model_name:
            df2 = pd.read_csv(model_path)
            df2["alpha"] = df2["alpha"].round(1)
            df2["model"] = model_name
        elif "PreADD" in model_name or "Model-Arithmetic" in model_name:
            df2 = pd.read_csv(model_path)
            df2["model"] = model_name
            df2["prompt"] = df2["prompt"].apply(
                lambda x: str(x).replace("<|begin_of_text|>", "")
            )
        else:
            raise ValueError("Invalid model_name")

        merged_df = pd.merge(
            df1[df1["alpha"] == alpha_1],
            df2[df2["alpha"] == alpha_2]
            if "sae" in model_name or "gsae" in model_name
            else df2,
            how="inner",
            on="orig_prompt"
            if "icv" in model_name or "diff" in model_name
            else "prompt",
            suffixes=("", "_1"),
        )[
            [
                "prompt",
                "completion",
                "alpha",
                "model",
                "completion_1",
                "model_1",
            ]
        ]
        print(len(merged_df), len(df1[df1["alpha"] == alpha_1]), len(df2))
        print(df1.columns)
        print(df2.columns)
        merged_df["completion"].fillna("", inplace=True)
        merged_df["completion_1"].fillna("", inplace=True)
        merged_df = eval(merged_df, "pii")
        merged_df.to_csv(
            f"./llama3_SAE/SAE_eval/winrates/PII-decrease-{gsae_version}-vs-{model_name}.csv",
            index=False,
        )

# RTP
for gsae_version, gsae_path in [
    (
        "gsae",
        "./llama3_SAE/SAE_eval/RTP-Block_v2/rtp-all-llama3-l24576-b11-k2048-4096.csv",
        
    ),
]:
    df1 = pd.read_csv(gsae_path)
    df1["alpha"] = df1["alpha"].round(1)
    for model_name, model_path in matches_rtp_tox_decrease:
        if model_path == "" or os.path.exists(
            f"./llama3_SAE/SAE_eval/winrates/RTP-tox_decrease-{gsae_version}-vs-{model_name}.csv"
        ):
            continue
        print(model_name, model_path)
        if "sae" in model_name or "gsae" in model_name or "Baseline" in model_name:
            df2 = pd.read_csv(model_path)
            df2["alpha"] = df2["alpha"].round(1)
        elif "icv" in model_name or "diff" in model_name:
            df2 = pd.read_json(model_path)
            df2["model"] = model_name
        elif "PreADD" in model_name or "Model-Arithmetic" in model_name:
            df2 = pd.read_csv(model_path)
            df2["model"] = model_name
        elif "Baseline" in model_name:
            df2 = pd.read_csv(model_path)
            df2["alpha"] = df2["alpha"].round(1)
            df2["model"] = model_name
        else:
            raise ValueError("Invalid model_name")

        merged_df = pd.merge(
            df1[df1["alpha"] == -0.4],
            df2[
                df2["alpha"] == -0.4
                if "Baseline" not in model_name
                else df2["alpha"] == 0.0
            ]
            if "sae" in model_name or "gsae" in model_name
            else df2,
            how="inner",
            on="prompt",
            suffixes=("", "_1"),
        )[
            [
                "prompt",
                "completion",
                "alpha",
                "start_tox",
                "model",
                "PersAPI",
                "completion_1",
                "model_1",
            ]
        ]
        merged_df["completion"].fillna("", inplace=True)
        merged_df["completion_1"].fillna("", inplace=True)
        print(len(merged_df))
        merged_df = eval(merged_df, "tox")
        merged_df.to_csv(
            f"./llama3_SAE/SAE_eval/winrates/RTP-tox_decrease-{gsae_version}-vs-{model_name}.csv",
            index=False,
        )

    for model_name, model_path in matches_rtp_tox_increase:
        if model_path == "" or os.path.exists(
            f"./llama3_SAE/SAE_eval/winrates/RTP-tox_increase-{gsae_version}-vs-{model_name}.csv"
        ):
            continue

        print(model_name, model_path)
        if "sae" in model_name or "gsae" in model_name or "Baseline" in model_name:
            df2 = pd.read_csv(model_path)
            df2["alpha"] = df2["alpha"].round(1)
        elif "icv" in model_name or "diff" in model_name:
            df2 = pd.read_json(model_path)
            df2["model"] = model_name
        elif "PreADD" in model_name or "Model-Arithmetic" in model_name:
            df2 = pd.read_csv(model_path)
            df2["model"] = model_name
        elif "Baseline" in model_name:
            df2 = pd.read_csv(model_path)
            df2["alpha"] = df2["alpha"].round(1)
            df2["model"] = model_name
        else:
            raise ValueError("Invalid model_name")

        merged_df = pd.merge(
            df1[df1["alpha"] == 0.4],
            df2[
                df2["alpha"] == 0.4
                if "Baseline" not in model_name
                else df2["alpha"] == 0.0
            ]
            if "sae" in model_name or "gsae" in model_name
            else df2,
            how="inner",
            on="prompt",
            suffixes=("", "_1"),
        )[
            [
                "prompt",
                "completion",
                "alpha",
                "start_tox",
                "model",
                "PersAPI",
                "completion_1",
                "model_1",
            ]
        ]
        merged_df["completion"].fillna("", inplace=True)
        merged_df["completion_1"].fillna("", inplace=True)
        print(len(merged_df))
        out = eval(merged_df, "non-tox")
        out.to_csv(
            f"./llama3_SAE/SAE_eval/winrates/RTP-tox_increase-{gsae_version}-vs-{model_name}.csv",
            index=False,
        )

# SP
for gsae_version, gsae_path in [
    (
        "gsae",
        "./llama3_SAE/SAE_eval/SP-Block_v2/SP-llama3-l24576-b03-k2048-4096.csv",
    ),
]:
    df1 = pd.read_csv(gsae_path)
    df1["alpha"] = df1["alpha"].round(1)
    for model_name, model_path in matches_sp_increase:
        if model_path == "" or os.path.exists(
            f"./llama3_SAE/SAE_eval/winrates/SP-increase-{gsae_version}-vs-{model_name}.csv"
        ):
            continue
        print(model_name, model_path)
        if "sae" in model_name or "gsae" in model_name or "Baseline" in model_name:
            df2 = pd.read_csv(model_path)
            df2["alpha"] = df2["alpha"].round(1)
        elif "icv" in model_name or "diff" in model_name:
            df2 = pd.read_json(model_path)
            df2["model"] = model_name
        elif "PreADD" in model_name or "Model-Arithmetic" in model_name:
            df2 = pd.read_csv(model_path)
            df2["model"] = model_name
        elif "Baseline" in model_name:
            df2 = pd.read_csv(model_path)
            df2["alpha"] = df2["alpha"].round(1)
            df2["model"] = model_name
        else:
            raise ValueError("Invalid model_name")

        merged_df = pd.merge(
            df1[df1["alpha"] == 0.2],
            df2[
                df2["alpha"] == 0.2
                if "Baseline" not in model_name
                else df2["alpha"] == 0.0
            ]
            if "sae" in model_name or "gsae" in model_name
            else df2,
            how="inner",
            on="prompt",
            suffixes=("", "_1"),
        )[
            [
                "prompt",
                "completion",
                "alpha",
                "label",
                "model",
                "completion_1",
                "model_1",
            ]
        ]
        merged_df["completion"].fillna("", inplace=True)
        merged_df["completion_1"].fillna("", inplace=True)
        merged_df = eval(merged_df, "sp")
        merged_df.to_csv(
            f"./llama3_SAE/SAE_eval/winrates/SP-increase-{gsae_version}-vs-{model_name}.csv",
            index=False,
        )

    for model_name, model_path in matches_sp_decrease:
        if model_path == "" or os.path.exists(
            f"./llama3_SAE/SAE_eval/winrates/SP-decrease-{gsae_version}-vs-{model_name}.csv"
        ):
            continue
        print(model_name, model_path)
        if "sae" in model_name or "gsae" in model_name:
            df2 = pd.read_csv(model_path)
            df2["alpha"] = df2["alpha"].round(1)
        elif "icv" in model_name or "diff" in model_name:
            df2 = pd.read_json(model_path)
            df2["model"] = model_name
        elif "PreADD" in model_name or "Model-Arithmetic" in model_name:
            df2 = pd.read_csv(model_path)
            df2["model"] = model_name
        elif "Baseline" in model_name:
            df2 = pd.read_csv(model_path)
            df2["alpha"] = df2["alpha"].round(1)
            df2["model"] = model_name
        else:
            raise ValueError("Invalid model_name")

        merged_df = pd.merge(
            df1[df1["alpha"] == -0.2],
            df2[
                df2["alpha"] == -0.2
                if "Baseline" not in model_name
                else df2["alpha"] == 0.0
            ]
            if "sae" in model_name or "gsae" in model_name
            else df2,
            how="inner",
            on="prompt",
            suffixes=("", "_1"),
        )[
            [
                "prompt",
                "completion",
                "alpha",
                "label",
                "model",
                "completion_1",
                "model_1",
            ]
        ]
        merged_df["completion"].fillna("", inplace=True)
        merged_df["completion_1"].fillna("", inplace=True)
        merged_df = eval(merged_df, "non-sp")
        merged_df.to_csv(
            f"./llama3_SAE/SAE_eval/winrates/SP-decrease-{gsae_version}-vs-{model_name}.csv",
            index=False,
        )
