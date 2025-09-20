from datasets import (
    load_dataset,
    load_from_disk,
)
from torch.utils.data import DataLoader
import torch
import transformers
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from typing import List
import gc
from time import sleep
import math

tqdm.pandas()
from collections import defaultdict
from typing import Tuple
import plotly.express as px
import plotly.graph_objects as go
import json
import pickle
import re
import os
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool
import requests

from utils.sae_loading import get_model
from sklearn.metrics import accuracy_score, f1_score
from sklearn import tree
import matplotlib.pyplot as plt
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import TransformersNlpEngine

from utils.tree_loader import get_root_node, get_tree_stats
from utils.persedio_config import analyzer
from presidio_analyzer.recognizer_result import RecognizerResult

from aleph_alpha_client import Client
from utils.instruction_grader import InstructionGrader

import logging
import inspect

from torch.profiler import profile, record_function, ProfilerActivity

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

client = Client(
    token="CLIENT_TOKEN",
    host="HOST",
)
StrucEval = InstructionGrader(client, "llama-3.1-70b-instruct")


def log_var(var):
    frame = inspect.currentframe().f_back  # Get caller's frame
    var_name = next(
        (name for name, val in frame.f_locals.items() if val is var), "unknown"
    )
    logger.debug(f"{var_name}: {var}")


logger.log_var = log_var

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

LATENT = 0
THRESH = 4096
FACTOR = 0.0

@torch.no_grad()
def hook_res_smart(model, input, output):
    output = list(output)

    # Compute latents and find active features
    # latents = torch.nn.functional.sigmoid(llama3_sae.SAE.encode_pre_act(output[0]))
    latents, _ = llama3_sae.SAE.encode(output[0])
    batch_indices, seq_indices, feat_indices = latents[:, :, LATENT_FEATURES].nonzero(
        as_tuple=True
    )
    
    # Efficiently extract decoder weight for selected features
    decoder_weight = llama3_sae.SAE.decoder.weight[:, LATENT_FEATURES][:, feat_indices]

    # Get relevant residuals
    selected_residuals = output[0][batch_indices, seq_indices]

    # Compute norms efficiently
    norm_res = selected_residuals.norm(p=2, dim=-1, keepdim=True)
    norm_gsae = decoder_weight.norm(p=2, dim=0, keepdim=True).T

    # Compute scaling factor
    factor = norm_res / (norm_gsae + 1e-8)  # Avoid division by zero

    # Compute scaled SAE output
    scaled_sae_out = (decoder_weight * factor.view(1, -1)).T

    torch.index_put_(
        output[0],
        (batch_indices, seq_indices),
        scaled_sae_out * FACTOR,
        accumulate=True,
    )

    return tuple(output)



@torch.no_grad()
def hook_res_smart_balanced_topk(model, input, output):
    output = list(output)

    # Compute latents and find active features
    latents, _ = llama3_sae.SAE.encode(output[0])
    
    batch_indices, seq_indices, feat_indices = latents[:, :, LATENT_FEATURES].nonzero(
        as_tuple=True
    )
    values = latents[:, :, LATENT_FEATURES][batch_indices, seq_indices, feat_indices]
    # Efficiently extract decoder weight for selected features
    decoder_weight = llama3_sae.SAE.decoder.weight[:, LATENT_FEATURES][:, feat_indices]

    # Get relevant residuals
    selected_residuals = output[0][batch_indices, seq_indices]

    # Compute norms efficiently
    norm_res = selected_residuals.norm(p=2, dim=-1, keepdim=True)
    norm_gsae = decoder_weight.norm(p=2, dim=0, keepdim=True).T

    # Compute scaling factor
    factor = norm_res / (norm_gsae + 1e-8)  # Avoid division by zero

    # Compute scaled SAE output
    scaled_sae_out = (decoder_weight * values * factor.view(1, -1)).T

    torch.index_put_(
        output[0],
        (batch_indices, seq_indices),
        scaled_sae_out * FACTOR,
        accumulate=True,
    )

    return tuple(output)


def filter_entities(
    entities: List[RecognizerResult],
) -> List[dict]:
    """
    Removes fully contained entities and resolves partial overlaps by confidence.

    :param entities: List of entities extracted by Presidio.

    :return: List of entities after filtering.
    """
    # Sort by confidence score (highest first)
    sorted_entities = sorted(entities, key=lambda x: x.score, reverse=True)

    final_entities = []

    for entity in sorted_entities:
        entity_kept = True  # Assume we will keep this entity

        for existing in final_entities:
            # Check for **full containment**
            if existing.start <= entity.start and entity.end <= existing.end:
                entity_kept = False  # Entity is fully inside an existing one
                break

            # Check for **partial overlap**
            elif (
                entity.start < existing.end and entity.end > existing.start
            ):  # Partial overlap detected
                if entity.score <= existing.score:
                    entity_kept = False  # Lower-confidence entity is removed
                    break
                else:
                    # If the new entity has a **higher score**, remove the old one
                    final_entities.remove(existing)

        if entity_kept:
            final_entities.append(entity)  # Add the entity if it survived checks

    return [ent.to_dict() for ent in final_entities]


def extract_value_for_ner(text: str, ner_tags: str) -> List[dict]:
    """
    Extracts the value of the named entities from the text.

    :param text: The text from which the entities were extracted.
    :param ner_tags: The named entities extracted from the text.

    :return: The named entities with the value extracted.
    """
    for tag in ner_tags:
        tag["value"] = text[tag["start"] : tag["end"]]

    return ner_tags


def filter_invalid_entries(entries):
    """
    Filters out entries where the 'value' field contains specific unwanted values.

    :param entries: list of dicts, each containing extracted data including 'value'
    :return: list of dicts, filtered entries
    """
    invalid_values = {
        "12345",
        "9999",
        "0000",
        "John Doe",
        "Jane Doe",
        "John Smith",
        "Jane Smith",
        "Main St",
        "Anytown",
        "xyz",
        "XYZ",
        "abc",
        "ABC",
        "example.com",
        "example.net",
        "example.org",
        "192.0.2.0",
        "2001:db8::",
        "999-99-9999",
        "123-45-6789",
        "000-12-3456",
    }

    return [
        entry
        for entry in entries
        if not any(invalid in entry["value"] for invalid in invalid_values)
    ]


def compare_predictions_to_ground_truth(predictions, ground_truth):
    """
    Compares predictions from Presidio (Entity objects) with ground truth labels and returns the number of found and not found labels.
    This version handles both exact and partial matches based only on start and end positions.

    :param predictions: List of predicted entities from Presidio (Entity objects).
    :param ground_truth: List of ground truth entities.

    :return: Tuple (found_count, not_found_count)
    """
    found_count = 0
    not_found_count = 0

    # Loop over the ground truth entries
    for gt in ground_truth:
        match_found = False

        # Check if any prediction matches the ground truth (exact or partial match)
        for prediction in predictions:
            # Extracting relevant attributes from Entity object
            pred_start, pred_end, pred_label = (
                prediction.start,
                prediction.end,
                prediction.entity_type,
            )
            gt_start, gt_end, gt_label = gt["start"], gt["end"], gt["label"]
            gt_label = "".join(filter(lambda x: not x.isdigit(), gt_label))

            # Exact match check
            if pred_label == gt_label and pred_start == gt_start and pred_end == gt_end:
                match_found = True
                break  # Once matched exactly, no need to check further predictions

            # Partial match check (overlap between predictions and ground truth)
            elif pred_label == gt_label and (
                (pred_start <= gt_end and pred_end >= gt_start)  # Overlap condition
                or (pred_start >= gt_start and pred_end <= gt_end)
            ):  # Predicted entity inside ground truth
                match_found = True
                break  # Once a partial match is found, break out of the loop

        if match_found:
            found_count += 1
        else:
            not_found_count += 1

    return found_count
    # return not_found_count


def LLM_Eval_struc(data):
    prompt = data[1]["prompt"]
    cont = data[1]["completion"]
    model = data[1]["model"]
    counter = 0
    failed = {"prompt": prompt, "completion": cont, "model": model, "failed": True}
    if (
        cont == ""
        or str(cont).isspace()
        or (isinstance(cont, float) and math.isnan(cont))
        or cont is None
        or pd.isnull(cont)
    ):
        return failed
    while True:
        try:
            return StrucEval.grade(
                prompt, cont, model, template_type="structure"
            ).__dict__
        except Exception as e:
            counter += 1
            if counter >= 5:
                print(
                    f"Stuck in a loop with error {e}"
                )
                return failed
            print(f"Waiting because of error {e}")
            sleep(1)


dataset = load_dataset(
    "ai4privacy/pii-masking-300k",
    split="validation",
    token=HF_TOKEN,
    cache_dir="./cache_3",
    streaming=False,
)
dataset = dataset.filter(
    lambda x: x["language"] == "en" or x["language"] == "English"
)

load_from_ckpt = True
ckpt_dirs = {
    "Checkpoint Path": "Result Dir",
}
ckpt_dir, save_dir = list(ckpt_dirs.items())[-1]
for ckpt_dir, save_dir in ckpt_dirs.items():
    save_path = f"./llama3_SAE/SAE_eval/{save_dir}/"
    if __name__ == "__main__":
        for model_name in [
            f"llama3-l{str(j)}-b{str(k)}-k{str(i)}{l}"
            for i in [2048]
            for j in [24576]
            for k in [11]
            for l in ["_s0", "_s1"]
        ]:
            model_base_type = model_name.split("-")[0].split("_")[0]
            scaling = (re.findall(r"_s\d+", model_name) or ["_s1"])[0]
            model_name = model_name.replace(scaling, "")
            llama3_sae, tokenizer = None, None
            del llama3_sae
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            llama3_sae, tokenizer = get_model(
                model_name,
                load_from_ckpt,
                scaling,
                "block",
                ckpt_dir,
                act="topk-sigmoid",
                epoch=100,
            )
            block_num = int(model_name.split("-")[-2][1:])
            llama3_sae.eval()
            model_name += scaling
            with open(
                f"{save_path}root_nodes/root_nodes-{model_name}.json",
                "r",
            ) as f:
                root_nodes = json.load(f)

            steering_methods = {
                "smart_balanced_topk": {
                    "function": hook_res_smart_balanced_topk,
                    "concept": "ALL",
                },
                "smart_topk": {"function": hook_res_smart, "concept": "ALL"},
            }
            for method, method_dict in steering_methods.items():
                if method_dict["concept"] == "ALL":
                    LATENT_FEATURES = [
                        j["feature_index"] for i, j in root_nodes.items() if i != "O"
                    ]
                else:
                    LATENT_FEATURES = [
                        root_nodes[method_dict["concept"]]["feature_index"]
                    ]

                h_B_1 = llama3_sae.model.layers[block_num].register_forward_hook(
                    method_dict["function"]
                )

                eval_out = defaultdict(list)
                df = pd.DataFrame()
                if os.path.isfile(
                    f"./{model_base_type}_SAE/SAE_eval/{save_dir}/pii_{method}_steering_scores-{model_name}-{str(THRESH)}.csv"
                ):
                    df = pd.read_csv(
                        f"./{model_base_type}_SAE/SAE_eval/{save_dir}/pii_{method}_steering_scores-{model_name}-{str(THRESH)}.csv"
                    )
                else:
                    batch_size = 32
                    for b in tqdm(range(0, len(dataset) // batch_size + 1), desc="Gen"):
                        elem = dataset[(b * batch_size) : (b * batch_size) + 32]
                        text = elem["source_text"]
                        toks = tokenizer(text, return_tensors="pt", padding=True)

                        toks = {
                            "input_ids": toks.input_ids[:, :-10].to("cuda"),
                            "attention_mask": toks.attention_mask[:, :-10].to("cuda"),
                        }
                        cut_text = tokenizer.batch_decode(toks["input_ids"])
                        cut_text = [
                            t.replace("<|end_of_text|>", "").replace(
                                "<|begin_of_text|>", ""
                            )
                            for t in cut_text
                        ]
                        for i in [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]:
                            FACTOR = i
                            with torch.no_grad():
                                new_toks = 32
                                out = llama3_sae.generate(
                                    **toks,
                                    do_sample=False,
                                    temperature=None,
                                    top_p=None,
                                    max_new_tokens=new_toks,
                                    pad_token_id=tokenizer.eos_token_id,
                                )
                                gen_text = tokenizer.batch_decode(
                                    out[:, -1 * new_toks :]
                                )
                                gen_text = [
                                    t.replace("<|end_of_text|>", "").replace(
                                        "<|begin_of_text|>", ""
                                    )
                                    for t in gen_text
                                ]
                            eval_out["orig_prompt"] += text
                            eval_out["prompt"] += cut_text
                            eval_out["completion"] += gen_text
                            eval_out["alpha"] += [i for _ in range(len(text))]
                            eval_out["pii_mask"] += elem["privacy_mask"]
                            eval_out["model"] += [model_name for _ in range(len(text))]
                            eval_out["language"] += elem["language"]

                    df_dictionary = pd.DataFrame(eval_out)
                    df = pd.concat([df, df_dictionary], ignore_index=True)
                    df.to_csv(
                        f"./{model_base_type}_SAE/SAE_eval/{save_dir}/pii_{method}_steering_scores-{model_name}-{str(THRESH)}.csv"
                    )
                h_B_1.remove()

                if "persedio_completion" not in df.columns:
                    def analyzer_func(cont):

                        if cont == "" or isinstance(cont, float):
                            return []
                        else:
                            try:
                                return analyzer.analyze(
                                    text=cont,
                                    language="en",
                                )
                            except:
                                return []

                    df["persedio_completion"] = df["completion"].progress_apply(
                        lambda x: analyzer_func(x)
                    )
                    df["persedio_completion"] = df["persedio_completion"].apply(
                        lambda x: filter_entities(x)
                    )
                    df["persedio_completion"] = df[
                        ["completion", "persedio_completion"]
                    ].apply(
                        lambda x: extract_value_for_ner(
                            x["completion"], x["persedio_completion"]
                        ),
                        axis=1,
                    )
                    df["persedio_completion_clean"] = df["persedio_completion"].apply(
                        lambda x: filter_invalid_entries(x)
                    )

                    df["len_persedio_completion"] = df["persedio_completion"].apply(
                        lambda x: len(x)
                    )
                    df["len_persedio_completion_clean"] = df[
                        "persedio_completion_clean"
                    ].apply(lambda x: len(x))
                    df["diff_unclean_clean"] = (
                        df["len_persedio_completion"]
                        - df["len_persedio_completion_clean"]
                    )
                    df.to_csv(
                        f"./{model_base_type}_SAE/SAE_eval/{save_dir}/pii_{method}_steering_scores-{model_name}-{str(THRESH)}.csv"
                    )
                print("Smart Steering done.")

                if "is_structurally_consistent" not in df.columns:
                    tmp2 = process_map(
                        LLM_Eval_struc,
                        df[["prompt", "completion", "model"]].iterrows(),
                        max_workers=32,
                        total=len(df),
                        desc="LLMJudge",
                    )
                    df2 = pd.DataFrame(tmp2)
                    df = df.join(df2, rsuffix="_digsaed")
                    df = df.loc[:, ~df.columns.str.endswith("_digsaed")]
                    df.to_csv(
                        f"./{model_base_type}_SAE/SAE_eval/{save_dir}/pii_{method}_steering_scores-{model_name}-{str(THRESH)}.csv"
                    )
                print("LLMJudge done.")
