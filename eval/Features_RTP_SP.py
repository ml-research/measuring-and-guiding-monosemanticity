from datasets import concatenate_datasets, load_from_disk
import torch
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import csv
from torch.utils.data import DataLoader
import re
from collections import defaultdict
import os
import logging
from utils.tree_loader import get_root_node
from utils.sae_loading import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = "HF_TOKEN"


def RTP_tox_feature(llama3_sae, tokenizer, feat: int = 0, all: bool = False) -> None:
    # test = load_dataset("allenai/real-toxicity-prompts", split="train", num_proc=4)
    test = load_from_disk("./datasets/RTP")[
        "test"
    ]

    def preprocess_1(example):
        is_toxic = (
            example["prompt"]["toxicity"] >= 0.75
            if example["prompt"]["toxicity"] is not None
            else False
        )
        return {
            "text": example["prompt"]["text"],
            "is_toxic": is_toxic,
            "toxcitity_level": example["prompt"]["toxicity"]
            if example["prompt"]["toxicity"] is not None
            else 0.0,
        }

    test = test.map(preprocess_1, num_proc=4)
    test = test.remove_columns(
        [
            "filename",
            "begin",
            "end",
            "challenging",
            "prompt",
            "continuation",
        ]
    )
    if not all:
        tmp = []
        for i in np.linspace(0.05, 1, 20):
            tmp.append(
                test.filter(
                    lambda x: x["toxcitity_level"] >= i - 0.05
                    and x["toxcitity_level"] < i
                )
                .shuffle(42)
                .select(range(100))
            )
        test = concatenate_datasets(tmp)
    test = DataLoader(test, batch_size=32, num_workers=16)

    res = defaultdict(list)
    for sent in tqdm(test):
        toks = tokenizer(
            sent["text"], return_tensors="pt", padding=True
        ).input_ids.cuda()
        llama3_sae(toks)
        acts = llama3_sae.hook.pop()
        pre_act, latents, recons = llama3_sae.SAE(acts)
        
        
        res["sent"] += sent["text"]
        res["label"] += sent["toxcitity_level"].tolist()
        res["feature_value"] += latents[:, :, feat].cpu().tolist()
        res["pre_feature_value"] += pre_act[:, :, feat].cpu().tolist()
        res["latent_min"] += latents.min(2).values.cpu().tolist()
        res["latent_max"] += latents.max(2).values.cpu().tolist()
        res["latent_mean"] += latents.mean(2).cpu().tolist()
        res["latent_std"] += latents.std(2).cpu().tolist()
        torch.cuda.empty_cache()

    df = pd.DataFrame.from_dict(res)
    df["input_ids"] = df["sent"].apply(lambda x: tokenizer(x).input_ids)
    df["feature_value_mean"] = df[["input_ids", "feature_value"]].apply(
        lambda x: np.mean(x["feature_value"][(len(x["input_ids"]) - 0) * -1 :]),
        axis=1,
    )
    df["pre_feature_value_mean"] = df[["input_ids", "pre_feature_value"]].apply(
        lambda x: np.mean(x["pre_feature_value"][(len(x["input_ids"]) - 0) * -1 :]),
        axis=1,
    )
    df = df.replace("\n", "", regex=True)
    return df


def SP_feature(llama3_sae, tokenizer, feat: int = 0, ds_type: str = "valid") -> None:
    test = load_from_disk(
        "./datasets/Shakespeare"
    )[ds_type]
    test = test.map(lambda x: {"label": 0 if x["label"] == "modern" else 1})
    test = DataLoader(test, batch_size=4, num_workers=16)

    res = defaultdict(list)
    for sent in tqdm(test):
        toks = tokenizer(
            sent["text"], return_tensors="pt", padding=True
        ).input_ids.cuda()
        llama3_sae(toks)
        acts = llama3_sae.hook.pop()
        pre_act, latents, recons = llama3_sae.SAE(acts)
        
        
        res["sent"] += sent["text"]
        res["label"] += sent["label"].tolist()
        res["feature_value"] += latents[:, :, feat].cpu().tolist()
        res["pre_feature_value"] += pre_act[:, :, feat].cpu().tolist()
        res["latent_min"] += latents.min(2).values.cpu().tolist()
        res["latent_max"] += latents.max(2).values.cpu().tolist()
        res["latent_mean"] += latents.mean(2).cpu().tolist()
        res["latent_std"] += latents.std(2).cpu().tolist()
        torch.cuda.empty_cache()

    df = pd.DataFrame.from_dict(res)
    df["input_ids"] = df["sent"].apply(lambda x: tokenizer(x).input_ids)
    df["feature_value_mean"] = df[["input_ids", "feature_value"]].apply(
        lambda x: np.mean(x["feature_value"][(len(x["input_ids"]) - 0) * -1 :]),
        axis=1,
    )
    df["pre_feature_value_mean"] = df[["input_ids", "pre_feature_value"]].apply(
        lambda x: np.mean(x["pre_feature_value"][(len(x["input_ids"]) - 0) * -1 :]),
        axis=1,
    )
    df = df.replace("\n", "", regex=True)
    return df


if __name__ == "__main__":
    THRESHOLD = 0.0
    load_from_ckpt = True
    ckpt_dirs = {
        "Checkpoint Path": "Result Dir",
    }

    for ckpt_dir, save_dir in list(ckpt_dirs.items()):
        for model_name in [
            f"llama3-gsae"
            for i in [2048]
            for j in [24576]
            for k in [3]
            for l in ["_s0", "_s1"]
        ]:
            skip = False
            llm_sae, tokenizer = None, None
            model_base_type = model_name.split("-")[0].split("_")[0]
            scaling = (re.findall(r"_s\d+", model_name) or ["_s5"])[0]
            model_name = model_name.replace(scaling, "")
            llm_sae, tokenizer = get_model(
                model_name,
                load_from_ckpt,
                scaling,
                "block",
                ckpt_dir,
                act="topk-sigmoid",
            )
            block_num = int(model_name.split("-")[-2][1:])
            model_name += scaling

            if (
                os.path.isfile(
                    f"./SAE/{model_base_type}_SAE/SAE_eval/{save_dir}/rtp-mix-tox_feature-{model_name}.csv"
                )
                or "SP" in save_dir
            ):
                pass
            else:
                # RealToxicPrompts 2000 evenly distributed - Tox Feature Value
                file = f"./SAE/llama3_SAE/SAE_eval/{save_dir}/rtp-mix-tox_feature-{model_name}.csv"
                if os.path.isfile(file) or skip:
                    logger.info(
                        f"{save_dir}/rtp-mix-tox_feature-{model_name}.csv exists"
                    )
                else:
                    LATENT = (
                        get_root_node(
                            f"./SAE/{model_base_type}_SAE/SAE_eval/{save_dir}/rtp_tree_balanced_{model_name}.pkl"
                        )["feature_index"]
                        if os.path.isfile(
                            f"./SAE/{model_base_type}_SAE/SAE_eval/{save_dir}/rtp_tree_balanced_{model_name}.pkl"
                        )
                        else 0
                    )
                    logger.info(
                        f"Using model: '{model_name}' with scaling: '{scaling}' and latent feature: '{LATENT}'"
                    )

                    res_tox_feature = RTP_tox_feature(llm_sae, tokenizer, feat=LATENT)
                    res_tox_feature.to_csv(
                        file,
                        quoting=csv.QUOTE_NONNUMERIC,
                        lineterminator="\n",
                        escapechar="\\",
                    )

            if (
                os.path.isfile(
                    f"./SAE/{model_base_type}_SAE/SAE_eval/{save_dir}/rtp-all-tox_feature-{model_name}.csv"
                )
                or "SP" in save_dir
            ):
                pass
            else:
                # RTP all test entries
                file = f"./SAE/llama3_SAE/SAE_eval/{save_dir}/rtp-all-tox_feature-{model_name}.csv"
                if os.path.isfile(file) or skip:
                    logger.info(
                        f"{save_dir}/rtp-all-tox_feature-{model_name}.csv exists"
                    )
                else:
                    LATENT = (
                        get_root_node(
                            f"./SAE/{model_base_type}_SAE/SAE_eval/{save_dir}/rtp_tree_test-cb_{model_name}.pkl"
                        )["feature_index"]
                        if os.path.isfile(
                            f"./SAE/{model_base_type}_SAE/SAE_eval/{save_dir}/rtp_tree_test-cb_{model_name}.pkl"
                        )
                        else 0
                    )
                    logger.info(
                        f"Using model: '{model_name}' with scaling: '{scaling}' and latent feature: '{LATENT}'"
                    )

                    res_tox_feature = RTP_tox_feature(
                        llm_sae, tokenizer, feat=LATENT, all=True
                    )
                    res_tox_feature.to_csv(
                        file,
                        quoting=csv.QUOTE_NONNUMERIC,
                        lineterminator="\n",
                        escapechar="\\",
                    )

            if (
                os.path.isfile(
                    f"./SAE/{model_base_type}_SAE/SAE_eval/{save_dir}/sp_valid-feature-{model_name}.csv"
                )
                or "RTP" in save_dir
            ):
                pass
            else:
                # SP valid
                file = f"./SAE/llama3_SAE/SAE_eval/{save_dir}/sp_valid-feature-{model_name}.csv"
                if os.path.isfile(file) or skip:
                    logger.info(f"{save_dir}/sp_valid-feature-{model_name}.csv exists")
                else:
                    LATENT = (
                        get_root_node(
                            f"./SAE/llama3_SAE/SAE_eval/{save_dir}/sp_tree_valid_{model_name}.pkl"
                        )["feature_index"]
                        if os.path.isfile(
                            f"./SAE/llama3_SAE/SAE_eval/{save_dir}/sp_tree_valid_{model_name}.pkl"
                        )
                        else 0
                    )
                    logger.info(
                        f"Using model: '{model_name}' with scaling: '{scaling}' and latent feature: '{LATENT}'"
                    )

                    res_sp_feature = SP_feature(llm_sae, tokenizer, feat=LATENT)
                    res_sp_feature.to_csv(
                        file,
                        quoting=csv.QUOTE_NONNUMERIC,
                        lineterminator="\n",
                        escapechar="\\",
                    )
