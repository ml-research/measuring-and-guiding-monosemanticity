from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import pickle
import os
import logging
from datasets import Dataset
from typing import Tuple
from random import shuffle
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = "HF_TOKEN"


from utils.sae_loading import get_model, get_ckpts
from utils.tree_loader import get_tree_stats, get_root_node


def RTP_latents(
    llama3_sae,
    tokenizer,
    k,
    all: bool = True,
    balanced: bool = False,
    top2k: bool = False,
    tox_levels: tuple = (0.5, 0.5),
) -> Tuple[list, list]:
    test = load_from_disk("./datasets/RTP")[
        "test"
    ]
    if balanced:

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
        if not top2k:
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
        else:
            test = test.sort("toxcitity_level")
            test = concatenate_datasets(
                [Dataset.from_dict(test[:2000]), Dataset.from_dict(test[-2000:])]
            )
        dataset_toxic = test.filter(
            lambda x: x["toxcitity_level"] is not None and x["toxcitity_level"] >= 0.5
        )
        dataset_non_toxic = test.filter(
            lambda x: x["toxcitity_level"] is not None and x["toxcitity_level"] < 0.5
        )

    else:
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
            tok_ids = tokenizer(line["text"], return_tensors="pt").input_ids.to(
                "cuda:0"
            )
            llama3_sae(tok_ids)
            acts = llama3_sae.hook.pop()
            latents = llama3_sae.SAE(acts)[1]
            toxic_latents.append([tok_ids.detach().cpu(), latents.detach().cpu(), None])

    torch.cuda.empty_cache()
    gc.collect()

    non_toxic_latents = []
    with torch.no_grad():
        for line in tqdm(dataset_non_toxic):
            tok_ids = tokenizer(line["text"], return_tensors="pt").input_ids.to(
                "cuda:0"
            )
            llama3_sae(tok_ids)
            acts = llama3_sae.hook.pop()
            latents = llama3_sae.SAE(acts)[1]
            non_toxic_latents.append(
                [tok_ids.detach().cpu(), latents.detach().cpu(), None]
            )

    torch.cuda.empty_cache()
    gc.collect()

    toxic_l_bin = []
    toxic_l_sum = []
    toxic_l_mean = []
    for ids, latents, inds in tqdm(toxic_latents):
        l_sum = latents.sum(1)
        l_mean = latents.mean(1)
        l_bin = (latents > 0).sum(1)
        toxic_l_bin.append(l_bin.detach().cpu().numpy()[0])
        toxic_l_sum.append(l_sum.detach().cpu().numpy()[0])
        toxic_l_mean.append(l_mean.detach().cpu().numpy()[0])

    non_toxic_l_bin = []
    non_toxic_l_sum = []
    non_toxic_l_mean = []
    for ids, latents, inds in tqdm(non_toxic_latents):
        l_sum = latents.sum(1)
        l_mean = latents.mean(1)
        l_bin = (latents > 0).sum(1)
        non_toxic_l_bin.append(l_bin.detach().cpu().numpy()[0])
        non_toxic_l_sum.append(l_sum.detach().cpu().numpy()[0])
        non_toxic_l_mean.append(l_mean.detach().cpu().numpy()[0])

    ls = toxic_l_mean + non_toxic_l_mean
    labels = [1 for _ in range(len(toxic_l_sum))] + [
        0 for _ in range(len(non_toxic_l_sum))
    ]

    return ls, labels


def RTP_tree(
    llama3_sae,
    tokenizer,
    file,
    k,
    all: bool = True,
    balanced: bool = False,
    top2k: bool = False,
    class_weight: bool = False,
    tox_levels: tuple = (0.5, 0.5),
):
    ls, labels = RTP_latents(llama3_sae, tokenizer, k, all, balanced, top2k, tox_levels)
    clf = tree.DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        class_weight="balanced" if class_weight else None,
    )
    clf = clf.fit(ls, labels)
    print(clf.get_depth())

    res = {"model": [], "depth": []}
    res["model"].append(model_name)
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
    llama3_sae,
    tokenizer,
    file,
    k,
    all: bool = True,
    balanced: bool = False,
    top2k: bool = False,
    class_weight: bool = False,
    tox_levels: tuple = (0.5, 0.5),
    label_shuffle: bool = False,
):
    ls, labels = RTP_latents(llama3_sae, tokenizer, k, all, balanced, top2k, tox_levels)

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

    res.to_csv(f"{file}_cut.csv")


def SP_latents(llama3_sae, tokenizer, k, valid: bool = True) -> Tuple[list, list]:
    # test = load_dataset("allenai/real-toxicity-prompts", split="train", num_proc=4)
    test = load_from_disk(
        "./datasets/Shakespeare"
    )[f"{'valid' if valid else 'test'}"]

    test = test.map(lambda x: {"label": 0 if x["label"] == "modern" else 1})
    dataset_modern = test.filter(lambda x: x["label"] == 0)
    dataset_original = test.filter(lambda x: x["label"] == 1)

    original_latents = []
    with torch.no_grad():
        for line in tqdm(dataset_original):
            tok_ids = tokenizer(line["text"], return_tensors="pt").input_ids.to(
                "cuda:0"
            )
            llama3_sae(tok_ids)
            acts = llama3_sae.hook.pop()
            latents = llama3_sae.SAE(acts)[1]
            original_latents.append(
                [tok_ids.detach().cpu(), latents.detach().cpu(), None]
            )

    modern_latents = []
    with torch.no_grad():
        for line in tqdm(dataset_modern):
            tok_ids = tokenizer(line["text"], return_tensors="pt").input_ids.to(
                "cuda:0"
            )
            llama3_sae(tok_ids)
            acts = llama3_sae.hook.pop()
            latents = llama3_sae.SAE(acts)[1]
            modern_latents.append(
                [tok_ids.detach().cpu(), latents.detach().cpu(), None]
            )

    modern_l_bin = []
    modern_l_sum = []
    modern_l_mean = []
    for ids, latents, inds in tqdm(modern_latents):
        l_sum = latents.sum(1)
        l_mean = latents.mean(1)
        l_bin = (latents > 0).sum(1)
        modern_l_bin.append(l_bin.detach().cpu().numpy()[0])
        modern_l_sum.append(l_sum.detach().cpu().numpy()[0])
        modern_l_mean.append(l_mean.detach().cpu().numpy()[0])

    original_l_bin = []
    original_l_sum = []
    original_l_mean = []
    for ids, latents, inds in tqdm(original_latents):
        l_sum = latents.sum(1)
        l_mean = latents.mean(1)
        l_bin = (latents > 0).sum(1)
        original_l_bin.append(l_bin.detach().cpu().numpy()[0])
        original_l_sum.append(l_sum.detach().cpu().numpy()[0])
        original_l_mean.append(l_mean.detach().cpu().numpy()[0])

    ls = modern_l_mean + original_l_mean
    labels = [0 for _ in range(len(modern_l_sum))] + [
        1 for _ in range(len(original_l_sum))
    ]

    return ls, labels


def SP_tree(llama3_sae, tokenizer, file, k, valid: bool = True):
    ls, labels = SP_latents(llama3_sae, tokenizer, k, valid)

    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=None)
    clf = clf.fit(ls, labels)
    print(clf.get_depth())

    res = {"model": [], "depth": []}
    res["model"].append(model_name)
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
    df.to_csv(f"{file}_stats.csv")


def cut_SP_tree(
    llama3_sae, tokenizer, file, k, valid: bool = True, label_shuffle: bool = False
):
    ls, labels = SP_latents(llama3_sae, tokenizer, k, valid)

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

    res.to_csv(f"{file}_cut.csv")


if __name__ == "__main__":
    print("Starting")
    load_from_ckpt = True
    model_name = "llama3-gsae"

    ckpt_dirs = {
        "Checkpoint Path": "Result Dir",
    }
    for ckpt_dir, res_path in ckpt_dirs.items():
        for i in [0]:
            logger.info(f"{model_name}_s{i}")
            k = int(model_name.split("-")[-1][1:])
            if res_path != "pretrained":
                _, e, _ = get_ckpts(f"{model_name}_s{i}", dir=ckpt_dir)
                if e != 100 or (i == 1 and res_path == "RTP-Block_v2"):
                    logger.warning(f"{model_name}_s{i} not found in {ckpt_dir}")
                    continue

            try:
                llama3_sae, tokenizer = get_model(
                    model_name,
                    load_from_ckpt,
                    f"_s{i}",
                    "block",
                    ckpt_dir,
                    act="topk-sigmoid" if res_path != "pretrained" else "topk",
                )
            except:
                logger.warning(f"{model_name}_s{i} not found in {ckpt_dir}")
                continue

            skip = False
            skip_sp = True if "RTP" in res_path else False
            skip_rtp = True if "SP" in res_path else False

            # RTP Tree
            file = f"./SAE/llama3_SAE/SAE_eval/{res_path}/rtp_tree_{model_name}_s{i}"
            if os.path.isfile(file + ".pkl"):
                logger.info(f"{res_path}/rtp_tree_{model_name}_s{i}.pkl exists")
            else:
                RTP_tree(llama3_sae, tokenizer, file, k, False)

            # cut RTP Tree
            file = f"./SAE/llama3_SAE/SAE_eval/{res_path}/rtp_tree_{model_name}_s{i}"
            if os.path.isfile(file + "_cut.csv") or skip_rtp:
                logger.info(f"{file}_cut.csv exists")
            else:
                cut_RTP_tree(llama3_sae, tokenizer, file, k, False)

            # cut RTP Tree random labels
            file = f"./SAE/llama3_SAE/SAE_eval/{res_path}/rtp_tree_{model_name}_s{i}r"
            if os.path.isfile(file + "_cut.csv") or skip_rtp:
                logger.info(f"{file}_cut.csv exists")
            else:
                cut_RTP_tree(llama3_sae, tokenizer, file, k, False, label_shuffle=True)

            # RTP Tree ALL
            file = f"./SAE/llama3_SAE/SAE_eval/{res_path}/rtp_tree_test_{model_name}_s{i}"
            if os.path.isfile(file + ".pkl") or skip:
                logger.info(f"{res_path}/rtp_tree_test_{model_name}_s{i}.pkl exists")
            else:
                RTP_tree(llama3_sae, tokenizer, file, k, True)

            # RTP Tree ALL class balanced
            file = f"./SAE/llama3_SAE/SAE_eval/{res_path}/rtp_tree_test_cb_{model_name}_s{i}"
            if os.path.isfile(file + ".pkl") or skip:
                logger.info(f"{res_path}/rtp_tree_test_cb_{model_name}_s{i}.pkl exists")
            else:
                RTP_tree(llama3_sae, tokenizer, file, k, True, class_weight=True)

            # cut RTP Tree ALL class balanced
            file = f"./SAE/llama3_SAE/SAE_eval/{res_path}/rtp_tree_test_cb_{model_name}_s{i}"
            if os.path.isfile(file + "_cut.csv") or skip_rtp:
                logger.info(f"{file}_cut.csv exists")
            else:
                cut_RTP_tree(llama3_sae, tokenizer, file, k, True, class_weight=True)

            file = f"./SAE/llama3_SAE/SAE_eval/{res_path}/rtp_tree_test_cb_{model_name}_s{i}r"
            if os.path.isfile(file + "_cut.csv") or skip_rtp:
                logger.info(f"{file}_cut.csv exists")
            else:
                cut_RTP_tree(
                    llama3_sae,
                    tokenizer,
                    file,
                    k,
                    True,
                    class_weight=True,
                    label_shuffle=True,
                )

            # SP Tree valid
            file = f"./SAE/llama3_SAE/SAE_eval/{res_path}/sp_tree_valid_{model_name}_s{i}"
            if os.path.isfile(file + ".pkl") or skip_sp:
                logger.info(f"{res_path}/sp_tree_valid_{model_name}_s{i}.pkl exists")
            else:
                SP_tree(llama3_sae, tokenizer, file, k, True)

            # cut SP Tree valid
            file = f"./SAE/llama3_SAE/SAE_eval/{res_path}/sp_tree_valid_{model_name}_s{i}"
            if os.path.isfile(file + "_cut.pkl") or skip_sp:
                logger.info(f"{file}_cut.csv exists")
            else:
                cut_SP_tree(llama3_sae, tokenizer, file, k, True)

            file = f"./SAE/llama3_SAE/SAE_eval/{res_path}/sp_tree_valid_{model_name}_s{i}r"
            if os.path.isfile(file + "_cut.pkl") or skip_sp:
                logger.info(f"{file}_cut.csv exists")
            else:
                cut_SP_tree(llama3_sae, tokenizer, file, k, True, label_shuffle=True)

            # # SP Tree test
            file = f"./SAE/llama3_SAE/SAE_eval/{res_path}/sp_tree_test_{model_name}_s{i}"
            if os.path.isfile(file + ".pkl") or skip_sp:
                logger.info(f"{res_path}/sp_tree_test_{model_name}_s{i}.pkl exists")
            else:
                SP_tree(llama3_sae, tokenizer, file, k, False)

            # cut SP Tree test
            file = f"./SAE/llama3_SAE/SAE_eval/{res_path}/sp_tree_test_{model_name}_s{i}"
            if os.path.isfile(file + "_cut.pkl") or skip_sp:
                logger.info(f"{file}_cut.csv exists")
            else:
                cut_SP_tree(llama3_sae, tokenizer, file, k, False)

            file = f"./SAE/llama3_SAE/SAE_eval/{res_path}/sp_tree_test_{model_name}_s{i}r"
            if os.path.isfile(file + "_cut.pkl") or skip_sp:
                logger.info(f"{file}_cut.csv exists")
            else:
                cut_SP_tree(llama3_sae, tokenizer, file, k, False, label_shuffle=True)
