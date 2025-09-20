from datasets import load_dataset, load_from_disk, concatenate_datasets
import torch
import transformers
import re
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from utils.Toxicity_API import tox_classifier
from time import sleep
import os
from torch.utils.data import DataLoader
from collections import defaultdict
import logging
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = "HF_TOKEN"

from eval import test_ppl, preprocess

from utils.sae_loading import get_model
from utils.tree_loader import get_root_node

from aleph_alpha_client import Client
from utils.instruction_grader import InstructionGrader

client = Client(
    token="CLIENT_TOKEN",
    host="HOST",
)

ToxEval = InstructionGrader(client, "llama-3.1-70b-instruct")


def test_tox(model, tokenizer, toks):
    toks = {
        "input_ids": toks.input_ids.cuda(),
        "attention_mask": toks.attention_mask.cuda(),
    }
    new_toks = 32

    with torch.no_grad():
        out = model.generate(
            **toks,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=new_toks,
            pad_token_id=tokenizer.eos_token_id,
        )

        gen_text = tokenizer.batch_decode(out[:, -1 * new_toks :])

        gen_text = [
            t.replace("<|end_of_text|>", "").replace("<|begin_of_text|>", "")
            for t in gen_text
        ]

        with Pool(64) as p:
            tox_new = p.map(tox_classifier, gen_text)

    return (
        tox_new,
        gen_text,
    )


def RTP_ppl(model_name: str, load_from_ckpt: bool, model=None) -> None:
    test = load_dataset(
        "allenai/real-toxicity-prompts",
        split="train",
        num_proc=4,
    )
    test = test.filter(
        lambda x: x["prompt"]["toxicity"] is not None
        and x["continuation"]["toxicity"] is not None

    )

    test = test.map(preprocess, num_proc=4)
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
    tmp = []
    for i in np.linspace(0.05, 1, 20):
        tmp.append(
            test.filter(
                lambda x: x["text_toxicity"] >= i - 0.05 and x["text_toxicity"] < i
            )
            .shuffle(42)
            .select(range(100))
        )
    test = concatenate_datasets(tmp)
    test = DataLoader(test, batch_size=128, num_workers=32)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, padding_side="left"
    )
    tokenizer_label = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer_label.pad_token = tokenizer_label.eos_token

    if model is None:
        llama3_sae, _ = get_model(model_name, load_from_ckpt)
    else:
        llama3_sae = model

    res = defaultdict(list)
    for sent in tqdm(test):
        llama3_sae.hook.mod_features = None
        llama3_sae.hook.mod_threshold = None
        llama3_sae.hook.mod_scaling = None
        toks = tokenizer(sent["text"], return_tensors="pt", padding=True)
        labels = tokenizer_label(
            sent["cont"], return_tensors="pt", padding=True
        ).input_ids.cuda()
        neg_log_likelihood = test_ppl(
            llama3_sae, tokenizer, toks.input_ids.cuda(), labels
        )
        tox, gen_text = test_tox(llama3_sae, tokenizer, toks)
        res["sent"] += sent["text"]
        res["cont"] += sent["cont"]
        res["sent_toxicity"] += sent["text_toxicity"].tolist()
        res["cont_toxicity"] += sent["cont_toxicity"].tolist()

        res["base"] += neg_log_likelihood.cpu().tolist()
        res["gen_text_base"] += gen_text
        res["gen_toxicity_base"] += tox

        for alpha in [-100.0, -50.0, -1.0, 0.0, 1.0, 50.0, 100.0]:
            llama3_sae.hook.mod_features = 0
            llama3_sae.hook.mod_threshold = THRESHOLD
            llama3_sae.hook.mod_scaling = alpha
            neg_log_likelihood = test_ppl(
                llama3_sae, tokenizer, toks.input_ids.cuda(), labels
            )
            tox, gen_text = test_tox(llama3_sae, tokenizer, toks)
            res[alpha] += neg_log_likelihood.cpu().tolist()
            res[f"gen_text_{alpha}"] += gen_text
            res[f"gen_toxicity_{alpha}"] += tox

    df = pd.DataFrame.from_dict(res)
    df = df.replace("\n", "", regex=True)
    return df


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


THRESHOLD = 0.0
load_from_ckpt = True
ckpt_dirs = {
    "Checkpoint Path": "Result Dir",
}
ckpt_dir, save_dir = list(ckpt_dirs.items())[-1]


LATENT = 0


def hook_res_filtered_topk(model, input, output):
    output = list(output)
    signs = torch.sign(llama3_sae.SAE.decoder.weight[:, LATENT])
    topk, indices = torch.topk(llama3_sae.SAE.decoder.weight[:, LATENT].abs(), k=THRESH)
    filtered = (
        torch.zeros_like(llama3_sae.SAE.decoder.weight[:, LATENT]).scatter(
            0, indices, topk
        )
        * signs
    )
    norm_res = output[0].norm(p=2, dim=2)
    norm_gsae = llama3_sae.SAE.decoder.weight[:, LATENT].norm(p=2)
    factor = norm_res / norm_gsae
    scaled_sae_out = (filtered[None, None].T * factor.T).T

    output[0] = output[0] + (scaled_sae_out * FACTOR)
    return tuple(output)


@torch.no_grad()
def hook_res_smart_balanced_topk(model, input, output):
    """
    Hook function to modify the output of a model by applying a smart balanced top-k scaling
    mechanism using latent features and decoder weights.

    Args:
        model (torch.nn.Module): The model to which the hook is attached.
        input (tuple): The input to the model (unused in this function).
        output (tuple): The output from the model, expected to be a tuple where the first
            element is a tensor containing residuals.

    Returns:
        tuple: The modified output tuple with the first element updated by applying the
            scaled SAE output.

    Notes:
        - This function operates with `torch.no_grad()` to avoid tracking gradients.
        - The function computes latent features using the `llama3_sae.SAE.encode` method.
        - Decoder weights are extracted and balanced using the latent values.
        - Norms of residuals and decoder weights are computed to calculate a scaling factor.
        - The scaling factor is applied to adjust the model's output, with an optional
        inversion of latent values based on the `FACTOR` variable.
        - A small epsilon (1e-8) is added to the denominator to prevent division by zero.
    """
    output = list(output)

    # Compute latents and find active features
    latents, _ = llama3_sae.SAE.encode(output[0])

    values = latents[:, :, LATENT : LATENT + 1]
    # Efficiently extract decoder weight for selected features
    decoder_weight = llama3_sae.SAE.decoder.weight[:, LATENT : LATENT + 1]

    # Get relevant residuals
    selected_residuals = output[0]

    # Compute norms efficiently
    norm_res = selected_residuals.norm(p=2, dim=-1, keepdim=True)
    norm_gsae = decoder_weight.norm(p=2, dim=0, keepdim=True).T

    # Compute scaling factor
    factor = norm_res / (norm_gsae + 1e-8)  # Avoid division by zero

    # switch weighing if we want to increase concept, avoid over steering into OOD
    values = 1 - values if FACTOR > 0 else values

    # Balanced decoder weight
    balanced_weights = decoder_weight * values.transpose(1, 2)
    # Compute scaled SAE output
    scaled_sae_out = balanced_weights * factor.transpose(1, 2)
    output[0] += scaled_sae_out.transpose(1, 2) * FACTOR

    return tuple(output)


def tox_eval(data):
    cont = data[1]["completion"]
    while True:
        try:
            return tox_classifier(cont)
        except:
            # print("Waiting...")
            sleep(2)


def LLM_Eval(data):
    prompt = data[1]["prompt"]
    cont = data[1]["completion"]
    model = data[1]["model"]
    counter = 0
    failed = {
        "coherence": False,
        "completion": "",
        "criticism": "",
        "fluency": False,
        "has_correct_grammar_and_spelling": False,
        "input_toxicity_rating": 0.0,
        "is_harmful": False,
        "is_repeating": False,
        "is_toxic_compared_to_input": 0,
        "model": "",
        "prompt": "",
        "quality": -1,
        "toxicity_rating": 0.0,
    }
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
            return ToxEval.grade(prompt, cont, model).__dict__
        except Exception as e:
            counter += 1
            if counter >= 5:
                print(
                    f"Stuck in a loop with error {e}"
                )
                return failed
            print(f"Waiting because of error {e}")
            sleep(2)


if __name__ == "__main__":
    test = load_from_disk("./datasets/RTP")[
        "test"
    ]
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

    tmp = []
    test = DataLoader(test, batch_size=128, num_workers=32)


    for model_name in [
        f"llama3-l{str(j)}-b{str(k)}-k{str(i)}{l}"
        for i in [2048]
        for j in [24576]
        for k in [11]
        for l in ["_s1", "_s0"]
    ]:
        llama3_sae, tokenizer = None, None
        scaling = (re.findall(r"_s\d+", model_name) or ["_s5"])[0]
        model_name = model_name.replace(scaling, "")
        llama3_sae, tokenizer = get_model(
            model_name, load_from_ckpt, scaling, "block", ckpt_dir, act="topk-sigmoid"
        )
        block_num = int(model_name.split("-")[-2][1:])


        LATENT = (
            get_root_node(
                f"./SAE/llama3_SAE/SAE_eval/{save_dir}/rtp_tree_{model_name}{scaling}.pkl"
            )["feature_index"]
            if os.path.isfile(
                f"./SAE/llama3_SAE/SAE_eval/{save_dir}/rtp_tree_{model_name}{scaling}.pkl"
            )
            else 0
        )
        logger.info(
            f"Using model: '{model_name}' with scaling: '{scaling}' and latent feature: '{LATENT}'"
        )
        h_B_2 = llama3_sae.model.layers[block_num].register_forward_hook(
            hook_res_filtered_topk
        )

        model_name = model_name if scaling == "_s5" else model_name + scaling
        # for f in tqdm([0, 1, 2, 3], desc=f"Overall {model_name}"):
        for f in tqdm([0], desc=f"Overall {model_name}"):
            m = llama3_sae.SAE.decoder.weight[:, LATENT].mean()  
            s = llama3_sae.SAE.decoder.weight[:, LATENT].std() * f
            THRESH = (
                (
                    (llama3_sae.SAE.decoder.weight[:, LATENT] > m + s)
                    | (llama3_sae.SAE.decoder.weight[:, LATENT] < m - s)
                )
                .sum()
                .item()
                if f != 0
                else llama3_sae.SAE.decoder.weight[:, LATENT].size()[0]
            )
            toxs = []
            eval_out = defaultdict(list)
            df = pd.DataFrame()
            if os.path.isfile(
                f"./SAE/llama3_SAE/SAE_eval/{save_dir}/rtp-all-{model_name}-{str(THRESH)}.csv"
            ):
                df = pd.read_csv(
                    f"./SAE/llama3_SAE/SAE_eval/{save_dir}/rtp-all-{model_name}-{str(THRESH)}.csv"
                )
            else:
                for elem in tqdm(test, desc="Gen"):
                    text = elem["text"]
                    toks = tokenizer(text, return_tensors="pt", padding=True)
                    toks = {
                        "input_ids": toks.input_ids.to("cuda"),
                        "attention_mask": toks.attention_mask.to("cuda"),
                    }

                    for i in np.linspace(-1.0, 1.0, 11):
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
                            gen_text = tokenizer.batch_decode(out[:, -1 * new_toks :])
                            gen_text = [
                                t.replace("<|end_of_text|>", "").replace(
                                    "<|begin_of_text|>", ""
                                )
                                for t in gen_text
                            ]
                        eval_out["prompt"] += text
                        eval_out["completion"] += gen_text
                        eval_out["alpha"] += [i for _ in range(len(text))]
                        eval_out["start_tox"] += [
                            j.item() for j in elem["toxcitity_level"]
                        ]
                        eval_out["model"] += [model_name for _ in range(len(text))]

                df_dictionary = pd.DataFrame(eval_out)
                df = pd.concat([df, df_dictionary], ignore_index=True)
                df.to_csv(
                    f"./SAE/llama3_SAE/SAE_eval/{save_dir}/rtp-all-{model_name}-{str(THRESH)}.csv"
                )

            if "PersAPI" not in df.columns:
                tmp = process_map(
                    tox_eval,
                    df[["completion"]].iterrows(),
                    max_workers=32,
                    total=len(df),
                )
                df["PersAPI"] = tmp
                df.to_csv(
                    f"./SAE/llama3_SAE/SAE_eval/{save_dir}/rtp-all-{model_name}-{str(THRESH)}.csv"
                )
            print("PersAPi done.")

            if "criticism" not in df.columns:
                tmp2 = process_map(
                    LLM_Eval,
                    df[["prompt", "completion", "model"]].iterrows(),
                    max_workers=32,
                    total=len(df),
                )
                df2 = pd.DataFrame(tmp2)
                df = df.join(df2, rsuffix="_digsaed")
                df = df.loc[:, ~df.columns.str.endswith("_digsaed")]
                df.to_csv(
                    f"./SAE/llama3_SAE/SAE_eval/{save_dir}/rtp-all-{model_name}-{str(THRESH)}.csv"
                )

            print("Llama done.")
        h_B_2.remove()
