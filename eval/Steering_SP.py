from datasets import load_from_disk
import torch
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
import math
from collections import defaultdict
import logging
from multiprocessing import Pool
from time import sleep
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from tqdm.contrib.concurrent import process_map
from utils.sae_loading import get_model
from utils.tree_loader import get_root_node
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = "HF_TOKEN"


from aleph_alpha_client import Client
from utils.instruction_grader import InstructionGrader

client = Client(
    token="CLIENT_TOKEN",
    host="HOST",
)
ToxEval = InstructionGrader(client, "llama-3.1-70b-instruct")

THRESHOLD = 0.0
load_from_ckpt = True
model_name = "llama3-l24576-b11-k2048"
ckpt_dirs = {
    "Checkpoint Path": "Result Dir",
}
ckpt_dir, save_dir = list(ckpt_dirs.items())[-1]


ds = load_from_disk("./datasets/Shakespeare")[
    "valid"  # Play: Twelfth Night - 1218 sentences
    # "test"  # Play: Romeo and Juliet - 1462 sentences
]
ds = ds.map(lambda x: {"label": 0 if x["label"] == "modern" else 1})
ppl_ds = load_from_disk(
    "./datasets/Shakespeare"
)[
    "valid"  # Play: Twelfth Night - 1218 sentences
    # "test"  # Play: Romeo and Juliet - 1462 sentences
]
ppl_ds = ds.map(lambda x: {"label": 0 if x["label"] == "modern" else 1})
num_rows = (ppl_ds.num_rows // 2) - 1

FACTOR = 0
THRESH = 0
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
        "has_correct_modern_grammar_and_spelling": False,
        "has_correct_shakespearean_grammar_and_spelling": False,
        "input_shakespearean_grade": 0.0,
        "is_harmonious_with_input": False,
        "style_shift_evaluation": 0,
        "is_repeating": False,
        "model": "llama3-l24576-b25-k2048",
        "prompt": "",
        "style_alignment": -1,
        "shakespearean_grade": 0.0,
    }
    if (
        cont == ""
        or str(cont).isspace()
        or (isinstance(cont, float) and math.isnan(cont))
    ):
        return failed
    while True:
        try:
            out = ToxEval.grade(prompt, cont, model, "SP").__dict__
            return out
        except Exception as e:
            counter += 1
            if counter >= 5:
                print(f"Waiting because of error {e}")
                print(f"Input eval input\nprompt: {prompt}\ncompletion:{cont}")
                print(f"Stuck in a loop with error {e}")
                return failed
            sleep(1)


def LLM_Grammar_Eval(data):
    prompt = data[1]["prompt"]
    cont = data[1]["completion"]
    model = data[1]["model"]
    counter = 0
    failed = {
        "is_grammatically_correct": False,
        "is_style_appropriate": False,
        "failed": True,
        "prompt": "",
        "completion": "",
    }
    if (
        cont == ""
        or str(cont).isspace()
        or (isinstance(cont, float) and math.isnan(cont))
    ):
        return failed
    while True:
        try:
            out = ToxEval.grade(prompt, cont, model, "grammar").__dict__
            return out
        except Exception as e:
            counter += 1
            if counter >= 5:
                print(f"Waiting because of error {e}")
                print(f"Input eval input\nprompt: {prompt}\ncompletion:{cont}")
                print(f"Stuck in a loop with error {e}")
                return failed
            sleep(1)


def get_loss(start, end):
    input_ids = torch.concat([start, end], dim=1)
    end = torch.where(end == 128001, -100, end)
    target_ids = torch.concat([torch.ones_like(start) * -100, end], dim=1)
    return llama3_sae(input_ids, labels=target_ids).loss.item()


id2label = {0: "modern", 1: "shakespearean"}
label2id = {"modern": 0, "shakespearean": 1}

sp_classifier_model = AutoModelForSequenceClassification.from_pretrained(
    # "distilbert/distilbert-base-uncased",
    "./SAE/llama3_SAE/Shakespeare_Classifier/checkpoint-4600",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)
sp_classifier_tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased"
)
sp_classifier = pipeline(
    "sentiment-analysis", model=sp_classifier_model, tokenizer=sp_classifier_tokenizer
)

test = DataLoader(ds, batch_size=128, num_workers=32)

for model_name in [
    f"llama3-l{str(j)}-b{k:02}-k{str(i)}{l}"
    for i in [2048]
    # for i in [32, 64, 128, 1024, 2048, 4096]
    # for j in [12288]
    for j in [24576]
    # for j in [49152]
    # for j in [12288, 24576, 49152]
    for k in [11]
    for l in ["_s1", "_s0"]
]:

    llama3_sae, tokenizer = None, None
    scaling = (re.findall(r"_s\d+", model_name) or ["_s1"])[0]
    model_name = model_name.replace(scaling, "")
    llama3_sae, tokenizer = get_model(
        model_name, load_from_ckpt, scaling, "block", ckpt_dir, act="topk-sigmoid"
    )
    block_num = int(model_name.split("-")[-2][1:])
    LATENT = (
        get_root_node(
            f"./SAE/llama3_SAE/SAE_eval/{save_dir}/sp_tree_valid_{model_name}{scaling}.pkl"
        )["feature_index"]
        if os.path.isfile(
            f"./SAE/llama3_SAE/SAE_eval/{save_dir}/sp_tree_valid_{model_name}{scaling}.pkl"
        )
        else 0
    )
    logger.info(
        f"Using model: '{model_name}' with scaling: '{scaling}' and latent feature: '{LATENT}'"
    )

    h_B_2 = llama3_sae.model.layers[block_num].register_forward_hook(
        hook_res_filtered_topk
    )


    model_name = model_name if scaling == "_s1" else model_name + scaling


    for f in tqdm([0], desc=f"Overall {model_name}"):
        m = llama3_sae.SAE.decoder.weight[:, LATENT].mean().item()
        s = llama3_sae.SAE.decoder.weight[:, LATENT].std().item() * f
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
        df = pd.DataFrame()
        eval_out = defaultdict(list)
        if os.path.isfile(
            f"./SAE/llama3_SAE/SAE_eval/{save_dir}/SP-{model_name}-{str(THRESH)}.csv"
        ):
            df = pd.read_csv(
                f"./SAE/llama3_SAE/SAE_eval/{save_dir}/SP-{model_name}-{str(THRESH)}.csv"
            )
        else:
            for elem in tqdm(test, desc="Gen"):
                text = elem["text"]
                toks = tokenizer(text, return_tensors="pt", padding=True)
                toks = {
                    "input_ids": toks.input_ids.cuda(),
                    "attention_mask": toks.attention_mask.cuda(),
                }

                for i in np.linspace(-2.0, 2.0, 21):
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
                    eval_out["label"] += [j.item() for j in elem["label"]]
                    eval_out["model"] += [model_name for _ in range(len(text))]

            df_dictionary = pd.DataFrame(eval_out)
            df = pd.concat([df, df_dictionary], ignore_index=True)
            df.to_csv(
                f"./SAE/llama3_SAE/SAE_eval/{save_dir}/SP-{model_name}-{str(THRESH)}.csv"
            )

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
                f"./SAE/llama3_SAE/SAE_eval/{save_dir}/SP-{model_name}-{str(THRESH)}.csv"
            )
        print("LLM eval done.")

        if "is_grammatically_correct" not in df.columns:
            tmp2 = process_map(
                LLM_Grammar_Eval,
                df[["prompt", "completion", "model"]].iterrows(),
                max_workers=32,
                total=len(df),
            )
            df2 = pd.DataFrame(tmp2)
            df = df.join(df2, rsuffix="_digsaed")
            df = df.loc[:, ~df.columns.str.endswith("_digsaed")]
            df.to_csv(
                f"./SAE/llama3_SAE/SAE_eval/{save_dir}/SP-{model_name}-{str(THRESH)}.csv"
            )
        print("LLM Grammar eval done.")

        if "class_label_bool" not in df.columns:
            TMP = df["completion"].apply(
                lambda text: sp_classifier(text)
                if isinstance(text, str)
                else [{"label": "", "score": 0.0}]
            )
            df["class_label"] = TMP.apply(
                lambda x: x[0]["label"] if isinstance(x, list) else x["label"]
            )
            df["class_score"] = TMP.apply(
                lambda x: x[0]["score"] if isinstance(x, list) else x["score"]
            )
            df["class_label_bool"] = df["class_label"].apply(
                lambda x: label2id[x] if x != "" else 0
            )
            df.to_csv(
                f"./SAE/llama3_SAE/SAE_eval/{save_dir}/SP-{model_name}-{str(THRESH)}.csv"
            )
        print("Classifier eval done.")

        if not os.path.isfile(
            f"./SAE/llama3_SAE/SAE_eval/{save_dir}/SP-ppl-{model_name}-{str(THRESH)}.csv"
        ):
            res = defaultdict(list)
            for j in tqdm(range(num_rows)):
                modern_start = ppl_ds[j]["text"]
                modern_end = ppl_ds[j + 1]["text"] + " "
                sp_start = ppl_ds[num_rows + j]["text"]
                sp_end = ppl_ds[num_rows + j + 1]["text"] + " "
                t_modern_start = tokenizer.encode(
                    modern_start, return_tensors="pt"
                ).cuda()
                t_modern_end = tokenizer.encode(modern_end, return_tensors="pt").cuda()[
                    :, 1:
                ]
                t_sp_start = tokenizer.encode(sp_start, return_tensors="pt").cuda()
                t_sp_end = tokenizer.encode(sp_end, return_tensors="pt").cuda()[:, 1:]
                for i in np.linspace(-1.0, 1.0, 11):
                    FACTOR = i
                    res["Modern -> Modern"].append(
                        get_loss(t_modern_start, t_modern_end)
                    )
                    res["Modern -> SP"].append(get_loss(t_modern_start, t_sp_end))
                    res["SP -> Modern"].append(get_loss(t_sp_start, t_modern_end))
                    res["SP -> SP"].append(get_loss(t_sp_start, t_sp_end))
                    res["alpha"].append(FACTOR)
                    res["modern_start"].append(modern_start)
                    res["modern_end"].append(modern_end)
                    res["sp_start"].append(sp_start)
                    res["sp_end"].append(sp_end)

            df_ppl = pd.DataFrame.from_dict(res)
            df_ppl.to_csv(
                f"./SAE/llama3_SAE/SAE_eval/{save_dir}/SP-ppl-{model_name}-{str(THRESH)}.csv"
            )
            print("PPL done.")

    h_B_2.remove()
