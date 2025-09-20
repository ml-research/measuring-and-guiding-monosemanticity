from collections import OrderedDict
import torch
import transformers
from transformers import AutoModelForCausalLM
from accelerate import (
    init_empty_weights,
    dispatch_model,
    infer_auto_device_map,
)
from llama3_SAE.configuration_llama3_SAE import LLama3_SAE_Config
from gemma2_SAE.configuration_gemma2_SAE import Gemma2_SAE_Config
import torch
import numpy as np
import json
from llama3_SAE.modeling_llama3_SAE import (
    LLama3_SAE,
    HookedTransformer_with_SAE_suppresion,
)
from gemma2_SAE.modeling_gemma2_SAE import (
    Gemma2_SAE,
)
from train_SAE import SAE_Train_config
from collections import defaultdict
from typing import Tuple
from glob import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = "YOUR_HF_TOKEN"


def move_sae_to_gpu(device_map: OrderedDict, hook_block_num: int) -> OrderedDict:
    """
    Moves 'SAE' to the corresponding GPU based on the specified layer number.

    Args:
        device_map (OrderedDict): PyTorch device map of model layers.
        hook_block_num (int): The layer number whose GPU assignment should be used for 'SAE'.

    Returns:
        OrderedDict: Updated device map with 'SAE' assigned to the correct GPU.
    """
    # Determine the GPU for the specified layer
    layer_key = f"model.layers.{hook_block_num}"
    if layer_key in device_map:
        sae_gpu = device_map[layer_key]

        # Update 'SAE' if it exists in the device map
        if "SAE" in device_map:
            device_map["SAE"] = sae_gpu

    return device_map


def get_ckpts_pretrained(model: str = None, dir: str = None):
    if dir is None:
        raise AttributeError

    path = f"./ckpt/{dir}/checkpoints/{model.split('-')[-2]}"

    if model is None:
        return glob(
            f"./ckpt/{dir}/checkpoints/*.pth"
        )
    else:
        return (
            f"{path}/state_dict.pth",
            -1,
            -1,
        )


def get_ckpts(model: str = None, dir: str = None, epoch: int = 100):
    ckpts = defaultdict(
        lambda: {"epochs": 0, "steps": 0, "ckpt_path": "", "loss_scaling": "1"}
    )
    if dir is None:
        raise AttributeError

    for path in glob(
        f"./ckpt/{dir}/checkpoints/*"
    ):
        try:
            with open(
                f"{path}/load_data.json",
                "r",
            ) as f:
                load_data = json.load(f)
                model_name = (
                    load_data["hparams"]["config_path"].split("/")[-1].split(".")[0]
                )
                try:
                    loss_scaling = "_s" + str(load_data["hparams"]["cond_loss_scaling"])
                except:
                    loss_scaling = ""

                model_name += str(loss_scaling)
        except FileNotFoundError:
            continue

        trial_state = np.load(
            f"{path}/trial_state.pkl",
            allow_pickle=True,
        )
        if trial_state["epochs_trained"] == epoch:
            ckpts[model_name]["ckpt_path"] = f"{path}/state_dict.pth"
            ckpts[model_name]["epochs"] = trial_state["epochs_trained"]
            ckpts[model_name]["steps"] = trial_state["batches_trained"]

    if model is None:
        return ckpts
    else:
        return (
            ckpts[model]["ckpt_path"],
            ckpts[model]["epochs"],
            ckpts[model]["steps"],
        )


def get_model(
    model_name,
    load_from_ckpt,
    scaling="",
    site="mlp",
    dir: str = None,
    act: str = None,
    epoch: int = 100,
) -> Tuple[AutoModelForCausalLM, transformers.AutoTokenizer]:
    model_base_type = model_name.split("-")[0].split("_")[0]
    with open(
        f"./{model_base_type}_SAE/SAE_config/{model_name}.json"
    ) as f:
        conf_as_json = json.load(f)
        conf = SAE_Train_config(**conf_as_json)

    model_type = model_name.split("-")[0]
    if model_type == "llama3":
        model_hf = "meta-llama/Meta-Llama-3-8B"
        llama_conf_path = (
            "./llama3_SAE/config.json"
        )
    elif model_type == "gemma2":
        model_hf = "google/gemma-2-9b"
        llama_conf_path = (
            "./gemma2_SAE/config.json"
        )
    elif model_type == "llama3_instruct":
        model_hf = "meta-llama/Meta-Llama-3-8B-Instruct"
        llama_conf_path = (
            "./llama3_SAE/config.json"
        )
    elif model_type == "llama3_70B":
        model_hf = "meta-llama/Meta-Llama-3-70B"
        llama_conf_path = (
            "./llama3_SAE/config_70B.json"
        )
    elif model_type == "llama3_70B_instruct":
        model_hf = "meta-llama/Meta-Llama-3-70B-Instruct"
        llama_conf_path = (
            "./llama3_SAE/config_70B.json"
        )
    else:
        raise NotImplementedError

    logger.info(f"Trying to initialize model: {model_hf}")

    with open(llama_conf_path) as f:
        llm_sae_config = json.load(f)

    with init_empty_weights():
        if model_base_type == "llama3":
            llm_sae_config = LLama3_SAE_Config(**llm_sae_config)
        elif model_base_type == "gemma2":
            llm_sae_config = Gemma2_SAE_Config(**llm_sae_config)

        llm_sae_config.n_latents = conf.n_latents
        llm_sae_config.hook_block_num = int(model_name.split("-")[-2][1:])
        llm_sae_config.activation = conf.activation if act is None else act
        llm_sae_config.activation_k = conf.k
        llm_sae_config.site = site
        llm_sae_config.base_model_name = model_hf
        if model_base_type == "llama3":
            llm_sae = LLama3_SAE(llm_sae_config)
        elif model_base_type == "gemma2":
            llm_sae = Gemma2_SAE(llm_sae_config)

    model_base = transformers.AutoModelForCausalLM.from_pretrained(
        model_hf,
        token="YOUR_HF_TOKEN",
        cache_dir="./upload",
    )
    logger.info(f"Loaded model: {model_hf}.")

    llm_sae.model = model_base.model
    llm_sae.lm_head = model_base.lm_head
    llm_sae = llm_sae.half()

    if load_from_ckpt:
        if "pretrained" in conf.ds_path:
            (ckpt_path, epoch, step) = get_ckpts_pretrained(model_name, dir)
            logger.info(
                f"Loading {model_name} from checkpoint at {epoch}, {step} and scaling {scaling} in dir {dir}"
            )
        else:
            (ckpt_path, epoch, step) = get_ckpts(model_name + scaling, dir, epoch)
            logger.info(
                f"Loading {model_name} from checkpoint at {epoch}, {step} and scaling {scaling} in dir {dir}"
            )

        sae_state_dict = torch.load(ckpt_path, weights_only=False)["models_state_dict"][
            0
        ]
        try:
            llm_sae.SAE.load_state_dict(sae_state_dict, strict=True, assign=True)
        except:
            # needs to be done because of tied weights of encoder and decoder layer
            sae_state_dict["decoder.weight"] = (
                sae_state_dict["decoder.linear.weight"].clone().T
            )
            sae_state_dict.pop("decoder.linear.weight")
            llm_sae.SAE.load_state_dict(sae_state_dict, strict=True, assign=True)
        llm_sae.hook.remove_handle
        llm_sae.hook = HookedTransformer_with_SAE_suppresion(
            block=llm_sae_config.hook_block_num,
            sae=llm_sae.SAE,
            mod_features=None,
            mod_threshold=None,
            mod_replacement=None,
            mod_scaling=None,
        ).register_with(llm_sae.model, site)

    llm_sae = llm_sae.half()
    device_map = infer_auto_device_map(
        llm_sae,
        no_split_module_classes=[
            "LlamaDecoderLayer",
            "LlamaMLP",
            "LlamaRMSNorm",
            "LlamaAttention",
            "LlamaFlashAttention2",
            "LlamaSdpaAttention",
            "Linear",
            "SAE",
        ],
        dtype="float16",
        max_memory={0: "75GB", 1: "75GB"},
    )

    device_map = move_sae_to_gpu(device_map, llm_sae_config.hook_block_num)
    logger.info(f"Dispatching model to {device_map}")

    llm_sae = dispatch_model(llm_sae, device_map=device_map, force_hooks=True)
    llm_sae.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_hf, token=HF_TOKEN, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(
        f"CUDA Memory Allocated: {torch.cuda.memory_allocated()}"
    )  # Memory currently allocated
    logger.info(
        f"CUDA Memory Reserved: {torch.cuda.memory_reserved()}"
    )  # Total memory reserved by caching allocator
    logger.info(torch.cuda.memory_summary())  # Full memory report

    return llm_sae, tokenizer
