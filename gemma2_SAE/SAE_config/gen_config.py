import json
from itertools import product

HIDDEN_DIM = 3584
BLOCKS = [i for i in range(1, 42)]
# BLOCKS = [5, 15, 24, 25, 26]
# BLOCKS = [62]
# BLOCKS = [11]
# LATENTS = [(12288, 3), (24576, 6), (49152, 12)]
# LATENTS = [(12288, 3)]
LATENTS = [(HIDDEN_DIM * 6, 6)]
# LATENTS = [(49152, 12)]
# ACT = [4096]
ACT = [2048]
# ACT = [1024]
# ACT = [128]
# ACT = [32, 64, 128, 1024, 2048, 4096]
# ACT = [32, 64, 128, 1024, 2048, 4096, 12288, 24576, 49152]

# name_postfix = "_instruct"
# name_postfix = "_70B"
name_postfix = ""
factor = 1
for BLOCK in BLOCKS:
    DATASET = (
        # f"./datasets/gemma2{name_postfix}-RTP_split-B{BLOCK:02}-block"
        # f"./datasets/gemma2{name_postfix}-SP-B{BLOCK:02}-block"
        f"./datasets_v2/gemma2{name_postfix}-RTP_split-B{BLOCK:02}-block"
        # f"./datasets_v2/gemma2{name_postfix}-Combi_SP-tox_211-B{BLOCK:02}-block"
        # f"./datasets_v2/gemma2{name_postfix}-Combi_SP-tox_211-B{BLOCK:02}-block"
        # f"./datasets_v2/gemma2{name_postfix}-PII_300k-B{BLOCK:02}-block"
        # f"./datasets_v2/gemma2{name_postfix}-PII_300k_mult_lang-B{BLOCK:02}-block"
        # f"./datasets_v2/gemma2{name_postfix}-SP-B{BLOCK:02}-block"
    )
    for l, a in product(LATENTS, ACT):
        if (
            l[0] < a
            or (l[0] == 24576 and a == 12288)
            or (l[0] == 49152 and a == 12288)
            or (l[0] == 49152 and a == 24576)
        ):
            continue

        name = f"gemma2{name_postfix}-l{l[0] * factor}-b{BLOCK:02}-k{a}"
        print(name)
        with open(
            f"./gemma2_SAE/SAE_config/{name}.json",
            "w",
        ) as f:
            json.dump(
                {
                    "batch_size": 2048,
                    "ds_path": DATASET,
                    "factor": l[1],
                    "k": a,
                    "lr": 0.00001,
                    "max_epochs": 100,
                    "n_inputs": 3584 * factor,
                    # "activation": "topk",
                    "activation": "topk-sigmoid",
                    # "ckpt_path": ckpt,
                },
                f,
            )

    # break
