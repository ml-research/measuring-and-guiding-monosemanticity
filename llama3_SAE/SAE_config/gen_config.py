import json
from itertools import product

BLOCKS = [11]
LATENTS = [(24576, 6)]
ACT = [2048]

# name_postfix = "_instruct"
# name_postfix = "_70B"
name_postfix = ""
factor = 1
for BLOCK in BLOCKS:
    DATASET = (
        f"./datasets/llama3{name_postfix}-RTP_split-B{BLOCK:02}-block"
    )
    for l, a in product(LATENTS, ACT):
        if (
            l[0] < a
            or (l[0] == 24576 and a == 12288)
            or (l[0] == 49152 and a == 12288)
            or (l[0] == 49152 and a == 24576)
        ):
            continue

        name = f"llama3{name_postfix}-l{l[0] * factor}-b{BLOCK:02}-k{a}"
        print(name)
        with open(
            f"./llama3_SAE/SAE_config/{name}.json",
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
                    "n_inputs": 4096 * factor,
                    "activation": "topk-sigmoid",
                },
                f,
            )

    # break
