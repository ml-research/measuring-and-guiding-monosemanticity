# Measuring and Guiding Monosemanticity

Supplementary material for the paper ["Measuring and Guiding Monosemanticity"](https://arxiv.org/abs/2506.19382).

The repository consits of a number of scripts that help reproduce the experiments and corresponding results of the paper. 

## Installation

The environment was created with the help of [poetry](https://python-poetry.org/).
To install the environment:
1. Install poetry: `pip install poetry`
2. Run: `poetry install` inside this folder

## Overview Dirs
- `create_data`: Scripts to create the used datasets from hugginface or other sources
- `eval`: Evaluation scripts for the three datasets
- `gemma2_SAE`: Wrapper for Huggingface transformer model [Gemma2](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)
- `llama3_SAE`: Wrapper for Huggingface transformer model [Llama3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6)
- `train`: Train script to use on [determined](https://docs.determined.ai/) clusters
- `utils`: Utilities used in the folders above


## Create Data
To speed up training or evaluation time, you can create the datasets before hand. 
The script `act_dataset.py` takes four arguments:
1. The shorthand for the dataset, e.g. `SP` for the Shakespeare dataset.
2. The hookpoint, e.g. `25` is the 25th block
3. The model name, in this case `llama3` for the llama3-8B model
4. Where to hook in the specified block, `block` is the residual stream after the 25th block. `mlp` would be the output of the mlp layer. 

```bash
poetry run python ./create_data/act_dataset.py SP 25 llama3 block
```

The result will be saved in `./datasets_v2`.

## Train
To train a SAE on one of the created datasets, you first need to create a config, either be looking at one of the sample configs in `./llama3_SAE/SAE_config` or generation one with `./llama3_SAE/SAE_config/gen_config.py`

For a determined cluster one sample determined config file can be found here: `./train/train_SAE_24k_k2048.yaml`. There you have to enter your config file or use on of the sample one.

## Eval
The files are named corresponding to what they evaluate. RTP, SP, PII stand for the datasets, named as in the paper described. Steering, FMS, Feautres correspond to the conducted experiments. 

If the SAEs where trained with the inclued train script, it suffices to insert the checkpoint path into the evaluation scipts, a loding method for this case is provided (`utils/sae_loading.py`). Otherwise, the safed SAEs need to be modified or another loding method needs to be implemented.

## Other Experiments
The repository of [ICV](https://github.com/shengliu66/ICV) was adapted to acomodate DiffVec and the datasets mentioned in the paper.
The repository of [Model Arithmetic](https://github.com/eth-sri/language-model-arithmetic) already included PreAdd but needed modification in order to accomodate all datasets from the paper.
Hyperparameters of all 4 methods can be found in Appendix D. 


# Citation

```bibtex
@inproceedings{harle2025monosemanticity,
    title = {Measuring and Guiding Monosemanticity},
    author = {Ruben H{\"a}rle and Felix Friedrich and Manuel Brack and Stephan W{\"a}ldchen and Bj{\"o}rn Deiseroth and
    Patrick Schramowski and Kristian Kersting},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2025},
    note = {Spotlight}
    }
```