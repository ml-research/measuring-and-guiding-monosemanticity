import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from datasets import Dataset as DS
from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    load_from_disk,
    concatenate_datasets,
)
import transformers
from typing import List, Tuple
from random import shuffle, randint
from tqdm import tqdm
from create_dataset import HookedTransformer, Encode
import multiprocess
import sys

HF_TOKEN = "YOUR_HF_TOKEN"

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
}


def get_SP_dataset():
    """
    Creates Shakespeare dataset modern vs original based on data from: https://github.com/harsh19/Shakespearizing-Modern-English.git
    """
    ds = {
        "train": {"text": [], "label": []},
        "valid": {"text": [], "label": []},
        "test": {"text": [], "label": []},
    }

    for stage in ["train", "valid", "test"]:
        for label in ["modern", "original"]:
            with open(
                f"./Shakespearizing-Modern-English/data/{stage}.{label}.nltktok",
                "r",
            ) as f:
                sents = [
                    sent.replace("\n", "")
                    for sent in tqdm(f.readlines(), desc=f"{stage}: {label}")
                ]
                labels = [label for _ in range(len(sents))]

            ds[stage]["text"] += sents
            ds[stage]["label"] += labels

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(ds["train"]),
            "valid": Dataset.from_dict(ds["valid"]),
            "test": Dataset.from_dict(ds["test"]),
        }
    )
    print(dataset)
    dataset.save_to_disk(
        "./datasets/Shakespeare",
        num_proc=32,
    )
    return dataset


class ActivationsDataset(Dataset):
    """Activation Dataset of ."""

    def __init__(
        self,
        ds_path: str = "wikipedia",
        ds_name: str = "20220301.simple",
        ds_split: str = "train",
        ds_cache_dir: str = "./cache",
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        hooks: int = -1,
        site: str = "mlp",
        batch_size: int = 64,
        chunk_size: int = 100000,
    ) -> None:
        """
        Arguments:

        """
        self.chunk_size = chunk_size
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, token=HF_TOKEN, device_map="auto"
        )
        self.model = model.half()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, token=HF_TOKEN
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

        for parameter in self.model.model.parameters():
            parameter.requires_grad = False

        self.Hook = HookedTransformer(hooks).register_with(self.model, site)

        self.ds_path = ds_path
        self.ds_name = ds_name
        self.ds_split = ds_split
        self.ds_cache_dir = ds_cache_dir
        if (
            ds_path == "allenai/real-toxicity-prompts"
        ):
            self.remove_columns = [
                "filename",
                "begin",
                "end",
                "challenging",
                "prompt",
                "continuation",
            ]
        self.toxic_counter = {"toxic": 0, "non-toxic": 0}
        self.loaded_dataset = self.get_dataset()
        self.iter_dataset = iter(self.loaded_dataset)
        self.samples = self.gen_samples()
        logger.info(f"Label Distribution: {self.toxic_counter}")
        self.counter: int = 0
        self.counter_limit = self.chunk_size // batch_size

    def get_dataset(self):
        dataset = load_dataset(
            self.ds_path,
            self.ds_name,
            split=self.ds_split,
            cache_dir=self.ds_cache_dir,
            streaming=True,
        )
        dataset = dataset.map(
            self.encode,
            batched=True,
            batch_size=100,
            remove_columns=self.remove_columns,
        )

        return dataset

    def encode(self, example):
        with torch.no_grad():
            toks = self.tokenizer(
                [
                    p["text"] + c["text"]
                    for p, c in zip(example["prompt"], example["continuation"])
                ],
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).input_ids

            text = [self.tokenizer.convert_ids_to_tokens(tok) for tok in toks]
            toxicity = []
            for p, c in zip(example["prompt"], example["continuation"]):
                if p["toxicity"] is not None and c["toxicity"] is not None:
                    toxicity.append((p["toxicity"] + c["toxicity"]) / 2)
                elif p["toxicity"] is None and c["toxicity"] is not None:
                    toxicity.append(c["toxicity"])
                elif p["toxicity"] is not None and c["toxicity"] is None:
                    toxicity.append(p["toxicity"])
                else:
                    toxicity.append(0)

            out = self.model(toks.cuda())
            acts = self.Hook.pop().detach().cpu()  

            encoded = {
                "text": [],
                "input_ids": [],
                "acts": [],
                "toxicity": [],
            }

            for sent, tok, act, tox in zip(text, toks, acts, toxicity):
                if tox >= 0.5:
                    toxic = 1
                    self.toxic_counter["toxic"] += 1
                else:
                    toxic = 0
                    self.toxic_counter["non-toxic"] += 1

                for s, t, a in zip(sent, tok, act):
                    if (
                        s == "<|end_of_text|>"
                        or s == "<|begin_of_text|>"
                        or s == "<bos>"
                        or s == "<eos>"
                    ):
                        continue
                    encoded["text"].append(s)
                    encoded["input_ids"].append(t)
                    encoded["acts"].append(a)
                    encoded["toxicity"].append(toxic)
        
        return encoded

    def encode_rtp(self, example):
        with torch.no_grad():
            toks = self.tokenizer(
                [p["text"] for p in example["prompt"]],
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).input_ids

            text = [self.tokenizer.convert_ids_to_tokens(tok) for tok in toks]
            toxicity = []
            for p in example["prompt"]:
                if p["toxicity"] is not None:
                    toxicity.append(p["toxicity"])
                else:
                    toxicity.append(0)

            out = self.model(toks.cuda())
            acts = self.Hook.pop().detach().cpu()  

            encoded = {
                "text": [],
                "input_ids": [],
                "acts": [],
                "toxicity": [],
            }

            for sent, tok, act, tox in zip(text, toks, acts, toxicity):
                if tox >= 0.5:
                    toxic = 1
                    self.toxic_counter["toxic"] += 1
                else:
                    toxic = 0
                    self.toxic_counter["non-toxic"] += 1

                for s, t, a in zip(sent, tok, act):
                    if (
                        s == "<|end_of_text|>"
                        or s == "<|begin_of_text|>"
                        or s == "<bos>"
                        or s == "<eos>"
                    ):
                        continue
                    encoded["text"].append(s)
                    encoded["input_ids"].append(t)
                    encoded["acts"].append(a)
                    encoded["toxicity"].append(toxic)

        
        return encoded

    # wiki dataset
    def encode_wiki(self, example):
        with torch.no_grad():
            torch.cuda.empty_cache()
            toks = self.tokenizer(
                example["text"],
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).input_ids

            text = [self.tokenizer.convert_ids_to_tokens(tok) for tok in toks]

            out = self.model(toks.cuda())
            acts = self.Hook.pop().detach().cpu()  # .half()

            encoded = {
                "text": [],
                "input_ids": [],
                "acts": [],
                "toxicity": [],
            }

            for sent, tok, act in zip(text, toks, acts):
                for s, t, a in zip(sent, tok, act):
                    if (
                        s == "<|end_of_text|>"
                        or s == "<|begin_of_text|>"
                        or s == "<bos>"
                        or s == "<eos>"
                    ):
                        continue
                    encoded["text"].append(s)
                    encoded["input_ids"].append(t)
                    encoded["acts"].append(a)
                    encoded["toxicity"].append(-1)
        
        return encoded

    def encode_SP(self, example):
        with torch.no_grad():
            torch.cuda.empty_cache()
            toks = self.tokenizer(
                example["text"],
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).input_ids

            text = [self.tokenizer.convert_ids_to_tokens(tok) for tok in toks]

            out = self.model(toks.cuda())
            acts = self.Hook.pop().detach().cpu()  # .half()

            encoded = {
                "text": [],
                "input_ids": [],
                "acts": [],
                "label": [],
            }

            
            for sent, tok, act, label in zip(text, toks, acts, example["label"]):
                l = 1 if label == "original" else 0
                for s, t, a in zip(sent, tok, act):
                    if (
                        s == "<|end_of_text|>"
                        or s == "<|begin_of_text|>"
                        or s == "<bos>"
                        or s == "<eos>"
                    ):
                        continue
                    encoded["text"].append(s)
                    encoded["input_ids"].append(t)
                    encoded["acts"].append(a)
                    encoded["label"].append(l)
        
        return encoded

    def encode_combi(self, example):
        with torch.no_grad():
            torch.cuda.empty_cache()
            toks = self.tokenizer(
                example["text"],
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).input_ids

            text = [self.tokenizer.convert_ids_to_tokens(tok) for tok in toks]

            out = self.model(toks.cuda())
            acts = self.Hook.pop().detach().cpu()

            encoded = {
                "text": [],
                "input_ids": [],
                "acts": [],
                "label": [],
            }

            for sent, tok, act, label in zip(text, toks, acts, example["label"]):
                for s, t, a in zip(sent, tok, act):
                    if (
                        s == "<|end_of_text|>"
                        or s == "<|begin_of_text|>"
                        or s == "<bos>"
                        or s == "<eos>"
                    ):
                        continue
                    encoded["text"].append(s)
                    encoded["input_ids"].append(t)
                    encoded["acts"].append(a)
                    encoded["label"].append(label)
        
        return encoded

    def encode_CSD(self, example):
        with torch.no_grad():
            torch.cuda.empty_cache()
            toks = self.tokenizer(
                example["text"],
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).input_ids

            text = [self.tokenizer.convert_ids_to_tokens(tok) for tok in toks]

            out = self.model(toks.cuda())
            acts = self.Hook.pop().detach().cpu()

            encoded = {
                "text": [],
                "input_ids": [],
                "acts": [],
                "toxicity": [],
            }

            
            for sent, tok, act, label in zip(text, toks, acts, example["toxicity"]):
                l = label
                for s, t, a in zip(sent, tok, act):
                    if (
                        s == "<|end_of_text|>"
                        or s == "<|begin_of_text|>"
                        or s == "<bos>"
                        or s == "<eos>"
                    ):
                        continue
                    encoded["text"].append(s)
                    encoded["input_ids"].append(t)
                    encoded["acts"].append(a)
                    encoded["toxicity"].append(l)
        
        return encoded

    def encode_PII(self, example):
        with torch.no_grad():
            torch.cuda.empty_cache()
            input_text = example["source_text"]
            masks = example["privacy_mask"]
            all_input_ids, labels = self.batch_wrapper(input_text, masks)

            text = [self.tokenizer.convert_ids_to_tokens(tok) for tok in all_input_ids]
            all_input_ids = torch.stack(all_input_ids, dim=0)

            out = self.model(all_input_ids.cuda())
            acts = self.Hook.pop().detach().cpu()

            encoded = {
                "text": [],
                "input_ids": [],
                "acts": [],
                "label": [],
            }

            for sent, tok, act, label in zip(text, all_input_ids, acts, labels):
                for s, t, a, l in zip(sent, tok, act, label):
                    if (
                        s == "<|end_of_text|>"
                        or s == "<|begin_of_text|>"
                        or s == "<bos>"
                        or s == "<eos>"
                    ):
                        continue

                    encoded["text"].append(s)
                    encoded["input_ids"].append(t)
                    encoded["acts"].append(a)
                    encoded["label"].append(
                        one_hot(
                            torch.tensor(PII_LABELS_TO_ID[l]),
                            num_classes=len(PII_LABELS_TO_ID),
                        )
                        if l != "O"
                        else torch.zeros(len(PII_LABELS_TO_ID), dtype=torch.int)
                    )

        return encoded

    def convert_to_sequence(self, input_list, total_length):
        """
        Convert a list of labeled segments into a sequential list with gaps labeled as 'O'.

        :param input_list: List of dictionaries with 'value', 'start', 'end', and 'label' keys.
        :param total_length: Total length of the sequence.
        :return: Transformed list with gaps labeled as 'O'.
        """
        result = []
        previous_end = 0

        for entry in input_list:
            start = entry["start"]
            end = entry["end"]
            label = entry["label"]

            # Remove numbers from the label
            label = "".join(filter(lambda x: not x.isdigit(), entry["label"]))

            # Add gap with label 'O' if there is a gap between entries
            if previous_end < start:
                result.append([previous_end, start, "O"])

            # Add the current entry
            result.append([start, end, label])

            # Update previous_end to the end of the current segment
            previous_end = end

        # Add trailing gap with label 'O' if needed
        if previous_end < total_length:
            result.append([previous_end, total_length, "O"])

        return result

    def get_input_ids_and_labels_PII(
        self, text: str, spans: list
    ) -> Tuple[torch.Tensor, list]:
        all_input_ids = torch.tensor([[self.tokenizer.bos_token_id]])
        labels = ["O"]
        for span in spans:
            input_ids = self.tokenizer(
                text[span[0] : span[1]],
                return_tensors="pt",
                truncation=True,
                padding=True,
                add_special_tokens=False,
            ).input_ids
            all_input_ids = torch.concat([all_input_ids, input_ids], dim=1)
            labels += [span[2] for _ in range(len(input_ids[0]))]

        return all_input_ids[0], labels

    def batch_padding(
        self, input_ids_list: list, labels_list: list
    ) -> Tuple[list, list]:
        max_len = max(len(x) for x in input_ids_list)
        new_iil = []
        new_ll = []
        for t, l in zip(input_ids_list, labels_list):
            cur_len = len(t)
            if cur_len < max_len:
                new_t = torch.concat(
                    [
                        torch.tensor(
                            [
                                self.tokenizer.pad_token_id
                                for _ in range(max_len - cur_len)
                            ]
                        ),
                        t,
                    ]
                )
                new_l = ["O" for _ in range(max_len - cur_len)] + l

                new_iil.append(new_t)
                new_ll.append(new_l)

            else:
                new_iil.append(t)
                new_ll.append(l)
        return new_iil, new_ll

    def batch_wrapper(self, text: list, masks: list) -> Tuple[list, list]:
        input_ids_list = []
        labels_list = []
        for t, m in zip(text, masks):
            spans = self.convert_to_sequence(m, len(t))
            input_ids, labels = self.get_input_ids_and_labels_PII(t, spans)
            input_ids_list.append(input_ids)
            labels_list.append(labels)

        input_ids_list, labels_list = self.batch_padding(input_ids_list, labels_list)
        return input_ids_list, labels_list

    def gen_samples(self) -> list:
        samples = None
        with torch.no_grad():
            torch.cuda.empty_cache()
        samples = []
        for _ in tqdm(range(self.chunk_size), desc="Generating token chunk"):
            try:
                samples.append(next(self.iter_dataset))
            except StopIteration:
                logger.info("Reload dataset.")
                self.iter_dataset = iter(self.get_dataset())
                samples.append(next(self.iter_dataset))

        return samples

    def __len__(self):
        return self.chunk_size

    def __getitem__(self, idx):
        new_idx = self.counter % (self.chunk_size * 10)  
        if new_idx == 0 and self.counter != 0:
            self.samples = self.gen_samples()
            self.counter = 0
            return self.samples[idx]
        else:
            self.counter += 1
            return self.samples[idx]



class ActivationsDataset_local(ActivationsDataset):
    """Activation Dataset of ."""

    def __init__(
        self,
        ds_path: str = "wikipedia",
        ds_name: str = "20220301.simple",
        ds_split: str = "train",
        ds_cache_dir: str = "./cache",
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        hooks: int = -1,
        site: str = "mlp",
        batch_size: int = 64,
    ) -> None:
        super().__init__(
            ds_path,
            ds_name,
            ds_split,
            ds_cache_dir,
            model_name,
            hooks,
            site,
            batch_size,
            0,
        )

        self.iter_dataset = self.loaded_dataset

    def get_dataset(self):
        if self.ds_name == "shakespeare":
            dataset = load_from_disk(self.ds_path)
            dataset = dataset.map(
                self.encode_SP,
                batched=True,
                batch_size=25,
            )
        elif self.ds_name == "Combi_SP-tox_":
            dataset = load_from_disk(self.ds_path + self.ds_split)
            dataset = dataset.map(
                self.encode_combi,
                batched=True,
                batch_size=25,
                remove_columns=["Tox", "SP"],
            )
        elif ds_path == "nvidia/Aegis-AI-Content-Safety-Dataset-1.0":
            dataset = load_dataset(
                self.ds_path,
                cache_dir=self.ds_cache_dir,
                streaming=False,
            )
            dataset = dataset.map(
                lambda x: {
                    "toxicity": sum(
                        [
                            "Safe" != x[f"labels_{i}"]
                            for i in range(x["num_annotations"])
                        ]
                    )
                    > x["num_annotations"] / 2
                }
            )
            dataset = dataset.map(
                self.encode_CSD,
                batched=True,
                batch_size=25,
                remove_columns=[
                    "num_annotations",
                    "id",
                    "text_type",
                    "labels_0",
                    "labels_1",
                    "labels_2",
                    "labels_3",
                    "labels_4",
                ],
            )
        elif "pii-masking" in ds_path:
            dataset = load_dataset(
                self.ds_path,
                cache_dir=self.ds_cache_dir,
                streaming=False,
            )
            if ds_split != "mult":
                dataset = dataset.filter(
                    lambda x: x["language"] == "en" or x["language"] == "English"
                )

            dataset = dataset.map(
                self.encode_PII,
                batched=True,
                batch_size=25,
                remove_columns=[
                    "source_text",
                    "target_text",
                    "privacy_mask",
                    "span_labels",
                    "mbert_text_tokens",
                    "mbert_bio_labels",
                    "id",
                    "language",
                    "set",
                ],
            )
        else:
            if self.ds_name == "wikitext-103-raw-v1":
                if "[" in self.ds_split:
                    self.ds_split, ds_split_range = self.ds_split.split("[")
                    ds_split_start, ds_split_end = ds_split_range.replace(
                        "]", ""
                    ).split(":")
                    ds_split_start = 0 if ds_split_start == "" else ds_split_start
                    logger.info(
                        f"Using subset of {self.ds_split}, with {int(ds_split_end) - int(ds_split_start)} samples."
                    )
                dataset = load_dataset(
                    self.ds_path,
                    self.ds_name,
                    split=self.ds_split,
                    cache_dir=self.ds_cache_dir,
                    streaming=False,
                )
                dataset = dataset.filter(lambda x: x["text"] != "")
                dataset = dataset.shuffle(42).select(
                    range(int(ds_split_start), int(ds_split_end))
                )
                dataset = dataset.map(
                    self.encode_wiki,
                    batched=True,
                    batch_size=25,
                )
            elif (
                self.ds_name
                == "./datasets/RTP"
            ):
                dataset = load_from_disk(
                    "./datasets/RTP"
                )
                dataset = dataset.map(
                    self.encode_rtp,
                    batched=True,
                    batch_size=4,
                    remove_columns=[
                        "filename",
                        "begin",
                        "end",
                        "challenging",
                        "prompt",
                        "continuation",
                        "sent_toxicity_bin",
                    ],
                )
            else:
                dataset = load_dataset(
                    self.ds_path,
                    self.ds_name,
                    split=self.ds_split,
                    cache_dir=self.ds_cache_dir,
                    streaming=False,
                )
                dataset = dataset.map(
                    self.encode,
                    batched=True,
                    batch_size=100,
                    remove_columns=self.remove_columns,
                )

        self.model = None
        self.tokenizer = None
        return dataset 

    def __len__(self):
        return len(self.iter_dataset)

    def __getitem__(self, idx):
        
        item = self.iter_dataset[idx]
        item["acts"] = torch.tensor(item["acts"])
        
        return item


def create_mixture_dataset(block: int, small: bool = False):
    suff = "_small" if small else ""
    ds_wiki = load_from_disk(
        f"./datasets/llama3-Wiki{suff}-B{block:02}"
    ).shuffle(42)

    ds_wiki = ds_wiki.with_format("torch").train_test_split(test_size=0.1)

    ds_rtp = load_from_disk(
        f"./datasets/llama3-RTP-B{block:02}"
    ).shuffle(42)
    ds_rtp = ds_rtp.with_format("torch").train_test_split(test_size=0.1)

    print(ds_wiki)
    print(ds_rtp)

    dataset_train = concatenate_datasets([ds_wiki["train"], ds_rtp["train"]]).shuffle()
    dataset_test = concatenate_datasets([ds_wiki["test"], ds_rtp["test"]]).shuffle()
    dataset = DatasetDict({"train": dataset_train, "test": dataset_test})
    print(dataset)
    dataset.save_to_disk(
        f"./datasets/llama3-Wiki{suff}_RTP-B{block:02}",
        num_proc=32,
    )


if __name__ == "__main__":
    # decide on the dataset
    if sys.argv[1] == "RTP":
        ds_path = "allenai/real-toxicity-prompts"  # 2949263
        ds_name = None
        ds_split = "train"
    elif sys.argv[1] == "RTP_split":
        ds_path = "./datasets/RTP"
        ds_name = "./datasets/RTP"
        ds_split = "train"
    elif sys.argv[1] == "SP":
        ds_path = "./datasets/Shakespeare"
        ds_name = "shakespeare"
        ds_split = None
    elif sys.argv[1] in "Combi_SP-tox_111":
        ds_path = "./datasets/Combi_SP-tox_"
        ds_name = "Combi_SP-tox_"
        ds_split = "111"
    elif sys.argv[1] in "Combi_SP-tox_211":
        ds_path = "./datasets/Combi_SP-tox_"
        ds_name = "Combi_SP-tox_"
        ds_split = "211"
    elif sys.argv[1] == "CSD":
        ds_path = "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"
        ds_name = None
        ds_split = None
    elif sys.argv[1] == "Wiki":
        ds_path = "wikitext"
        ds_name = "wikitext-103-raw-v1"
        ds_split = "train[:100000]"
    elif sys.argv[1] == "Wiki_small":
        ds_path = "wikitext"
        ds_name = "wikitext-103-raw-v1"
        ds_split = "train[:30000]"
    elif sys.argv[1] == "PII_300k":
        ds_path = "ai4privacy/pii-masking-300k"
        ds_name = "PII_300k"
        ds_split = None
    elif sys.argv[1] == "PII_300k_mult_lang":
        ds_path = "ai4privacy/pii-masking-300k"
        ds_name = "PII_300k"
        ds_split = "mult"
    else:
        raise NotImplementedError

    block = int(sys.argv[2])

    # decide on the model
    if sys.argv[3] == "llama3":
        model_hf = "meta-llama/Meta-Llama-3-8B"
    elif sys.argv[3] == "llama3_instruct":
        model_hf = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif sys.argv[3] == "llama3_70B":
        model_hf = "meta-llama/Meta-Llama-3-70B"
    elif sys.argv[3] == "llama3_70B_instruct":
        model_hf = "meta-llama/Meta-Llama-3-70B-Instruct"
    elif sys.argv[3] == "gemma2":
        model_hf = "google/gemma-2-9b"
    elif sys.argv[3] == "gemma2_2b":
        model_hf = "google/gemma-2-2b"
    else:
        raise NotImplementedError

    # Decide on the location of the forward hook
    if sys.argv[4] == "mlp":
        site = "mlp"
    elif sys.argv[4] == "block":
        site = "block"
    else:
        raise NotImplementedError

    logger.info(
        f"Generating {sys.argv[1]} dataset for block {block:02} and site '{site}' of {sys.argv[3]}."
    )
    dataset = ActivationsDataset_local(
        model_name=model_hf,
        hooks=block,
        site=site,
        ds_path=ds_path,
        ds_name=ds_name,
        ds_split=ds_split,
        ds_cache_dir="./cache_3",
        batch_size=2048,
    )
    logger.info(f"Length of dataset: {len(dataset)}")
    
    dataset.iter_dataset.save_to_disk(
        f"./datasets_v2/{sys.argv[3]}-{sys.argv[1]}-B{str(block) if len(str(block)) > 1 else '0' + str(block)}-{site}",
        num_proc=32,
    )

