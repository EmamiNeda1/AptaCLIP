import gc
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel


class ProteinEncoder:
    def __init__(
        self,
        model_name="facebook/esm2_t12_35M_UR50D",
        max_len=1500,
        device="cuda"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

        self.max_len = max_len
        self.device = device

    def encode(self, sequences):
        with torch.no_grad():
            inputs = self.tokenizer(
                sequences,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_len
            ).to(self.device)

            outputs = self.model(**inputs)

            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings.cpu().numpy()

            torch.cuda.empty_cache()
            gc.collect()

        return embeddings


class AptamerEncoder:
    def __init__(
        self,
        model_name="zhihan1996/DNA_bert_6",
        max_len=150,
        device="cuda"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(device)

        self.max_len = max_len
        self.device = device
        self.k = 6

    def encode(self, sequences):
    return X, y
