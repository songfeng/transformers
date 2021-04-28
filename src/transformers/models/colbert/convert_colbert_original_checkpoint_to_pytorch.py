# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
from pathlib import Path
from typing import OrderedDict

import torch
from torch.serialization import default_restore_location

from transformers import BertConfig, ColBERTConfig, ColBERTContextEncoder, ColBERTQuestionEncoder, ColBERTReader


CheckpointState = collections.namedtuple(
    "CheckpointState", ["batch", "arguments", "model_dict", "optimizer_dict", "epoch"]
    # , "scheduler_dict", "offset", "encoder_params"]
)


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    print("Reading saved model from %s", model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    for k in [('model_dict', 'model_state_dict'), ('optimizer_dict', 'optimizer_state_dict')]:
        state_dict[k[0]] = state_dict.pop(k[-1], None)
    new_model_dict = OrderedDict()
    for key, value in state_dict['model_dict'].items():
        if key.startswith("bert."):
            key = 'bert_model.' + key[5:]
        # key = key.replace("bert_model.encoder", "bert_model")
        new_model_dict[key] = value
    state_dict['model_dict'] = new_model_dict
    for key in ["linear.weight"]:
        new_model_dict.pop(key)
    return CheckpointState(**state_dict)


class ColBERTState:
    def __init__(self, src_file: Path):
        self.src_file = src_file

    def load_colbert_model(self):
        raise NotImplementedError

    @staticmethod
    def from_type(comp_type: str, *args, **kwargs) -> "ColBERTState":
        if comp_type.startswith("c"):
            return ColBERTContextEncoderState(*args, **kwargs)
        if comp_type.startswith("q"):
            return ColBERTQuestionEncoderState(*args, **kwargs)
        if comp_type.startswith("r"):
            return ColBERTReaderState(*args, **kwargs)
        else:
            raise ValueError("Component type must be either 'ctx_encoder', 'question_encoder' or 'reader'.")


class ColBERTContextEncoderState(ColBERTState):
    def load_colbert_model(self):
        model = ColBERTContextEncoder(ColBERTConfig(**BertConfig.get_config_dict("bert-base-uncased")[0]))
        print("Loading ColBERT biencoder from {}".format(self.src_file))
        saved_state = load_states_from_checkpoint(self.src_file)
        encoder, prefix = model.ctx_encoder, "ctx_model."
        # Fix changes from https://github.com/huggingface/transformers/commit/614fef1691edb806de976756d4948ecbcd0c0ca3
        state_dict = {"bert_model.embeddings.position_ids": model.ctx_encoder.bert_model.embeddings.position_ids}
        for key, value in saved_state.model_dict.items():
            if key.startswith(prefix):
                key = key[len(prefix) :]
                if not key.startswith("encode_proj."):
                    key = "bert_model." + key
                # key = key.replace("bert_model.encoder", "bert_model")
            state_dict[key] = value
        encoder.load_state_dict(state_dict)
        return model


class ColBERTQuestionEncoderState(ColBERTState):
    def load_colbert_model(self):
        model = ColBERTQuestionEncoder(ColBERTConfig(**BertConfig.get_config_dict("bert-base-uncased")[0]))
        print("Loading ColBERT biencoder from {}".format(self.src_file))
        saved_state = load_states_from_checkpoint(self.src_file)
        encoder, prefix = model.question_encoder, "question_model."
        # Fix changes from https://github.com/huggingface/transformers/commit/614fef1691edb806de976756d4948ecbcd0c0ca3
        state_dict = {"bert_model.embeddings.position_ids": model.question_encoder.bert_model.embeddings.position_ids}
        for key, value in saved_state.model_dict.items():
            if key.startswith(prefix):
                key = key[len(prefix) :]
                if not key.startswith("encode_proj."):
                    key = "bert_model." + key
                key = key.replace("bert_model.encoder", "bert_model")
            state_dict[key] = value
        encoder.load_state_dict(state_dict)
        return model


class ColBERTReaderState(ColBERTState):
    def load_colbert_model(self):
        model = ColBERTReader(ColBERTConfig(**BertConfig.get_config_dict("bert-base-uncased")[0]))
        print("Loading ColBERT reader from {}".format(self.src_file))
        saved_state = load_states_from_checkpoint(self.src_file)
        # Fix changes from https://github.com/huggingface/transformers/commit/614fef1691edb806de976756d4948ecbcd0c0ca3
        state_dict = {
            "encoder.bert_model.embeddings.position_ids": model.span_predictor.encoder.bert_model.embeddings.position_ids
        }
        for key, value in saved_state.model_dict.items():
            if key.startswith("encoder.") and not key.startswith("encoder.encode_proj"):
                key = "encoder.bert_model." + key[len("encoder.") :]
            state_dict[key] = value
        model.span_predictor.load_state_dict(state_dict)
        return model


def convert(comp_type: str, src_file: Path, dest_dir: Path):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)

    colbert_state = ColBERTState.from_type(comp_type, src_file=src_file)
    model = colbert_state.load_colbert_model()
    model.save_pretrained(dest_dir)
    model.from_pretrained(dest_dir)  # sanity check


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--type", type=str, 
        default="ctx_encoder",
        help="Type of the component to convert: 'ctx_encoder', 'question_encoder' or 'reader'."
    )
    parser.add_argument(
        "--src",
        default="/Users/songfeng/Downloads/colbert-60000.dnn",
        type=str,
        help="Path to the colbert checkpoint file.",
    )
    parser.add_argument("--dest", type=str, default=None, help="Path to the output PyTorch model directory.")
    args = parser.parse_args()

    src_file = Path(args.src)
    dest_dir = f"bconverted-{src_file.name}" if args.dest is None else args.dest
    dest_dir = Path(dest_dir)
    assert src_file.exists()
    assert (
        args.type is not None
    ), "Please specify the component type of the model to convert: 'ctx_encoder', 'question_encoder' or 'reader'."
    convert(args.type, src_file, dest_dir)
