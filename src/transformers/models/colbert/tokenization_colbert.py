import collections
from typing import List, Optional, Union

from ...file_utils import TensorType, add_end_docstrings, add_start_docstrings
from ...tokenization_utils_base import BatchEncoding
from ...utils import logging
from ..bert.tokenization_bert import BertTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/dpr-ctx_encoder-single-nq-base": "https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base/resolve/main/vocab.txt",
        "facebook/dpr-ctx_encoder-multiset-base": "https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "facebook/dpr-ctx_encoder-single-nq-base": "https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base/resolve/main/tokenizer.json",
        "facebook/dpr-ctx_encoder-multiset-base": "https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base/resolve/main/tokenizer.json",
    },
}
QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/dpr-question_encoder-single-nq-base": "https://huggingface.co/facebook/dpr-question_encoder-single-nq-base/resolve/main/vocab.txt",
        "facebook/dpr-question_encoder-multiset-base": "https://huggingface.co/facebook/dpr-question_encoder-multiset-base/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "facebook/dpr-question_encoder-single-nq-base": "https://huggingface.co/facebook/dpr-question_encoder-single-nq-base/resolve/main/tokenizer.json",
        "facebook/dpr-question_encoder-multiset-base": "https://huggingface.co/facebook/dpr-question_encoder-multiset-base/resolve/main/tokenizer.json",
    },
}
READER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/dpr-reader-single-nq-base": "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/vocab.txt",
        "facebook/dpr-reader-multiset-base": "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "facebook/dpr-reader-single-nq-base": "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/tokenizer.json",
        "facebook/dpr-reader-multiset-base": "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/tokenizer.json",
    },
}

CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-ctx_encoder-single-nq-base": 512,
    "facebook/dpr-ctx_encoder-multiset-base": 512,
}
QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-question_encoder-single-nq-base": 512,
    "facebook/dpr-question_encoder-multiset-base": 512,
}
READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-reader-single-nq-base": 512,
    "facebook/dpr-reader-multiset-base": 512,
}


CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-ctx_encoder-single-nq-base": {"do_lower_case": True},
    "facebook/dpr-ctx_encoder-multiset-base": {"do_lower_case": True},
}
QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-question_encoder-single-nq-base": {"do_lower_case": True},
    "facebook/dpr-question_encoder-multiset-base": {"do_lower_case": True},
}
READER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-reader-single-nq-base": {"do_lower_case": True},
    "facebook/dpr-reader-multiset-base": {"do_lower_case": True},
}


class ColBERTContextEncoderTokenizer(BertTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION


class ColBERTQuestionEncoderTokenizer(BertTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION
