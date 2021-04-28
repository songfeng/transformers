from typing import TYPE_CHECKING

from ...file_utils import _BaseLazyModule, is_tf_available, is_tokenizers_available, is_torch_available


_import_structure = {
    "configuration_colbert": ["COLBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ColBERTConfig"],
    "tokenization_ColBERT": [
        "ColBERTContextEncoderTokenizer",
        "ColBERTQuestionEncoderTokenizer",
        "ColBERTReaderOutput",
        "ColBERTReaderTokenizer",
    ],
}


if is_tokenizers_available():
    _import_structure["tokenization_colbert_fast"] = [
        "ColBERTContextEncoderTokenizerFast",
        "ColBERTQuestionEncoderTokenizerFast",
        "ColBERTReaderTokenizerFast",
    ]

if is_torch_available():
    _import_structure["modeling_colbert"] = [
        "ColBERT_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ColBERT_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ColBERT_READER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ColBERTContextEncoder",
        "ColBERTPretrainedContextEncoder",
        "ColBERTPretrainedQuestionEncoder",
        "ColBERTPretrainedReader",
        "ColBERTQuestionEncoder",
        "ColBERTReader",
    ]

if is_tf_available():
    _import_structure["modeling_tf_colbert"] = [
        "TF_ColBERT_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TF_ColBERT_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TF_ColBERT_READER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFColBERTContextEncoder",
        "TFColBERTPretrainedContextEncoder",
        "TFColBERTPretrainedQuestionEncoder",
        "TFColBERTPretrainedReader",
        "TFColBERTQuestionEncoder",
        "TFColBERTReader",
    ]


if TYPE_CHECKING:
    from .configuration_colbert import ColBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, ColBERTConfig
    from .tokenization_colbert import (
        ColBERTContextEncoderTokenizer,
        ColBERTQuestionEncoderTokenizer,
        ColBERTReaderOutput,
        ColBERTReaderTokenizer,
    )

    if is_tokenizers_available():
        from .tokenization_colbert_fast import (
            ColBERTContextEncoderTokenizerFast,
            ColBERTQuestionEncoderTokenizerFast,
            ColBERTReaderTokenizerFast,
        )

    if is_torch_available():
        from .modeling_colbert import (
            ColBERT_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
            ColBERT_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
            ColBERT_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
            ColBERTContextEncoder,
            ColBERTPretrainedContextEncoder,
            ColBERTPretrainedQuestionEncoder,
            ColBERTPretrainedReader,
            ColBERTQuestionEncoder,
            ColBERTReader,
        )

    if is_tf_available():
        from .modeling_tf_colbert import (
            TF_ColBERT_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TF_ColBERT_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TF_ColBERT_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFColBERTContextEncoder,
            TFColBERTPretrainedContextEncoder,
            TFColBERTPretrainedQuestionEncoder,
            TFColBERTPretrainedReader,
            TFColBERTQuestionEncoder,
            TFColBERTReader,
        )

else:
    import importlib
    import os
    import sys

    class _LazyModule(_BaseLazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)
