from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..bert.modeling_bert import BertModel
from .configuration_colbert import ColBERTConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "ColBERTConfig"

ColBERT_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-ctx_encoder-single-nq-base",
    "facebook/dpr-ctx_encoder-multiset-base",
]
ColBERT_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-question_encoder-single-nq-base",
    "facebook/dpr-question_encoder-multiset-base",
]
ColBERT_READER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-reader-single-nq-base",
    "facebook/dpr-reader-multiset-base",
]


##########
# Outputs
##########


@dataclass
class ColBERTContextEncoderOutput(ModelOutput):
    """
    Class for outputs of :class:`~transformers.ColBERTQuestionEncoder`.

    Args:
        pooler_output: (:obj:``torch.FloatTensor`` of shape ``(batch_size, embeddings_size)``):
            The ColBERT encoder outputs the `pooler_output` that corresponds to the context representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed contexts for nearest neighbors queries with questions embeddings.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    pooler_output: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ColBERTQuestionEncoderOutput(ModelOutput):

    pooler_output: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ColBERTReaderOutput(ModelOutput):

    start_logits: torch.FloatTensor
    end_logits: torch.FloatTensor = None
    relevance_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ColBERTEncoder(PreTrainedModel):

    base_model_prefix = "bert_model"

    def __init__(self, config: ColBERTConfig):
        super().__init__(config)
        self.bert_model = BertModel(config)
        assert self.bert_model.config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.projection_dim = config.projection_dim
        if self.projection_dim > 0:
            self.encode_proj = nn.Linear(self.bert_model.config.hidden_size, config.projection_dim)
        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[BaseModelOutputWithPooling, Tuple[Tensor, ...]]:
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output, pooled_output = outputs[:2]
        pooled_output = sequence_output[:, 0, :]
        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)

        if not return_dict:
            return (sequence_output, pooled_output) + outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def embeddings_size(self) -> int:
        if self.projection_dim > 0:
            return self.encode_proj.out_features
        return self.bert_model.config.hidden_size

    def init_weights(self):
        self.bert_model.init_weights()
        if self.projection_dim > 0:
            self.encode_proj.apply(self.bert_model._init_weights)


class ColBERTSpanPredictor(PreTrainedModel):

    base_model_prefix = "encoder"

    def __init__(self, config: ColBERTConfig):
        super().__init__(config)
        self.encoder = ColBERTEncoder(config)
        self.qa_outputs = nn.Linear(self.encoder.embeddings_size, 2)
        self.qa_classifier = nn.Linear(self.encoder.embeddings_size, 1)
        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[ColBERTReaderOutput, Tuple[Tensor, ...]]:
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        n_passages, sequence_length = input_ids.size() if input_ids is not None else inputs_embeds.size()[:2]
        # feed encoder
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        # compute logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        relevance_logits = self.qa_classifier(sequence_output[:, 0, :])

        # resize
        start_logits = start_logits.view(n_passages, sequence_length)
        end_logits = end_logits.view(n_passages, sequence_length)
        relevance_logits = relevance_logits.view(n_passages)

        if not return_dict:
            return (start_logits, end_logits, relevance_logits) + outputs[2:]

        return ColBERTReaderOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            relevance_logits=relevance_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def init_weights(self):
        self.encoder.init_weights()


##################
# PreTrainedModel
##################


class ColBERTPretrainedContextEncoder(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ColBERTConfig
    load_tf_weights = None
    base_model_prefix = "ctx_encoder"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def init_weights(self):
        self.ctx_encoder.init_weights()


class ColBERTPretrainedQuestionEncoder(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ColBERTConfig
    load_tf_weights = None
    base_model_prefix = "question_encoder"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def init_weights(self):
        self.question_encoder.init_weights()


class ColBERTPretrainedReader(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ColBERTConfig
    load_tf_weights = None
    base_model_prefix = "span_predictor"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def init_weights(self):
        self.span_predictor.encoder.init_weights()
        self.span_predictor.qa_classifier.apply(self.span_predictor.encoder.bert_model._init_weights)
        self.span_predictor.qa_outputs.apply(self.span_predictor.encoder.bert_model._init_weights)


###############
# Actual Models
###############


class ColBERTContextEncoder(ColBERTPretrainedContextEncoder):
    def __init__(self, config: ColBERTConfig):
        super().__init__(config)
        self.config = config
        self.ctx_encoder = ColBERTEncoder(config)
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[ColBERTContextEncoderOutput, Tuple[Tensor, ...]]:
        r"""
        Return:

        Examples::

            >>> from transformers import ColBERTContextEncoder, ColBERTContextEncoderTokenizer
            >>> tokenizer = ColBERTContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
            >>> model = ColBERTContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
            >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors='pt')["input_ids"]
            >>> embeddings = model(input_ids).pooler_output
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        outputs = self.ctx_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs[1:]
        return ColBERTContextEncoderOutput(
            pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class ColBERTQuestionEncoder(ColBERTPretrainedQuestionEncoder):
    def __init__(self, config: ColBERTConfig):
        super().__init__(config)
        self.config = config
        self.question_encoder = ColBERTEncoder(config)
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[ColBERTQuestionEncoderOutput, Tuple[Tensor, ...]]:
        r"""
        Return:

        Examples::

            >>> from transformers import ColBERTQuestionEncoder, ColBERTQuestionEncoderTokenizer
            >>> tokenizer = ColBERTQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
            >>> model = ColBERTQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
            >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors='pt')["input_ids"]
            >>> embeddings = model(input_ids).pooler_output
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        outputs = self.question_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs[1:]
        return ColBERTQuestionEncoderOutput(
            pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class ColBERTReader(ColBERTPretrainedReader):
    def __init__(self, config: ColBERTConfig):
        super().__init__(config)
        self.config = config
        self.span_predictor = ColBERTSpanPredictor(config)
        self.init_weights()


    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict=None,
    ) -> Union[ColBERTReaderOutput, Tuple[Tensor, ...]]:
        r"""
        Return:

        Examples::

            >>> from transformers import ColBERTReader, ColBERTReaderTokenizer
            >>> tokenizer = ColBERTReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
            >>> model = ColBERTReader.from_pretrained('facebook/dpr-reader-single-nq-base')
            >>> encoded_inputs = tokenizer(
            ...         questions=["What is love ?"],
            ...         titles=["Haddaway"],
            ...         texts=["'What Is Love' is a song recorded by the artist Haddaway"],
            ...         return_tensors='pt'
            ...     )
            >>> outputs = model(**encoded_inputs)
            >>> start_logits = outputs.stat_logits
            >>> end_logits = outputs.end_logits
            >>> relevance_logits = outputs.relevance_logits

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        return self.span_predictor(
            input_ids,
            attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
