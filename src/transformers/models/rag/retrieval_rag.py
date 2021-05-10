# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
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
"""RAG Retriever model implementation."""

import os
import pickle
import time
from typing import Iterable, List, Optional, Tuple

import pdb

import torch
import numpy as np

from ...file_utils import cached_path, is_datasets_available, is_faiss_available, is_remote_url, requires_backends
from ...tokenization_utils_base import BatchEncoding
from ...utils import logging
from .configuration_rag import RagConfig
from .tokenization_rag import RagTokenizer


if is_datasets_available():
    from datasets import Dataset, load_dataset, load_from_disk

if is_faiss_available():
    import faiss


logger = logging.get_logger(__name__)


LEGACY_INDEX_PATH = "https://storage.googleapis.com/huggingface-nlp/datasets/wiki_dpr/"


class Index:
    """
    A base class for the Indices encapsulated by the :class:`~transformers.RagRetriever`.
    """

    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        """
        Returns a list of dictionaries, containing titles and text of the retrieved documents.

        Args:
            doc_ids (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs)`):
                A tensor of document indices.
        """
        raise NotImplementedError

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each query in the batch, retrieves ``n_docs`` documents.

        Args:
            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size):
                An array of query vectors.
            n_docs (:obj:`int`):
                The number of docs retrieved per query.

        Returns:
            :obj:`np.ndarray` of shape :obj:`(batch_size, n_docs)`: A tensor of indices of retrieved documents.
            :obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`: A tensor of vector representations of
            retrieved documents.
        """
        raise NotImplementedError

    def is_initialized(self):
        """
        Returns :obj:`True` if index is already initialized.
        """
        raise NotImplementedError

    def init_index(self):
        """
        A function responsible for loading the index into memory. Should be called only once per training run of a RAG
        model. E.g. if the model is trained on multiple GPUs in a distributed setup, only one of the workers will load
        the index.
        """
        raise NotImplementedError


class LegacyIndex(Index):
    """
    An index which can be deserialized from the files built using https://github.com/facebookresearch/DPR. We use
    default faiss index parameters as specified in that repository.

    Args:
        vector_size (:obj:`int`):
            The dimension of indexed vectors.
        index_path (:obj:`str`):
            A path to a `directory` containing index files compatible with
            :class:`~transformers.models.rag.retrieval_rag.LegacyIndex`
    """

    INDEX_FILENAME = "hf_bert_base.hnswSQ8_correct_phi_128.c_index"
    PASSAGE_FILENAME = "psgs_w100.tsv.pkl"

    def __init__(self, vector_size, index_path):
        self.index_id_to_db_id = []
        self.index_path = index_path
        self.passages = self._load_passages()
        self.vector_size = vector_size
        self.index = None
        self._index_initialized = False

    def _resolve_path(self, index_path, filename):
        assert os.path.isdir(index_path) or is_remote_url(index_path), "Please specify a valid ``index_path``."
        archive_file = os.path.join(index_path, filename)
        try:
            # Load from URL or cache if already cached
            resolved_archive_file = cached_path(archive_file)
        except EnvironmentError:
            msg = (
                f"Can't load '{archive_file}'. Make sure that:\n\n"
                f"- '{index_path}' is a correct remote path to a directory containing a file named {filename}"
                f"- or '{index_path}' is the correct path to a directory containing a file named {filename}.\n\n"
            )
            raise EnvironmentError(msg)
        if resolved_archive_file == archive_file:
            logger.info(f"loading file {archive_file}")
        else:
            logger.info(f"loading file {archive_file} from cache at {resolved_archive_file}")
        return resolved_archive_file

    def _load_passages(self):
        logger.info(f"Loading passages from {self.index_path}")
        passages_path = self._resolve_path(self.index_path, self.PASSAGE_FILENAME)
        with open(passages_path, "rb") as passages_file:
            passages = pickle.load(passages_file)
        return passages

    def _deserialize_index(self):
        logger.info(f"Loading index from {self.index_path}")
        resolved_index_path = self._resolve_path(self.index_path, self.INDEX_FILENAME + ".index.dpr")
        self.index = faiss.read_index(resolved_index_path)
        resolved_meta_path = self._resolve_path(self.index_path, self.INDEX_FILENAME + ".index_meta.dpr")
        with open(resolved_meta_path, "rb") as metadata_file:
            self.index_id_to_db_id = pickle.load(metadata_file)
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    def is_initialized(self):
        return self._index_initialized

    def init_index(self):
        index = faiss.IndexHNSWFlat(self.vector_size + 1, 512)
        index.hnsw.efSearch = 128
        index.hnsw.efConstruction = 200
        self.index = index
        self._deserialize_index()
        self._index_initialized = True

    def get_doc_dicts(self, doc_ids: np.array):
        doc_list = []
        for doc_ids_i in doc_ids:
            ids = [str(int(doc_id)) for doc_id in doc_ids_i]
            docs = [self.passages[doc_id] for doc_id in ids]
            doc_list.append(docs)
        doc_dicts = []
        for docs in doc_list:
            doc_dict = {}
            doc_dict["title"] = [doc[1] for doc in docs]
            doc_dict["text"] = [doc[0] for doc in docs]
            doc_dicts.append(doc_dict)
        return doc_dicts

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        aux_dim = np.zeros(len(question_hidden_states), dtype="float32").reshape(-1, 1)
        query_nhsw_vectors = np.hstack((question_hidden_states, aux_dim))
        _, docs_ids = self.index.search(query_nhsw_vectors, n_docs)
        vectors = [[self.index.reconstruct(int(doc_id))[:-1] for doc_id in doc_ids] for doc_ids in docs_ids]
        ids = [[int(self.index_id_to_db_id[doc_id]) for doc_id in doc_ids] for doc_ids in docs_ids]
        return np.array(ids), np.array(vectors)


class HFIndexBase(Index):
    def __init__(self, vector_size, dataset, index_initialized=False):
        self.vector_size = vector_size
        self.dataset = dataset
        self._index_initialized = index_initialized
        self._check_dataset_format(with_index=index_initialized)
        dataset.set_format("numpy", columns=["embeddings"], output_all_columns=True, dtype="float32")

    def _check_dataset_format(self, with_index: bool):
        if not isinstance(self.dataset, Dataset):
            raise ValueError(f"Dataset should be a datasets.Dataset object, but got {type(self.dataset)}")
        if len({"title", "text", "embeddings"} - set(self.dataset.column_names)) > 0:
            raise ValueError(
                "Dataset should be a dataset with the following columns: "
                "title (str), text (str) and embeddings (arrays of dimension vector_size), "
                f"but got columns {self.dataset.column_names}"
            )
        if with_index and "embeddings" not in self.dataset.list_indexes():
            raise ValueError(
                "Missing faiss index in the dataset. Make sure you called `dataset.add_faiss_index` to compute it "
                "or `dataset.load_faiss_index` to load one from the disk."
            )

    def init_index(self):
        raise NotImplementedError()

    def is_initialized(self):
        return self._index_initialized

    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        return [self.dataset[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, ids = self.dataset.search_batch("embeddings", question_hidden_states, n_docs)
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))])
        return np.array(ids), np.array(vectors), np.array(scores)  # shapes (batch_size, n_docs), (batch_size, n_docs, d) and (batch_size, n_docs)

    @staticmethod
    def filter_ids(common_ids, ids, scores):
        new_ids = []
        new_scores = []
        for i in range(len(ids)):
            if ids[i] in common_ids:
                new_ids.append(ids[i])
                new_scores.append(scores[i])

        return new_ids, new_scores

    def get_top_docs_multihandle(self, current_hidden_states: np.ndarray, history_hidden_states: np.ndarray,
                                 scoring_func, n_docs=5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        total_docs = len(self.dataset)
        scores_current, ids_current = self.dataset.search_batch("embeddings", current_hidden_states, 500)
        scores_history, ids_history = self.dataset.search_batch("embeddings", history_hidden_states, 500)

        final_scores = []
        final_ids = []
        for i in range(len(ids_current)):
            ids_current_i, scores_current_i = ids_current[i], scores_current[i]
            ids_history_i, scores_history_i = ids_history[i], scores_history[i]

            ## common ids between question and history
            common_ids = set(ids_current_i).intersection(set(ids_history_i))
            common_ids = {i for i in common_ids if i >= 0}
            if len(common_ids) < n_docs:
                print("Error: only {} common ids found".format(len(common_ids)))

            ## only keep ids and scores that are common between question and history
            ids_current_i_common, scores_current_i_common = self.filter_ids(common_ids, ids_current_i, scores_current_i)
            ids_history_i_common, scores_history_i_common = self.filter_ids(common_ids, ids_history_i, scores_history_i)

            assert len(ids_current_i_common) == len(ids_history_i_common)

            # try:
            #     assert len(ids_current_i_common) == len(ids_history_i_common)
            # except AssertionError:
            #     logger.info("assert failed {} {}".format(len(q_doc_ids), len(h_doc_ids)))
            #     pdb.set_trace()

            ## sort by ids
            q_doc_ids, q_doc_scores = zip(*sorted(zip(ids_current_i_common, scores_current_i_common)))
            h_doc_ids, h_doc_scores = zip(*sorted(zip(ids_history_i_common, scores_history_i_common)))

            q_doc_ids, q_doc_scores = list(q_doc_ids), list(q_doc_scores)
            h_doc_ids, h_doc_scores = list(h_doc_ids), list(h_doc_scores)

            assert q_doc_ids == h_doc_ids
            # try:
            #     assert q_doc_ids == h_doc_ids
            # except TypeError:
            #     logger.info("assert failed {} {}".format(len(q_doc_ids), len(h_doc_ids)))
            #     pdb.set_trace()

            ## Combine scores using scoring function
            rescored_ids = []
            rescored_scores = []
            for id, q_score, h_score in zip(q_doc_ids, q_doc_scores, h_doc_scores):
                rescored_ids.append(id)
                inp = torch.Tensor([q_score, h_score])
                rescored_scores.append(scoring_func(inp).tolist())

            rescored_scores, rescored_ids = zip(*sorted(zip(rescored_scores, rescored_ids), reverse=True))
            rescored_scores, rescored_ids = list(rescored_scores), list(rescored_ids)

            final_ids.append(rescored_ids[:n_docs])
            final_scores.append(rescored_scores[:n_docs])

        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in final_ids]
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))])
        return np.array(final_ids), np.array(vectors), np.array(final_scores)  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def get_top_docs_rerank(self, combined_hidden_states: np.ndarray, current_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores1, ids1 = self.dataset.search_batch("embeddings", combined_hidden_states, n_docs)
        scores2, ids2 = self.dataset.search_batch("embeddings", current_hidden_states, n_docs)
        ids3 = [[None] * (n_docs * 2)] * len(ids1)
        scores3 = [[0] * (n_docs * 2)] * len(ids1)
        scores = []
        ids = []
        for r in range(len(ids1)):
            n1, n2 = len(ids1[r]), len(ids2[r])
            i = j = k = 0
            while i < n1 and j < n2:
                if scores1[r][i] >= scores2[r][j]:
                    ids3[r][k] = ids1[r][i]
                    scores3[r][k] = scores1[r][i]
                    k, i = k + 1, i + 1
                else:
                    ids3[r][k] = ids2[r][j]
                    scores3[r][k] = scores2[r][i]
                    k, j = k + 1, j + 1
            while i < n1:
                ids3[r][k] = ids1[r][i]
                scores3[r][k] = scores1[r][i]
                k, i = k + 1, i + 1
            while j < n2:
                ids3[r][k] = ids2[r][j]
                scores3[r][k] = scores2[r][j]
                k, j = k + 1, j + 1
            ids_new = []
            scores_new = []
            for ii, ele in enumerate(ids3[r]):
                if ele not in ids_new:
                    ids_new.append(ele)
                    scores_new.append(scores3[r][ii])
            ids.append(ids_new[:5])
            scores.append(scores_new[:5])
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))])
        return np.array(ids), np.array(vectors), np.array(scores)

class CanonicalHFIndex(HFIndexBase):
    """
    A wrapper around an instance of :class:`~datasets.Datasets`. If ``index_path`` is set to ``None``, we load the
    pre-computed index available with the :class:`~datasets.arrow_dataset.Dataset`, otherwise, we load the index from
    the indicated path on disk.

    Args:
        vector_size (:obj:`int`): the dimension of the passages embeddings used by the index
        dataset_name (:obj:`str`, optional, defaults to ``wiki_dpr``):
            A dataset identifier of the indexed dataset on HuggingFace AWS bucket (list all available datasets and ids
            with ``datasets.list_datasets()``).
        dataset_split (:obj:`str`, optional, defaults to ``train``)
            Which split of the ``dataset`` to load.
        index_name (:obj:`str`, optional, defaults to ``train``)
            The index_name of the index associated with the ``dataset``. The index loaded from ``index_path`` will be
            saved under this name.
        index_path (:obj:`str`, optional, defaults to ``None``)
            The path to the serialized faiss index on disk.
        use_dummy_dataset (:obj:`bool`, optional, defaults to ``False``): If True, use the dummy configuration of the dataset for tests.
    """

    def __init__(
        self,
        vector_size: int,
        dataset_name: str = "wiki_dpr",
        dataset_split: str = "train",
        index_name: Optional[str] = None,
        index_path: Optional[str] = None,
        use_dummy_dataset=False,
    ):
        if int(index_path is None) + int(index_name is None) != 1:
            raise ValueError("Please provide `index_name` or `index_path`.")
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.index_name = index_name
        self.index_path = index_path
        self.use_dummy_dataset = use_dummy_dataset
        logger.info(f"Loading passages from {self.dataset_name}")
        dataset = load_dataset(
            self.dataset_name, with_index=False, split=self.dataset_split, dummy=self.use_dummy_dataset
        )
        super().__init__(vector_size, dataset, index_initialized=False)

    def init_index(self):
        if self.index_path is not None:
            logger.info(f"Loading index from {self.index_path}")
            self.dataset.load_faiss_index("embeddings", file=self.index_path)
        else:
            logger.info(f"Loading index from {self.dataset_name} with index name {self.index_name}")
            self.dataset = load_dataset(
                self.dataset_name,
                with_embeddings=True,
                with_index=True,
                split=self.dataset_split,
                index_name=self.index_name,
                dummy=self.use_dummy_dataset,
            )
            self.dataset.set_format("numpy", columns=["embeddings"], output_all_columns=True)
        self._index_initialized = True


class CustomHFIndex(HFIndexBase):
    """
    A wrapper around an instance of :class:`~datasets.Datasets`. The dataset and the index are both loaded from the
    indicated paths on disk.

    Args:
        vector_size (:obj:`int`): the dimension of the passages embeddings used by the index
        dataset_path (:obj:`str`):
            The path to the serialized dataset on disk. The dataset should have 3 columns: title (str), text (str) and
            embeddings (arrays of dimension vector_size)
        index_path (:obj:`str`)
            The path to the serialized faiss index on disk.
    """

    def __init__(self, vector_size: int, dataset, index_path=None):
        super().__init__(vector_size, dataset, index_initialized=index_path is None)
        self.index_path = index_path

    @classmethod
    def load_from_disk(cls, vector_size, dataset_path, index_path):
        logger.info(f"Loading passages from {dataset_path}")
        if dataset_path is None or index_path is None:
            raise ValueError(
                "Please provide ``dataset_path`` and ``index_path`` after calling ``dataset.save_to_disk(dataset_path)`` "
                "and ``dataset.get_index('embeddings').save(index_path)``."
            )
        dataset = load_from_disk(dataset_path)
        return cls(vector_size=vector_size, dataset=dataset, index_path=index_path)

    def init_index(self):
        if not self.is_initialized():
            logger.info(f"Loading index from {self.index_path}")
            self.dataset.load_faiss_index("embeddings", file=self.index_path)
            self._index_initialized = True


class RagRetriever:
    """
    Retriever used to get documents from vector queries. It retrieves the documents embeddings as well as the documents
    contents, and it formats them to be used with a RagModel.

    Args:
        config (:class:`~transformers.RagConfig`):
            The configuration of the RAG model this Retriever is used with. Contains parameters indicating which
            ``Index`` to build. You can load your own custom dataset with ``config.index_name="custom"`` or use a
            canonical one (default) from the datasets library with ``config.index_name="wiki_dpr"`` for example.
        question_encoder_tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer that was used to tokenize the question. It is used to decode the question and then use the
            generator_tokenizer.
        generator_tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer used for the generator part of the RagModel.
        index (:class:`~transformers.models.rag.retrieval_rag.Index`, optional, defaults to the one defined by the configuration):
            If specified, use this index instead of the one built using the configuration

    Examples::

        >>> # To load the default "wiki_dpr" dataset with 21M passages from wikipedia (index name is 'compressed' or 'exact')
        >>> from transformers import RagRetriever
        >>> retriever = RagRetriever.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', dataset="wiki_dpr", index_name='compressed')

        >>> # To load your own indexed dataset built with the datasets library. More info on how to build the indexed dataset in examples/rag/use_own_knowledge_dataset.py
        >>> from transformers import RagRetriever
        >>> dataset = ...  # dataset must be a datasets.Datasets object with columns "title", "text" and "embeddings", and it must have a faiss index
        >>> retriever = RagRetriever.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', indexed_dataset=dataset)

        >>> # To load your own indexed dataset built with the datasets library that was saved on disk. More info in examples/rag/use_own_knowledge_dataset.py
        >>> from transformers import RagRetriever
        >>> dataset_path = "path/to/my/dataset"  # dataset saved via `dataset.save_to_disk(...)`
        >>> index_path = "path/to/my/index.faiss"  # faiss index saved via `dataset.get_index("embeddings").save(...)`
        >>> retriever = RagRetriever.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', index_name='custom', passages_path=dataset_path, index_path=index_path)

        >>> # To load the legacy index built originally for Rag's paper
        >>> from transformers import RagRetriever
        >>> retriever = RagRetriever.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', index_name='legacy')

    """

    def __init__(self, config, question_encoder_tokenizer, generator_tokenizer, index=None, init_retrieval=True):
        self._init_retrieval = init_retrieval
        requires_backends(self, ["datasets", "faiss"])
        super().__init__()
        self.index = index or self._build_index(config)
        self.generator_tokenizer = generator_tokenizer
        self.question_encoder_tokenizer = question_encoder_tokenizer

        self.n_docs = config.n_docs
        self.batch_size = config.retrieval_batch_size

        self.config = config
        if self._init_retrieval:
            self.init_retrieval()

    @staticmethod
    def _build_index(config):
        if config.index_name == "legacy":
            return LegacyIndex(
                config.retrieval_vector_size,
                config.index_path or LEGACY_INDEX_PATH,
            )
        elif config.index_name == "custom":
            return CustomHFIndex.load_from_disk(
                vector_size=config.retrieval_vector_size,
                dataset_path=config.passages_path,
                index_path=config.index_path,
            )
        else:
            return CanonicalHFIndex(
                vector_size=config.retrieval_vector_size,
                dataset_name=config.dataset,
                dataset_split=config.dataset_split,
                index_name=config.index_name,
                index_path=config.index_path,
                use_dummy_dataset=config.use_dummy_dataset,
            )

    @classmethod
    def from_pretrained(cls, retriever_name_or_path, indexed_dataset=None, **kwargs):
        requires_backends(cls, ["datasets", "faiss"])
        config = kwargs.pop("config", None) or RagConfig.from_pretrained(retriever_name_or_path, **kwargs)
        rag_tokenizer = RagTokenizer.from_pretrained(retriever_name_or_path, config=config)
        question_encoder_tokenizer = rag_tokenizer.question_encoder
        generator_tokenizer = rag_tokenizer.generator
        if indexed_dataset is not None:
            config.index_name = "custom"
            index = CustomHFIndex(config.retrieval_vector_size, indexed_dataset)
        else:
            index = cls._build_index(config)
        return cls(
            config,
            question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer,
            index=index,
        )

    def save_pretrained(self, save_directory):
        if isinstance(self.index, CustomHFIndex):
            if self.config.index_path is None:
                index_path = os.path.join(save_directory, "hf_dataset_index.faiss")
                self.index.dataset.get_index("embeddings").save(index_path)
                self.config.index_path = index_path
            if self.config.passages_path is None:
                passages_path = os.path.join(save_directory, "hf_dataset")
                # datasets don't support save_to_disk with indexes right now
                faiss_index = self.index.dataset._indexes.pop("embeddings")
                self.index.dataset.save_to_disk(passages_path)
                self.index.dataset._indexes["embeddings"] = faiss_index
                self.config.passages_path = passages_path
        self.config.save_pretrained(save_directory)
        rag_tokenizer = RagTokenizer(
            question_encoder=self.question_encoder_tokenizer,
            generator=self.generator_tokenizer,
        )
        rag_tokenizer.save_pretrained(save_directory)

    def init_retrieval(self):
        """
        Retriever initialization function. It loads the index into memory.
        """

        logger.info("initializing retrieval")
        self.index.init_index()

    def postprocess_docs(self, docs, input_strings, prefix, n_docs, return_tensors=None):
        r"""
        Postprocessing retrieved ``docs`` and combining them with ``input_strings``.

        Args:
            docs  (:obj:`dict`):
                Retrieved documents.
            input_strings (:obj:`str`):
                Input strings decoded by ``preprocess_query``.
            prefix (:obj:`str`):
                Prefix added at the beginning of each input, typically used with T5-based models.

        Return:
            :obj:`tuple(tensors)`: a tuple consisting of two elements: contextualized ``input_ids`` and a compatible
            ``attention_mask``.
        """

        def cat_input_and_doc(doc_title, doc_text, input_string, prefix):
            # TODO(Patrick): if we train more RAG models, I want to put the input first to take advantage of effortless truncation
            # TODO(piktus): better handling of truncation
            if doc_title.startswith('"'):
                doc_title = doc_title[1:]
            if doc_title.endswith('"'):
                doc_title = doc_title[:-1]
            if prefix is None:
                prefix = ""
            if self.config.segmentation == "token":
                out = (
                    prefix + doc_title + self.config.title_sep + doc_text + self.config.doc_sep + input_string
                ).replace("  ", " ")
            else:
                out = (
                    prefix + input_string.strip()[: self.config.max_source_length] + self.config.doc_sep + doc_text
                ).replace("  ", " ")
            return out

        rag_input_strings = [
            cat_input_and_doc(
                docs[i]["title"][j],
                docs[i]["text"][j],
                input_strings[i],
                prefix,
            )
            for i in range(len(docs))
            for j in range(n_docs)
        ]

        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(
            rag_input_strings,
            max_length=self.config.max_combined_length,
            return_tensors=return_tensors,
            padding="max_length",
            truncation=True,
        )

        return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]

    def _chunk_tensor(self, t: Iterable, chunk_size: int) -> List[Iterable]:
        return [t[i : i + chunk_size] for i in range(0, len(t), chunk_size)]

    def _main_retrieve(self, combined_hidden_states: np.ndarray, current_hidden_states: np.ndarray, history_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        def linear(a: List[int]):
            return sum(a)

        def linear2(a: List[int]):
            return a[0] + 0.5 * a[1]

        def nonlinear(nnet: torch.nn.Module, a):
            return nnet(a)

        combined_hidden_states_batched = self._chunk_tensor(combined_hidden_states, self.batch_size)
        current_hidden_states_batched = self._chunk_tensor(current_hidden_states, self.batch_size)
        history_hidden_states_batched = self._chunk_tensor(history_hidden_states, self.batch_size)
        ids_batched = []
        vectors_batched = []
        scores_batched = []
        for comb_h_s, curr_h_s, hist_h_s in zip(combined_hidden_states_batched, current_hidden_states_batched, history_hidden_states_batched):
            start_time = time.time()
            if self.config.scoring_func in ["linear", "nonlinear"]:
                scoring_func = linear if self.config.scoring_func == "linear" else nonlinear
                ids, vectors, scores = self.index.get_top_docs_multihandle(curr_h_s, hist_h_s, scoring_func, n_docs)
            elif self.config.scoring_func == "reranking":
                ids, vectors, scores = self.index.get_top_docs_rerank(comb_h_s, curr_h_s, n_docs)
            else:
                ids, vectors, scores = self.index.get_top_docs(comb_h_s, n_docs)
            logger.debug(
                f"index search time: {time.time() - start_time} sec, batch size {comb_h_s.shape}"
            )
            ids_batched.extend(ids)
            vectors_batched.extend(vectors)
            scores_batched.extend(scores)
        return (
            np.array(ids_batched),
            np.array(vectors_batched),
            np.array(scores),
        )  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def retrieve(self, combined_hidden_states: np.ndarray,current_hidden_states: np.ndarray, history_hidden_states: np.ndarray, n_docs: int) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        Retrieves documents for specified ``question_hidden_states``.

        Args:
            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`):
                A batch of query vectors to retrieve with.
            n_docs (:obj:`int`):
                The number of docs retrieved per query.

        Return:
            :obj:`Tuple[np.ndarray, np.ndarray, List[dict]]`: A tuple with the following objects:

            - **retrieved_doc_embeds** (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs, dim)`) -- The retrieval
              embeddings of the retrieved docs per query.
            - **doc_ids** (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs)`) -- The ids of the documents in the
              index
            - **doc_dicts** (:obj:`List[dict]`): The :obj:`retrieved_doc_embeds` examples per query.
        """

        doc_ids, retrieved_doc_embeds, doc_scores = self._main_retrieve(combined_hidden_states, current_hidden_states, history_hidden_states, n_docs)
        return retrieved_doc_embeds, doc_ids, doc_scores, self.index.get_doc_dicts(doc_ids)

    def __call__(
        self,
        question_input_ids: List[List[int]],
        combined_hidden_states: List[List[int]],
        current_hidden_states: np.ndarray,
        history_hidden_states: np.ndarray,
        prefix=None,
        n_docs=None,
        return_tensors=None,
    ) -> BatchEncoding:
        """
        Retrieves documents for specified :obj:`question_hidden_states`.

        Args:
            question_input_ids: (:obj:`List[List[int]]`) batch of input ids
            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`:
                A batch of query vectors to retrieve with.
            prefix: (:obj:`str`, `optional`):
                The prefix used by the generator's tokenizer.
            n_docs (:obj:`int`, `optional`):
                The number of docs retrieved per query.
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`, defaults to "pt"):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.

        Returns: :class:`~transformers.BatchEncoding`: A :class:`~transformers.BatchEncoding` with the following
        fields:

            - **context_input_ids** -- List of token ids to be fed to a model.

              `What are input IDs? <../glossary.html#input-ids>`__

            - **context_attention_mask** -- List of indices specifying which tokens should be attended to by the model
            (when :obj:`return_attention_mask=True` or if `"attention_mask"` is in :obj:`self.model_input_names`).

              `What are attention masks? <../glossary.html#attention-mask>`__

            - **retrieved_doc_embeds** -- List of embeddings of the retrieved documents
            - **doc_ids** -- List of ids of the retrieved documents
        """

        n_docs = n_docs if n_docs is not None else self.n_docs
        prefix = prefix if prefix is not None else self.config.generator.prefix
        retrieved_doc_embeds, doc_ids, doc_scores, docs = self.retrieve(combined_hidden_states=combined_hidden_states,
        current_hidden_states=current_hidden_states,history_hidden_states=history_hidden_states,n_docs=n_docs)

        input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)
        context_input_ids, context_attention_mask = self.postprocess_docs(
            docs, input_strings, prefix, n_docs, return_tensors=return_tensors
        )

        return BatchEncoding(
            {
                "context_input_ids": context_input_ids,
                "context_attention_mask": context_attention_mask,
                "retrieved_doc_embeds": retrieved_doc_embeds,
                "doc_ids": doc_ids,
                "doc_scores": doc_scores,
            },
            tensor_type=return_tensors,
        )