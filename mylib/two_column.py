from typing import Dict, List, Sequence, Iterable
import itertools
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path

from allennlp.data import Batch, Vocabulary, Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer, TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, PretrainedTransformerTokenizer
from allennlp.models import SimpleTagger
from allennlp.modules import FeedForward
from allennlp.modules.seq2seq_encoders import PassThroughEncoder, FeedForwardEncoder
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training import util
from allennlp.training.trainer import GradientDescentTrainer
from allennlp.common import Params

import os
import random
import shutil
import sys



logger = logging.getLogger(__name__)


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ""
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":
            return True
        else:
            return False


@DatasetReader.register("ner_two_column")
class NerTwoColumnDatasetReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        coding_scheme: str = "IOB1",
        label_namespace: str = "labels",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if coding_scheme not in ("IOB1", "BIOUL"):
            raise ConfigurationError("unknown coding_scheme: {}".format(coding_scheme))

        self.coding_scheme = coding_scheme
        self.label_namespace = label_namespace
        self._original_coding_scheme = "IOB1"

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [[line.strip().split()[0],line.strip().split()[-1]] for line in lines]
                    
                    # unzipping trick returns tuples, but our Fields need lists
                    fields = [list(field) for field in zip(*fields)]
                    tokens_, ner_tags = fields
                    # TextField requires `Token` objects
                    tokens = [Token(token) for token in tokens_]

                    yield self.text_to_instance(tokens, ner_tags)

    def text_to_instance(  # type: ignore
        self,
        tokens: List[Token],
        ner_tags: List[str] = None,
    ) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """

        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {"tokens": sequence}
        

        # Recode the labels if necessary.
        if self.coding_scheme == "BIOUL":
            coded_ner = (
                to_bioul(ner_tags, encoding=self._original_coding_scheme)
                if ner_tags is not None
                else None
            )
        else:
            # the default IOB1
            coded_ner = ner_tags

        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens], "tags" : coded_ner})
        instance_fields["tags"] = SequenceLabelField(coded_ner, sequence, self.label_namespace)

        return Instance(instance_fields)
