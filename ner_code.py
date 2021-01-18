from typing import Dict, List, Sequence, Iterable
import itertools
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path

from allennlp.data import allennlp_collate, Batch, Vocabulary, Instance, PyTorchDataLoader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers import Conll2003DatasetReader
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

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

#  CONFIG SECTION
expname = "exp1"

logger = logging.getLogger(__name__)

model_name = "bert-base-cased"
indexers = {"bert" : PretrainedTransformerMismatchedIndexer(model_name, namespace="bert")}

reader = Conll2003DatasetReader(token_indexers = indexers)
train_dataset = reader.read("conll2003/eng.train")
validation_dataset = reader.read("conll2003/eng.testa")
test_dataset = reader.read("conll2003/eng.testb")

all_insts = train_dataset + validation_dataset + test_dataset


vocab = Vocabulary.from_instances(all_insts)

dataset = Batch(all_insts)
dataset.index_instances(vocab)

embedder = PretrainedTransformerMismatchedEmbedder(model_name, last_layer_only = True)
token_embedder = BasicTextFieldEmbedder({"bert" : embedder})
embedding_dim = 768
encoder = PassThroughEncoder(input_dim=embedding_dim)

model = SimpleTagger(vocab = vocab,
                     text_field_embedder = token_embedder,
                     encoder = encoder,
                     calculate_span_f1 = True,
                     label_encoding = "IOB1")

optimizer = optim.Adam(model.parameters(), lr=3e-05)

if torch.cuda.is_available():
    print("Using GPU")
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1


# don't put a slash after this?
serialization_dir = f"tmp-{expname}/"

if os.path.exists(serialization_dir):
    print("serialization directory exists, removing...")
    shutil.rmtree(serialization_dir)

batch_size = 32
validation_batch_size = 64

data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=allennlp_collate)
validation_data_loader = DataLoader(validation_dataset, batch_size=validation_batch_size, collate_fn=allennlp_collate)

trainer = GradientDescentTrainer(model=model,
                  optimizer=optimizer,
                  data_loader=data_loader,
                  validation_data_loader=validation_data_loader,
                  patience=10,
                  num_epochs=75,
                  validation_metric="+f1-measure-overall",
                  cuda_device=cuda_device,
                  serialization_dir=serialization_dir)

metrics = trainer.train()
