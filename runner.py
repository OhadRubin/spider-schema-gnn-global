import os

# os.environ[“CUDA_DEVICE_ORDER”]=“PCI_BUS_ID”
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# experiment_name = "3_heads_lr3_keep_op_identity+agenda_enriched_all+lr_e3+mult_scalar_per_action+glove"
# experiment_name = "crappy-red-dhole"
# train_dataset = reader.read("dataset/train_spider.json")
from models.semantic_parsing.ratsql_encoder import RatsqlEncoder

# from models.semantic_parsing.gnn_encoder import GnnEncoder
from models.semantic_parsing.spider_decoder import SpiderParser
from allennlp.modules.seq2vec_encoders.boe_encoder import BagOfEmbeddingsEncoder

from allennlp.modules.attention import DotProductAttention
from allennlp.nn.beam_search import BeamSearch
from allennlp.modules.seq2seq_encoders.pass_through_encoder import PassThroughEncoder
from allennlp.data.token_indexers import PretrainedTransformerIndexer

from allennlp.modules.token_embedders import (
    PretrainedTransformerMismatchedEmbedder,
    PretrainedTransformerEmbedder,
)

from allennlp.modules.text_field_embedders import (
    TextFieldEmbedder,
    BasicTextFieldEmbedder,
)
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper

import torch.optim as optim
from allennlp.training.trainer import Trainer
import torch
from allennlp.models.archival import Archive
import torch
from allennlp.common import Params
from allennlp.models.model import Model
from allennlp.common.params import with_fallback
from dataset_readers.spider_ratsql import SpiderRatsqlDatasetReader
from allennlp.data.vocabulary import Vocabulary


# with Timer("List Comprehension Example"):

# from models.semantic_parsing.spider_parser import SpiderParser
reader = SpiderRatsqlDatasetReader(
    question_token_indexers={
        "tokens": PretrainedTransformerIndexer("bert-base-uncased")
    },
    tables_file="dataset/tables.json",
    cache_directory="cache/train",
    max_instances=None,
)
# reader = SpiderRatsqlDatasetReader(tables_file="dataset/tables.json",max_instances=1000)
# settings = Params.from_file(f"experiments/{experiment_name}/config.json")
# model = Model.load(config=settings, serialization_dir=f"experiments/{experiment_name}")


train_dataset = reader.read("dataset/train_spider.json")
vocab = Vocabulary.from_instances(train_dataset)
# exit(0)
EMBEDDING_DIM = 768
HIDDEN_DIM = 768
# token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
#                             embedding_dim=EMBEDDING_DIM)
# word_embeddings = BasicTextFieldEmbedder({"tokens": PretrainedTransformerEmbedder("bert-base-uncased")})
# word_embeddings = {"tokens": PretrainedTransformerEmbedder("bert-base-uncased")}
word_embeddings = PretrainedTransformerEmbedder("bert-base-uncased")


beam = BeamSearch(end_index=0, beam_size=10)

schema_encoder = RatsqlEncoder(
    question_embedder=word_embeddings, action_embedding_dim=768
)
# schema_encoder = GnnEncoder(encoder=PassThroughEncoder(200),entity_encoder=BagOfEmbeddingsEncoder(200),question_embedder=word_embeddings,action_embedding_dim=200)
model = SpiderParser(
    vocab=vocab,
    schema_encoder=schema_encoder,
    decoder_beam_search=beam,
    input_attention=DotProductAttention(),
    past_attention=DotProductAttention(),
    max_decoding_steps=200,
)

model.cuda()
a = None
for i, inst in enumerate(train_dataset):
    a = inst
    res_list = model.forward_on_instances([inst])
    print(i, res_list)
