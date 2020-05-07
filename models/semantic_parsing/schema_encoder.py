import torch

from overrides import overrides
from allennlp.common.registrable import Registrable
from allennlp.nn.util import masked_softmax


class SchemaEncoder(torch.nn.Module, Registrable):
    def __init__(self) -> None:
        super().__init__()
        # pas