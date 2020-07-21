
from typing import Dict, List, Tuple

import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder, Embedding, TimeDistributed
# from torch_geometric.data import Data, Batch

# from modules.gated_graph_conv import GatedGraphConv
from allennlp.training.metrics import Average

from semparse.worlds.spider_world import SpiderWorld
import difflib
import os
from functools import partial
from typing import Dict, List, Tuple, Any, Mapping, Sequence

import sqlparse
import torch
from allennlp.common.util import pad_sequence_to_length

from allennlp.data import Vocabulary
from allennlp_semparse.fields.production_rule_field import ProductionRule, ProductionRuleArray
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder, Embedding, Attention, FeedForward, \
    TimeDistributed
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util, Activation
from allennlp_semparse.state_machines import BeamSearch
from allennlp_semparse.state_machines.states import GrammarStatelet
from torch.nn import Parameter

# from models.semantic_parsing.graph_pruning import GraphPruning
# from models.semantic_parsing.spider_base import SpiderBase
from semparse.worlds.evaluate_spider import evaluate
from state_machines.states.rnn_statelet import RnnStatelet
from allennlp_semparse.state_machines.trainers import MaximumMarginalLikelihood
from allennlp.training.metrics import Average
from overrides import overrides

from semparse.contexts.spider_context_utils import action_sequence_to_sql
from semparse.worlds.spider_world import SpiderWorld
from state_machines.states.grammar_based_state import GrammarBasedState
from state_machines.states.sql_state import SqlState
from state_machines.transition_functions.attend_past_schema_items_transition import \
    AttendPastSchemaItemsTransitionFunction
from state_machines.transition_functions.linking_transition_function import LinkingTransitionFunction


def get_neighbor_indices(worlds: List[SpiderWorld],
                            num_entities: int,
                            device: torch.device) -> torch.LongTensor:
    """
    This method returns the indices of each entity's neighbors. A tensor
    is accepted as a parameter for copying purposes.

    Parameters
    ----------
    worlds : ``List[SpiderWorld]``
    num_entities : ``int``
    tensor : ``torch.Tensor``
        Used for copying the constructed list onto the right device.

    Returns
    -------
    A ``torch.LongTensor`` with shape ``(batch_size, num_entities, num_neighbors)``. It is padded
    with -1 instead of 0, since 0 is a valid neighbor index. If all the entities in the batch
    have no neighbors, None will be returned.
    """

    num_neighbors = 0
    for world in worlds:
        for entity in world.db_context.knowledge_graph.entities:
            if len(world.db_context.knowledge_graph.neighbors[entity]) > num_neighbors:
                num_neighbors = len(world.db_context.knowledge_graph.neighbors[entity])

    batch_neighbors = []
    no_entities_have_neighbors = True
    for world in worlds:
        # Each batch instance has its own world, which has a corresponding table.
        entities = world.db_context.knowledge_graph.entities
        entity2index = {entity: i for i, entity in enumerate(entities)}
        entity2neighbors = world.db_context.knowledge_graph.neighbors
        neighbor_indexes = []
        for entity in entities:
            entity_neighbors = [entity2index[n] for n in entity2neighbors[entity]]
            if entity_neighbors:
                no_entities_have_neighbors = False
            # Pad with -1 instead of 0, since 0 represents a neighbor index.
            padded = pad_sequence_to_length(entity_neighbors, num_neighbors, lambda: -1)
            neighbor_indexes.append(padded)
        neighbor_indexes = pad_sequence_to_length(neighbor_indexes,
                                                    num_entities,
                                                    lambda: [-1] * num_neighbors)
        batch_neighbors.append(neighbor_indexes)
    # It is possible that none of the entities has any neighbors, since our definition of the
    # knowledge graph allows it when no entities or numbers were extracted from the question.
    if no_entities_have_neighbors:
        return None
    return torch.tensor(batch_neighbors, device=device, dtype=torch.long)



def query_difficulty(targets: torch.LongTensor, action_mapping, batch_index):
    number_tables = len([action_mapping[(batch_index, int(a))] for a in targets if
                            a >= 0 and action_mapping[(batch_index, int(a))].startswith('table_name')])
    return number_tables > 1


def get_type_vector(worlds: List[SpiderWorld],
                        num_entities: int,
                        device) -> Tuple[torch.LongTensor, Dict[int, int]]:
    """
    Produces the encoding for each entity's type. In addition, a map from a flattened entity
    index to type is returned to combine entity type operations into one method.

    Parameters
    ----------
    worlds : ``List[AtisWorld]``
    num_entities : ``int``
    tensor : ``torch.Tensor``
        Used for copying the constructed list onto the right device.

    Returns
    -------
    A ``torch.LongTensor`` with shape ``(batch_size, num_entities, num_types)``.
    entity_types : ``Dict[int, int]``
        This is a mapping from ((batch_index * num_entities) + entity_index) to entity type id.
    """
    entity_types = {}
    batch_types = []

    column_type_ids = ['boolean', 'foreign', 'number', 'others', 'primary', 'text', 'time']

    for batch_index, world in enumerate(worlds):
        types = []

        for entity_index, entity in enumerate(world.db_context.knowledge_graph.entities):
            parts = entity.split(':')
            entity_main_type = parts[0]
            if entity_main_type == 'column':
                column_type = parts[1]
                entity_type = column_type_ids.index(column_type)
            elif entity_main_type == 'string':
                # cell value
                entity_type = len(column_type_ids)
            elif entity_main_type == 'table':
                entity_type = len(column_type_ids) + 1
            else:
                raise (Exception("Unkown entity"))
            types.append(entity_type)

            # For easier lookups later, we're actually using a _flattened_ version
            # of (batch_index, entity_index) for the key, because this is how the
            # linking scores are stored.
            flattened_entity_index = batch_index * num_entities + entity_index
            entity_types[flattened_entity_index] = entity_type
        padded = pad_sequence_to_length(types, num_entities, lambda: 0)
        batch_types.append(padded)

    return torch.tensor(batch_types, dtype=torch.long, device=device), entity_types


def get_graph_adj_lists(device, world, global_entity_id, global_node=False):
    entity_mapping = {}
    for i, entity in enumerate(world.db_context.knowledge_graph.entities):
        entity_mapping[entity] = i
    entity_mapping['_global_'] = global_entity_id
    adj_list_own = []  # column--table
    adj_list_link = []  # table->table / foreign->primary
    adj_list_linked = []  # table<-table / foreign<-primary
    adj_list_global = []  # node->global

    # TODO: Prepare in advance?
    for key, neighbors in world.db_context.knowledge_graph.neighbors.items():
        idx_source = entity_mapping[key]
        for n_key in neighbors:
            idx_target = entity_mapping[n_key]
            if n_key.startswith("table") or key.startswith("table"):
                adj_list_own.append((idx_source, idx_target))
            elif n_key.startswith("string") or key.startswith("string"):
                adj_list_own.append((idx_source, idx_target))
            elif key.startswith("column:foreign"):
                adj_list_link.append((idx_source, idx_target))
                src_table_key = f"table:{key.split(':')[2]}"
                tgt_table_key = f"table:{n_key.split(':')[2]}"
                idx_source_table = entity_mapping[src_table_key]
                idx_target_table = entity_mapping[tgt_table_key]
                adj_list_link.append((idx_source_table, idx_target_table))
            elif n_key.startswith("column:foreign"):
                adj_list_linked.append((idx_source, idx_target))
                src_table_key = f"table:{key.split(':')[2]}"
                tgt_table_key = f"table:{n_key.split(':')[2]}"
                idx_source_table = entity_mapping[src_table_key]
                idx_target_table = entity_mapping[tgt_table_key]
                adj_list_linked.append((idx_source_table, idx_target_table))
            else:
                assert False

        adj_list_global.append((idx_source, entity_mapping['_global_']))

    all_adj_types = [adj_list_own, adj_list_link, adj_list_linked]

    if global_node:
        all_adj_types.append(adj_list_global)

    return [torch.tensor(l, device=device, dtype=torch.long).transpose(0, 1) if l
            else torch.tensor(l, device=device, dtype=torch.long)
            for l in all_adj_types]



def is_nonterminal(token: str):
    if token[0] == '"' and token[-1] == '"':
        return False
    return True


def action_history_match(predicted: List[int], targets: torch.LongTensor) -> int:
    # TODO(mattg): this could probably be moved into a FullSequenceMatch metric, or something.
    # Check if target is big enough to cover prediction (including start/end symbols)
    if len(predicted) > targets.size(0):
        return 0
    predicted_tensor = targets.new_tensor(predicted)
    targets_trimmed = targets[:len(predicted)]
    # Return 1 if the predicted sequence is anywhere in the list of targets.
    return torch.max(torch.min(targets_trimmed.eq(predicted_tensor), dim=0)[0]).item()

def get_linking_probabilities(  worlds: List[SpiderWorld],
                                linking_scores: torch.FloatTensor,
                                question_mask: torch.LongTensor,
                                entity_type_dict: Dict[int, int],
                                num_entity_types) -> torch.FloatTensor:
    """
    Produces the probability of an entity given a question word and type. The logic below
    separates the entities by type since the softmax normalization term sums over entities
    of a single type.

    Parameters
    ----------
    worlds : ``List[WikiTablesWorld]``
    linking_scores : ``torch.FloatTensor``
        Has shape (batch_size, num_question_tokens, num_entities).
    question_mask: ``torch.LongTensor``
        Has shape (batch_size, num_question_tokens).
    entity_type_dict : ``Dict[int, int]``
        This is a mapping from ((batch_index * num_entities) + entity_index) to entity type id.

    Returns
    -------
    batch_probabilities : ``torch.FloatTensor``
        Has shape ``(batch_size, num_question_tokens, num_entities)``.
        Contains all the probabilities for an entity given a question word.
    """
    _, num_question_tokens, num_entities = linking_scores.size()
    batch_probabilities = []

    for batch_index, world in enumerate(worlds):
        all_probabilities = []
        num_entities_in_instance = 0

        # NOTE: The way that we're doing this here relies on the fact that entities are
        # implicitly sorted by their types when we sort them by name, and that numbers come
        # before "date_column:", followed by "number_column:", "string:", and "string_column:".
        # This is not a great assumption, and could easily break later, but it should work for now.
        for type_index in range(num_entity_types):
            # This index of 0 is for the null entity for each type, representing the case where a
            # word doesn't link to any entity.
            entity_indices = [0]
            entities = world.db_context.knowledge_graph.entities
            for entity_index, _ in enumerate(entities):
                if entity_type_dict[batch_index * num_entities + entity_index] == type_index:
                    entity_indices.append(entity_index)

            if len(entity_indices) == 1:
                # No entities of this type; move along...
                continue

            # We're subtracting one here because of the null entity we added above.
            num_entities_in_instance += len(entity_indices) - 1

            # We separate the scores by type, since normalization is done per type.  There's an
            # extra "null" entity per type, also, so we have `num_entities_per_type + 1`.  We're
            # selecting from a (num_question_tokens, num_entities) linking tensor on _dimension 1_,
            # so we get back something of shape (num_question_tokens,) for each index we're
            # selecting.  All of the selected indices together then make a tensor of shape
            # (num_question_tokens, num_entities_per_type + 1).
            indices = linking_scores.new_tensor(entity_indices, dtype=torch.long)
            entity_scores = linking_scores[batch_index].index_select(1, indices)

            # We used index 0 for the null entity, so this will actually have some values in it.
            # But we want the null entity's score to be 0, so we set that here.
            entity_scores[:, 0] = 0

            # No need for a mask here, as this is done per batch instance, with no padding.
            type_probabilities = torch.nn.functional.softmax(entity_scores, dim=1)
            all_probabilities.append(type_probabilities[:, 1:])

        # We need to add padding here if we don't have the right number of entities.
        if num_entities_in_instance != num_entities:
            zeros = linking_scores.new_zeros(num_question_tokens,
                                                num_entities - num_entities_in_instance)
            all_probabilities.append(zeros)

        # (num_question_tokens, num_entities)
        probabilities = torch.cat(all_probabilities, dim=1)
        batch_probabilities.append(probabilities)
    batch_probabilities = torch.stack(batch_probabilities, dim=0)
    return batch_probabilities * question_mask.unsqueeze(-1).float()


def get_schema_graph_encoding(gnn,
                                worlds: List[SpiderWorld],
                                initial_graph_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    max_num_entities = max([len(world.db_context.knowledge_graph.entities) for world in worlds])
    batch_size = initial_graph_embeddings.size(0)

    graph_data_list = []

    for batch_index, world in enumerate(worlds):
        x = initial_graph_embeddings[batch_index]

        adj_list = get_graph_adj_lists(initial_graph_embeddings.device,
                                                world, initial_graph_embeddings.size(1) - 1)
        graph_data = Data(x)
        for i, l in enumerate(adj_list):
            graph_data[f'edge_index_{i}'] = l
        graph_data_list.append(graph_data)

    batch = Batch.from_data_list(graph_data_list)

    gnn_output = gnn(batch.x, [batch[f'edge_index_{i}'] for i in range(gnn.num_edge_types)])

    num_nodes = max_num_entities
    gnn_output = gnn_output.view(batch_size, num_nodes, -1)
    # entities_encodings = gnn_output
    entities_encodings = gnn_output[:, :max_num_entities]
    # global_node_encodings = gnn_output[:, max_num_entities]

    return entities_encodings