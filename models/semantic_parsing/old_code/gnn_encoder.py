from typing import Dict, List, Tuple

import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder, Embedding, TimeDistributed
from torch_geometric.data import Data, Batch

from modules.gated_graph_conv import GatedGraphConv
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

from models.semantic_parsing.graph_pruning import GraphPruning
from models.semantic_parsing import parser_utils
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

from models.semantic_parsing.schema_encoder import SchemaEncoder

@SchemaEncoder.register("gnn")
class GNNEncoder(SchemaEncoder):
    def __init__(self,
                 encoder: Seq2SeqEncoder,
                 entity_encoder: Seq2VecEncoder,
                 question_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 decoder_use_graph_entities: bool = True,
                 gnn_timesteps: int = 2,
                 pruning_gnn_timesteps: int = 2,
                 parse_sql_on_decoding: bool = True,
                 add_action_bias: bool = True,
                 use_neighbor_similarity_for_linking: bool = True,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self._encoder = encoder
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._question_embedder = question_embedder
        self._entity_encoder = TimeDistributed(entity_encoder)

        self._num_entity_types = 9
        self._embedding_dim = question_embedder.get_output_dim()

        self._entity_type_encoder_embedding = Embedding(self._embedding_dim, self._num_entity_types)

        self._linking_params = torch.nn.Linear(16, 1)
        torch.nn.init.uniform_(self._linking_params.weight, 0, 1)
        self._gnn = GatedGraphConv(self._embedding_dim, gnn_timesteps, num_edge_types=3, dropout=dropout)

        self._neighbor_params = torch.nn.Linear(self._embedding_dim, self._embedding_dim)
        self._add_action_bias = add_action_bias

        self._parse_sql_on_decoding = parse_sql_on_decoding
        self._decoder_use_graph_entities = decoder_use_graph_entities
        self._use_neighbor_similarity_for_linking = use_neighbor_similarity_for_linking
        self._action_embedding_dim = action_embedding_dim
        num_actions = 112
        
        if self._add_action_bias:
            input_action_dim = action_embedding_dim + 1
        else:
            input_action_dim = action_embedding_dim
        self._action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=input_action_dim)
        self._output_action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=action_embedding_dim)

        self._encoder_output_dim = self._action_embedding_dim+encoder.get_output_dim()
        # self.encoder_output_dim = 

        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(self._action_embedding_dim))
        self._first_attended_utterance = torch.nn.Parameter(torch.FloatTensor(self._encoder_output_dim))
        self._first_attended_output = torch.nn.Parameter(torch.FloatTensor(self._action_embedding_dim))
        torch.nn.init.normal_(self._first_action_embedding)
        torch.nn.init.normal_(self._first_attended_utterance)
        torch.nn.init.normal_(self._first_attended_output)

        self._entity_type_decoder_embedding = Embedding(self._num_entity_types, action_embedding_dim)

        self._graph_pruning = GraphPruning(3, self._embedding_dim, encoder.get_output_dim(), dropout,
                                           timesteps=pruning_gnn_timesteps)

        self._ent2ent_ff = FeedForward(action_embedding_dim, 1, action_embedding_dim, Activation.by_name('relu')())


    def _get_initial_state(self,
                           utterance: Dict[str, torch.LongTensor],
                           worlds: List[SpiderWorld],
                           schema: Dict[str, torch.LongTensor],
                           actions: List[List[ProductionRule]],
                           action_sequence=None) -> GrammarBasedState:

        valid_actions = actions
        max_len_entities = max([len(w.db_context.knowledge_graph.entities) for w in worlds])
        batch_size = len(worlds)
        # print(utterance['tokens']['tokens'])
        device = utterance['tokens']['tokens'].device
        
        oracle_entities = []
        oracle_relevance_score = None
        if action_sequence is not None:
            
            # we want oracle supervision for which entities should be in the query, for the loss calculation
            for batch_index, batch_actions in enumerate(action_sequence.squeeze(-1)):
                oracle_entities.append(set([valid_actions[batch_index][action][0].split(' -> ')[1].strip('["]') for
                                            action in batch_actions if
                                            not valid_actions[batch_index][action][1] and action >= 0]))
            oracle_relevance_score = [pad_sequence_to_length(w.get_oracle_relevance_score(oe), max_len_entities)
                                            for w, oe in zip(worlds, oracle_entities)]
            oracle_relevance_score = torch.tensor(oracle_relevance_score, dtype=torch.float,
                                                        device=device)

        schema_text = schema['text']
        embedded_schema = self._question_embedder(schema_text, num_wrapping_dims=1)
        schema_mask = util.get_text_field_mask(schema_text, num_wrapping_dims=1).float()

        embedded_utterance = self._question_embedder(utterance)
        utterance_mask = util.get_text_field_mask(utterance).float()

        batch_size, num_entities, num_entity_tokens, _ = embedded_schema.size()
        num_entities = max([len(world.db_context.knowledge_graph.entities) for world in worlds])
        num_question_tokens = utterance['tokens']['tokens'].size(1)

        # entity_types: tensor with shape (batch_size, num_entities), where each entry is the
        # entity's type id.
        # entity_type_dict: Dict[int, int], mapping flattened_entity_index -> type_index
        # These encode the same information, but for efficiency reasons later it's nice
        # to have one version as a tensor and one that's accessible on the cpu.
        entity_types, entity_type_dict = parser_utils.get_type_vector(worlds, num_entities, embedded_schema.device)

        entity_type_embeddings = self._entity_type_encoder_embedding(entity_types)

        # Compute entity and question word similarity.  We tried using cosine distance here, but
        # because this similarity is the main mechanism that the model can use to push apart logit
        # scores for certain actions (like "n -> 1" and "n -> -1"), this needs to have a larger
        # output range than [-1, 1].
        question_entity_similarity = torch.bmm(embedded_schema.view(batch_size,
                                                                    num_entities * num_entity_tokens,
                                                                    self._embedding_dim),
                                               torch.transpose(embedded_utterance, 1, 2))

        question_entity_similarity = question_entity_similarity.view(batch_size,
                                                                     num_entities,
                                                                     num_entity_tokens,
                                                                     num_question_tokens)
        # (batch_size, num_entities, num_question_tokens)
        question_entity_similarity_max_score, _ = torch.max(question_entity_similarity, 2)

        # (batch_size, num_entities, num_question_tokens, num_features)
        linking_features = schema['linking']

        linking_scores = question_entity_similarity_max_score

        feature_scores = self._linking_params(linking_features).squeeze(3)

        linking_scores = linking_scores + feature_scores

        # (batch_size, num_question_tokens, num_entities)
        linking_probabilities = parser_utils.get_linking_probabilities(worlds, linking_scores.transpose(1, 2),
                                                                utterance_mask, entity_type_dict,self._num_entity_types)

        # (batch_size, num_entities, num_neighbors) or None
        neighbor_indices = parser_utils.get_neighbor_indices(worlds, num_entities, linking_scores.device)

        if self._use_neighbor_similarity_for_linking and neighbor_indices is not None:
            # (batch_size, num_entities, embedding_dim)
            encoded_table = self._entity_encoder(embedded_schema, schema_mask)

            # Neighbor_indices is padded with -1 since 0 is a potential neighbor index.
            # Thus, the absolute value needs to be taken in the index_select, and 1 needs to
            # be added for the mask since that method expects 0 for padding.
            # (batch_size, num_entities, num_neighbors, embedding_dim)
            embedded_neighbors = util.batched_index_select(encoded_table, torch.abs(neighbor_indices))

            neighbor_mask = util.get_text_field_mask({"tokens":{'ignored': neighbor_indices + 1}},
                                                     num_wrapping_dims=1).float()

            # Encoder initialized to easily obtain a masked average.
            neighbor_encoder = TimeDistributed(BagOfEmbeddingsEncoder(self._embedding_dim, averaged=True))
            # (batch_size, num_entities, embedding_dim)
            embedded_neighbors = neighbor_encoder(embedded_neighbors, neighbor_mask)
            projected_neighbor_embeddings = self._neighbor_params(embedded_neighbors.float())

            # (batch_size, num_entities, embedding_dim)
            entity_embeddings = torch.tanh(entity_type_embeddings + projected_neighbor_embeddings)
        else:
            # (batch_size, num_entities, embedding_dim)
            entity_embeddings = torch.tanh(entity_type_embeddings)

        link_embedding = util.weighted_sum(entity_embeddings, linking_probabilities)
        encoder_input = torch.cat([link_embedding, embedded_utterance], 2)

        # (batch_size, utterance_length, encoder_output_dim)
        encoder_outputs = self._dropout(self._encoder(encoder_input, utterance_mask))
        # compute the relevance of each entity with the relevance GNN
        ent_relevance, ent_relevance_logits, ent_to_qst_lnk_probs = self._graph_pruning(worlds,
                                                                                        encoder_outputs,
                                                                                        entity_type_embeddings,
                                                                                        linking_scores,
                                                                                        utterance_mask,
                                                                                        parser_utils.get_graph_adj_lists)
        # save this for loss calculation
        self.predicted_relevance_logits = ent_relevance_logits

        # multiply the embedding with the computed relevance
        graph_initial_embedding = entity_type_embeddings * ent_relevance

        encoder_output_dim = self._encoder.get_output_dim()
        entities_graph_encoding = parser_utils.get_schema_graph_encoding(self._gnn, worlds,graph_initial_embedding)
        graph_link_embedding = util.weighted_sum(entities_graph_encoding, linking_probabilities)
        encoder_outputs = torch.cat((
            encoder_outputs,
            graph_link_embedding
        ), dim=-1)
        encoder_output_dim = self._action_embedding_dim + self._encoder.get_output_dim()

        entities_ff = self._ent2ent_ff(entities_graph_encoding)
        linked_actions_linking_scores = torch.bmm(entities_ff, entities_ff.transpose(1, 2))


        # This will be our initial hidden state and memory cell for the decoder LSTM.
        final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                             utterance_mask.long(),
                                                             self._encoder.is_bidirectional())
        memory_cell = encoder_outputs.new_zeros(batch_size, encoder_output_dim)
        initial_score = embedded_utterance.data.new_zeros(batch_size)

        # To make grouping states together in the decoder easier, we convert the batch dimension in
        # all of our tensors into an outer list.  For instance, the encoder outputs have shape
        # `(batch_size, utterance_length, encoder_output_dim)`.  We need to convert this into a list
        # of `batch_size` tensors, each of shape `(utterance_length, encoder_output_dim)`.  Then we
        # won't have to do any index selects, or anything, we'll just do some `torch.cat()`s.
        initial_score_list = [initial_score[i] for i in range(batch_size)]
        encoder_output_list = [encoder_outputs[i] for i in range(batch_size)]
        utterance_mask_list = [utterance_mask[i] for i in range(batch_size)]
        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(RnnStatelet(final_encoder_output[i],
                                                 memory_cell[i],
                                                 self._first_action_embedding,
                                                 self._first_attended_utterance,
                                                 encoder_output_list,
                                                 utterance_mask_list))

        initial_grammar_state = [self._create_grammar_state(worlds[i],
                                                            actions[i],
                                                            linking_scores[i],
                                                            linked_actions_linking_scores[i],
                                                            entity_types[i],
                                                            entities_graph_encoding[
                                                                i] if entities_graph_encoding is not None else None)
                                 for i in range(batch_size)]

        initial_sql_state = [SqlState(actions[i], self._parse_sql_on_decoding) for i in range(batch_size)]

        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_state,
                                          grammar_state=initial_grammar_state,
                                          sql_state=initial_sql_state,
                                          possible_actions=actions,
                                          action_entity_mapping=[w.get_action_entity_mapping() for w in worlds])

        
        loss = torch.tensor([0]).float().to(device)

        if action_sequence is not None:

            graph_mask = util.get_mask_from_sequence_lengths(torch.tensor([len(w.entities_names) for w in worlds],
                                                                           device=device),
                                                              max_len_entities).float()
            graph_loss = torch.nn.functional.binary_cross_entropy_with_logits(ent_relevance_logits.squeeze(-1),
                                                                              oracle_relevance_score, reduction='none')
            graph_loss = (graph_loss * graph_mask).sum() / graph_mask.sum()
            loss += graph_loss
        return initial_state, loss
            

    def _create_grammar_state(self,
                              world: SpiderWorld,
                              possible_actions: List[ProductionRule],
                              linking_scores: torch.Tensor,
                              linked_actions_linking_scores: torch.Tensor,
                              entity_types: torch.Tensor,
                              entity_graph_encoding: torch.Tensor) -> GrammarStatelet:
        action_map = {}
        for action_index, action in enumerate(possible_actions):
            action_string = action[0]
            action_map[action_string] = action_index

        valid_actions = world.valid_actions
        entity_map = {}
        entities = world.entities_names

        for entity_index, entity in enumerate(entities):
            entity_map[entity] = entity_index

        translated_valid_actions: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]] = {}
        for key, action_strings in valid_actions.items():
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.  We'll first split those productions by global vs.
            # linked action.

            action_indices = [action_map[action_string] for action_string in action_strings]
            production_rule_arrays = [(possible_actions[index], index) for index in action_indices]
            global_actions = []
            linked_actions = []
            for production_rule_array, action_index in production_rule_arrays:
                if production_rule_array[1]:
                    global_actions.append((production_rule_array[2], action_index))
                else:
                    linked_actions.append((production_rule_array[0], action_index))

            if global_actions:
                global_action_tensors, global_action_ids = zip(*global_actions)
                global_action_tensor = torch.cat(global_action_tensors, dim=0).to(
                    global_action_tensors[0].device).long()
                global_input_embeddings = self._action_embedder(global_action_tensor)
                global_output_embeddings = self._output_action_embedder(global_action_tensor)
                translated_valid_actions[key]['global'] = (global_input_embeddings,
                                                           global_output_embeddings,
                                                           list(global_action_ids))
            if linked_actions:
                linked_rules, linked_action_ids = zip(*linked_actions)
                entities = [rule.split(' -> ')[1].strip('[]\"') for rule in linked_rules]

                entity_ids = [entity_map[entity] for entity in entities]

                entity_linking_scores = linking_scores[entity_ids]

                if linked_actions_linking_scores is not None:
                    entity_action_linking_scores = linked_actions_linking_scores[entity_ids]

                if not self._decoder_use_graph_entities:
                    entity_type_tensor = entity_types[entity_ids]
                    entity_type_embeddings = (self._entity_type_decoder_embedding(entity_type_tensor)
                                              .to(entity_types.device)
                                              .float())
                else:
                    entity_type_embeddings = entity_graph_encoding.index_select(
                        dim=0,
                        index=torch.tensor(entity_ids, device=entity_graph_encoding.device)
                    )
                translated_valid_actions[key]['linked'] = (entity_linking_scores,
                                                            entity_type_embeddings,
                                                            list(linked_action_ids),
                                                            entity_action_linking_scores)


        return GrammarStatelet(['statement'],
                               translated_valid_actions,
                               parser_utils.is_nonterminal)
