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
from models.semantic_parsing import parser_utils 
from semparse.worlds.evaluate_spider import evaluate
from state_machines.states.rnn_statelet import RnnStatelet
from allennlp_semparse.state_machines.trainers import MaximumMarginalLikelihood
from allennlp.training.metrics import Average
from overrides import overrides
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from semparse.contexts.spider_context_utils import action_sequence_to_sql
from semparse.worlds.spider_world import SpiderWorld
from state_machines.states.grammar_based_state import GrammarBasedState
from state_machines.states.sql_state import SqlState
from state_machines.transition_functions.attend_past_schema_items_transition import \
    AttendPastSchemaItemsTransitionFunction
from state_machines.transition_functions.linking_transition_function import LinkingTransitionFunction
import numpy as np
import  modules.transformer as transformer
from models.semantic_parsing.schema_encoder import SchemaEncoder
from allennlp.modules.matrix_attention import BilinearMatrixAttention


@SchemaEncoder.register("ratsql")
class RatsqlEncoder(SchemaEncoder):
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
        # self._question_embedder = question_embedder
        self._question_embedder = PretrainedTransformerMismatchedEmbedder("distilbert-base-uncased")
        num_layers = 4
        num_heads = 8
        hidden_size = 768
        ff_size = 768
        tie_layers = False
        n_relations = 51
        self.rat_encoder = transformer.Encoder(
            lambda: transformer.EncoderLayer(
                hidden_size,
                transformer.MultiHeadedAttentionWithRelations(
                    num_heads,
                    hidden_size,
                    dropout),
                transformer.PositionwiseFeedForward(
                    hidden_size,
                    ff_size,
                    dropout),
                n_relations,
                dropout),
            hidden_size,
            num_layers,
            tie_layers)
            

        self._entity_encoder = TimeDistributed(entity_encoder)

        self._num_entity_types = 9
        self._embedding_dim = self._question_embedder.get_output_dim()
        self._att_question_schema = BilinearMatrixAttention(self._embedding_dim,self._embedding_dim)
        self._att_schema_schema = BilinearMatrixAttention(self._embedding_dim,self._embedding_dim)

        # self._embedding_dim = 200
        self._entity_type_encoder_embedding = Embedding(self._embedding_dim, self._num_entity_types)

        self._linking_params = torch.nn.Linear(16, 1)
        torch.nn.init.uniform_(self._linking_params.weight, 0, 1)
        # self._gnn = GatedGraphConv(self._embedding_dim, gnn_timesteps, num_edge_types=3, dropout=dropout)

        self._neighbor_params = torch.nn.Linear(self._embedding_dim, self._embedding_dim)
        self._action_embedding_dim = action_embedding_dim
        

        self._add_action_bias = add_action_bias

        self._parse_sql_on_decoding = parse_sql_on_decoding
        self._decoder_use_graph_entities = decoder_use_graph_entities
        self._use_neighbor_similarity_for_linking = use_neighbor_similarity_for_linking
        
        num_actions = 112
        
        if self._add_action_bias:
            input_action_dim = action_embedding_dim + 1
        else:
            input_action_dim = action_embedding_dim
        self._action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=input_action_dim)
        self._output_action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=action_embedding_dim)
        
        self._encoder_output_dim = self._action_embedding_dim+self._embedding_dim
        self._to_emb = torch.nn.Linear( self._embedding_dim, self._encoder_output_dim)
        # # self.encoder_output_dim = 

        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(self._action_embedding_dim))
        self._first_attended_utterance = torch.nn.Parameter(torch.FloatTensor(self._encoder_output_dim))
        self._first_attended_output = torch.nn.Parameter(torch.FloatTensor(self._action_embedding_dim))
        torch.nn.init.normal_(self._first_action_embedding)
        torch.nn.init.normal_(self._first_attended_utterance)
        torch.nn.init.normal_(self._first_attended_output)

        self._entity_type_decoder_embedding = Embedding(self._num_entity_types, action_embedding_dim)

        # self._graph_pruning = GraphPruning(3, self._embedding_dim, encoder.get_output_dim(), dropout,
        #                                    timesteps=pruning_gnn_timesteps)

        self._ent2ent_ff = FeedForward(action_embedding_dim, 1,
                                       action_embedding_dim,
                                       Activation.by_name('relu')())

        


    def _get_initial_state(self,
                           utterance: Dict[str, torch.LongTensor],
                           worlds: List[SpiderWorld],
                           schema: Dict[str, torch.LongTensor],
                           actions: List[List[ProductionRule]],
                           action_sequence=None,
                           schema_strings=None,
                           lengths=None,
                           offsets=None,
                           relation = None) -> GrammarBasedState:

        utterance_schema =utterance
        batch_size = len(worlds)
        device = utterance_schema['tokens']["token_ids"].device
        utterance_schema['tokens']['offsets']=offsets
        embedded_utterance_schema = self._question_embedder(**utterance_schema['tokens'])
        
        relation_mask = torch.ones_like(relation) #TODO: fixme

        enriched_utterance_schema = self.rat_encoder(embedded_utterance_schema, relation.long(),relation_mask)

        #TODO: validate that this is working as intented
        utterance_schema, utterance_schema_mask = parser_utils.batched_span_select(enriched_utterance_schema,lengths) 
        
        utterance,schema = torch.split(utterance_schema,1,dim=1)
        utterance_mask,schema_mask = torch.split(utterance_schema_mask,1,dim=1)

        utterance_mask = torch.squeeze(utterance_mask, 1)
        schema_mask = torch.squeeze(schema_mask, 1)
        utterance = torch.squeeze(utterance, 1)
        schema = torch.squeeze(schema, 1)
        # for x in [utterance_mask,schema_mask,utterance,schema]:
            # print(x.size())

        
        
        # replace with avg of the question concat with avg of the schema
        final_encoder_output = self._to_emb(util.masked_mean(utterance,utterance_mask.unsqueeze(-1),dim=1))

        # replace with rat_encoder outputs for the question concatenated with the schema weighted by the linking scores
        encoder_outputs = self._to_emb(utterance)
        memory_cell = torch.zeros([batch_size, self._encoder_output_dim],device=device)  #OK

        linking_scores  = self._att_question_schema(utterance,schema)
        linked_actions_linking_scores = self._att_schema_schema(schema,schema)
        
        initial_score = torch.zeros([batch_size],device=device) #OK
        
        
        # 
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
                                                            schema[i],
                                                            schema_strings[i])
                                 for i in range(batch_size)]

        initial_sql_state = [SqlState(actions[i], self._parse_sql_on_decoding) for i in range(batch_size)]

        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_state,
                                          grammar_state=initial_grammar_state,
                                          sql_state=initial_sql_state,
                                          possible_actions=actions,
                                          action_entity_mapping=[w.get_action_entity_mapping() for w in worlds]) #TODO: fixme (get_action_entity_mapping)
        
        loss = torch.tensor([0]).float().to(device)

        return initial_state, loss
            

    def _create_grammar_state(self,
                              world: SpiderWorld,
                              possible_actions: List[ProductionRule],
                              linking_scores: torch.Tensor,
                              linked_actions_linking_scores: torch.Tensor,
                              entity_graph_encoding: torch.Tensor,
                              schema_strings) -> GrammarStatelet:
        action_map = {}
        for action_index, action in enumerate(possible_actions):
            action_string = action[0]
            action_map[action_string] = action_index

        valid_actions = world.valid_actions
        entity_map = {}
        # entities = world.entities_names

        for entity_index, entity in enumerate(schema_strings):
            entity_map[entity] = entity_index

        translated_valid_actions: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]] = {}

        for key, action_strings in valid_actions.items():
            # print(key, action_strings)
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.  We'll first split those productions by global vs.
            # linked action.

            action_indices = [action_map[action_string] for action_string in action_strings]
            production_rule_arrays = [(possible_actions[index], index) for index in action_indices]
            global_actions = []
            linked_actions = []
            for production_rule_array, action_index in production_rule_arrays:
                # print(production_rule_array, action_index)
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
                # print(entities)
                entity_ids = []
                for entity in entities:
                    if entity in entity_map:
                        entity_ids.append(entity_map[entity])
                if not entity_ids:
                    #TODO: for non_literal_num we just take the first column -  need to fix this!!! maybe move this to global?
                    entity_ids = [0] 
                    # continue

                entity_type_embeddings = entity_graph_encoding.index_select(
                    dim=0,
                    index=torch.tensor(entity_ids, device=entity_graph_encoding.device)
                )
                entity_linking_scores = linking_scores[entity_ids]
                entity_action_linking_scores = linked_actions_linking_scores[entity_ids]
                translated_valid_actions[key]['linked'] = (entity_linking_scores,
                                                            entity_type_embeddings,
                                                            list(linked_action_ids),
                                                            entity_action_linking_scores)
        # print(translated_valid_actions)

        return GrammarStatelet(['statement'],
                               translated_valid_actions,
                               parser_utils.is_nonterminal)
