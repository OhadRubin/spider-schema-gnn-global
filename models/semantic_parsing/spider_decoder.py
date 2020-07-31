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
# from dataset_readers.fields.production_rule_field import ProductionRuleField

from allennlp_semparse.fields.production_rule_field  import ProductionRule, ProductionRuleArray
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

from semparse.contexts.spider_context_utils import action_sequence_to_sql
from semparse.worlds.spider_world import SpiderWorld
from state_machines.states.grammar_based_state import GrammarBasedState
from state_machines.states.sql_state import SqlState
from state_machines.transition_functions.attend_past_schema_items_transition import \
    AttendPastSchemaItemsTransitionFunction
from state_machines.transition_functions.linking_transition_function import LinkingTransitionFunction

from models.semantic_parsing.schema_encoder import SchemaEncoder
# from models.semantic_parsing.gnn_encoder import GNNEncoder
# from models.semantic_parsing.bert_encoder import BertEncoder

# from models.semantic_parsing.spider_encoder_decoder import SpiderParser

@Model.register("spider")
class SpiderParser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 schema_encoder: SchemaEncoder,
                 decoder_beam_search: BeamSearch,
                 input_attention: Attention,
                 past_attention: Attention,
                 max_decoding_steps: int,
                 graph_loss_lambda: float = 0.5,
                 dataset_path: str = 'dataset',
                 training_beam_size: int = None,
                 decoder_num_layers: int = 1,
                 dropout: float = 0.0
                 ) -> None:
        super().__init__(vocab)
        self.vocab = vocab
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        self._acc_single = Average()
        self._acc_multi = Average()

        self._schema_encoder = schema_encoder
        self._schema_encoder._vocab = vocab

        self._max_decoding_steps = max_decoding_steps
        if "_add_action_bias" in self._schema_encoder.__dict__: 
            self._add_action_bias = self._schema_encoder._add_action_bias
        self._action_padding_index = -1  # the padding value used by IndexField

        self._exact_match = Average()
        self._sql_evaluator_match = Average()
        self._action_similarity = Average()
        self._beam_hit = Average()

        self._graph_loss_lambda = graph_loss_lambda



        self._decoder_num_layers = decoder_num_layers
        self._beam_search = decoder_beam_search
        self._decoder_trainer = MaximumMarginalLikelihood(training_beam_size)

        # self._transition_function = AttendPastSchemaItemsTransitionFunction(encoder_output_dim=self._schema_encoder._encoder_output_dim,
        #                                                                     action_embedding_dim=self._schema_encoder._action_embedding_dim,
        #                                                                     input_attention=input_attention,
        #                                                                     past_attention=past_attention,
        #                                                                     predict_start_type_separately=False,
        #                                                                     add_action_bias=self._add_action_bias,
        #                                                                     dropout=dropout,
        #                                                                     num_layers=self._decoder_num_layers)
        self._transition_function = LinkingTransitionFunction(encoder_output_dim=self._schema_encoder._encoder_output_dim,
                                                                action_embedding_dim=self._schema_encoder._action_embedding_dim,
                                                                input_attention=input_attention,
                                                                predict_start_type_separately=False,
                                                                add_action_bias=self._add_action_bias,
                                                                dropout=dropout,
                                                                num_layers=self._decoder_num_layers)


        # TODO: Remove hard-coded dirs
        self._evaluate_func = partial(evaluate,
                                      db_dir=os.path.join(dataset_path, 'database'),
                                      table=os.path.join(dataset_path, 'tables.json'),
                                      check_valid=False)

    @overrides
    def forward(self,  # type: ignore
                valid_actions: List[List[ProductionRule]],
                world: List[SpiderWorld],
                utterance: Dict[str, torch.LongTensor] = None,
                schema: Dict[str, torch.LongTensor]= None,
                action_sequence: torch.LongTensor = None,
                enc=None,
                relation=None,
                schema_strings=None,
                lengths=None,
                offsets=None
                ) -> Dict[str, torch.Tensor]:
        batch_size = len(world)
        initial_state, loss = self._schema_encoder._get_initial_state(enc, world, schema,
                valid_actions, action_sequence, schema_strings, lengths, offsets, relation)
        if action_sequence is not None:
            # Remove the trailing dimension (from ListField[ListField[IndexField]]).
            action_sequence = action_sequence.squeeze(-1)
            action_mask = action_sequence != self._action_padding_index
        else:
            action_mask = None

        if action_sequence is not None:
            loss = self._graph_loss_lambda*loss
    

        if self.training:
            try:
                decode_output = self._decoder_trainer.decode(initial_state,
                                                             self._transition_function,
                                                             (action_sequence.unsqueeze(1), action_mask.unsqueeze(1)))
                query_loss = decode_output['loss']
            except ZeroDivisionError:
                # print("hi")
                return {'loss': Parameter(torch.tensor([0]).float()).to(action_sequence.device)}

            loss += ((1-self._graph_loss_lambda) * query_loss)

            return {'loss': loss}
        else:
            if action_sequence is not None and action_sequence.size(1) > 1:
                try:
                    query_loss = self._decoder_trainer.decode(initial_state,
                                                              self._transition_function,
                                                              (action_sequence.unsqueeze(1),
                                                               action_mask.unsqueeze(1)))['loss']
                    loss += query_loss
                except ZeroDivisionError:
                    pass

            outputs: Dict[str, Any] = {
                'loss': loss
            }

            num_steps = self._max_decoding_steps
            # This tells the state to start keeping track of debug info, which we'll pass along in
            # our output dictionary.
            initial_state.debug_info = [[] for _ in range(batch_size)]

            best_final_states = self._beam_search.search(num_steps,
                                                         initial_state,
                                                         self._transition_function,
                                                         keep_final_unfinished_states=False)

            self._compute_validation_outputs(valid_actions,
                                             best_final_states,
                                             world,
                                             action_sequence,
                                             outputs)
            return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            '_match/exact_match': self._exact_match.get_metric(reset),
            'sql_match': self._sql_evaluator_match.get_metric(reset),
            '_others/action_similarity': self._action_similarity.get_metric(reset),
            '_match/match_single': self._acc_single.get_metric(reset),
            '_match/match_hard': self._acc_multi.get_metric(reset),
            'beam_hit': self._beam_hit.get_metric(reset)
        }

    def _compute_validation_outputs(self,

                                    actions: List[List[ProductionRuleArray]],
                                    best_final_states: Mapping[int, Sequence[GrammarBasedState]],
                                    world: List[SpiderWorld],
                                    target_list: List[List[str]],
                                    outputs: Dict[str, Any]) -> None:
        batch_size = len(actions)

        outputs['predicted_sql_query'] = []
        outputs['candidates'] = []

        action_mapping = {}
        for batch_index, batch_actions in enumerate(actions):
            for action_index, action in enumerate(batch_actions):
                action_mapping[(batch_index, action_index)] = action[0]

        for i in range(batch_size):
            # gold sql exactly as given
            original_gold_sql_query = ' '.join(world[i].get_query_without_table_hints())

            if i not in best_final_states:
                self._exact_match(0)
                self._action_similarity(0)
                self._sql_evaluator_match(0)
                self._acc_multi(0)
                self._acc_single(0)
                outputs['predicted_sql_query'].append('')
                continue

            best_action_indices = best_final_states[i][0].action_history[0]

            action_strings = [action_mapping[(i, action_index)]
                              for action_index in best_action_indices]
            predicted_sql_query = action_sequence_to_sql(action_strings, add_table_names=True)
            ref_predicted_sql_query =  sqlparse.format(predicted_sql_query, reindent=False)
            print(ref_predicted_sql_query)
            outputs['predicted_sql_query'].append(ref_predicted_sql_query)

            if target_list is not None:
                targets = target_list[i].data
            target_available = target_list is not None and targets[0] > -1

            if target_available:
                sequence_in_targets = parser_utils.action_history_match(best_action_indices, targets)
                self._exact_match(sequence_in_targets)

                sql_evaluator_match = self._evaluate_func(original_gold_sql_query, predicted_sql_query, world[i].db_id)
                self._sql_evaluator_match(sql_evaluator_match)

                similarity = difflib.SequenceMatcher(None, best_action_indices, targets)
                self._action_similarity(similarity.ratio())

                difficulty = parser_utils.query_difficulty(targets, action_mapping, i)
                if difficulty:
                    self._acc_multi(sql_evaluator_match)
                else:
                    self._acc_single(sql_evaluator_match)

            beam_hit = False
            candidates = []
            for pos, final_state in enumerate(best_final_states[i]):
                action_indices = final_state.action_history[0]
                action_strings = [action_mapping[(i, action_index)]
                                  for action_index in action_indices]
                candidate_sql_query = action_sequence_to_sql(action_strings, add_table_names=True)

                correct = False
                if target_available:
                    correct = self._evaluate_func(original_gold_sql_query, candidate_sql_query, world[i].db_id)
                    if correct:
                        beam_hit = True
                    self._beam_hit(beam_hit)
                candidates.append({
                    'query': action_sequence_to_sql(action_strings, add_table_names=True),
                    'correct': correct
                })
            outputs['candidates'].append(candidates)

