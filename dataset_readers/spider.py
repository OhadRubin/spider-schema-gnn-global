import json
import logging
import os
from typing import List, Dict
import codecs

import dill
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Tokenizer, TokenIndexer, Field, Instance
from allennlp.data.fields import TextField, ListField, IndexField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
# from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from overrides import overrides
from spacy.symbols import ORTH, LEMMA

from dataset_readers.dataset_util.spider_utils import fix_number_value, disambiguate_items
from dataset_readers.fields.spider_knowledge_graph_field import SpiderKnowledgeGraphField
from dataset_readers.fields.production_rule_field import ProductionRuleField
from semparse.contexts.spider_db_context import SpiderDBContext
from semparse.worlds.spider_world import SpiderWorld
import pickle
logger = logging.getLogger(__name__)
import jsonpickle


from time import time


class Timer(object):
    def __init__(self, description):
        self.description = description
    def __enter__(self):
        self.start = time()
    def __exit__(self, type, value, traceback):
        self.end = time()
        print(f"{self.description}: {self.end - self.start}")



import pathlib
@DatasetReader.register("spider")
class SpiderDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 keep_if_unparsable: bool = True,
                 tables_file: str = None,
                 dataset_path: str = 'dataset/database',
                 cache_directory: str = "cache/train",
                #  load_cache: bool = True,
                #  save_cache: bool = True,
                 max_instances = None):
        super().__init__(lazy=lazy,cache_directory=cache_directory,max_instances=max_instances)
        # super().__init__(lazy=lazy,max_instances=max_instances)

        # default spacy tokenizer splits the common token 'id' to ['i', 'd'], we here write a manual fix for that
        spacy_tokenizer = SpacyTokenizer(pos_tags=True)
        spacy_tokenizer.spacy.tokenizer.add_special_case(u'id', [{ORTH: u'id', LEMMA: u'id'}])
        self._tokenizer = spacy_tokenizer

        self._utterance_token_indexers = question_token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._keep_if_unparsable = keep_if_unparsable

        self._tables_file = tables_file
        self._dataset_path = dataset_path
        # self._load_cache = load_cache
        # self._save_cache = save_cache
        # self._loading_limit = loading_limit

    @overrides
    def _read(self, file_path: str):
        if file_path.endswith('.json'):
            yield from self._read_examples_file(file_path)
        else:
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

    def _read_examples_file(self, file_path: str):
        cache_dir = os.path.join('cache', file_path.split("/")[-1])

        # if self._load_cache:
        #     logger.info(f'Trying to load cache from {cache_dir}')
        # if self._save_cache:
        #     os.makedirs(cache_dir, exist_ok=True)

        cnt = 0
        with open(file_path, "r") as data_file:
            
            json_obj = json.load(data_file)
            for total_cnt, ex in enumerate(json_obj):
                # cache_filename = f'instance-{total_cnt}.pt'
                # cache_filepath = os.path.join(cache_dir, cache_filename)
                # if self._loading_limit == cnt:
                #     break

                # if self._load_cache:
                #     # pathlib.Path(cache_filepath).exists()
                #     try:
                #         ins = dill.load(open(cache_filepath, 'rb'))
                #         ins = self.process_instance(ins, total_cnt)
                #         # print("hi")
                #         if ins is None and not self._keep_if_unparsable:
                #             # skip unparsed examples
                #             continue
                #         yield ins
                #         cnt += 1
                #         continue
                #     except Exception as e:
                #         print(e)
                #         # could not load from cache - keep loading without cache
                #         pass

                query_tokens = None
                if 'query_toks' in ex:
                    # fix for examples: we want to use the 'query_toks_no_value' field of the example which anonymizes
                    # values. However, it also anonymizes numbers (e.g. LIMIT 3 -> LIMIT 'value', which is not good
                    # since the official evaluator does expect a number and not a value
                    ex = fix_number_value(ex)

                    # we want the query tokens to be non-ambiguous (i.e. know for each column the table it belongs to,
                    # and for each table alias its explicit name)
                    # we thus remove all aliases and make changes such as:
                    # 'name' -> 'singer@name',
                    # 'singer AS T1' -> 'singer',
                    # 'T1.name' -> 'singer@name'
                    try:
                        query_tokens = disambiguate_items(ex['db_id'], ex['query_toks_no_value'],
                                                                self._tables_file, allow_aliases=False)
                    except Exception as e:
                        # there are two examples in the train set that are wrongly formatted, skip them
                        print(f"error with {ex['query']}")
                        print(e)

                ins = self.text_to_instance(
                    utterance=ex['question'],
                    db_id=ex['db_id'],
                    sql=query_tokens)
                ins = self.process_instance(ins, total_cnt)
                # if ins is not None:
                    # cnt += 1
                # if self._save_cache:
                    # dill.dump(ins, open(cache_filepath, 'wb'))

                if ins is not None:
                    yield ins

    def text_to_instance(self,
                         utterance: str,
                         db_id: str,
                         sql: List[str] = None):
        # print("bye")
        fields: Dict[str, Field] = {}

        db_context = SpiderDBContext(db_id, utterance, tokenizer=self._tokenizer,
                                     tables_file=self._tables_file, dataset_path=self._dataset_path)
        table_field = SpiderKnowledgeGraphField(db_context.knowledge_graph,
                                                db_context.tokenized_utterance,
                                                self._utterance_token_indexers,
                                                entity_tokens=db_context.entity_tokens,
                                                include_in_vocab=False,
                                                max_table_tokens=None)

        world = SpiderWorld(db_context, query=sql)
        fields["utterance"] = TextField(db_context.tokenized_utterance, self._utterance_token_indexers)

        action_sequence, all_actions = world.get_action_sequence_and_all_actions()

        if action_sequence is None and self._keep_if_unparsable:
            # print("Parse error")
            action_sequence = []
        elif action_sequence is None:
            return None

        index_fields: List[Field] = []
        production_rule_fields: List[Field] = []

        for production_rule in all_actions:
            nonterminal, rhs = production_rule.split(' -> ')
            production_rule = ' '.join(production_rule.split(' '))
            field = ProductionRuleField(production_rule,
                                        world.is_global_rule(rhs),
                                        nonterminal=nonterminal)
            production_rule_fields.append(field)

        valid_actions_field = ListField(production_rule_fields)
        fields["valid_actions"] = valid_actions_field

        action_map = {action.rule: i  # type: ignore
                      for i, action in enumerate(valid_actions_field.field_list)}

        for production_rule in action_sequence:
            index_fields.append(IndexField(action_map[production_rule], valid_actions_field))
        if not action_sequence:
            index_fields = [IndexField(-1, valid_actions_field)]

        action_sequence_field = ListField(index_fields)
        fields["action_sequence"] = action_sequence_field
        fields["world"] = MetadataField(world)
        fields["schema"] = table_field
        for key,value  in table_field.__dict__.items():
        # for key,value  in fields.items():
            pickled =  dill.dumps(value)
            res = jsonpickle.encode(pickled)
            # print(f"{key}:{len(res)}")
        ins = Instance(fields)
        # print(ins)
        # import jsonpickle
        # ins_p = jsonpickle.encode(ins)
        # ins_d = jsonpickle.decode(ins_p)
        return  ins

    def process_instance(self, instance: Instance, index: int):
        return instance

    # def serialize_instance(self, instance: Instance) -> str:
    #     # with Timer():
    #     # if True:
    #     pickled =  dill.dumps(instance)
    #     res = jsonpickle.encode(pickled)
    #     # print(len(res))
    #     # res = pickle.loads(codecs.decode(pickled.encode(), 'base64')).decode()


    #     # jsonpickle.decode(jsonpickle.encode(instance))
    #     return res
    #     # return  codecs.encode(pickle.dumps(instance), "base64").decode()

    #     # return pickled.encode()
        

    # def deserialize_instance(self, string: str) -> Instance:
    #     # pickled = codecs.decode(string, "base64")
    #     # with Timer("pickled = jsonpickle.decode(string)"):
    #     pickled = jsonpickle.decode(string)

    #     res = dill.loads(pickled)
    #     # res = pickle.loads(codecs.decode(string.encode(), 'base64'))
    #     return  res  # type: ignore
    #     return pickle.loads(codecs.decode(string.encode(), "base64"))

    @overrides
    def _instances_from_cache_file(self, cache_filename: str):
        with open(cache_filename, "rb") as cache_file:
            yield from dill.load(cache_file)
            # for line in cache_file:
                # yield self.deserialize_instance(line.strip())

    @overrides
    def _instances_to_cache_file(self, cache_filename, instances) -> None:
        with open(cache_filename, "wb") as cache:
            dill.dump(instances,cache)
            # for instance in Tqdm.tqdm(instances):
                # cache.write(self.serialize_instance(instance) + "\n")