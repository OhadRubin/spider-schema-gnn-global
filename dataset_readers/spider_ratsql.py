import itertools
import numpy as np
import collections
import itertools
import json
import os
import sys

import attr
import numpy as np
import torch
from transformers import BertModel, BertTokenizer

import re
import string


import nltk.corpus
import json
import re
import sqlite3
from copy import copy
from pathlib import Path
from typing import List, Dict

import attr
import torch
from tqdm import tqdm



import json
import logging
import os
from typing import List, Dict
import codecs

import dill
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Tokenizer, TokenIndexer, Field, Instance
from allennlp.data.fields import TextField, ListField, IndexField, MetadataField, ArrayField
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
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from collections import defaultdict
from allennlp.data import Tokenizer, Token
from semparse.contexts.spider_context_utils import action_sequence_to_sql
import pickle
import jsonpickle
from time import time
import pathlib
from tqdm import tqdm

logger = logging.getLogger(__name__)

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNKS = set(a for a in string.punctuation)



def normalize_schema_constant(entity):
    col = "_".join(entity)
    col =  col.split("_<table-sep>_")
    if len(col)==1:
        return "_".join(entity)
    else:
        table = col[1]
        col = col[0].split(">_")[1]
        return f"{table}@{col}"

def clamp(value, abs_max):
    value = max(-abs_max, value)
    value = min(abs_max, value)
    return value

def to_dict_with_sorted_values(d, key=None):
    return {k: sorted(v, key=key) for k, v in d.items()}


def to_dict_with_set_values(d):
    result = {}
    for k, v in d.items():
        hashable_v = []
        for v_elem in v:
            if isinstance(v_elem, list):
                hashable_v.append(tuple(v_elem))
            else:
                hashable_v.append(v_elem)
        result[k] = set(hashable_v)
    return result


def tuplify(x):
    if not isinstance(x, (tuple, list)):
        return x
    return tuple(tuplify(elem) for elem in x)

@attr.s
class SpiderItem:
    text = attr.ib()
    code = attr.ib()
    schema = attr.ib()
    orig = attr.ib()
    orig_schema = attr.ib()


@attr.s
class Column:
    id = attr.ib()
    table = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    type = attr.ib()
    foreign_key_for = attr.ib(default=None)


@attr.s
class Table:
    id = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    columns = attr.ib(factory=list)
    primary_keys = attr.ib(factory=list)


@attr.s
class Schema:
    db_id = attr.ib()
    tables = attr.ib()
    columns = attr.ib()
    foreign_key_graph = attr.ib()
    orig = attr.ib()
    connection = attr.ib(default=None)


@attr.s
class PreprocessedSchema:
    column_names = attr.ib(factory=list)
    table_names = attr.ib(factory=list)
    table_bounds = attr.ib(factory=list)
    column_to_table = attr.ib(factory=dict)
    table_to_columns = attr.ib(factory=dict)
    foreign_keys = attr.ib(factory=dict)
    foreign_keys_tables = attr.ib(factory=lambda: collections.defaultdict(set))
    primary_keys = attr.ib(factory=list)


@DatasetReader.register("spider_ratsql")
class SpiderRatsqlDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 keep_if_unparsable: bool = True,
                 tables_file: str = None,
                 dataset_path: str = 'dataset/database',
                 cache_directory: str = "cache/train",
                 include_table_name_in_column=True,
                 fix_issue_16_primary_keys=False,
                 qq_max_dist=2,
                 cc_max_dist=2,
                 tt_max_dist=2,
                 max_instances=None):

        super().__init__(lazy=lazy, cache_directory=cache_directory, max_instances=max_instances)
        # super().__init__(lazy=lazy,cache_directory=cache_directory,max_instances=None)
        self._max_instances = max_instances
        # default spacy tokenizer splits the common token 'id' to ['i', 'd'], we here write a manual fix for that
        # spacy_tokenizer = SpacyTokenizer(pos_tags=True)
        # spacy_tokenizer.spacy.tokenizer.add_special_case(u'id', [{ORTH: u'id', LEMMA: u'id'}])
        # self._utterance_token_indexers = {"tokens":PretrainedTransformerIndexer("distilbert-base-uncased")}
        # self._utterance_token_indexers = {"tokens":PretrainedTransformerMismatchedIndexer("bert-base-uncased")}


        # self._utterance_token_indexers = {"tokens":PretrainedTransformerIndexer("bert-base-uncased")}
        self._utterance_token_indexers = question_token_indexers
        
        self._tokenizer = self._utterance_token_indexers['tokens']._allennlp_tokenizer
        # self._utterance_token_indexers['tokens']._tokenizer = self._utterance_token_indexers['tokens']._allennlp_tokenizer
        # self._tokenizer = spacy_tokenizer

        # self._utterance_token_indexers = question_token_indexers or {'tokens': SingleIdTokenIndexer()}
        
        self._keep_if_unparsable = keep_if_unparsable

        self._tables_file = tables_file
        self._dataset_path = dataset_path

        #ratsql
        self.include_table_name_in_column = include_table_name_in_column
        self.fix_issue_16_primary_keys = fix_issue_16_primary_keys
        self.texts = collections.defaultdict(list)
        self.counted_db_ids = set()
        self.preprocessed_schemas = {}
        self.qq_max_dist = qq_max_dist
        self.cc_max_dist = cc_max_dist
        self.tt_max_dist = tt_max_dist

        self.relation_ids = {}

        def add_relation(name):
            self.relation_ids[name] = len(self.relation_ids)

        def add_rel_dist(name, max_dist):
            for i in range(-max_dist, max_dist + 1):
                add_relation((name, i))

        add_rel_dist('qq_dist', qq_max_dist)
        add_rel_dist('cc_dist', cc_max_dist)
        add_rel_dist('tt_dist', tt_max_dist)

        rel_names = ['qc_default','qt_default','cq_default','cc_default','cc_foreign_key_forward',
                    'cc_foreign_key_backward','cc_table_match','ct_default','ct_foreign_key',
                    'ct_primary_key','ct_table_match','ct_any_table','tq_default','tc_default','tc_primary_key',
                    'tc_table_match','tc_any_table','tc_foreign_key',
                    'tt_default','tt_foreign_key_forward','tt_foreign_key_backward','tt_foreign_key_both',
                    'qcCEM','cqCEM','qtTEM','tqTEM','qcCPM','cqCPM','qtTPM','tqTPM',
                    "qcNUMBER","cqNUMBER","qcTIME","cqTIME","qcCELLMATCH","cqCELLMATCH",]
        for rel in rel_names:
            add_relation(rel)

        self.schemas=None
        self.eval_foreign_key_maps=None
        self.schemas, self.eval_foreign_key_maps = self.load_tables([self._tables_file])

        for db_id, schema in self.schemas.items():
            sqlite_path = Path(self._dataset_path) / db_id / f"{db_id}.sqlite"
            source: sqlite3.Connection
            with sqlite3.connect(sqlite_path) as source:
                dest = sqlite3.connect(':memory:')
                dest.row_factory = sqlite3.Row
                source.backup(dest)
            schema.connection = dest
        # super().__init__(lazy=lazy,max_instances=max_instances)


    


    @overrides
    def _read(self, file_path: str):
        if file_path.endswith('.json'):
            yield from self._read_examples_file(file_path)
        else:
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

    def _read_examples_file(self, file_path: str):
        cache_dir = os.path.join('cache', file_path.split("/")[-1])


        cnt = 0
        with open(file_path, "r") as data_file:
            
            json_obj = json.load(data_file)
            for total_cnt, ex in tqdm(enumerate(json_obj)):
                # if total_cnt<2150:
                    # continue
                if total_cnt==self._max_instances:
                    break
                query_tokens = None
                if 'query_toks' in ex:

                    

                    try:
                        ex = fix_number_value(ex)
                        query_tokens = disambiguate_items(ex['db_id'], ex['query_toks_no_value'],
                                                                self._tables_file, allow_aliases=False)
                    except Exception as e:
                        # there are two examples in the train set that are wrongly formatted, skip them
                        print(f"error with {ex['query']}")
                        print(e)
                        continue

                ins = self.text_to_instance(
                    utterance=ex['question'],
                    db_id=ex['db_id'],
                    sql=query_tokens,orig=ex)
                ins = self.process_instance(ins, total_cnt)

                if ins is not None:
                    yield ins
                # sys.exc_clear()
                # sys.exc_traceback = sys.last_traceback = None

    def text_to_instance(self,
                         utterance: str,
                         db_id: str,
                         sql: List[str] = None,
                         orig=None):
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

        item = SpiderItem(
            # text=orig['question_toks'],
            text=[x.text for x in world.db_context.tokenized_utterance[1:-1]],
            code=orig['sql'],
            schema=self.schemas[db_id],
            orig=orig,
            orig_schema=self.schemas[db_id].orig)
        desc = self.preprocess_item(item, "train")

        schema_strings = [normalize_schema_constant(x) for x in desc['columns']+desc['tables']]
        fields["schema_strings"] = MetadataField(schema_strings)
        q = [x.lower() for x in desc['question']]
        c = ["_".join(x).lower() for x in desc['columns']]
    
        t = ["_".join(x).lower() for x in desc['tables']]
        t = ["<table>" + x for x in t]
        enc = q + c + t
        relation = self.compute_relations(desc, len(enc), len(q), len(c), range(len(c)+1), range(len(t)+1))

        rel_dict = defaultdict(dict)
        for i,x in  enumerate(list(range(len(q)))+schema_strings):
            for j,y in  enumerate(list(range(len(q)))+schema_strings):
                rel_dict[x][y]= relation[i,j]


        entities = []
        for entity_text,entity in zip(world.db_context.entity_texts,world.db_context.knowledge_graph.entities):
            entity_text = entity_text.replace(" ","_")
            if entity.split(":")[0] == "column":
                table = entity.split(":")[2]
                table = world.db_context.schema[table].text.lower().replace(" ","_")
                # col = world.db_context.schema[table].text.lower().replace(" ","_")
                entities.append(f"{table}@{entity_text}")
                
            else:
                entities.append(entity_text)

        new_enc = list(range(len(q)))+entities
        new_relation = np.zeros([len(new_enc),len(new_enc)])
        try:
            for i,x in  enumerate(new_enc):
                for j,y in  enumerate(new_enc):
                    new_relation[i][j] = rel_dict[x][y]
        except:
            print("err")
            return None
        fields['relation'] = ArrayField(new_relation,padding_value=-1,dtype=np.int32)

        # ebc
        cls_token = self._tokenizer.tokenize('a')[0]
        eos_token = self._tokenizer.tokenize('a')[-1]
        enc_field_list = []
        sizes_list = []
        # for x in self._tokenizer.batch_tokenize(enc):
        #     token_list = [y for y in x[1:-1] if y.text not in ['_']]
        #     sizes_list.append(len(token_list))
        #     enc_field_list.extend(token_list) 
        # world.db_context.tokenized_utterance
        # enc__fi
        #TODO: add sep and update the lengths accordingly... i.e change to [1:]
        q= [[x] for x in world.db_context.tokenized_utterance[1:-1]]
        # q= [[x] for x in world.db_context.tokenized_utterance[1:]]
        # sizes_list.append(len(q))
        # schema_tokens = [x[1:-1] for x in world.db_context.entity_tokens]
        schema_tokens = [[eos_token]+self._tokenizer.tokenize(x)[1:] for x in world.db_context.knowledge_graph.entities[:1]]
        # schema_tokens
        schema_tokens.extend([self._tokenizer.tokenize(x)[1:] for x in world.db_context.knowledge_graph.entities[1:]])
        schema_tokens = [[y for y in x if y.text not in ["_"]] for x in schema_tokens]
        # print(schema_tokens)
        # exit(0)
        enc_field_list = []
        # sizes_list = []
        # print(q+schema_tokens)
        for x in [[cls_token]]+q+schema_tokens+[[eos_token]]:
            sizes_list.append(len(x))
            enc_field_list.extend(x) 

        if len(enc_field_list)>512:
            return None

        def get_offsets(lengths):
            e = np.cumsum(([0]+list(lengths))[:-1])
            return list(zip(e+1,e+np.array(lengths)))
        offsets = get_offsets(sizes_list[1:-1])
        # print(offsets)
        # fields['lengths'] = ArrayField(np.array([[0,len(q)-1],[len(q),len(q)+len(schema_tokens)-1]]),dtype=np.int32)
        fields['lengths'] = ArrayField(np.array([[0,len(q)-1],[len(q),len(q)+len(schema_tokens)]]),dtype=np.int32)
        fields['offsets'] = ArrayField(np.array(offsets),padding_value=0,dtype=np.int32)
        fields["enc"] = TextField(enc_field_list, self._utterance_token_indexers)


        
        # print(desc)
        # print(rel_dict)
        # # # print(enc)
        # print(world.db_context.entity_tokens)
        # print(world.db_context.tokenized_utterance)
        # print(world.db_context.entity_texts)
        # print(world.db_context.utterance)
        # print()
        # enc = 
        # assert len([x.text for x in world.db_context.tokenized_utterance[1:-1]]) ==len(q)
        # print(list(world.entities_names.keys()))
        # print(world.entities_names)
        # assert len(list(world.entities_names.keys())) == (len(schema_strings) -1)
        

        

        # world.entities_names = {j:i for i,j in enumerate(schema_strings)}
        # fields["utterance"] = TextField(db_context.tokenized_utterance, self._utterance_token_indexers)
        action_sequence, all_actions = world.get_action_sequence_and_all_actions()
        if action_sequence is None and self._keep_if_unparsable:
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
        
            
        # print(action_sequence_to_sql([production_rule for production_rule in action_sequence],add_table_names=True))
        for production_rule in action_sequence:
            index_fields.append(IndexField(action_map[production_rule], valid_actions_field))
        if not action_sequence:
            index_fields = [IndexField(-1, valid_actions_field)]
        # print(' '.join(world.get_query_without_table_hints()))
        action_sequence_field = ListField(index_fields)
        fields["action_sequence"] = action_sequence_field

        fields["world"] = MetadataField(world)
        # fields["schema"] = table_field
        ins = Instance(fields)

        return  ins

    def process_instance(self, instance: Instance, index: int):
        return instance


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


    def compute_relations(self, desc, enc_length, q_enc_length, c_enc_length, c_boundaries, t_boundaries):
        sc_link = desc.get('sc_link', {'q_col_match': {}, 'q_tab_match': {}})
        cv_link = desc.get('cv_link', {'num_date_match': {}, 'cell_match': {}})

        # Catalogue which things are where
        loc_types = {}
        for i in range(q_enc_length):
            loc_types[i] = ('question',)

        c_base = q_enc_length
        for c_id, (c_start, c_end) in enumerate(zip(c_boundaries, c_boundaries[1:])):
            for i in range(c_start + c_base, c_end + c_base):
                loc_types[i] = ('column', c_id)
        t_base = q_enc_length + c_enc_length
        for t_id, (t_start, t_end) in enumerate(zip(t_boundaries, t_boundaries[1:])):
            
            for i in range(t_start + t_base, t_end + t_base):
                loc_types[i] = ('table', t_id)
        relations = np.empty((enc_length, enc_length), dtype=np.int64)

        for i, j in itertools.product(range(enc_length), repeat=2):
            def set_relation(name):
                relations[i, j] = self.relation_ids[name]

            i_type, j_type = loc_types[i], loc_types[j]
            if i_type[0] == 'question':
                if j_type[0] == 'question':
                    set_relation(('qq_dist', clamp(j - i, self.qq_max_dist)))
                elif j_type[0] == 'column':
                    # set_relation('qc_default')
                    j_real = j - c_base
                    if f"{i},{j_real}" in sc_link["q_col_match"]:
                        set_relation("qc" + sc_link["q_col_match"][f"{i},{j_real}"])
                    elif f"{i},{j_real}" in cv_link["cell_match"]:
                        set_relation("qc" + cv_link["cell_match"][f"{i},{j_real}"])
                    elif f"{i},{j_real}" in cv_link["num_date_match"]:
                        set_relation("qc" + cv_link["num_date_match"][f"{i},{j_real}"])
                    else:
                        set_relation('qc_default')
                elif j_type[0] == 'table':
                    j_real = j - t_base
                    if f"{i},{j_real}" in sc_link["q_tab_match"]:
                        set_relation("qt" + sc_link["q_tab_match"][f"{i},{j_real}"])
                    else:
                        set_relation('qt_default')

            elif i_type[0] == 'column':
                if j_type[0] == 'question':
                    i_real = i - c_base
                    if f"{j},{i_real}" in sc_link["q_col_match"]:
                        set_relation("cq" + sc_link["q_col_match"][f"{j},{i_real}"])
                    elif f"{j},{i_real}" in cv_link["cell_match"]:
                        set_relation("cq" + cv_link["cell_match"][f"{j},{i_real}"])
                    elif f"{j},{i_real}" in cv_link["num_date_match"]:
                        set_relation("cq" + cv_link["num_date_match"][f"{j},{i_real}"])
                    else:
                        set_relation('cq_default')
                elif j_type[0] == 'column':
                    col1, col2 = i_type[1], j_type[1]
                    if col1 == col2:
                        set_relation(('cc_dist', clamp(j - i, self.cc_max_dist)))
                    else:
                        set_relation('cc_default')
                        if desc['foreign_keys'].get(str(col1)) == col2:
                            set_relation('cc_foreign_key_forward')
                        if desc['foreign_keys'].get(str(col2)) == col1:
                            set_relation('cc_foreign_key_backward')
                        if (desc['column_to_table'][str(col1)] == desc['column_to_table'][str(col2)]):
                            set_relation('cc_table_match')

                elif j_type[0] == 'table':
                    col, table = i_type[1], j_type[1]
                    set_relation('ct_default')
                    if self.match_foreign_key(desc, col, table):
                        set_relation('ct_foreign_key')
                    col_table = desc['column_to_table'][str(col)]
                    if col_table == table:
                        if col in desc['primary_keys']:
                            set_relation('ct_primary_key')
                        else:
                            set_relation('ct_table_match')
                    elif col_table is None:
                        set_relation('ct_any_table')

            elif i_type[0] == 'table':
                if j_type[0] == 'question':
                    i_real = i - t_base
                    if f"{j},{i_real}" in sc_link["q_tab_match"]:
                        set_relation("tq" + sc_link["q_tab_match"][f"{j},{i_real}"])
                    else:
                        set_relation('tq_default')
                elif j_type[0] == 'column':
                    table, col = i_type[1], j_type[1]
                    set_relation('tc_default')

                    if self.match_foreign_key(desc, col, table):
                        set_relation('tc_foreign_key')
                    col_table = desc['column_to_table'][str(col)]
                    if col_table == table:
                        if col in desc['primary_keys']:
                            set_relation('tc_primary_key')
                        else:
                            set_relation('tc_table_match')
                    elif col_table is None:
                        set_relation('tc_any_table')
                elif j_type[0] == 'table':
                    table1, table2 = i_type[1], j_type[1]
                    if table1 == table2:
                        set_relation(('tt_dist', clamp(j - i, self.tt_max_dist)))
                    else:
                        set_relation('tt_default')
                        forward = table2 in desc['foreign_keys_tables'].get(str(table1), ())
                        backward = table1 in desc['foreign_keys_tables'].get(str(table2), ())
                        if forward and backward:
                            set_relation('tt_foreign_key_both')
                        elif forward:
                            set_relation('tt_foreign_key_forward')
                        elif backward:
                            set_relation('tt_foreign_key_backward')
        return relations

    @classmethod
    def match_foreign_key(cls, desc, col, table):
        foreign_key_for = desc['foreign_keys'].get(str(col))
        if foreign_key_for is None:
            return False

        foreign_table = desc['column_to_table'][str(foreign_key_for)]
        return desc['column_to_table'][str(col)] == foreign_table

    def validate_item(self, item, section):
        return True, None

    def preprocess_item(self, item, validation_info):
        question, question_for_copying = item.text,item.text
        preproc_schema = self._preprocess_schema(item.schema)
        assert preproc_schema.column_names[0][0].startswith("<type:")
        column_names_without_types = [col[1:] for col in preproc_schema.column_names]
        sc_link = self.compute_schema_linking(question, column_names_without_types, preproc_schema.table_names)

        cv_link = self.compute_cell_value_linking(question, item.schema)

        return {
            'raw_question': item.orig['question'],
            'question': question,
            'question_for_copying': question_for_copying,
            'db_id': item.schema.db_id,
            'sc_link': sc_link,
            'cv_link': cv_link,
            'columns': preproc_schema.column_names,
            'tables': preproc_schema.table_names,
            'table_bounds': preproc_schema.table_bounds,
            'column_to_table': preproc_schema.column_to_table,
            'table_to_columns': preproc_schema.table_to_columns,
            'foreign_keys': preproc_schema.foreign_keys,
            'foreign_keys_tables': preproc_schema.foreign_keys_tables,
            'primary_keys': preproc_schema.primary_keys,
        }

    def _preprocess_schema(self, schema):
        if schema.db_id in self.preprocessed_schemas:
            return self.preprocessed_schemas[schema.db_id]
        result = self.preprocess_schema_uncached(schema, self._tokenize,
                                            self.include_table_name_in_column, self.fix_issue_16_primary_keys)
        self.preprocessed_schemas[schema.db_id] = result
        return result

    def _tokenize(self, presplit, unsplit):

        return presplit

    def _tokenize_for_copying(self, presplit, unsplit):
        return presplit, presplit

    # schema linking, similar to IRNet
    @classmethod
    def compute_schema_linking(cls,question, column, table):
        def partial_match(x_list, y_list):
            x_str = " ".join(x_list)
            y_str = " ".join(y_list)
            if x_str in STOPWORDS or x_str in PUNKS:
                return False
            if re.match(rf"\b{re.escape(x_str)}\b", y_str):
                assert x_str in y_str
                return True
            else:
                return False

        def exact_match(x_list, y_list):
            x_str = " ".join(x_list)
            y_str = " ".join(y_list)
            if x_str == y_str:
                return True
            else:
                return False

        q_col_match = dict()
        q_tab_match = dict()

        col_id2list = dict()
        for col_id, col_item in enumerate(column):
            if col_id == 0:
                continue
            col_id2list[col_id] = col_item

        tab_id2list = dict()
        for tab_id, tab_item in enumerate(table):
            tab_id2list[tab_id] = tab_item

        # 5-gram
        n = 5
        while n > 0:
            for i in range(len(question) - n + 1):
                n_gram_list = question[i:i + n]
                n_gram = " ".join(n_gram_list)
                if len(n_gram.strip()) == 0:
                    continue
                # exact match case
                for col_id in col_id2list:
                    if exact_match(n_gram_list, col_id2list[col_id]):
                        for q_id in range(i, i + n):
                            q_col_match[f"{q_id},{col_id}"] = "CEM"
                for tab_id in tab_id2list:
                    if exact_match(n_gram_list, tab_id2list[tab_id]):
                        for q_id in range(i, i + n):
                            q_tab_match[f"{q_id},{tab_id}"] = "TEM"

                # partial match case
                for col_id in col_id2list:
                    if partial_match(n_gram_list, col_id2list[col_id]):
                        for q_id in range(i, i + n):
                            if f"{q_id},{col_id}" not in q_col_match:
                                q_col_match[f"{q_id},{col_id}"] = "CPM"
                for tab_id in tab_id2list:
                    if partial_match(n_gram_list, tab_id2list[tab_id]):
                        for q_id in range(i, i + n):
                            if f"{q_id},{tab_id}" not in q_tab_match:
                                q_tab_match[f"{q_id},{tab_id}"] = "TPM"
            n -= 1
        return {"q_col_match": q_col_match, "q_tab_match": q_tab_match}

    @classmethod
    def load_tables(cls,paths):
        schemas = {}
        eval_foreign_key_maps = {}

        for path in paths:
            schema_dicts = json.load(open(path))
            for schema_dict in schema_dicts:
                tables = tuple(
                    Table(
                        id=i,
                        name=name.split(),
                        unsplit_name=name,
                        orig_name=orig_name,
                    )
                    for i, (name, orig_name) in enumerate(zip(
                        schema_dict['table_names'], schema_dict['table_names_original']))
                )
                columns = tuple(
                    Column(
                        id=i,
                        table=tables[table_id] if table_id >= 0 else None,
                        name=col_name.split(),
                        unsplit_name=col_name,
                        orig_name=orig_col_name,
                        type=col_type,
                    )
                    for i, ((table_id, col_name), (_, orig_col_name), col_type) in enumerate(zip(
                        schema_dict['column_names'],
                        schema_dict['column_names_original'],
                        schema_dict['column_types']))
                )

                # Link columns to tables
                for column in columns:
                    if column.table:
                        column.table.columns.append(column)

                for column_id in schema_dict['primary_keys']:
                    # Register primary keys
                    column = columns[column_id]
                    column.table.primary_keys.append(column)

                foreign_key_graph = None
                for source_column_id, dest_column_id in schema_dict['foreign_keys']:
                    # Register foreign keys
                    source_column = columns[source_column_id]
                    dest_column = columns[dest_column_id]
                    source_column.foreign_key_for = dest_column

                db_id = schema_dict['db_id']
                assert db_id not in schemas
                schemas[db_id] = Schema(db_id, tables, columns, foreign_key_graph, schema_dict)
                eval_foreign_key_maps[db_id] = cls.build_foreign_key_map(schema_dict)

        return schemas, eval_foreign_key_maps

    @classmethod
    def build_foreign_key_map(cls,entry):
        cols_orig = entry["column_names_original"]
        tables_orig = entry["table_names_original"]

        # rebuild cols corresponding to idmap in Schema
        cols = []
        for col_orig in cols_orig:
            if col_orig[0] >= 0:
                t = tables_orig[col_orig[0]]
                c = col_orig[1]
                cols.append("__" + t.lower() + "." + c.lower() + "__")
            else:
                cols.append("__all__")

        def keyset_in_list(k1, k2, k_list):
            for k_set in k_list:
                if k1 in k_set or k2 in k_set:
                    return k_set
            new_k_set = set()
            k_list.append(new_k_set)
            return new_k_set

        foreign_key_list = []
        foreign_keys = entry["foreign_keys"]
        for fkey in foreign_keys:
            key1, key2 = fkey
            key_set = keyset_in_list(key1, key2, foreign_key_list)
            key_set.add(key1)
            key_set.add(key2)

        foreign_key_map = {}
        for key_set in foreign_key_list:
            sorted_list = sorted(list(key_set))
            midx = sorted_list[0]
            for idx in sorted_list:
                foreign_key_map[cols[idx]] = cols[midx]

        return foreign_key_map

    @classmethod
    def compute_cell_value_linking(cls, tokens, schema):
        def isnumber(word):
            try:
                float(word)
                return True
            except:
                return False

        def db_word_match(word, column, table, db_conn):
            cursor = db_conn.cursor()

            p_str = f"select {column} from {table} where {column} like '{word} %' or {column} like '% {word}' or " \
                    f"{column} like '% {word} %' or {column} like '{word}'"
            return False #TODO: fixme
            try:
                cursor.execute(p_str)
                p_res = cursor.fetchall()
                if len(p_res) == 0:
                    return False
                else:
                    return p_res
            except sqlite3.OperationalError as e:
                # print(p_str)
                return False

        num_date_match = {}
        cell_match = {}

        for q_id, word in enumerate(tokens):
            if len(word.strip()) == 0:
                continue
            if word in STOPWORDS or word in PUNKS:
                continue

            num_flag = isnumber(word)

            CELL_MATCH_FLAG = "CELLMATCH"

            for col_id, column in enumerate(schema.columns):
                if col_id == 0:
                    assert column.orig_name == "*"
                    continue

                # word is number 
                if num_flag:
                    if column.type in ["number", "time"]:  # TODO fine-grained date
                        num_date_match[f"{q_id},{col_id}"] = column.type.upper()
                else:
                    ret = db_word_match(word, column.orig_name, column.table.orig_name, schema.connection)
                    if ret:
                        # print(word, ret)
                        cell_match[f"{q_id},{col_id}"] = CELL_MATCH_FLAG

        cv_link = {"num_date_match": num_date_match, "cell_match": cell_match}
        return cv_link

    @classmethod
    def preprocess_schema_uncached(cls, schema,
                                tokenize_func,
                                include_table_name_in_column,
                                fix_issue_16_primary_keys):

        r = PreprocessedSchema()

        last_table_id = None
        for i, column in enumerate(schema.columns):
            col_toks = tokenize_func(
                column.name, column.unsplit_name)

            # assert column.type in ["text", "number", "time", "boolean", "others"]
            type_tok = f'<type: {column.type}>'

            column_name = [type_tok] + col_toks

            if include_table_name_in_column:
                if column.table is None:
                    table_name = ['<any-table>']
                else:
                    table_name = tokenize_func(
                        column.table.name, column.table.unsplit_name)
                column_name += ['<table-sep>'] + table_name
            r.column_names.append(column_name)

            table_id = None if column.table is None else column.table.id
            r.column_to_table[str(i)] = table_id
            if table_id is not None:
                columns = r.table_to_columns.setdefault(str(table_id), [])
                columns.append(i)
            if last_table_id != table_id:
                r.table_bounds.append(i)
                last_table_id = table_id

            if column.foreign_key_for is not None:
                r.foreign_keys[str(column.id)] = column.foreign_key_for.id
                r.foreign_keys_tables[str(column.table.id)].add(column.foreign_key_for.table.id)

        r.table_bounds.append(len(schema.columns))
        assert len(r.table_bounds) == len(schema.tables) + 1

        for i, table in enumerate(schema.tables):
            table_toks = tokenize_func(
                table.name, table.unsplit_name)
            r.table_names.append(table_toks)

        last_table = schema.tables[-1]

        r.foreign_keys_tables = to_dict_with_sorted_values(r.foreign_keys_tables)
        r.primary_keys = [
            column.id
            for table in schema.tables
            for column in table.primary_keys
        ] if fix_issue_16_primary_keys else [
            column.id
            for column in last_table.primary_keys
            for table in schema.tables
        ]

        return r