from flask import Flask, render_template, request
app = Flask(__name__)

from allennlp.models.archival import Archive
import torch
import json
from allennlp.common import Params
from allennlp.models.model import Model
from allennlp.common.params import with_fallback
from dataset_readers.spider import SpiderDatasetReader

from dataset_readers.spider_rerank import SpiderRerankDatasetReader
from predictors.spider_predictor import WikiTablesParserPredictor
from predictors.spider_predictor_complete import SpiderParserPredictor

from models.semantic_parsing.spider_parser import SpiderParser
from models.semantic_parsing.spider_reranker import SpiderReranker

from allennlp.predictors.predictor import Predictor
from glob import glob
import base64
import random

reader = SpiderDatasetReader(tables_file="dataset/tables.json")
predictor = Predictor.from_path("experiments/experiment", "spider_predict_complete")

def predict(utterance,db_id="department_management"):
    q = reader.text_to_instance(utterance=utterance,db_id=db_id)
    return predictor.predict_instance(q)['predicted_sql_query']

# reader = SpiderDatasetReader(tables_file="dataset/tables.json",loading_limit=-1)
# predictor = Predictor.from_path("experiments/experiment", "spider")

# def predict(utterance,db_id="department_management"):
#     q = reader.text_to_instance(utterance=utterance,db_id=db_id)
#     return predictor.predict_instance(q)['predicted_sql_query']


@app.route("/schema/<img_path>", methods=['POST','GET'])
def nr(img_path):
    with open(f"schema//{img_path}.png", "rb") as image_file:
        return base64.b64encode(image_file.read())

@app.route('/',methods = ['POST', 'GET'])
def query():
    if request.method == 'POST':
        result = request.form
        print(result)
    img=''
    db_id_list = glob("schema/*.png")
#     random.shuffle(db_id_list)
    db_id_list = sorted(db_id_list)[::-1]
    db_id_list = [x.split("/")[-1][:-4] for x in db_id_list]
    db_id_list = {x:x for x  in db_id_list}
    return render_template('query.html',db_id_list=db_id_list,img=img)

@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        print(result)
#         result['Prediction'] = predict(result['Question'])
        pred = predict(result['Question'],db_id=result['db_id'])
        result = {'Question':result['Question'],"Prediction":pred,"db_id":result['db_id']}
        
        return render_template("result.html",result = result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8786,debug=True)