# mkdir backup
import json
import os
import argparse
from pprint import pprint
import torch

# from sh import sed
# from sh import mkdir
import sh
import subprocess
import pathlib
import allennlp

# from dataset_readers.spider import SpiderDatasetReader
from dataset_readers.spider_ratsql import SpiderRatsqlDatasetReader
from models.semantic_parsing.spider_decoder import SpiderParser
from models.semantic_parsing.ratsql_encoder import RatsqlEncoder
from models.semantic_parsing.schema_encoder import SchemaEncoder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder

from allennlp.commands.train import train_model_from_file, train_model
from allennlp.common import Params
from allennlp.common.params import with_fallback
import namegenerator
import glob
import warnings

# warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore",category=DeprecationWarning)
inc_packages = [
    "models.semantic_parsing.ratsql_encoder",
    "models.semantic_parsing.spider_decoder",
    "models.semantic_parsing.schema_encoder",
    "dataset_readers.spider",
    "dataset_readers.spider_ratsql",
    # "predictors.sparc_predictor",
    # "predictors.sparc_predictor_full"
]
inc_str = " ".join([" ".join(["--include-package", x]) for x in inc_packages])
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", nargs="?")
    parser.add_argument("-p", "--partials", nargs="+", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gpu", type=str, default="0")

    parser.add_argument("--recover", action="store_true")
    parser.add_argument("-s", "--settings", default=())
    args = parser.parse_args()
    # TODO: makedir named experiments
    default_partials = []
    partials = args.partials
    partials.extend(default_partials)
    default_config_file = "train_configs/defaults.jsonnet"
    # default_config_file = 'train_configs/defaults.jsonnet'

    experiment_name_parts = []
    if args.name:
        experiment_name_parts.append(args.name)
    else:
        experiment_name_parts.append(namegenerator.gen())
    if partials:
        experiment_name_parts.append(
            "+".join([x for x in partials if x not in default_partials])
        )
    experiment_name = "_".join(experiment_name_parts)

    settings = Params.from_file(
        default_config_file,
        ext_vars={"gpu": args.gpu, "experiment_name": experiment_name},
    ).params

    _cuda_device = int(settings["trainer"]["cuda_device"])
    settings["trainer"]["cuda_device"] = _cuda_device

    for partial_config_name in partials:
        partial_config_path = glob.glob(
            f"train_configs/partials/**/{partial_config_name}.json", recursive=True
        )[0]
        partial_config = Params.from_file(partial_config_path)

        settings = with_fallback(partial_config.params, settings)

    new_config_path = os.path.join("train_configs", "+".join(partials) + ".json")
    json.dump(settings, open(new_config_path, "wt"), indent=2)

    allennlp_settings = " ".join(args.settings)
    force = "--force" if args.force else ""
    recover = "--recover" if args.recover else ""
    assert not pathlib.Path(f"experiments/{experiment_name}").exists()
    sh.mkdir(f"experiments/{experiment_name}")
    sh.rm("-rf", "last_experiment")
    sh.ln("-s", f"experiments/{experiment_name}", "last_experiment")
    #     s = """{{"trainer": {"cuda_device": %s  }}}""".format(_cuda_device)
    subprocess.check_call(
        f"git ls-files | tar Tzcf - backup/{experiment_name}.tgz", shell=True
    )  # TODO: fixme
    # torch.autograd.set_detect_anomaly(True)
    train_model_from_file(
        new_config_path,
        f"experiments/{experiment_name}",
        recover=args.recover,
        include_package=inc_packages,
        force=True,
    )
#     settings = allennlp.common.params.Params(settings)
#     train_model(settings, f'experiments/{experiment_name}', recover=args.recover, force=args.force)


#     cmd_predict = f"""allennlp predict experiments/{experiment_name} dataset/sparc/dev.json --predictor sparc --use-dataset-reader --cuda-device={_cuda_device} --output-file experiments/{experiment_name}/prediction.sql {inc_str} --weights-file experiments/{experiment_name}/best.th --silent""".split(" ")
#     subprocess.check_call(cmd_predict)
#     sed('-i',"1d",f"experiments/{experiment_name}/prediction.sql")
#     cmd_predict = f"""allennlp predict experiments/{experiment_name} dataset/sparc/dev.json --predictor sparc_full --use-dataset-reader --cuda-device={_cuda_device} --output-file experiments/{experiment_name}/full_prediction.json {inc_str} --weights-file experiments/{experiment_name}/best.th --silent""".split(" ")
#     subprocess.check_call(cmd_predict)
#     cmd_predict= f"""python evaluation_sqa.py --gold dataset/sparc/dev_gold.txt --pred experiments/{experiment_name}/prediction.sql --table dataset/sparc/tables.json --etype match --db dataset/sparc/database --questions dataset/sparc/questions.txt""".split(" ")
#     with open(f"experiments/{experiment_name}/eval.txt","wb") as f:
#         subprocess.check_call(cmd_predict,stdout=f)
# #     subprocess.check_call(f"""sed -i '1d' experiments/{experiment_name}/prediction.sql""".split(' '))
