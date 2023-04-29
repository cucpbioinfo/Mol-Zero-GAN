from bayes_opt import BayesianOptimization
from utils import utils
from bayes_opt.util import load_logs
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import torch
import numpy as np
import tensorflow as tf
import random
from bayes_opt import SequentialDomainReductionTransformer
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")


import moses
import torch
import pickle
from moses.models_storage import ModelsStorage

from utils import utils

MODELS = ModelsStorage()
model_config = torch.load("pretrained/latentgan_config.pt")
model_vocab = torch.load("pretrained/latentgan_vocab.pt")
model_state = torch.load("pretrained/latentgan_model.pt")
model = MODELS.get_model_class("latentgan")(model_vocab, model_config)
model.load_state_dict(model_state)
model = model.cuda()


vector = []

layers = []


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-target', help='Specify the target',default = "4BTK")
parser.add_argument('-iteration', help='Iteration of Bayesian Optimzation',default = 50)
parser.add_argument('-sample_per_iteration', help='Number of Samples per Iteration',default = 3072)
parser.add_argument('-singular_size', help='Number of Singular Values ',default = 5)
parser.add_argument('-output_log', help='Number of Singular Values ',default = 'bayesian_result/output_weighted_sum.json')
parser.add_argument('-output', help='Number of Singular Values ',default = 'optimized_model/optimized_model2.pt')
parser.add_argument('-ba_optimization_log', help='Path of ba optimization parameters result',default = 'bayesian_result/ba_optimization.json')
parser.add_argument('-alpha',help='alpha value',default=0.2)
parser.add_argument('-beta',help='beta value',default=1)




args = parser.parse_args()

target = args.target
iteration = args.iteration
sample_per_iteration = args.sample_per_iteration
singular_size = args.singular_size
output_log = args.output_log
output = args.output
alpha = args.alpha
beta = args.beta
ba_optimization_log = args.ba_optimization_log
print(target,iteration,sample_per_iteration,singular_size,output)


# add singular value to be parameters for optmization process
for c in model.Generator.model:

    if "Linear" in str(type(c)):
        v, s, u = utils.svdNeural(c)
        vector += [float(e) for e in list(torch.diag(s.weight, 0)[0:singular_size])]
        layers.append(c)



print(utils.list_to_dict(vector))
vector_dict = utils.list_to_dict(vector)

df = pd.read_json(ba_optimization_log, lines=True)

# convert df to list
def df_to_list(df):
    l = []
    for i in range(len(df)):
        l.append(df.iloc[i])
    return l


df = df_to_list(df)

# get highest score param of df
def get_highest_score(df):
    max_score = 0
    max_param = {}
    for i in range(len(df)):
        if df[i]["target"] > max_score:
            max_score = df[i]["target"]
            max_param = df[i]["params"]
    return max_score, max_param


max_score, _ = get_highest_score(df)
first_score = df[0]["target"]


#calculation for scaling BA. alue
print("max_score", max_score)
print("first_score", first_score)


# %%
def black_box_function(**v):

    utils.clear_tmp()

    global singular_size
    global max_score
    global first_score

    num = sample_per_iteration
    batch_size = 256

    v = {int(k): v[k] for k in v}
    vec = utils.dict_to_list(v)
    vec = torch.cuda.FloatTensor(vec)
    tmp = utils.replaceLayers(vec, layers, singular_size)
    check = False
    result = []
    qed = []

    smiles = []

    for i in range(num // batch_size):

        print(i, "/", num // batch_size)

        s = utils.latentGanSample(model, batch_size)
        smiles += s.copy()
        print(len(list(set(smiles))))

        # calculate qed of the sample
        qed += [utils.calculateQED(e) for e in s]
        utils.convertSmilesToLigand(s, "")

        start = 0
        stop = 100

        while True:
            print(start)
            result += utils.runVina(start=start, stop=stop)
            print(
                sum(
                    [
                        float(e) if e != "" else 0
                        for e in [e.decode("utf-8") for e in result]
                    ]
                )
                / len(result),
                len(result),
            )
            if stop == batch_size:
                break

            start += 100
            stop += 100
            stop = min(stop, batch_size)

        # ignore empty string


    for i, layer in enumerate(layers):
        layer.weight = torch.nn.Parameter(tmp[i])

    result = [e.decode("utf-8") for e in result]
    result = [float(e) if e != "" else 0 for e in result]

    print("result", len(result))

    score = sum(result) / num
    qed_score = sum(qed) / num

    ba_score = -score
    ba_score = (ba_score - first_score) / (max_score - first_score)

 
    utils.clear_tmp()


    return alpha * ba_score + beta * qed_score



print("Start Bayesian Optimization")
score = utils.bayesianNeural(
    vector,
    black_box_function,
    singular_size,
    output_path=output_log,
    n_iter=iteration,
)
param = score["params"]
vec = utils.dict_to_list({int(k): param[k] for k in param})
vec = torch.cuda.FloatTensor(vec)
tmp = utils.replaceLayers(vec, layers, singular_size)
torch.save(model.state_dict(), output )
