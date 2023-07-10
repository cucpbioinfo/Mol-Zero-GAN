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
import os
import pandas as pd





import moses
import torch
import pickle
from moses.models_storage import ModelsStorage

# from IPython.display import display, Markdown, HTML, clear_output
from utils import utils

MODELS = ModelsStorage()
model_config = torch.load("pretrained/latentgan_config.pt")
model_vocab = torch.load("pretrained/latentgan_vocab.pt")
model_state = torch.load("pretrained/latentgan_model.pt")


model = MODELS.get_model_class("latentgan")(model_vocab, model_config)
model.load_state_dict(model_state)
model = model.cuda()




import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-model_param', help='Iteration of Bayesian Optimzation',default = 'bayesian_result/output.json')
parser.add_argument('-sample', help='Iteration of Bayesian Optimzation',default = 3072)
parser.add_argument('-singular_size', help='Number of Singular Values of Optimized Model ',default = 5)
parser.add_argument('-output', help='Number of Singular Values ',default = 'generated_result/smilesx.txt')

args = parser.parse_args()
model_param = args.model_param
sample = int(args.sample)
singular_size = int(args.singular_size)
output = args.output


df = pd.read_json(model_param, lines=True)


args = parser.parse_args()

singular_size = 5
vector = []

layers = []


for c in model.Generator.model:

    if "Linear" in str(type(c)):
        v, s, u = utils.svdNeural(c)
        vector += [float(e) for e in list(torch.diag(s.weight, 0)[0:singular_size])]
        layers.append(c)







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
    return max_param


max_param = get_highest_score(df)



from rdkit import Chem
from moses.metrics.SA_Score import sascorer
def calculateSA(smi):
    try:
        return sascorer.calculateScore(smi)
    except Exception as e:
        print(e)
        return 10



def samp(**v):

    # utils.clear_tmp()

    global singular_size
    global sample

    num =  sample
    batch_size = 256

    v = {int(k): v[k] for k in v}
    vec = utils.dict_to_list(v)
    vec = torch.cuda.FloatTensor(vec)
    tmp = utils.replaceLayers(vec, layers, singular_size)
    result = []
    
    for i in range(num // batch_size):

        print(i, "/", num // batch_size)
       
        s = utils.latentGanSample(model, batch_size)
        result += s.copy()
        # write smiles to files
        with open(output.format(i), "a") as f:
            for j in range(len(s)):
                f.write(
                    s[j] + "\n"
                )

    return


samp(**max_param)
