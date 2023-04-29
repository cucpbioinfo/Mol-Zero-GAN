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

parser.add_argument('-iteration', help='Iteration of Bayesian Optimzation',default = 5)
parser.add_argument('-sample_per_iteration', help='Number of Samples per Iteration',default = 3072)
parser.add_argument('-singular_size', help='Number of Singular Values ',default = 5)
parser.add_argument('-output_log', help='Number of Singular Values ',default = 'bayesian_result/output_qed.json')
parser.add_argument('-output', help='Number of Singular Values ',default = 'optimized_model/optimized_model.pt')



args = parser.parse_args()

iteration = args.iteration
sample_per_iteration = args.sample_per_iteration
singular_size = args.singular_size
output_log = args.output_log
output = args.output
singular_size = args.singular_size

vector = []
layers = []


for c in model.Generator.model:
    if "Linear" in str(type(c)):
        v, s, u = utils.svdNeural(c)
        vector += [float(e) for e in list(torch.diag(s.weight, 0)[0:singular_size])]
        layers.append(c)



print(utils.list_to_dict(vector))
vector_dict = utils.list_to_dict(vector)




def qedOp(**v):
    global singular_size

    num = sample_per_iteration

    v = {int(k): v[k] for k in v}
    vec = utils.dict_to_list(v)
    vec = torch.cuda.FloatTensor(vec)
    tmp = utils.replaceLayers(vec, layers, singular_size)
    check = False

    s = utils.latentGanSample(model, num)

    for i, layer in enumerate(layers):
        layer.weight = torch.nn.Parameter(tmp[i])

    score = utils.fitnessQED(s)

    return score


# %%
from utils import utils


# %%
# ignore warning
import warnings

warnings.filterwarnings("ignore")


print("Start Bayesian Optimization")
score = utils.bayesianNeural(
    vector,
    qedOp,
    singular_size,
    output_path=output_log,
    n_iter=iteration,
)
param = score["params"]
vec = utils.dict_to_list({int(k): param[k] for k in param})
vec = torch.cuda.FloatTensor(vec)
tmp = utils.replaceLayers(vec, layers, singular_size)
torch.save(model.state_dict(), output )

