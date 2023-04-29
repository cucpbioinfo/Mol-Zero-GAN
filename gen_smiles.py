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




import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-model', help='Iteration of Bayesian Optimzation',default = 'optimized_model/optimized_model.pt')
parser.add_argument('-sample', help='Iteration of Bayesian Optimzation',default = 3072*256)
parser.add_argument('-output', help='Number of Singular Values ',default = 'generated_result/smiles.txt')

args = parser.parse_args()
loaded_model = args.model
sample = args.sample
output = args.output



MODELS = ModelsStorage()
model_config = torch.load("pretrained/latentgan_config.pt")
model_vocab = torch.load("pretrained/latentgan_vocab.pt")
model_state = torch.load(loaded_model)


model = MODELS.get_model_class("latentgan")(model_vocab, model_config)
model.load_state_dict(model_state)
model = model.cuda()







def samp():


    num = sample
    batch_size = 256

  

    for i in range(num // batch_size):

        print(i, "/", num // batch_size)
       
        s = utils.latentGanSample(model, batch_size)

        # write smiles to files
        with open(output, "a") as f:
            for j in range(len(s)):
                f.write(
                    s[j]
                )

  
    return


samp()
