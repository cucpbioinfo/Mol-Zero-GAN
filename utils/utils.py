import torch
import numpy as np
import rdkit
from rdkit.Chem.QED import qed

# import moses
import torch

# from moses.models_storage import ModelsStorage
import pickle
import random

# from IPython.display import display, Markdown, HTML, clear_output
import copy
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
import os

import tempfile

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


from bayes_opt import BayesianOptimization

from bayes_opt.util import load_logs
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import torch
import numpy as np
import tensorflow as tf
import random
from bayes_opt import SequentialDomainReductionTransformer
import argparse
import concurrent.futures

import subprocess
import multiprocessing
from multiprocessing import Pool
import time


class LayerPointer:
    def __init__(self, layer_weight, layer_bias):
        self.weight = layer_weight
        self.bias = layer_bias



def svdNeural(c):

    if type(c) == torch.nn.modules.sparse.Embedding:
        return svdEmbedding(c)

    u, s, v = torch.svd(c.weight)
    v_nn = torch.nn.Linear(torch.t(v).shape[1], torch.t(v).shape[0], bias=False)
    s_nn = torch.nn.Linear(torch.diag(s).shape[1], torch.diag(s).shape[0], bias=False)
    u_nn = torch.nn.Linear(u.shape[1], u.shape[0])
    v_nn.weight = torch.nn.Parameter(torch.t(v))
    s_nn.weight = torch.nn.Parameter(torch.diag(s))
    u_nn.weight = torch.nn.Parameter(u)
    u_nn.bias = torch.nn.Parameter(c.bias)
    return v_nn, s_nn, u_nn



def calculateQED(smiles):
    try:

        m = rdkit.Chem.MolFromSmiles(smiles)
        # print(smiles,m is None,'gggggg')
        if m is None:
            return 0
        return qed(m)
    except:
        # print(smiles,'invalid','gggggg')
        # print(e)
        return 0


def fitnessQED(smiles):
    try:
        result = [calculateQED(e) for e in smiles]
        if len(result) == 0:
            return 0
        return np.mean(result)
    except:
        return 0

def replaceLayers(vv, layers, singular_size):

    i = 0
    print(len(vv))
    layers_copy = [layer.weight for layer in layers]
    for layer in layers:
        print('x')
        if type(layer) == torch.nn.modules.sparse.Embedding:
            v, s, u = svdEmbedding(layer)
            new_diag = torch.zeros(len(torch.diag(s, 0))).cuda()

            new_diag[:singular_size] = vv[i : i + singular_size].cuda()
            new_diag[singular_size:] = torch.diag(s, 0)[singular_size:].cuda()
            s = torch.diag_embed(new_diag)

            layer.weight = torch.nn.Parameter(torch.mm(u, torch.mm(s, v)))

            # new_diag = torch.diag(torch.diag(s,0)[:singular_size]).cuda()
            # new_diag[0:sigular_size] = vv[i:i+sigular_size].cuda()
            # new_diag[sigular_size:] =

        else:
            v, s, u = svdNeural(layer)
            new_diag = torch.zeros(len(torch.diag(s.weight, 0))).cuda()

            new_diag[0:singular_size] = vv[i : i + singular_size].cuda()
            new_diag[singular_size:] = torch.diag(s.weight, 0)[singular_size:].cuda()
            s.weight = torch.nn.Parameter(torch.diag_embed(new_diag))

            layer.weight = torch.nn.Parameter(
                torch.matmul(u.weight, torch.matmul(s.weight, v.weight))
            )
        i += singular_size

    return layers_copy







def list_to_dict(l):
    d = {}
    for i in range(len(l)):
        d[i] = l[i]
    return d


def dict_to_list(d):
    l = []
    for i in range(len(d)):
        l.append(d[i])
    return l


def bayesianNeural(
    vector,
    black_box_function,
    singular_size,
    rd_state=1234,
    mul=2,
    output_path="",
    n_iter=10,
):
    vector_dict = list_to_dict(vector)
    pbounds = {str(idx): (0, vector_dict[idx] * mul) for idx in range(len(vector_dict))}
    bounds_transformer = SequentialDomainReductionTransformer()
    print(vector_dict)
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=rd_state,
        verbose=1,
        bounds_transformer=bounds_transformer,
    )
    if output_path:
        print("load log", output_path)

        logger = JSONLogger(path=output_path)

        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    print("a")
    optimizer.probe(
        params={str(idx): vector_dict[idx] for idx in range(len(vector_dict))},
        lazy=True,
    )
    print("b")
    print("c")
    optimizer.maximize(init_points=0, n_iter=n_iter, acq="ucb")
    return optimizer.max






def convertSmilesToLigand(smiles, output_folder="ligands"):
    mols = []
    for smi in smiles:
        # print(smi)
        try:
            mol = Chem.MolFromSmiles(smi)
            mols.append(mol)
        except:
            pass

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(len(mols)):
            try:
                mol = mols[i]
                smi = smiles[i]
                mol = Chem.AddHs(mol)

                future = executor.submit(optimize_molecule, i, mol)
                futures.append(future)

            except:
                pass

        for future in concurrent.futures.as_completed(futures):
            i, pdb = future.result()

            if pdb is not None:
                with open(
                    os.path.join(output_folder, "smile_" + str(i) + ".pdb"), "w"
                ) as f:
                    f.write(pdb)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for file in os.listdir():
            if file.endswith(".pdb"):
                input_file = os.path.join(output_folder, file)
                output_file = os.path.join(
                    output_folder, file.replace(".pdb", ".pdbqt")
                )
                future = executor.submit(convert_to_pdbqt, i, input_file, output_file)
                futures.append(future)

    return


def optimize_molecule(i, mol):
    try:
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        pdb = Chem.MolToPDBBlock(mol)
        return i, pdb
    except:
        return i, None


def convert_to_pdbqt(i, input_file, output_file):

    os.system(
        "timeout 30 prepare_ligand4.py -l " + input_file + " -o " + output_file,
    )
    # if
    os.system("rm " + input_file)

    return i, input_file, output_file



def latentGanSample(model, sample=256):
    check = False

    s = []
    i = 0
    while len(s) <= sample:

        i += 1
        try:
            new_mol = model.sample(256)

            # add valid new mol to s check by rdkit
            for mol in new_mol:
                if mol is not None:
                    try:
                        molx = Chem.MolFromSmiles(mol)
                        if molx is not None:
                            s.append(mol)
                    except:
                        pass

        except:
            pass
    s = s[:sample]

    return s


cmd = "./scripts/autodock_vina_1_1_2_linux_x86/bin/vina --config conf.txt |  grep '(kcal/mol)' -A 3 | head -3 | tail -1| awk -e '{print $2}' "


# run cmd parallely
def runCmd(cmd):
    result = subprocess.check_output(cmd, shell=True)
    return result

def parallelRun(cmds):
    pool = Pool(processes=multiprocessing.cpu_count())
    results = pool.map(runCmd, cmds)
    pool.close()
    pool.join()
    return results


# run vina
def runVina(stop, start=0,target_conf = 'conf'):
    cmds = []
    for i in range(start, stop):
        cmd = (
            "vina --config conf/" + target_conf +".txt --ligand smile_"
            + str(i)
            + ".pdbqt |  grep '(kcal/mol)' -A 3 | head -3 | tail -1| awk -e '{print $2}'"
        )
        cmds.append(cmd)
    results = parallelRun(cmds)

    return results

def clear_tmp():
    try:
        os.system("rm *.pdbqt")
    except:
        pass

    try:
        os.system("rm *.pdb")
    except:
        pass
