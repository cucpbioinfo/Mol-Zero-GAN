# Mol-Zero-GAN: Zero-Shot Adaptation of Molecular Generative Adversarial Network for Specific Protein Targets
Mol-Zero-GAN is a framework that aims to optimized pretrained generative model based on Bayesian optimization (BO) to find the model optimal weights' singular values, factorized by singular value decomposition, and can generate drug candidates with desired properties with no additional data. The proposed framework can produce drugs with the desired properties on protein targets of interest by optimizing the model's weights


## Installation

[Install miniconda](https://docs.conda.io/en/latest/miniconda.html) and run the following command.

```conda env create --file env.yml```

```conda install -c bioconda mgltool```

## Optimizing Based Model

To optimize based model you can run following commands of each objective function

### QED optimization

```python3 qed_optmization.py -iteration 50 -sample_per_iteration 3072 -singular_size 5```

### BA optimization

```python3 ba_optmization.py -iteration 50 -sample_per_iteration 3072 -singular_size 5 -target TTBK1```

### Weighted Sum optimization

```python3 weighted_sum_optmization.py -iteration 50 -sample_per_iteration 3072 -singular_size 5 -target TTBK1 -ba_optmization_model bayesian_result/TTBK1.json```

## Generating Compounds

To generate compounds by optimized model you can run following command

```python3 gen_smiles.py -model_pram bayesian_result/TTBK1.json -sample 1000000 -singular_size 5```
