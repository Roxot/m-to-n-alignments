# m-to-m-alignments

Disclaimer: This code is under development and is not a finished product. As such, some functionality might not work or contain bugs.

### Installation
Needs at least Python 3.6 due to some features used, known to work with Python 3.7. To install dependencies run:

```
pip install -r requirements.txt
```

An additional dependency not installed through the requirements file is [probabll/dists.pt](https://github.com/probabll/dists.pt). This can be installed as follows:

```
git clone https://github.com/probabll/dists.pt.git
cd dists.pt/
pip install -r requirements.txt
pip install .
```

### Generate Toy Data
For debugging purposes we can use a randomly generated toy dataset that consists of made up words constructed from one or more fixed word pieces. The goal of the aligner is to align each target merged word consisting of potentially multiple word pieces, to each word piece it belongs to in the source split data. This (aligning to multiple source words) is an impossible task for the Neural IBM 1 model, whereas the Bernoulli bit-vector model should be able to do this. The toy dataset can be generated as:  
```
./generate_toy_data.sh
```

### Train a Neural IBM 1 model
Using default parameters one can train a Neural IBM 1 model on the toy dataset as:
```
mkdir experiments/
python -m alignments.train --training_prefix toy-data/train \
                           --validation_prefix toy-data/dev \
                           --src split \
                           --tgt merged \
                           --model_type neuralibm1 \
                           --use_gpu True \
                           --output_dir experiments/neuralibm1
```

### Train an m-to-m alignment model with bit vectors and REINFORCE
In order to train the Bernoulli bit-vector model using REINFORCE on the toy data one can run:
```
mkdir experiments/
python -m alignments.train --training_prefix toy-data/train \
                           --validation_prefix toy-data/dev \
                           --src split \
                           --tgt merged \
                           --model_type bernoulli-RF \
                           --prior_param_1 1.0 \
                           --use_gpu True \
                           --output_dir experiments/bernoulli-RF
```
Seting either `prior_param_1` or `prior_param_2` for the Bernoulli REINFORCE model .


If `prior_param_1 > 0`: On average align to `prior_param_1` source words.


If `prior_param_2 > 0`: Fixed alignment probability for all source words of `0 < prior_param_2 < 1`

This one needs a bit longer to converge due to variance of the REINFORCE estimator (and by default we only use a moving average baseline).

### Short overview of the code
* [alignments/train.py](alignments/train.py) is a general purpose training file used for all alignment models.
* The [alignments/neuralibm1_helper.py](alignments/neuralibm1_helper.py) and [alignments/alignmentvae_helper.py](alignments/alignmentvae_helper.py) files implement the model-specific creation, training and validation steps for each model.
* The [alignments/models](alignments/models) folder contains implementations for each specific model, the bit-vector model is contained in `alignmentvae.py`, with several ways to modle the bit vector implemented.
* All hyperparameters are constructed in [alignments/hparams/hparams.py](alignments/hparams/hparams.py)
* Re-usable architecture components are contained in [alignments/components](alignments/components)
* All data-loading is implemented in [alignments/data](alignments/data)
