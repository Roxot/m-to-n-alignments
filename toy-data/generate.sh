#!/bin/bash
set -ex

# Download the data.
wget -O data.tgz https://surfdrive.surf.nl/files/index.php/s/kTOqAxYrvyA1Py0/download
tar -xzvf data.tgz
rm data.tgz

# Pre-process the data into the right format.
cd data/
rm training.de dev.de README.md bpe_codes.en bpe_codes.de comparable.en comparable.de
mv training.en training_bpe.en
mv dev.en dev_bpe.en
sed -E 's/(@@ )|(@@ ?$)//g' < training_bpe.en > training.en
sed -E 's/(@@ )|(@@ ?$)//g' < dev_bpe.en > dev.en
rm training_bpe.en dev_bpe.en

# Generate the synthetic data.
python ../create_toy_data.py training.en training.ensp
python ../create_toy_data.py dev.en dev.ensp dev.wa.nonullalign
