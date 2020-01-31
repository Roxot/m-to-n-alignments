#!/bin/bash

TRAIN_SIZE=50000
DEV_SIZE=100
OUTPUT_FOLDER="toy-data"

mkdir $OUTPUT_FOLDER
python scripts/generate_toy_data.py $TRAIN_SIZE $OUTPUT_FOLDER/train
python scripts/generate_toy_data.py $DEV_SIZE $OUTPUT_FOLDER/dev
python scripts/create_naacl_for_bpe.py $OUTPUT_FOLDER/dev.split $OUTPUT_FOLDER/dev.wa.nonullalign
