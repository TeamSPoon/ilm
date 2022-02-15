#!/bin/bash
set -x
tensorboard --logdir=runs &

rm pickle_files/train.pkl
ln pickle_files/words.pkl pickle_files/train.pkl
python train_ilm.py --train_num_epochs 2 $*

rm pickle_files/train.pkl
ln pickle_files/ngrams.pkl pickle_files/train.pkl
python train_ilm.py --train_num_epochs 2 $*

rm pickle_files/train.pkl
ln pickle_files/sentences.pkl pickle_files/train.pkl
python train_ilm.py --train_num_epochs 2 $*

rm pickle_files/train.pkl
ln pickle_files/paragraphs.pkl pickle_files/train.pkl
python train_ilm.py --train_num_epochs 2 $*

rm pickle_files/train.pkl
ln pickle_files/documents.pkl pickle_files/train.pkl
python train_ilm.py --train_num_epochs 2 $*
