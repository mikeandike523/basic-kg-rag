#!/bin/bash

export BASHRC_NONINT_OK=1

source $HOME/.bashrc

conda_base

conda activate basic-kg-rag-env

python build_conceptnet_graph.py --edges