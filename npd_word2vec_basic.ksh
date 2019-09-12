#!/bin/ksh
rm -rf /abinitio/dev/data/ml/embed/embeddings.ckpt

python /home/aidev/dev/npd_batch/machine_learning/npd_word2vec_basic.py
exit 0
