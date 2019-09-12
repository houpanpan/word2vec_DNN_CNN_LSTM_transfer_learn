#!/bin/ksh
echo "Program started :$1"
python text_cnn_joint_train.py $1
python text_cnn_embed_itemdesc.py $1
echo "Program Ended"
