#! /bin/bash

echo "Current path is $PATH"
echo "Running"
export PATH=$PATH:/users/students/r0691656/r0691656/exp/master_thesis/condor

#Training
cd /users/students/r0691656/r0691656/exp/master_thesis; /users/students/r0691656/.cache/pypoetry/virtualenvs/audio-classification-thesis-Y9KjJEmP-py3.7/bin/python src/train_cross_val.py\
 				                --batch_size 32\
                                --epochs 10\
                                --learning_rate 0.0001\
                                --hop_length 512\
                                --dft_window_size 1024\
                                --dataset_name "data_50"\
                                --log_mel
