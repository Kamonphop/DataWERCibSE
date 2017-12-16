#!/bin/bash

for i in {1..10}
do
	echo "Running Fold $i"
	python3 train.py Fold_train$i
	python3 predict_cnn.py cnn_runs/Fold_train$i/checkpoints/ Fold_test$i
done