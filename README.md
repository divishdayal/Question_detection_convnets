# Question_detection_convnets

A Convolutional Neural Network model to identify questions from other sentences. It reaches perfect accuracy in just 1 epoch, 
maximum 2 epochs as the task is fairly simple.

## Dataset
5k sentences are selected each from the quora duplicate questions dataset(taking only one of the pair of questions in a single
data point of the dataset) and AG News Corpus.
AG news corpus was selected as the topics are varied and each data point sentence length is fairly comparable to the quora
dataset for questions.

## Run
Extract the data.zip file in the same directory as the python script.
Run using
> python model_sum.py

## Dependencies
The model is coded in Tensorflow library.
Other python libraries used are nltk, yaml, scikit-learn.

## Hyper-parameters
The model hyper-parameters can be changed inside the config.yml file. This file is read by the python script for the model's hyperparameters.
