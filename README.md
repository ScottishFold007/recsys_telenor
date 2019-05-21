# Cluster analysis and RNN based next-interaction prediction in a web session environment
This repository contains code used for experiments in my master's project.   
The experiments in this project were conducted on a Dell XPS 8700 computer with the Ubuntu 18.04 operating system.   
The computer has a Nvidia GeForce GTX 750 Ti graphics card.
 
## Requirements
Python 3  
Pytorch with support for CUDA  
numpy  
pandas   
matplotlib  
jupyter  
hdbscan  
gensim  
scikit-learn 

## Running the code
First, open a terminal window and navigate to the file you want to run.   
In order to run the python scripts, type: `python [filename]`  
In order to run a jupyter notebook, type: `jupyter notebook`. This opens up the jupyter notebook interface in the default web browser. Then, select the notebook you want to run.

## Preprocessing
Prepreprocessing of the dataset have been carried out in several steps.  
  
read_data_set.py: Reads the data from file into a pandas dataframe and removes empty rows.  
preprocessing/preprocess_v1.py: Creating general descriptive action labels.  
preprocessing/session.py: Defines a unique ID for each session and splits long sessions.  
data_analysis/consecutive_actions_per_session.ipynb: Removes consecutive repeating actions from the sessions an removes sessions with only one action.

## Experiments
data_analysis/lda/actions_lda.ipynb: LDA model for creating a topic model of the interactions in sessions.  
data_analysis/action2vec.py: Word2vec model used for creating embedded representations of interactions in sessions.  
data_analysis/HDBSCAN.ipynb: HDBSCAN model used for clustering embedded representations produced by word2vec of interactions i sessions.  
  
prediction/model.py: RNN model for next-interaction prediction.  
prediction/main.py: Script for training the RNN model.  
prediction/testing.py: Script for testing the RNN model.  
prediction/average_precision_for_the_nth_interaction.ipynb: Measuring the average precision@K for K in [1,2,3] for each of the interactions at nth position in the sessions.  
