# CS5228 Team 26 Rental Mining Final Project

This repository contains the code for data mining methods and models to predict HDB flat rental prices in Singapore.

### Given data sets

The files "train.csv", "test.csv" and the folder "auxiliary-data" were provided to us as the raw datasets to be utilized.

### Auxiliary Data and Hyperparameter Tuning.ipynb

This notebook contains the code for selecting the parameters to use in auxiliary data incorporation, and XGBoost Regressor model hyperparamter tuning.

### Neural Network Model Folder

This folder contains 5 files: "neural_net_model.ipynb" and "neural_net_model.py" contain the same code which includes all data preprocessing and addition of auxillary data incorporation needed specially for the neural network model. For example, we had to scale the values in the preprocessing here. The ipynv file has the cell outputs which we got and .py file is provided if some other notebook wanted to directly call or execute the same for comparision. The "pretrained_weights.pth" is a file containing the pretrained weights of our regression model and the "final_weights.pth" is the weights of our final model. The "submission_neural_net.csv" file is the submission file we have submitted on kaggle to get the test loss.
