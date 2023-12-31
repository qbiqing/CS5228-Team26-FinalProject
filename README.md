# CS5228 Team 26 Rental Mining Final Project

This repository contains the code for data mining methods and models to predict HDB flat rental prices in Singapore.

### Given data sets

The files "train.csv", "test.csv" and the folder "auxiliary-data" were provided to us as the raw datasets to be utilized.

### Auxiliary Data and Hyperparameter Tuning.ipynb

This notebook contains the code for selecting the parameters to use in auxiliary data incorporation, and XGBoost Regressor model hyperparamter tuning.

### Encoding.ipynb
This notebook compares the different types of encoding for the categorical columns

### Figures.ipynb
This notebook consists of EDA plots and other graphs/figures.

### Figures-2.ipynb
This notebook consists of additional EDA plots and other graphs/figures used in the report.

### Kaggle Submission Code.ipynb
This notebook consists of code to generate the csv to be uploaded to the kaggle competition.

### Model Comparison.ipynb
This notebook compares three different tree ensemble methods

### Neural Network Model Folder

This folder contains 5 files: "neural_net_model.ipynb" and "neural_net_model.py" contain the same code which includes all data preprocessing and addition of auxillary data incorporation needed specially for the neural network model. For example, we had to scale the values in the preprocessing here. The ipynv file has the cell outputs which we got and .py file is provided if some other notebook wanted to directly call or execute the same for comparision. The "pretrained_weights.pth" is a file containing the pretrained weights of our regression model and the "final_weights.pth" is the weights of our final model. The "submission_neural_net.csv" file is the submission file we have submitted on kaggle to get the test loss.

### Stock_nb.ipynb
This notebook contains all the steps in the way of analysing and incorporating the stock data, along with correlation heatmaps.

### xgboost_utils.py
This file contains all the methods that transform the data using finalized methods for data cleaning, preprocessing, encoding, and auxiliary data addition. The resultant data can be passed to the models for training and generating the final prediction.
