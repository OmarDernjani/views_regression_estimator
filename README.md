# Views Regression Estimator
## Overview

This repository contains code for the development and evaluation of machine learning models aimed at estimating the number of views received by YouTube Shorts. The primary objective of the project is to investigate whether statistical and neural models can meaningfully predict view counts based on the features available in the dataset.

The project makes use of a Multi-Layer Perceptron (MLP) implemented in PyTorch Lightning, along with preprocessing steps such as one-hot encoding for categorical features and standardized data processing pipelines.

## Project Structure

The repository is organized as follows:
.
├── notebooks/               # Jupyter notebooks for analysis and experiments
├── src/                     # Source code for models and training logic
├── data/                    # Raw and processed datasets (not tracked in Git)
├── lightning_logs/          # PyTorch Lightning logs
├── README.md                # Project documentation
├── requirements.txt         # Project dependencies
└── .gitignore               # Files excluded from version control

## Problem Definition

The task addressed in this project is the regression of a continuous variable: the number of views obtained by YouTube Shorts videos.

Multiple problem formulations were explored, including:

- direct regression on raw view counts,

- regression in logarithmic scale to reduce target variance,

- discretization of views into buckets for classification experiments.

These experiments were carried out to assess both the predictive capacity of the models and the limitations imposed by the dataset structure.

## Data Preparation

Data preprocessing follows the steps below:

Importing the dataset from a CSV file containing short-form video performance data.

Separating the target variable (views) from the input features.

Applying one-hot encoding to categorical variables.

Splitting the dataset into training and validation sets using train_test_split.

Optionally converting the target into discrete buckets using pandas.qcut for classification experiments.

Converting feature and target arrays into PyTorch tensors and wrapping them into Dataset and DataLoader objects.

## Model Architecture

The main predictive model is a Multi-Layer Perceptron (MLP) implemented in PyTorch Lightning. The model architecture consists of:

fully-connected linear layers (nn.Linear),

ReLU activation functions,

dropout layers for regularization,

CrossEntropyLoss for classification experiments,

regression loss functions for continuous prediction in log-space.

Different network sizes were tested to compare model capacity and generalization behavior.

## Training and Evaluation

Model training is carried out using PyTorch Lightning via the Trainer API, with:

multiple training epochs,

validation monitoring,

optional early stopping based on validation loss,

logging of training and validation metrics.

Evaluation metrics vary depending on the formulation of the problem, and include:

MAE and RMSE for regression tasks,

accuracy and confusion matrix for classification experiments,

correlation-based metrics for log-space regression.

During experimentation, various bucket-based classification schemes (5-class, 3-class, and binary) were tested. Results indicate that the dataset does not exhibit strong class separability with respect to view count ranges, while regression in logarithmic space provides a more stable formulation.

## Results Summary

The experimental results highlight several key observations:

direct regression on raw view counts is affected by high variance and skewed distributions;

regression in logarithmic scale improves numerical stability and interpretability;

classification into multiple view-count buckets results in low generalization performance;

in binary classification (low vs high), the model tends to predict a dominant class, indicating weak discriminative signal in the available features.

Overall, the experiments suggest that much of the variance in video performance cannot be explained using the current feature set, and is likely driven by external or unobserved factors.

## Usage

To run the project:

Clone the repository.

Install the dependencies listed in requirements.txt.

Place the dataset in the appropriate directory within data/.

Execute the notebooks or Python scripts for preprocessing, training, and evaluation.

## Dependencies

The project relies on the following main libraries:

Python (≥ 3.8)

pandas

numpy

scikit-learn

PyTorch

PyTorch Lightning

torchmetrics

Full dependency versions are listed in requirements.txt.

## Conclusions

This project explores the feasibility of predicting YouTube Shorts view counts using neural network models applied to tabular metadata features. The empirical findings indicate that, given the current dataset and feature representation, the predictive signal remains limited. View dynamics appear to depend on external, platform-level, or behavioral factors that are not captured in the available input variables.

The results should therefore be interpreted not only as a model performance outcome, but also as evidence of structural limitations in the underlying prediction problem.