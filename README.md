Customer Churn Prediction

This repository contains the code and documentation for a customer churn predictioject using logistic regression and decision tree models. The goal of this project is to predict whether a customer will churn (i.e., stop using a service) based on various features.

Table of Contents

    Introduction
    Dataset
    Installation
    Usage
    Modeling
    Results
    Contributing

Introduction

Customer churn is a critical issue for businesses, as acquiring new customers is often more expensive than retaining existing ones. This project aims to build predictive models to identify customers who are likely to churn, enabling businesses to take proactive measures to retain them.

Dataset
The dataset used in this project is from Kaggle's customer churn competition. It contains various features about customers, including demographic information, account information, and service usage details.

   Link to the dataset: Kaggle Customer Churn Dataset
Features include:

    customer_id: Unique ID for each customer
    credit_score: Customer's credit score
    country: Country of the customer
    gender: Gender of the customer
    age: Age of the customer
    tenure: Number of years the customer has been with the company
    balance: Account balance of the customer
    products_number: Number of products the customer has
    credit_card: Whether the customer has a credit card (1) or not (0)
    active_member: Whether the customer is an active member (1) or not (0)
    estimated_salary: Estimated annual salary of the customer
    churn: Whether the customer churned (1) or not (0)

Installation

To run this project, you need to have Python and the following libraries installed:

    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    jupyter

You can install these libraries using pip:

bash

pip install pandas numpy scikit-learn matplotlib seaborn jupyter

Usage

    Clone the repository:

bash

git clone https://github.com/Hartyplaza/churn-prediction.git
cd churn-prediction

    Open the Jupyter Notebook:

bash

jupyter notebook

    Run the notebook CHURN DATASET.ipynb to see the data analysis, model training, and evaluation.

Modeling
Logistic Regression

Logistic regression is a linear model used for binary classification problems. It estimates the probability that a given input point belongs to a certain class.
Decision Tree

Decision tree is a non-linear model that splits the data into subsets based on feature values. Each node represents a feature, each branch represents a decision rule, and each leaf represents an outcome.
Evaluation

Both models are evaluated using accuracy score

The results of the models are as follows:
    Logistic Regression:

    Accuracy: 0.78

    Decision Tree:
        Accuracy: 0.81
        

The decision tree model performs slightly better than The logistic regression model in terms of accuracy 

Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or bug fixes.
