# Credit Risk Analysis Project

## Overview

In the highly competitive credit card industry, financial institutions face the constant challenge of identifying potential defaulters. Extending credit to individuals who are likely to default on their payments can lead to significant financial losses and reputational damage. Credit card companies must carefully balance the risk of lending while ensuring they approve credit for deserving customers—those who are financially responsible and likely to make timely payments.

However, misjudging a client's creditworthiness not only increases the risk of default but also denies credit to individuals who would otherwise manage their financial obligations effectively. This can harm customer relationships and reduce revenue opportunities. Therefore, accurately predicting whether a customer will default is crucial to maintaining a healthy balance sheet and fostering customer trust.

We will develop a robust machine learning algorithm capable of accurately predicting customer defaults. By leveraging data-driven insights, the model will help financial institutions identify high-risk individuals before they are issued a credit card, thereby reducing the likelihood of defaults. This will allow credit card companies to mitigate risk while maximizing their portfolio of reliable customers, improving profitability and customer satisfaction.


## Table of Contents

- [Project Structure](#project-structure)
- [Data Overview](#data-overview)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [Handling Imbalanced Data](#handling-imbalanced-data)
- [ROC Curve Analysis](#roc-curve-analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The project includes the following files and directories:

- `Credit_Risk_Assesment.ipynb`: Jupyter notebook containing the complete codebase and analysis for assessing credit risk.
- `Credit_Risk_AI.ipynb`: Jupyter notebook integrating a language model AI to address your questions and concerns about the credit risk process.
- `Credit_Risk_KPI.ipynb`:Jupyter notebook focused on data analysis, showcasing critical Key Performance Indicators (KPIs) related to credit card data.
- `Credit_Risk_Test.ipynb`: A notebook where you can interactively test the developed model to predict whether a customer (or yourself) is likely to default.
- `README.md`: Detailed description of the project.
- `datasets`: Directory designated for storing datasets and associated pickle files.
- `models/`: Directory for saving trained machine learning models.
- `images/`: Directory to save plots and visualizations.

![alt text](<Screenshot 2024-11-05 at 2.59.54 PM.png>)
## Data Overview

The Credit Risk Dataset, sourced from Kaggle, is a publicly available dataset that simulates credit bureau data for analyzing credit risk. It includes features such as personal details (age, income, home ownership), loan specifics (intent, grade, amount, interest rate), and credit history information (default status, credit history length). 

The target variable we will be picking is `loan_status`, where `0` indicates non-default (the applicant repaid the loan) and `1` indicates default (the applicant failed to repay the loan).

## Exploratory Data Analysis

Initial data analysis focuses on understanding the distribution of key variables:

- **Distribution of Loan Status**: A pie chart is used to visualize the proportion of defaults vs. non-defaults. The data is heavily imbalanced, with significantly more non-defaults.
- **Box Plots**:
  - `person_income`: To understand the distribution of applicant income.
  - `person_emp_length`: To analyze the length of employment of the applicants.

## Modeling and Evaluation

The project involves training multiple models to predict loan defaults, including:

- Logistic Regression
- Random Forest
- Gradient Boosting

### Handling Imbalanced Data

Given the imbalance in the target variable, the project uses the following techniques:

- **SMOTE (Synthetic Minority Over-sampling Technique)**: To oversample the minority class and balance the dataset.
- **Class Weighting**: Adjusting weights in the loss function to give more importance to the minority class.

A `GridSearchCV` is employed to optimize the `f1` score and find the best parameters for the models.

## ROC Curve Analysis

To evaluate the performance of the models, ROC curves are plotted. The ROC curve provides a visual representation of the trade-off between the true positive rate (recall) and the false positive rate (1 - specificity). The AUC (Area Under the Curve) score is used as a metric to compare models.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/credit-risk-analysis.git
   cd credit-risk-analysis