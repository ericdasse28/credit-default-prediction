# credit-default-prediction

## About the project

In this project, we assume we are some bank in the United States.

The aim of this project is to predict whether a person will be able to pay off their loan given some information about them.

To achieve this, I developed an ML pipeline to train and evaluate a machine learning model to perform said task.

The dataset used in this project is originally from DataCamp's course [Credit Risk Modeling in Python](https://app.datacamp.com/learn/courses/credit-risk-modeling-in-python)

I use [DVC](https://dvc.org/) for data and model versioning as well to orchestrate the ML pipeline.

### Experiment tracking

All the experiments performed across this project can be tracked [here](https://studio.dvc.ai/user/ericdasse28/projects/credit-default-prediction-r4c9vq41ky)

## Development

To get started on this project:

1. Clone or fork this repository

2. Install the dependencies

```
poetry install
```

3. (Optional) To ensure your code quality is okay and all the tests are green before pushing your code to the remote repository, you can activate the pre-push hooks:

```
git config --local core.hooksPath .githooks/
```

If you want to remove the hooks:

```
git config --local --unset core.hooksPath
```

To bypass the hooks when you push:

```
git push --no-verify
```

## Training pipeline

### Exploratory Data Analysis

Notebook: https://colab.research.google.com/drive/1t5E7FJKD-UvGw3-vXyJKsJ8LM2fazLns?usp=sharing

### Data split

To avoid data leakage from test data to training data, we split the raw data into training
and test **first**.

### Data preprocessing

The **data preprocessing** phase is all about cleaning and preparing raw data to ensure it's in a usable format for modeling.
In this project, the data preprocessing is done following these steps:

1. Missing values handling: Imputing missing data or removing rows/columns with too many missing values (or missing values at suspicious locations)
2. Outlier treatment
3. Turn `cb_person_default_on_file` into an actual boolean feature

### Feature engineering

The intent behind the **feature engineering** stage is to create new features or modify existing
ones to _improve model performance_. More specifically, the goal is to enhance the
predictive power of the dataset by generating features that better represent the
underlying patterns in the data.

In this project, feature engineering is done according to the following steps:

1. Feature selection: to identify the most relevant features of the model
2. One-hot encoding of the categorical features
3. Log transformation of features with a large distribution (`person_income`, `loan_amnt`)
4. Min-max scaling of the numerical features

### Model validation

At this step, we perform a 5-fold cross-validation of the model on the training data.
The goal is to enhance our idea of how the model performs on unseen data.

The following metrics are computed:

- Average model accuracy on the folds
- Average precision
- Average recall

## Testing the pipeline

TODO
