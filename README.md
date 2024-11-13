# credit-default-prediction

## About the project

In this project, we assume we are some bank in the United States.

The aim of this project is to predict whether a person will be able to pay off their loan given some information about them.

To achieve this, I developed an ML pipeline to train and evaluate a machine learning model to perform said task.

The dataset used in this project is originally from DataCamp's course [Credit Risk Modeling in Python](https://app.datacamp.com/learn/courses/credit-risk-modeling-in-python)

I use [DVC](https://dvc.org/) for data and model versioning as well to orchestrate the ML pipeline.

### Experiment tracking

All the experiments performed across this project can be retrieved [here](https://studio.dvc.ai/user/ericdasse28/projects/credit-default-prediction-r4c9vq41ky)

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

## Training pipeline

### Data preprocessing

The **data preprocessing** phase is all about cleaning and preparing raw data to ensure it's in a usable format for modeling. In this project, the data preprocessing is done following these steps:

1. Missing values handling: Imputing missing data or removing rows/columns with too many missing values (or missing values at suspicious locations)
2. Outlier treatment
3. Log transformation of features with a large distribution (`person_income`)
4. Min-max scaling of the numerical features
5. One-hot encoding of the categorical features

### Data split

Once the data have been preprocessed, we split them into training and test sets.
