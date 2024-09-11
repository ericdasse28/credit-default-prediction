tune-train-params:
	poetry run params_tuning --dataset-path data/train.csv

lint:
	black --check .
	isort --check . --profile black
	flake8

