tune-train-params:
	poetry run params_tuning --dataset-path data/train.csv

lint:
	poetry run black --check .
	poetry run isort --check .
	poetry run flake8

