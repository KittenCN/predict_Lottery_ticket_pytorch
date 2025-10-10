# Makefile - 一键任务入口

setup:
	pip install -r requirements.txt

lint:
	python -m compileall scripts/ src/

test:
	pytest -q

run:
	python scripts/train_model.py --name kl8 --seq_len 5 --red_epochs 2 --batch_size 2

smoke:
	python scripts/smoke_train_predict.py

ci: lint test
