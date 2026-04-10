train:
	python3 src/train.py

serve:
	uvicorn src.main:app --reload --port 8000

docker-build:
	docker build -t fraud-model-api .

docker-run:
	docker run -p 8000:8000 fraud-model-api

compose-up:
	docker-compose up