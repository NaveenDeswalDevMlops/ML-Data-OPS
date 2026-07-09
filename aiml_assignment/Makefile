up:
	docker compose up --build -d
down:
	docker compose down
test:
	pytest tests/ --cov=app --cov=pipelines
# This forces the command to run INSIDE the Docker container, ignoring your Mac's Python 3.13
pipeline:
	docker exec -it aiml_assignment-api-1 python pipelines/data_pipeline.py

train:
	docker exec -it aiml_assignment-api-1 python pipelines/ml_pipeline.py