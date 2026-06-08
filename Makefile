.PHONY: install test run docker-build compose-up compose-down

install:
	pip install -e .[dev]

test:
	pytest -q

run:
	uvicorn wmcp_jepa_service.server:app --host 0.0.0.0 --port 8080 --reload

docker-build:
	docker build -t wmcp-jepa-service:local .

compose-up:
	cd deployment && docker compose up --build

compose-down:
	cd deployment && docker compose down
