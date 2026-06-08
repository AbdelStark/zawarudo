.PHONY: install test run docker-build compose-up compose-down bench

install:
	pip install -e .[dev]

test:
	pytest -q

bench:
	python benchmarks/run_benchmark.py --base-url $(or $(WMCP_BASE_URL),http://localhost:8080) --profile score-medium

run:
	uvicorn wmcp_jepa_service.server:app --host 0.0.0.0 --port 8080 --reload

docker-build:
	docker build -t wmcp-jepa-service:local .

compose-up:
	cd deployment && docker compose up --build

compose-down:
	cd deployment && docker compose down
