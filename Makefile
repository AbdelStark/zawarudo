.PHONY: install test run docker-build compose-up compose-down package package-verify bench demo demo-gpu demo-down

install:
	pip install -e .[dev]

test:
	pytest -q

package:
	python scripts/build_model_package.py synthetic --out .artifacts/model-package/lewm-pusht

package-verify:
	python scripts/build_model_package.py verify --package .artifacts/model-package/lewm-pusht

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

# One-command demo: client + backend + monitoring (mock backend by default).
demo:
	cd deployment && docker compose up --build

# Same, with an NVIDIA GPU reservation (needs the NVIDIA Container Toolkit).
demo-gpu:
	cd deployment && docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build

demo-down:
	cd deployment && docker compose down -v
