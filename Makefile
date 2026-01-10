# Infera Makefile
# Commands for development, testing, and deployment

.PHONY: install run run-full api test eval lint clean docker docker-up docker-down fetch help

# Default target
help:
	@echo "Infera - SEC 10-K Risk Analysis Pipeline"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Development:"
	@echo "  install      Install dependencies"
	@echo "  run          Run pipeline on AAPL (skip GPT summary)"
	@echo "  run-full     Run pipeline with GPT summary"
	@echo "  api          Start the FastAPI server"
	@echo ""
	@echo "SEC EDGAR Fetch:"
	@echo "  fetch TICKER=AAPL [YEAR=2023]     Fetch & analyze a 10-K"
	@echo "  fetch-only TICKER=AAPL [YEAR=2023] Download only (no analysis)"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run pytest test suite"
	@echo "  eval         Run evaluation harness"
	@echo "  lint         Run linter (ruff)"
	@echo ""
	@echo "Docker:"
	@echo "  docker       Build Docker image"
	@echo "  docker-up    Start containers"
	@echo "  docker-down  Stop containers"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean        Remove generated files"

# Development
install:
	pip install -r requirements.txt
	pip install pytest ruff

run:
	python services/pipeline_service.py --file data/AAPL_10K.html --skip-summary

run-full:
	python services/pipeline_service.py --file data/AAPL_10K.html

run-tsla:
	python services/pipeline_service.py --file data/TSLA_10K.html --skip-summary

run-msft:
	python services/pipeline_service.py --file data/MSFT_10K.html --skip-summary

# Fetch from SEC EDGAR (usage: make fetch TICKER=NVDA YEAR=2023)
fetch:
	@if [ -z "$(TICKER)" ]; then \
		echo "Usage: make fetch TICKER=AAPL [YEAR=2023]"; \
		exit 1; \
	fi
	python ingest/sec_fetcher.py --ticker $(TICKER) $(if $(YEAR),--year $(YEAR),) --analyze

fetch-only:
	@if [ -z "$(TICKER)" ]; then \
		echo "Usage: make fetch-only TICKER=AAPL [YEAR=2023]"; \
		exit 1; \
	fi
	python ingest/sec_fetcher.py --ticker $(TICKER) $(if $(YEAR),--year $(YEAR),)

api:
	python -m api.main

# Testing
test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=analyze --cov=services --cov-report=term-missing

eval:
	python evaluation/eval_scorer.py

compare:
	python evaluation/compare_methods.py

lint:
	ruff check . --ignore E501

format:
	ruff format .

# Docker
docker:
	docker build -t infera:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-test:
	docker-compose --profile testing up test

# Cleanup
clean:
	rm -rf __pycache__ **/__pycache__ .pytest_cache .ruff_cache
	rm -f infera.db
	rm -rf reports/*.md
	rm -rf data/clean/* data/scored/* data/segments/* data/summarized/*

clean-all: clean
	rm -rf evaluation/plots/*.png
	rm -f evaluation/eval_results.json evaluation/error_analysis.json

# Dependencies
freeze:
	pip freeze > requirements-lock.txt

