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
	@echo "  install         Install all dependencies"
	@echo "  install-backend Install backend dependencies"
	@echo "  run             Run pipeline on AAPL (skip GPT summary)"
	@echo "  run-full        Run pipeline with GPT summary"
	@echo "  api             Start the FastAPI server"
	@echo ""
	@echo "Frontend:"
	@echo "  frontend        Start frontend dev server"
	@echo ""
	@echo "SEC EDGAR Fetch:"
	@echo "  fetch TICKER=AAPL [YEAR=2023]     Fetch & analyze a 10-K"
	@echo "  fetch-only TICKER=AAPL [YEAR=2023] Download only (no analysis)"
	@echo ""
	@echo "Testing:"
	@echo "  test            Run pytest test suite"
	@echo "  eval            Run evaluation harness"
	@echo "  lint            Run linter (ruff)"
	@echo ""
	@echo "Docker:"
	@echo "  docker          Build API Docker image"
	@echo "  docker-db       Build PostgreSQL image (pgvector + TimescaleDB)"
	@echo "  docker-up       Start all containers"
	@echo "  docker-up-db    Start only PostgreSQL"
	@echo "  docker-down     Stop containers"
	@echo "  docker-logs     View container logs"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean           Remove generated files"

# Development
install: install-backend install-frontend

install-backend:
	cd backend && pip install -r requirements.txt
	pip install pytest ruff

install-frontend:
	@if [ -d "frontend/package.json" ]; then \
		cd frontend && npm install; \
	else \
		echo "Frontend not yet installed. Add Lovable export to frontend/ first."; \
	fi

run:
	cd backend && python services/pipeline_service.py --file data/AAPL_10K.html --skip-summary

run-full:
	cd backend && python services/pipeline_service.py --file data/AAPL_10K.html

run-tsla:
	cd backend && python services/pipeline_service.py --file data/TSLA_10K.html --skip-summary

run-msft:
	cd backend && python services/pipeline_service.py --file data/MSFT_10K.html --skip-summary

# Fetch from SEC EDGAR (usage: make fetch TICKER=NVDA YEAR=2023)
fetch:
	@if [ -z "$(TICKER)" ]; then \
		echo "Usage: make fetch TICKER=AAPL [YEAR=2023]"; \
		exit 1; \
	fi
	cd backend && python ingest/sec_fetcher.py --ticker $(TICKER) $(if $(YEAR),--year $(YEAR),) --analyze

fetch-only:
	@if [ -z "$(TICKER)" ]; then \
		echo "Usage: make fetch-only TICKER=AAPL [YEAR=2023]"; \
		exit 1; \
	fi
	cd backend && python ingest/sec_fetcher.py --ticker $(TICKER) $(if $(YEAR),--year $(YEAR),)

api:
	cd backend && python -m api.main

frontend:
	cd frontend && npm run dev

# Run both backend and frontend
dev:
	@echo "Starting backend and frontend..."
	@echo "Run 'make api' in one terminal and 'make frontend' in another"

# Testing
test:
	cd backend && python -m pytest tests/ -v

test-cov:
	cd backend && python -m pytest tests/ -v --cov=analyze --cov=services --cov-report=term-missing

eval:
	cd backend && python evaluation/eval_scorer.py

compare:
	cd backend && python evaluation/compare_methods.py

lint:
	cd backend && ruff check . --ignore E501

format:
	cd backend && ruff format .

# Docker
docker:
	docker build -t infera:latest ./backend

docker-db:
	docker build -t infera-postgres:latest ./docker/postgres

docker-up:
	docker-compose up -d

docker-up-db:
	docker-compose up -d db

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-test:
	docker-compose --profile testing up test

# Cleanup
clean:
	rm -rf backend/__pycache__ backend/**/__pycache__ .pytest_cache .ruff_cache
	rm -f infera.db
	rm -rf reports/*.md
	rm -rf backend/data/clean/* backend/data/scored/* backend/data/segments/* backend/data/summarized/*

clean-all: clean
	rm -rf backend/evaluation/plots/*.png
	rm -f backend/evaluation/eval_results.json backend/evaluation/error_analysis.json

# Dependencies
freeze:
	cd backend && pip freeze > requirements-lock.txt
