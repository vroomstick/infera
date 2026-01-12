# Infera Makefile
# Commands for development, testing, and deployment

.PHONY: install run run-full api test eval lint clean docker docker-up docker-down fetch help setup doctor demo

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
	@echo "Setup & Verification:"
	@echo "  setup           One-command setup (install deps, start DB)"
	@echo "  doctor          Check system health (Python, Docker, DB, env vars)"
	@echo "  demo            Run end-to-end demo with fixture"
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

# Setup & Verification
setup: install-backend docker-up-db
	@echo ""
	@echo "✅ Setup complete!"
	@echo "  - Dependencies installed"
	@echo "  - PostgreSQL container started"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Set DATABASE_URL in .env (or use default)"
	@echo "  2. Run 'make doctor' to verify setup"
	@echo "  3. Run 'make demo' to test the pipeline"

doctor:
	@echo "🔍 Infera System Health Check"
	@echo "================================"
	@echo ""
	@echo "Python:"
	@python3 --version || echo "❌ Python not found"
	@python3 -c "import sys; print(f'  Version: {sys.version}')" || echo "❌ Python not working"
	@echo ""
	@echo "Docker/OrbStack:"
	@docker --version 2>/dev/null || echo "❌ Docker not found (install Docker or OrbStack)"
	@docker ps >/dev/null 2>&1 && echo "  ✅ Docker daemon running" || echo "  ⚠️  Docker daemon not running (start Docker Desktop or OrbStack)"
	@echo ""
	@echo "PostgreSQL Container:"
	@docker ps --filter "name=infera-db" --format "{{.Names}}" | grep -q infera-db && echo "  ✅ Container 'infera-db' is running" || echo "  ⚠️  Container 'infera-db' not running (run 'make docker-up-db')"
	@echo ""
	@echo "Database Connectivity:"
	@cd backend && python3 -c "from data.database import SessionLocal; from sqlalchemy import text; db = SessionLocal(); db.execute(text('SELECT 1')); print('  ✅ Database connection successful')" 2>/dev/null || echo "  ❌ Database connection failed (check DATABASE_URL in .env)"
	@echo ""
	@echo "Environment Variables:"
	@test -f backend/.env && echo "  ✅ .env file exists" || echo "  ⚠️  .env file not found (copy from .env.example)"
	@cd backend && python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print(f'  DATABASE_URL: {\"✅ set\" if os.getenv(\"DATABASE_URL\") else \"❌ missing\"}'); print(f'  OPENAI_API_KEY: {\"✅ set\" if os.getenv(\"OPENAI_API_KEY\") else \"⚠️  optional (for summaries)\"}')" 2>/dev/null || echo "  ⚠️  Could not check env vars (install python-dotenv)"
	@echo ""
	@echo "Dependencies:"
	@cd backend && python3 -c "import sentence_transformers; print('  ✅ sentence-transformers installed')" 2>/dev/null || echo "  ❌ sentence-transformers not installed (run 'make install-backend')"
	@cd backend && python3 -c "import fastapi; print('  ✅ fastapi installed')" 2>/dev/null || echo "  ❌ fastapi not installed (run 'make install-backend')"
	@cd backend && python3 -c "import sqlalchemy; print('  ✅ sqlalchemy installed')" 2>/dev/null || echo "  ❌ sqlalchemy not installed (run 'make install-backend')"
	@echo ""
	@echo "================================"
	@echo "Run 'make demo' to test the pipeline"

demo:
	@echo "🚀 Running Infera Demo"
	@echo "======================"
	@echo ""
	@echo "This will run the full pipeline on a test fixture..."
	@echo ""
	@cd backend && python3 -m pytest tests/test_integration.py::TestPipelineIntegration::test_pipeline_standard_2023 -v -s || (echo ""; echo "❌ Demo failed. Run 'make doctor' to diagnose issues."; exit 1)
	@echo ""
	@echo "✅ Demo completed successfully!"
	@echo ""
	@echo "Next steps:"
	@echo "  - Run 'make fetch TICKER=AAPL' to analyze a real filing"
	@echo "  - Run 'make api' to start the API server"
	@echo "  - Check backend/reports/ for generated reports"

# Dependencies
freeze:
	cd backend && pip freeze > requirements-lock.txt
