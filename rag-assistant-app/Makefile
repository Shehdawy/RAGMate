# Handy shortcuts (macOS/Linux, or Windows with make installed). Run `make help`.
.DEFAULT_GOAL := help
PY ?= python

help:            ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:         ## install every dependency
	$(PY) -m pip install -r requirements-all.txt

run:             ## notebook + tests + backend + frontend
	$(PY) run_all.py

ingest:          ## build the vector store from data/raw (no notebook needed)
	cd backend && $(PY) -m app.cli ingest

up:              ## local Docker stack (with Ollama)
	docker compose up --build

prod-up:         ## production stack with HTTPS (needs .env)
	docker compose -f docker-compose.prod.yml up -d --build

prod-logs:       ## follow production logs
	docker compose -f docker-compose.prod.yml logs -f backend frontend

prod-down:       ## stop the production stack
	docker compose -f docker-compose.prod.yml down
