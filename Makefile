.PHONY: install test test-llm-core test-agent-search lint fmt

install:
	uv sync

test:
	uv run pytest packages/

test-llm-core:
	uv run pytest packages/llm_core/

test-agent-search:
	uv run pytest packages/agent_search/

lint:
	uv run flake8 packages/agent_search/ packages/llm_core/

fmt:
	uv run ruff format packages/

run:
	uv run uvicorn semantic_search.app:app --host 0.0.0.0 --port 8085 --reload \
		--reload-dir packages/semantic-search/semantic_search --reload-dir packages/llm-core/llm_core
