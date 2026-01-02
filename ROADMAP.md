# Roadmap: py-llm-kit

## Vision

Provide a small, consistent LLM integration layer for FlexInfer Python services:
- shared env var conventions
- shared LangChain client factories
- shared orchestration utilities (SpecDecode, structured outputs)

## Current Status

- Speculative decode workflow (draft → verify → revise)
- OpenAI-compatible client + ChatOpenAI factories

## Immediate Priorities

- [ ] Add optional cloud fallback (OpenAI) parity with service configs
- [ ] Add request/response logging hooks compatible with `py-observability`
- [ ] Add retry hooks compatible with `py-resilience`

## Maintenance

- [ ] Keep `openai`, `langchain-openai`, and `langgraph` versions in sync with services
