# py-llm-kit Roadmap

> Last Updated: 2026-07-02
> Tier: 2 (see workspace AGENTS.md "Portfolio Tiers")
> Tracking Issue: none — backlog is the [issues list](https://gitlab.flexinfer.ai/libs/py-llm-kit/-/issues)

## Current Status

Shared LLM integration layer for FlexInfer Python services: env-var
conventions, LangChain/OpenAI-compatible client factories, speculative-decode
workflow (draft → verify → revise), with request/retry hooks into
`flexinfer-observability` and `flexinfer-resilience`. Last organic activity
2026-01-08 (resilience/observability hooks); 2026-03-03 commits were CI fixes
and langchain typing alignment. Maintain mode: no active development.
Evidence: last-20-commit window via GitLab API, inspected 2026-07-02.

- **Plan store**: plan-workspace-portfolio-refresh-2026-h2-roadmaps-quality-baselin-f3db23
- **Deployed**: not deployed (library; consumed by flexinfer services)
- **CI**: python (shared platform/gitops templates)

## Now

_Maintenance only — no active development._

## Next

- [ ] Keep `openai`, `langchain-openai`, and `langgraph` versions in sync
  with consuming services (#1)
- [ ] Quality wave: pre-commit config + Makefile targets (queued by
  portfolio-refresh slice 7)

## Later

- None planned — revive on demand

## Backlog

Full backlog: [P1 issues](https://gitlab.flexinfer.ai/libs/py-llm-kit/-/issues/?label_name[]=P1) ·
[P2](https://gitlab.flexinfer.ai/libs/py-llm-kit/-/issues/?label_name[]=P2) ·
[P3](https://gitlab.flexinfer.ai/libs/py-llm-kit/-/issues/?label_name[]=P3) ·
[Milestones](https://gitlab.flexinfer.ai/libs/py-llm-kit/-/milestones)
