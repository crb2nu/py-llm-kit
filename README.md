# flexinfer-llm-kit

Reusable LLM client helpers and speculative decoding workflows for FlexInfer Python services.

## Features

- OpenAI-compatible client configuration (LiteLLM / vLLM / OpenAI)
- LangChain `ChatOpenAI` factories for common workloads
- Speculative decode workflow (draft → verify → revise) via LangGraph

## Installation

From PyPI:

```bash
pip install flexinfer-llm-kit
```

From GitLab:

```bash
pip install git+https://gitlab.flexinfer.ai/libs/py-llm-kit.git
```

## Usage

### Speculative decode

```python
from llm_kit.spec_decode import spec_decode

result = await spec_decode("Write 3 storyboard panel prompts in JSON.")
```

### Model factories

```python
from llm_kit.clients import get_textgen_model

model = get_textgen_model()
response = model.invoke([{"role": "user", "content": "Hello"}])
print(response.content)
```

## Configuration

| Env Var | Description | Default |
| --- | --- | --- |
| `LLM_BASE_URL` | OpenAI-compatible base URL (ex: `http://litellm.../v1`) | `http://litellm.ai.svc:8000/v1` |
| `LLM_API_KEY` | API key for the endpoint | `sk-local` |
| `LLM_TEXTGEN_MODEL` | LiteLLM model id/alias | `textgen` |
| `LLM_AGENT_MODEL` | LiteLLM model id/alias | `agent` |
| `LLM_VISION_MODEL` | LiteLLM model id/alias | `vision` |

## License

MIT

## Publishing (GitLab PyPI)

1. Bump `version` in `pyproject.toml`.
2. Tag and push:

```bash
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

3. In GitLab CI for `libs/py-llm-kit`, run the manual `publish` job for that tag pipeline.

## Publishing (Public PyPI)

1. Set GitLab CI variables: `PYPI_API_TOKEN` and `PUBLISH_PUBLIC_PYPI=true`.
2. Run the manual `publish:pypi` job for the tag pipeline.
