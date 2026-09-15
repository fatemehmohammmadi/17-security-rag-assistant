# Security RAG Assistant (local, no API key)

Bare-bones retrieval over a Markdown-ish security knowledge base: TF-IDF search → stitch top chunks into an answer.

No OpenAI/Anthropic calls. Good for demos where you want the *shape* of RAG without billing or leaking tickets to a third party.

## Run

```bash
pip install -r requirements.txt
python build_kb.py
python ask.py
```

## Outputs

- `outputs/sample_qa.json` — questions, grounded answers, source ids
- `outputs/kb_stats.json`

## Make it yours

Edit the docs in `build_kb.py` or append JSONL lines to `data/kb.jsonl`. For a “real” assistant, swap the templated answer for an LLM call that must cite the retrieved chunks.

## License

MIT
