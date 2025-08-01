# Interview Assistant

This module provides a simple command line tool for practicing interview questions using a
local knowledge base. Answers can be customised via `config.yaml`.

## Usage

```bash
python main.py [--audio question.wav]
```

If `--audio` is provided, the question is transcribed with SpeechRecognition.
Otherwise, you will be prompted to type your question.

Edit `config.yaml` to change the answer style parameters.
Add your own documents under `knowledge_base/` to improve retrieval.
