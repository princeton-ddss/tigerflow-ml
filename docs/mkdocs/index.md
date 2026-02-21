---
hide:
  - navigation
---
<style>
  .md-typeset h1,
  .md-content__button {
    /* display: none; */
    margin-bottom: 0;
  }
</style>

#

<p align="center">
  <img alt="tigerflow-ml-logo" src="assets/img/logo.png" width="350" />
</p>

<p align="center">
    <a href="https://www.python.org">
      <img alt="python-shield" src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB.svg?style=flat&logo=python&logoColor=white"/>
    </a>
    <a href="https://opensource.org/licenses/MIT">
      <img alt="mit-license" src="https://img.shields.io/badge/License-MIT-yellow.svg"/>
    </a>
</p>

**TigerFlow ML** is a [TigerFlow](https://github.com/princeton-ddss/tigerflow) task library extension that makes large-scale machine learning inference a breeze. Think of it as your replacement for expensive, opaque private cloud services. TigerFlow supports *batch inference* by running tasks on your organization's existing HPC resources using TigerFlow orchestration to optimize throughput and take advantage of unused compute cycles. Task logic is completely transparent and open source and relies on open-weight models available from the Hugging Face model hub (or bring your own). Tasks can be used on their own or incorporated into larger workflows using TigerFlow's pipeline construction.

## Available Tasks

- [OCR](tasks/ocr.md) — Extract text from images and PDFs
- [Translation](tasks/translate.md) — Translate text documents
- [Transcription](tasks/transcribe.md) — Transcribe audio to text
- [Object Detection](tasks/detect.md) — Detect objects in images and videos

## Installation

TigerFlow ML requires [TigerFlow](https://github.com/princeton-ddss/tigerflow) to be installed.

```bash
pip install tigerflow tigerflow-ml
```

## Next Steps

Check out the [Task Guide](tasks/ocr.md) for detailed usage, parameters, and examples.
