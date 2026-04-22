# Document Translation

Translate `.txt` documents into English (or another target language) using any compatible HuggingFace model. Supports automatic language detection and token-aware chunking.

## How it works

1. The source language for each `.txt` file in the input directory is detected via `langdetect` (or supplied with `--source-lang`).
2. If the document exceeds the token budget, it is split into chunks at paragraph and/or sentence boundaries.
3. Chunks are translated via the HuggingFace Transformers pipeline. Truncated output is automatically detected and retried with smaller chunks.
4. Translated chunks are reassembled and written to the output directory.

### Setup (login node)

Once you have tigerflow ready to go, you'll want to download the HuggingFace model you wish to use. You can use any LLM, which will use the prompt:

    "Translate the following text from {source_lang} to {target_lang}. Output only the translated text, nothing else. Text: {text}"


Or you can use any translation specific model you like. For the purpose of this tutorial, we will use the `google/translategemma-27b-it` model (to use this family of model, you will need to accept the [TranslateGemma license](https://huggingface.co/google/translategemma-27b-it) on HuggingFace).

First, make sure you are in a directory/virtual environment with `tigerflow-ml` installed. Authenticate and download the HuggingFace model to the local cache:

```bash
hf auth login
HF_HOME=./.hf hf download google/translategemma-27b-it
```

Setting `HF_HOME=./.hf` defines the directory where the model's files will be downloaded. Using `./.hf` means the files will be downloaded to the current working directory (`./`) and will be in a new directory named `.hf`. If you encounter any issues with downloadind the model, even after accepting the license, you can also try:

```bash
export HF_TOKEN="hf_token"
HF_HOME=./.hf hf download google/translategemma-27b-it
```

### Running the task

Once this model is downloaded, you can run the translation task. To see the options you have, you can enter the following in your terminal:

```
python -m tigerflow_ml.text.translate.slurm --help
```

To run this task directly, run:

```
python -m tigerflow_ml.text.translate.slurm --input-dir path/to/inputs/ --input-ext .txt --output-dir path/to/outputs/ --output-ext .txt --max-workers 1 --cpus 1 --memory 10G --time 24:00:00 --gpus 1 --sbatch-option "--constraint=gpu80" --setup-command "export HF_HOME=./.hf" --setup-command "source .venv/bin/activate" --setup-command "export TRANSFORMERS_OFFLINE=1" --model google/translategemma-27b-it
```

Here's a breakdown of what each of these arguments does:

- `--input-dir` : Specifies the path to the input data. All `.txt` files in this directory will be translated.
- `--input-ext` : Specifies the file format of the input data -- this needs to be `.txt`.
- `--output-dir` : Specifies the path where the translated output files will be saved.
- `--output-ext .txt` : Specifies the file format of the output data -- this needs to be `.txt`.
- `--max-workers` : The maximum number of workers for autoscaling.
- `--cpus` : The number of CPUs allocated per worker.
- `--memory` : The memory allocated per worker
- `--gpus` : The number of GPUs allocated per worker.
- `--sbatch-option` : Additional Slurm options for workers -- when using large models, setting `--constraint=gpu80` ensures the GPUs will have sufficient memory.
- `--setup-command` : Shell command to run before the task starts -- this is where you would activate your virtual environment and make sure `HF_HOME` is pointing to the directory where the model is saved.
- `--model` : The HuggingFace model repo ID for your translation model.

Some other arguments you can use are:

- `--source-lang` : Source language code (e.g. 'en', 'de', 'zh'). If this is not specified, the input files' language is detected via `langdetect`
- `--target-lang` : Target language code (e.g. 'de', 'en', 'fr'). This defaults to english (en).
- `--chunk-size` : The maximum tokens per chunk. The documents are translated one chunk at a time, so if your model cannot handle large inputs, this should be small. If no value is provided, the context window of the model will be determined using the downloaded `config.json` and an appropriate chunk size will be computed. If this process fails, it will fall back to a chunk size of `900`, which is optimized for the `translategemma` models.
- `--batch-size` : The number of chunks to translate in parallel. If not specified, there will be an attempt to identify an optimal number based on resources available and model size. If this process fails, a batch size of `1` will be used.
- `--fetch` : If included, allows downloading from the HuggingFace Hub. Only include `--fetch` if your hardware will have internet access.
- `--task-name` : The task name, which defaults to "Translate"
- `--revision` : The model revision
- `--cache-dir` : The HuggingFace cache directory for model files. This would be an alternative for specifying the path using `HF_HOME` in a `setup-command`. It is important to note that, when using `cache-dir`, you need to specify the pathway to `/hub` (ie. `./.hf/hub`).
- `--device` : Device to use (cuda, cpu, or auto)

### Running as a part of a pipeline
