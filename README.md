# SarcLM

SarcLM is a research codebase for multimodal sarcasm and sentiment analysis using LLaVA-based models and Chain-of-Thought (CoT) rationale generation. The repository contains data preparation scripts, fine-tuning utilities (with LoRA/PEFT), evaluation code, and notebooks for experiment tracking and reproduction.

This project was used to prepare datasets (notably MELD), generate rationales, fine-tune LLaVA-style multimodal models, and evaluate model robustness for sarcasm/sentiment detection tasks.

## Repository structure

- `main.py` — project entry (depending on your workflow).
- `src/` — core scripts and utilities:
  - `eval.py` — evaluation utilities.
  - `finetune_with_lora.py` — fine-tuning script with LoRA/PEFT.
  - `robust_llava_loader.py` — dataset / dataloader utilities for robust training and multimodal inputs.
  - `meld_data_preparation_with_cot_generation.py` — MELD data preprocessing and CoT rationale generation.
  - `test_compare_base_adapter_model.py` — comparisons between base and adapter models.
- `datasets/` — dataset files and metadata (MELD, annotations, and generated rationales).
- `notebooks/` — Jupyter notebooks used for data prep, fine-tuning, and results analysis.
- `assets/` — images and other static assets used in experiments.
- `requirements.txt` — Python dependencies for the project.

## Quick start

1. Create and activate a Python virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

1. Prepare datasets (example for MELD):

```powershell
python src/meld_data_preparation_with_cot_generation.py
```

1. Fine-tune a model with LoRA / PEFT (example):

```powershell
python src/finetune_with_lora.py --config configs/finetune.yaml
```

1. Run evaluation:

```powershell
python src/eval.py --preds path/to/preds.jsonl --gold datasets/annotations.json
```

Note: The above commands are illustrative. Check the script headers and notebook cells for exact CLI arguments and config file expectations.

## Notebooks

Key notebooks are in `notebooks/` and include:

- `MELD_Data_Preparation_with_CoT_Generation.ipynb` — data prep and chain-of-thought rationale generation.
- `Multimodal_LLaVA_Fine_tuning.ipynb` — interactive fine-tuning of LLaVA models.
- `LLaVA_Fine_tuning_with_PEFT_LoRA.ipynb` — experiment with PEFT/LoRA adapters.
- `Results.ipynb` — analysis and visualizations of experiments.

Open notebooks in Jupyter or VS Code's notebook editor.

## Datasets

The `datasets/` folder contains MELD data and generated rationale files. The repository includes example CSV/JSONL files used to train and evaluate models. If you need to download the original MELD dataset or other sources, follow the dataset-specific instructions inside the notebooks or `datasets/` README (if present).

## Development notes

- This codebase targets Python 3.8+ and relies on PyTorch and Hugging Face ecosystems for model training and PEFT/LoRA.
- Use GPU-enabled environments for training. Adjust batch sizes and precision (fp16) in your training configs.
- Many scripts expect environment variables or config files for model paths, dataset locations, and logging destinations — inspect the top of each script for required arguments.

## Running tests

If the repository contains tests, run them with:

```powershell
python -m pytest -q
```

Or run the provided `test.py` for quick sanity checks:

```powershell
python test.py
```

## Contributing

If you'd like to contribute, open issues or PRs on the project's GitHub repository. Please include reproducible steps and small focused changes.

## Contact

For questions about the code, data preparation, or experiments, inspect the notebooks and script docstrings first. If that doesn't help, reach out to the repository owner / maintainer.

---

Generated README — adapt examples and CLI arguments to match your local workflow and any config files you add.
