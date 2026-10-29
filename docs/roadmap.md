# Roadmapa znalostí a nástrojů

Roadmapa kopíruje fáze stavby nového frameworku. Každá fáze má: co se naučíš (koncepty),
jaké nástroje použijeme, a co si k tomu přečíst. Materiály jsou řazené od nejkratšího.
Označení: 📖 čtení, 🎥 video/kurz, 🔧 dokumentace nástroje.

---

## Fáze 0 – Prostředí a projekt jako balíček

**Koncepty:** virtuální prostředí a lockfile (reprodukovatelné závislosti), `pyproject.toml`
jako jediný zdroj metadat, editable install (`pip install -e .` / `uv sync`) místo `sys.path` hacků,
proč chybějící `__init__.py` způsobovalo trápení v noteboocích, formátování a lint jako automatická
kontrola, testy jako záchranná síť při refaktoru.

**Nástroje:** `uv` (prostředí + závislosti + lock, Linux i Windows), `ruff` (lint + format),
`pytest`, `pre-commit`.

**Materiály:**
- 🔧 uv – „Working on projects“: https://docs.astral.sh/uv/guides/projects/
- 📖 Python Packaging User Guide – „Writing your pyproject.toml“: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- 🔧 ruff: https://docs.astral.sh/ruff/tutorial/
- 📖 pytest „Get started“: https://docs.pytest.org/en/stable/getting-started.html

---

## Fáze 1 – Konfigurace

**Koncepty:** validace na hranici systému (pydantic: typy, `Literal`, `extra="forbid"`, vlastní
validátory), rozdíl mezi *template* (výchozí hodnoty) a *experiment* (přepisy), CLI override
(`--set training.lr=1e-3`), *resolved config* uložený s výsledky = reprodukovatelnost, hash configu.
Srovnání s Hydra/OmegaConf: co dělají navíc a za jakou cenu.

**Nástroje:** `pydantic` v2, `PyYAML`, `typer` (CLI).

**Materiály:**
- 🔧 pydantic – Models, Validators, `model_config`: https://docs.pydantic.dev/latest/concepts/models/
- 🔧 typer tutorial: https://typer.tiangolo.com/tutorial/
- 📖 Hydra – „Getting started“ (jen ke srovnání, nepoužijeme): https://hydra.cc/docs/intro/

---

## Fáze 2 – Data

**Koncepty:** proč je CSV drahé (parsování textu, žádné typy), binární formáty (`.npy`, memmap,
Parquet), cache klíčovaná hashem vstupu, deterministický split řízený seedem, celý dataset jako
GPU tensor a batching indexací (proč byl `DataLoader` s workery pro tabulární data špatně),
odvozené fyzikální featury jako registr funkcí, ověření normalizace (mean/std per sloupec),
únik informace mezi train/val/test (data leakage).

**Nástroje:** `numpy`, `pandas` (volitelně `polars` – řádově rychlejší čtení CSV), `torch`.

**Materiály:**
- 📖 numpy `memmap`: https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
- 📖 Apache Parquet – co to je: https://parquet.apache.org/docs/overview/
- 📖 polars „Getting started“: https://docs.pola.rs/user-guide/getting-started/
- 📖 PyTorch „Performance Tuning Guide“ (sekce data loading, `pin_memory`, `non_blocking`): https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

---

## Fáze 3 – Model, loss, task

**Koncepty:** registr (dekorátor + slovník jméno→třída) jako rozšiřitelnost bez úprav jádra,
`nn.Sequential` vs. vlastní `forward`, výstupní hlavy (regrese: identity; binární: logit + BCEWithLogits;
multiclass: logity + CrossEntropy – proč nedávat sigmoid/softmax do modelu), vážené a multitask
lossy (pevné váhy, nejistotní vážení, GradNorm), inicializace vah, mrtvé ReLU a jejich diagnostika.

**Nástroje:** `torch.nn`, vlastní `losses/`.

**Materiály:**
- 📖 PyTorch „What is torch.nn really?“: https://pytorch.org/tutorials/beginner/nn_tutorial.html
- 📖 Ruder – „An Overview of Multi-Task Learning in Deep Neural Networks“ (2017): https://arxiv.org/abs/1706.05098
- 📖 Kendall et al. – „Multi-Task Learning Using Uncertainty to Weigh Losses“ (2018): https://arxiv.org/abs/1705.07115
- 📖 Chen et al. – „GradNorm“ (2018): https://arxiv.org/abs/1711.02257
- 📖 Gorishniy et al. – „Revisiting Deep Learning Models for Tabular Data“ (2021): https://arxiv.org/abs/2106.11959

---

## Fáze 4 – Trénovací smyčka a tracking

**Koncepty:** anatomie kroku (zero_grad → forward → loss → backward → step), callbacky
(early stopping, checkpoint, scheduler) jako oddělení „co se děje v epoše“ od „kdy zasáhnout“,
`model.train()`/`eval()` a `no_grad`/`inference_mode`, seedování a determinismus (co jde a nejde
zaručit na GPU), mixed precision (AMP), `torch.compile`, sledování experimentů: lokální
strukturované výstupy → TensorBoard → W&B; co je MLflow a kdy dává smysl.
Srovnání s PyTorch Lightning: `LightningModule`, `Trainer`, callbacky – ukážeme, že naše smyčka
je totéž bez magie.

**Nástroje:** `torch`, `tensorboard`, `wandb`, (ukázka `mlflow`, `lightning`).

**Materiály:**
- 📖 Karpathy – „A Recipe for Training Neural Networks“: https://karpathy.github.io/2019/04/25/recipe/
- 📖 Google – „Deep Learning Tuning Playbook“: https://github.com/google-research/tuning_playbook
- 🔧 PyTorch AMP: https://pytorch.org/docs/stable/amp.html
- 🔧 PyTorch reprodukovatelnost: https://pytorch.org/docs/stable/notes/randomness.html
- 🔧 TensorBoard s PyTorch: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
- 🎥 W&B „Effective MLOps: Model Development“ (zdarma): https://www.wandb.courses/
- 🔧 Lightning „Basic skills“ (až po naší smyčce): https://lightning.ai/docs/pytorch/stable/levels/core_skills.html

---

## Fáze 5 – Automatizace: sweepy a Optuna

**Koncepty:** grid vs. random vs. bayesovské hledání (TPE), pruning neperspektivních běhů,
search space v YAML, objective = validační metrika (ne testovací!), multi-seed jako odhad rozptylu,
paralelní běhy a sdílená studie (SQLite storage), jak výsledky sweepu ukládat, aby se daly
porovnat bez notebooku.

**Nástroje:** `optuna`, vlastní `sweep` CLI, `optuna-dashboard`.

**Materiály:**
- 🔧 Optuna tutorial (Key features → Pruning → Storage): https://optuna.readthedocs.io/en/stable/tutorial/index.html
- 📖 Optuna + PyTorch příklad: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
- 📖 Bergstra & Bengio – „Random Search for Hyper-Parameter Optimization“ (2012): https://www.jmlr.org/papers/v13/bergstra12a.html

---

## Fáze 6 – Vyhodnocení a reporting

**Koncepty:** oddělení val (výběr modelu) a test (finální číslo), metriky pro regresi (MSE, MAE,
MAPE a její pasti u nul, MedAE, MaxAE) a klasifikaci (accuracy, precision/recall, F1, ROC-AUC,
kalibrace), správné měření inference (warm-up, `cuda.synchronize`, medián z opakování),
řetězené modely a chyba, která se přenáší, grafy generované skriptem ze strukturovaných výstupů.

**Nástroje:** `scikit-learn.metrics`, `matplotlib`, vlastní `report` CLI.

**Materiály:**
- 🔧 scikit-learn „Metrics and scoring“: https://scikit-learn.org/stable/modules/model_evaluation.html
- 📖 Raschka – „Model Evaluation, Model Selection, and Algorithm Selection in ML“: https://arxiv.org/abs/1811.12808

---

## Průřezově (kdykoli během práce)

- 📖 Goku Mohandas – „Made With ML“ (MLOps kurz, zdarma): https://madewithml.com/
- 📖 „The Missing Semester of Your CS Education“ (git, shell, debugging): https://missing.csail.mit.edu/
- 📖 PyTorch internals pro zvídavé – ezyang „PyTorch internals“: http://blog.ezyang.com/2019/05/pytorch-internals/
- 🎥 Karpathy „Neural Networks: Zero to Hero“ (backprop od nuly, když chceš jít hlouběji): https://karpathy.ai/zero-to-hero.html
