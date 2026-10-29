# Analýza legacy repozitáře MathCAS (stav k 2026-09-11)
Kopie pracovních poznámek z repozitáře MathCAS pořízená při založení VortexNet (2026-10-29).
Slouží jako referenční analýza starého kódu a archiv rozhodnutí z 2026-09-11. Aktuální rozhodnutí
jsou v `CLAUDE.md` v kořeni VortexNet; tento soubor se dál neupravuje.


## O co jde

PyTorch framework pro trénování MLP regresorů na tabulárních CFD datech (stlačitelné proudění).

- **Vstupy:** 9 složek tenzoru gradientu rychlosti `A11, A21, A31, A12, A22, A32, A13, A23, A33`.
- **Cíle:** `Shear`, `ResStrain`, `ResVort` (dekompozice gradientu rychlosti na smyk + reziduální strain/vorticita).
- **Data:** CSV `dataset_compressible_flow_60M_training_nstep180.csv` (60M řádků), testovací split
  `dataset_compressible_flow_5M_test.csv` (řádky 50M–55M) a varianta `_with_predshear.csv`,
  kde je navíc sloupec `PredShear` (predikce Shear z upstream modelu, používaný jako vstupní feature).
  Datasety a `outputs/` jsou gitignorované; na tomto stroji nejsou.
- **Výzkumná otázka:** multitask model (všechny 3 cíle najednou) vs. single-task (jen Shear)
  vs. řetězení dvou modelů (shear model → residual model dostane predikovaný Shear jako vstup).
  Porovnává se MSE vs. počet parametrů vs. inference čas, přes více seedů.

### Pojmenování experimentů (v `outputs/`)

- `dan_250 / dan_750 / dan_1500` – multitask modely (tři výstupy), číslo = velikost sítě.
- `shear_dan_*` – jen Shear.
- `res_predShear{250,750,1500}_dan_*` – reziduální model trénovaný s `PredShear` z daného upstream modelu.
- `res_trueShear_dan_*` – reziduální model trénovaný s pravým `Shear`, testovaný s predikovaným.
- Sufix `_v2 … _v5` = opakování s jiným seedem (123, 456, 789, 9999); bez sufixu = seed 42.
- Adresáře experimentů se tvoří jako `<timestamp>_<name>`, ale v noteboocích se často odkazují
  ručně přejmenované složky bez timestampu (`outputs/dan_750`).

## Struktura a implementace (stav `main`, 838 řádků Pythonu)

| Soubor | Role |
|---|---|
| `main.py` | vstup: `python main.py <config.yaml>` → vytvoří exp dir, zkopíruje config, trénuje, vyhodnotí (test CSV cesta natvrdo) |
| `config/schema.py` | pydantic modely `ArchitectureConfig`, `TrainingConfig`, `DataConfig`, `FullConfig` (bez validace hodnot) |
| `config/__init__.py` | `load_config`: YAML → `FullConfig`, dopočítá `name` z názvu souboru a defaulty |
| `src/architecture.py` | `NeuralNetwork` (Linear+aktivace+Dropout z configu), `check_dying_relus` přes forward hooky |
| `src/dataset.py` | `extract_data` (pandas CSV) → `split_data` (sklearn, `random_state=42` natvrdo) → `DataLoader` (num_workers=8); sliding window; nepoužitý `FastTensorDataLoader` |
| `src/trainer.py` | `set_seed`, `train_one_epoch`, `validate`, `train` (early stopping, ReduceLROnPlateau, best_model.pt, logování) |
| `src/evaluation.py` | `load_model`, `load_data`, `evaluate` (měří jen forward čas s cuda.synchronize), `write_results_to_file` → `evaluation_results.txt` |
| `utils/` | `experiment.py` (exp dir, kopie configu, md5 hash), `logger.py`, `metrics.py` (MAPE/MAE/MSE/MedAE/MaxAE jako stringy), `modules.py` (getattr lookup do torch.nn/optim), `profiler.py` (NVTX) |
| `notebooks/` | `trainer` (ruční tréninková smyčka, duplikát trainer.py), `evaluation` (dvoumodelové a řetězené vyhodnocení přes všechny kombinace seedů, ~500 řádků), `visualize` (~900 řádků grafů, parsuje logy regexem), `analyse` (EDA, tvorba test splitu, tvorba `PredShear` sloupce), `tools` (kopie NeuralNetwork) |
| `scripts/run_experiments.ps1` | PowerShell multi-seed sweep: přepisuje `seed:` a `name:` v YAML do temp souborů a volá `main.py` |
| `tests/test_architecture.py` | jediný test (forward shape) |

Historie: první commit 2025-06-02, jádro hotové 2025-06-18, NVTX 2025-07, scheduler + sliding
window 2025-08, evaluation modul 2025-08-26, dying-ReLU + sweep skript 2026-03, README 2026-03,
poslední výzkumný commit na main 2026-05-04. Vyvíjeno na Windows (RTX 3070), běželo i na
vzdáleném stroji **gaia2**.

## Větve (stav k 2026-09-11)

- **`main`** – referenční stav popsaný výše.
- **`origin/dataloading_refactor`** (1 commit, 2026-05-04, odbočka přímo z tipu main)
  - `prepare_dataloaders` rozděleno na `prepare_tensor_split` (CSV se čte **jednou** před smyčkou),
    `_select_train_window`, `build_train_loader_for_epoch`, `build_val_loader`.
  - V `train()` natvrdo `use_fast_loaders = True` → používá `FastTensorDataLoader` (indexace tensorů, žádné workery).
  - Val loader se staví jednou; train loader každou epochu (kvůli sliding window).
  - `main.py` vrací test CSV zpět na `..._5M_test.csv` (bez predshear) – pravděpodobně omyl / jiný stav.
  - Čistá, funkční změna. **Toto je správný směr** a měl by se stát základem nového data modulu.
- **`origin/gaia2`** (2 commity „updt“, oba pushnuté 2026-09-11 11:14, odbočka z 2026-03-23 `06c1c78`, tedy chybí jí 4 commity z main)
  - commit `e74e25b`: `num_workers=8 → 0` (pin_memory zůstalo True), odstraněn `verbose=True`
    z `ReduceLROnPlateau` (v novějším torchi neexistuje), sweep skript přepnut na `res_predShear1500_dan_750`,
    test CSV na `_with_predshear`.
  - commit `d304a48`: `_DATA_CACHE` (modulový cache klíčovaný cestou+sloupci, aby se CSV nečetlo každou epochu),
    `pd.read_csv(dtype=float32)` + `torch.from_numpy` (bez kopie), vlastní `split_data` bez sklearn
    (`np.random.default_rng(42)`), parametry `num_workers`/`pin_memory` protažené do `prepare_dataloaders`,
    trainer volá `num_workers=0, pin_memory=False`, extra logování (torch build, cuDNN, sys.executable).
  - **Obsahuje zacommitované konfliktní značky** (`<<<<<<< Updated upstream` / `>>>>>>> Stashed changes`)
    v `src/dataset.py` řádky 125–144 → větev v tomto stavu nelze importovat.
  - Specifika gaia2 (odvozeno z diffu): problém s DataLoader workery (multiprocessing) → `num_workers=0`;
    novější torch bez `verbose=` u scheduleru; potřeba vidět, který Python/torch build běží.
- **`origin/multi-level-config-management`** (1 commit, 2025-06-17, odbočka z velmi raného stavu)
  - Pokus o dvouúrovňové configy: `templates/regression.yaml` jako seznam vrstev s `${placeholder}`
    a `experiments/experiment1.yaml` s `architecture.template: regression`. Nikdy nedokončeno,
    žádná implementace substituce. Myšlenka (template + override) stojí za oživení, kód ne.

Obě větve `dataloading_refactor` a `gaia2` řeší tentýž problém (CSV čtené každou epochu, drahé
workery) dvěma různými způsoby a vzájemně se nekonzistentně liší v cestě k test CSV.

## Známé problémy na `main`

1. `prepare_dataloaders` se volá **každou epochu** → CSV se znovu načítá a splituje; `DataLoader`
   s `num_workers=8` na malých in-memory tensorech = spawn 8 procesů každou epochu. V logu
   z 2025-08-19: 5 000 vzorků, 2 819 parametrů, **5 s na epochu** – téměř vše je režie.
2. `FastTensorDataLoader` napsán, ale na main nepoužit.
3. `split_data` používá `random_state=42` bez ohledu na `config.seed`.
4. Log „Learning rate reduced to …“ se vypisuje při každém zlepšení (podmínka `num_bad_epochs == 0`), ne při snížení LR.
5. `ReduceLROnPlateau(verbose=True)` – deprecated/odstraněno v novějším torchi.
6. Cesta k test CSV natvrdo v `main.py`; evaluation ihned po tréninku bez možnosti vypnout.
7. Žádná normalizace/standardizace vstupů ani cílů (možná záměr, ale nikde nezaznamenáno).
8. Historie loss se neukládá strukturovaně (jen text log); notebooky parsují logy regexem.
9. Metriky se vrací jako formátované stringy, notebooky je převádí zpět na float.
10. `load_config` přidává klíč `log_file`, který schema nezná → pydantic ho tiše zahodí.
11. Chybí `__init__.py` v `src/`, `utils/`; každý notebook začíná 16řádkovým `sys.path` hackem.
12. Windows-only: backslash cesty v noteboocích, PowerShell sweep, `requirements.txt` je UTF-16 s `pywin32` a `torch+cu121` piny → na Linuxu neinstalovatelné.
13. Velká výzkumná logika (řetězené vyhodnocení, multi-seed kombinace, grafy) žije jen v noteboocích, duplicitně a bez testů.
14. Multi-seed sweep = PowerShell přepisující YAML textově.
15. Bez `pyproject.toml`, bez CI, 1 test.

## Prostředí (2026-09-11)

- Fedora Linux, systémový Python 3.14.7, **žádný venv, torch není nainstalován**.
- GPU: NVIDIA GeForce RTX 4060 Laptop, 8 GB.
- Vzdálený stroj gaia2: detaily doplnit od uživatele (OS, torch verze, důvod `num_workers=0`, Slurm?).

## Rozhodnutí

_(doplňovat průběžně, s datem)_

- 2026-09-11: Poznámky, rozhodnutí a detaily z konzultací se ukládají do tohoto souboru.
- 2026-09-11: **Požadavky na nový framework** (zadal uživatel):
  - co nejvíc lightweight, přehledný, rozšiřitelný;
  - regrese i klasifikace (jiná loss, jiná výstupní vrstva, jiné metriky) bez přepisování jádra;
  - vlastní loss funkce (vážené, jinak korigované) jako první-třídní rozšíření;
  - snadné střídání datasetů a sad feature/label sloupců;
  - YAML configy zůstávají (čitelnost, dohledatelnost) – uživateli se ten systém líbí;
  - **automatizace je klíčová**: časté běhy Optuny, plné tréninky různých velikostí, hyperparametrů a seedů, vše spouštěné skripty/CLI;
  - vyhnout se chaosu v Jupyter noteboocích – výzkumná logika patří do balíčku, notebooky jen na ad-hoc nahlížení;
  - musí běžet na Linuxu (tento stroj) i Windows 10/11 → žádné PowerShell/bash skripty jako součást workflow, vše v Pythonu, `pathlib`, multiprocessing-safe.
- 2026-09-11: **Styl spolupráce (učení je hlavní cíl vedle výzkumu):**
  - u každého modulu nejdřív vysvětlit, co bylo v původním repu špatně a proč to nyní děláme jinak;
  - kód psaný Claudem musí mít detailní vysvětlení (ideálně význam každého řádku);
  - někdy nechat psát kód uživatele: Claude dá přesnou specifikaci / řádek po řádku, uživatel přepisuje, Claude reviewuje;
  - přesný režim se ještě doladí (viz otevřené otázky).
- 2026-09-11: Uživatel zvažuje **stavbu od nuly** s inspirací a přenesením dobrých kousků ze starého repa,
  místo refaktoru na místě. Claude doporučuje totéž (viz níže). Čeká se na potvrzení a na volbu
  „nový repozitář“ vs. „nový balíček v tomto repu + `legacy/`“.

- 2026-09-11: **Profil uživatele:** juniorní ML/DL engineer, hlavní cíl je praxe a přenositelné know-how.
  Není vázaný na knihovny ze starého repa – vhodnější nástroje jsou vítané. Roadmapa znalostí a nástrojů: `docs/roadmap.md`.
- 2026-09-11: **Odpovědi na otázky z analýzy:**
  - Framework: uživatel je v tomto nováček, nechal volbu na doporučení. Claude doporučuje **čistý PyTorch**
    s vlastní tenkou trénovací smyčkou a callbacky (učení + výkon na malých MLP), **bez Lightning a Hydra**.
    Lightning se naučí později jako srovnávací cvičení (port jednoho experimentu). Vlastní minimalistický
    systém template + override + `--set` místo Hydry.
  - Data: **CSV zůstává zdrojem pravdy** (generátor dat v jiném projektu ho produkuje, je čitelné).
    Framework si z něj poprvé vytvoří binární cache (`.npy`, memmap-ovatelné), klíčovanou hashem
    (cesta, sloupce, mtime), a dál čte jen z cache. Parquet zvážit později jako výstup generátoru.
  - gaia2: **mimo scope**. Primární je tento Linux stroj; návrh musí zůstat otevřený nasazení na Windows 10/11.
  - Tracking: uživatel se chce naučit TensorBoard, W&B i MLflow. Návrh: vlastní rozhraní `Tracker`
    (log_scalar / log_params / log_artifact) s backendy: lokální JSON/CSV (vždy), TensorBoard, W&B.
    Pořadí učení: lokální → TensorBoard → W&B; MLflow jen ukázat.
  - Data preprocessing žije v jiném projektu; framework **předpokládá předzpracovaná CSV**.
    Ve frameworku budou jen pomocné „derived features“ (fyzikálně dopočitatelné z existujících sloupců,
    např. invarianty tenzoru gradientu rychlosti), řešené jako registr funkcí nad načtenými sloupci.
  - Normalizace: uživatel se domnívá, že CSV už byla normalizovaná. **Ověřit při prvním načtení dat**
    (framework bude logovat mean/std každého sloupce) a podle toho rozhodnout, zda transformaci zapnout.
- 2026-09-11: **Doporučený toolchain nového frameworku (ke schválení):** `uv` (prostředí + pyproject, Linux i Windows),
  PyTorch, `pydantic` v2, `PyYAML`, `typer` (CLI), `optuna`, `numpy`/`pandas` (+ volitelně `polars` na rychlé čtení CSV),
  `tensorboard`, `wandb`, `pytest`, `ruff`, `pre-commit`.

- 2026-09-11: **Nový repozitář s novým názvem.** „MathCAS“ = Matematický ústav AV ČR (místo výzkumu), byl to
  dočasný název. Starý repo zůstane read-only reference. Název nového repa ještě vybrat (návrhy v chatu).
- 2026-09-11: **Přenos kontextu do nového repa:** zdrojem pravdy jsou soubory (`CLAUDE.md`, `docs/roadmap.md`,
  budoucí `docs/lessons.md`) – zkopírují se do nového repa. Auto-paměť Claude Code je klíčovaná git repem
  (`~/.claude/projects/<encoded-path>/memory/`), zkopíruje se do nového encoded-path adresáře.
  Historie chatu VS Code rozšíření je oddělená od CLI; nespoléhat na resume, ale na soubory.
- 2026-09-11: **Dělba psaní kódu (upřesněno):** Claude píše většinu – kostry projektu i modulů, vše předchystá
  s vysvětlením. Uživatel dopisuje vybrané části přímým přepisem z chatu nebo propojuje moduly přesně podle
  instrukcí. **Časové omezení:** framework nesmí trvat měsíc; cíl je co nejdřív „první trénink“ a pak
  iterovat. Učení ano, ale efektivně.
- 2026-09-11: Python pro nový projekt: **3.12** (jisté wheely pro torch/wandb; systémový 3.14 je příliš nový),
  instaluje `uv`. Torch s CUDA 12.x wheelem pro RTX 4060.

- 2026-09-11: **Název – požadavky:** zřejmé, že jde o identifikaci proudění tekutin (důraz na vířivost, ne výhradně)
  pomocí neuronových sítí; delší název + zkratka z iniciál. Kontext: kolegové vyvíjejí `valib` (Vortex Analysis
  LIBrary, C/Fortran); dřívější experimenty pod názvem `flow-analysis`; `flownet` používá NVIDIA.
  Kandidáti (PyPI ověřeno 2026-09-11): **NIVA** – Neural Identification of Vortices and flow Analysis (PyPI volné),
  **FLINN** – FLow Identification with Neural Networks (PyPI volné), **NEFI** – NEural Flow Identification (PyPI volné);
  `nevis`, `vine` na PyPI obsazené.
- 2026-09-11: **Kde stavět (portfolio):** Claude doporučuje pokračovat v tomto repozitáři – přejmenovat na GitHubu
  (GitHub přesměruje starou URL), označit současný stav tagem `v1-legacy`, starý kód dočasně ponechat v `legacy/`
  jako referenci pro Claude i uživatele, po dokončení odstranit. Důvod: souvislá historie ~15 měsíců je pro
  portfolio silnější než dva repozitáře; auto-paměť Claude Code a `CLAUDE.md` zůstanou na místě. Čeká na rozhodnutí.

## Otevřené otázky k rozhodnutí

- Od nuly: nový repozitář, nebo nový balíček zde a starý kód přesunout do `legacy/`?
- Kam ukládat „lekce“ (co bylo špatně / jak nově): návrh `docs/lessons.md`, aby CLAUDE.md zůstal štíhlý.

- Zachovat čistý PyTorch, nebo přejít na Lightning / vlastní Trainer třídu?
- Formát dat: zůstat u CSV, nebo jednorázově převést na Parquet / `.npy` memmap?
- Tracking experimentů: strukturovaný JSON/CSV v exp dir stačí, nebo TensorBoard / MLflow / W&B?
- Musí zůstat podpora Windows a PowerShell sweepu, nebo stačí Linux + gaia2?
- Co přesně gaia2 vyžaduje (verze torch, CUDA, omezení multiprocessingu, plánovač úloh)?
- Normalizace vstupů/cílů – záměrně vynechána?
- Přejmenované složky v `outputs/` (bez timestampu) – zachovat konvenci, nebo zavést stabilní ID experimentu?

## Co přenést ze starého repa (kandidáti na kopírování)

- `FastTensorDataLoader` (src/dataset.py) – indexace tensorů bez workerů; bude výchozí loader.
- Cache + float32 čtení CSV a split bez sklearn z větve `gaia2` (bez konfliktních značek).
- `check_dying_relus` (src/architecture.py).
- NVTX profiler (utils/profiler.py) beze změn.
- Měření inference s `cuda.synchronize` (src/evaluation.py).
- Metriky z utils/metrics.py, ale vracet floaty, ne stringy.
- Exp dir + kopie configu + hash (utils/experiment.py).
- Logika řetězeného vyhodnocení a kombinací seedů z `notebooks/evaluation.ipynb` (buňky 7 a 10) → přepsat do balíčku.
- Myšlenka template + override configů z větve `multi-level-config-management`.

## Návrh plánu refaktoru (draft, k odsouhlasení)

0. Prostředí: `pyproject.toml`, venv/uv, torch pro Linux+CUDA, opravit requirements, rozběhat test.
1. Balíček `mathcas/` s `__init__.py`, CLI (`train`, `evaluate`, `sweep`), pryč se `sys.path` hacky.
2. Data: sloučit `dataloading_refactor` + `gaia2` → načíst jednou, seed-konzistentní split,
   `FastTensorDataLoader` (data rovnou na GPU), volitelná normalizace se uloženými statistikami.
3. Trainer: jedna smyčka pro skript i notebook, callbacky (early stop, checkpoint, scheduler),
   `history.json`/`metrics.json`, volitelně AMP a `torch.compile`.
4. Config: striktní pydantic (`Literal`, `extra="forbid"`), template + override (myšlenka z
   `multi-level-config-management`), CLI overrides, multi-seed sweep v Pythonu.
5. Evaluation: přenést řetězené / multi-seed vyhodnocení z notebooku do balíčku, strukturované výstupy,
   grafy generované z nich.
6. Testy + CI + README aktualizace.
