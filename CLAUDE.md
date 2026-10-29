# VortexNet – pracovní poznámky pro Claude

Sdílený zápisník pro spolupráci na stavbě frameworku. Ukládají se sem rozhodnutí z konzultací
a detaily, které nejsou odvoditelné z kódu. Při každé nové dohodě s uživatelem doplň sekci
**Rozhodnutí** (s datem). Soubor držet štíhlý – delší texty patří do `docs/`.

## O co jde

Od nuly stavěný nástupce frameworku **MathCAS** (`../MathCAS`, read-only reference) pro trénování
malých neuronových sítí na tabulárních CFD datech (stlačitelné proudění).

- **Vstupy:** 9 složek tenzoru gradientu rychlosti `A11 … A33`.
- **Cíle:** `Shear`, `ResStrain`, `ResVort` (dekompozice na smyk + reziduální strain/vorticita).
- **Data:** předzpracovaná CSV z jiného projektu (60M řádků trénink, 5M test, varianta s `PredShear`).
  Datasety a výstupy se nikdy necommitují.
- **Výzkumná otázka:** multitask (3 cíle najednou) vs. single-task vs. řetězení dvou modelů;
  srovnání MSE vs. počet parametrů vs. inference čas, přes více seedů.
- Pojmenování experimentů, známé problémy starého kódu a seznam „co přenést“ jsou v
  `docs/mathcas-analysis.md`.

## Soubory, které drží kontext

| Soubor | Obsah |
|---|---|
| `CLAUDE.md` | tento soubor: rozhodnutí, otevřené otázky |
| `docs/roadmap.md` | fáze stavby + koncepty, nástroje a studijní materiály ke každé |
| `docs/lessons.md` | u každého modulu: co bylo v MathCAS špatně, proč, jak nově |
| `docs/mathcas-analysis.md` | archivní kopie analýzy starého repa (2026-09-11), neupravovat |

## Styl spolupráce

- Učení je hlavní cíl vedle výzkumu, ale framework nesmí trvat měsíc – cíl je co nejdřív
  „první trénink“ a pak iterovat.
- U každého modulu nejdřív vysvětlit, co bylo ve starém repu špatně a proč to nyní děláme jinak
  (zapsat do `docs/lessons.md`), potom kód s vysvětlením (ideálně význam každého řádku).
- Claude píše většinu (kostry projektu i modulů). Uživatel dopisuje vybrané části přepisem
  z chatu nebo propojuje moduly podle přesných instrukcí; Claude reviewuje. Osvědčený vzor:
  Claude napíše první instanci vzoru, uživatel druhou.
- Nedumpovat velké nevysvětlené soubory.
- Komunikace česky; kód, komentáře v kódu a README anglicky.

## Požadavky na framework (2026-09-11)

- Co nejvíc lightweight, přehledný, rozšiřitelný.
- Regrese i klasifikace (jiná loss, výstupní vrstva, metriky) bez přepisování jádra.
- Vlastní loss funkce (vážené, multitask) jako první-třídní rozšíření.
- Snadné střídání datasetů a sad feature/label sloupců.
- YAML configy zůstávají (čitelnost, dohledatelnost).
- Automatizace: Optuna, plné tréninky různých velikostí/hyperparametrů/seedů, vše ze skriptů/CLI.
- Výzkumná logika patří do balíčku, notebooky jen na ad-hoc nahlížení.
- Linux (tento stroj) i Windows 10/11: žádné shell skripty ve workflow, vše v Pythonu, `pathlib`,
  multiprocessing-safe.

## Prostředí

- Fedora Linux, systémový Python 3.14.7 (příliš nový pro torch wheely), `uv` zatím nenainstalován.
- GPU: NVIDIA GeForce RTX 4060 Laptop, 8 GB.
- Vzdálený stroj gaia2: mimo scope.

## Rozhodnutí

- 2026-09-11: Architektura: **čistý PyTorch**, vlastní tenká trénovací smyčka s callbacky; bez Lightning
  a Hydra (Lightning později jako srovnávací cvičení). Vlastní systém config template + override + `--set`.
- 2026-09-11: Data: **CSV je zdroj pravdy**; framework si poprvé vytvoří binární cache (`.npy`, memmap)
  klíčovanou hashem (cesta, sloupce, mtime) a dál čte jen z ní. Celý dataset jako tensor (na GPU),
  batching indexací, žádní DataLoader workeři. Split deterministický podle seedu z configu.
- 2026-09-11: Preprocessing žije jinde; framework předpokládá předzpracovaná CSV. Ve frameworku jen
  „derived features“ (fyzikálně dopočitatelné, např. invarianty tenzoru) jako registr funkcí.
- 2026-09-11: Normalizace: uživatel se domnívá, že CSV už je normalizované. **Ověřit při prvním načtení**
  (logovat mean/std sloupců) a podle toho zapnout transformaci se uloženými statistikami.
- 2026-09-11: Tracking: vlastní rozhraní `Tracker` (log_scalar / log_params / log_artifact) s backendy
  lokální JSON/CSV (vždy), TensorBoard, W&B. Pořadí učení: lokální → TensorBoard → W&B; MLflow jen ukázat.
- 2026-09-11: Toolchain: `uv` + Python **3.12**, PyTorch (CUDA 12.x wheel), `pydantic` v2, `PyYAML`,
  `typer`, `optuna`, `numpy`/`pandas` (volitelně `polars`), `tensorboard`, `wandb`, `pytest`, `ruff`, `pre-commit`.
- 2026-10-29: **Název VortexNet, nový veřejný repozitář** `cervinka479/vortexnet` na osobním GitHubu
  (místo pokračování v repu MathCAS). MathCAS zůstává read-only reference v `../MathCAS`.
  Kontext přenesen soubory: `docs/roadmap.md` (kopie), `docs/mathcas-analysis.md` (kopie CLAUDE.md
  z MathCAS), paměť Claude Code zkopírována do adresáře tohoto repa.
- 2026-10-29: Lekce z starého kódu se zapisují do `docs/lessons.md` (jeden záznam na modul).

## Otevřené otázky

- Konvence ID experimentu: `<timestamp>_<name>` jako dřív, nebo stabilní jméno z configu + seed
  (aby se složky nemusely ručně přejmenovávat)?
- `pandas` vs. `polars` pro první načtení CSV (rozhodnout při fázi 2 podle měření).
- Jak řešit notebooky v gitu (nbstripout vs. žádné notebooky v repu).

## Plán stavby (fáze = `docs/roadmap.md`)

0. Prostředí: `uv`, `pyproject.toml`, Python 3.12, torch+CUDA, ruff, pytest, první test.
1. Config: pydantic schema (`extra="forbid"`), template + override, CLI `--set`, resolved config + hash.
2. Data: CSV → `.npy` cache, split podle seedu, tensor na GPU, batching indexací, derived features, kontrola normalizace.
3. Model/loss/task: registr, MLP, výstupní hlavy pro regresi/klasifikaci, vážené a multitask lossy, dying-ReLU diagnostika.
4. Trainer + tracking: smyčka s callbacky, `history.json`, lokální tracker → TensorBoard → W&B.
5. Automatizace: multi-seed sweep a Optuna v Pythonu.
6. Evaluation + report: řetězené/multi-seed vyhodnocení z notebooků do balíčku, grafy ze strukturovaných výstupů.
7. Testy, CI, README.
