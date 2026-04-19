# spoken-numbers-recognition

Распознавание произнесённых чисел (соревнование ASR). Ниже — как скачать данные с Kaggle и подготовить аудио для обучения.

## Требования

- [uv](https://docs.astral.sh/uv/) и Python 3.11+ (версия указана в `.python-version`).
- Учётная запись Kaggle и API-токен.

## Установка зависимостей

Из корня репозитория:

```bash
uv sync
```

## Скачивание датасета

1. Создайте файл `.env` по образцу `.env.example` и укажите учётные данные Kaggle:
   - `KAGGLE_API_TOKEN` — [API Tokens](https://www.kaggle.com/settings).

2. Запустите скрипт загрузки:

   ```bash
   uv run python src/dataset/download.py
   ```

   Скрипт вызовет `kagglehub.competition_download` для соревнования `asr-2026-spoken-numbers-recognition-challenge` и выведет путь к папке с данными (часто что-то вроде `data/.../asr-2026-spoken-numbers-recognition-challenge`).

3. В этой папке должны лежать `train.csv`, `dev.csv`, `test.csv` и каталоги `train/`, `dev/`, `test/` с аудиофайлами.

## Preprocessing аудио

Скрипт `src/dataset/preprocess_audio.py`:

- приводит все файлы к **моно**;
- **ресэмплирует** в **16 kHz** (16000 Гц);
- сохраняет **WAV 16-bit PCM** с той же относительной структурой путей, расширение в имени — `.wav`;
- по умолчанию **обрезает** записи длиннее **`--clip-seconds`** (7 с) для сплитов из **`--clip-splits`** (по умолчанию: `train`, `dev`, `test`). Чтобы обрезать только **dev** и **test**, передайте `--clip-splits dev,test`.

Пример (подставьте `--data-root` из вывода скрипта загрузки):

```bash
uv run python src/dataset/preprocess_audio.py /
  --data-root путь/к/asr-2026-spoken-numbers-recognition-challenge /
  --output-dir data/processed_16k /
  --copy-csv /
  --overwrite
```

Для обучения укажите корень данных как **`--output-dir`** (если использовали `--copy-csv`), чтобы пути из CSV совпадали с расположением файлов.

**MP3:** декодирование идёт через librosa; для некоторых систем может понадобиться **ffmpeg** в `PATH`.

## Baseline обучение

В репозитории есть первый baseline для CTC-распознавания по русским токенам числительных:

- tokenizer по русским компонентам числа: `сто`, `двадцать`, `пять`, `тысяча` и т.д.;
- log-mel признаки;
- компактная `Conv + BiGRU + CTC` модель;
- валидация по `dev` с `CER`, метриками по `spk_id` и сохранением русской расшифровки.

Запуск из корня репозитория:

```bash
uv run python -m src.train_baseline \
  --data-root data/processed_16k \
  --output-dir artifacts/baseline_ctc_words \
  --device mps \
  --batch-size 8 \
  --eval-batch-size 16 \
  --epochs 5
```

Артефакты сохраняются в `--output-dir`:

- `best.pt` — лучший чекпоинт по `dev CER`;
- `best_preview.json` — примеры предсказаний;
- `history.json` — история обучения по эпохам.

Это именно стартовый baseline. Следующие улучшения логично делать поверх него: аугментации, более сильный декодинг и альтернативную токенизацию для числительных.

## Sweep По Гиперпараметрам (Размер Модели + LR)

Для быстрого перебора baseline-конфигураций добавлены:

- Python-обертка: `src/sweep_baseline.py`;
- bash-скрипт запуска: `scripts/run_baseline_sweep.sh`.
- YAML-конфиг sweep: `configs/sweep_baseline.yaml`.

Запуск:

```bash
bash scripts/run_baseline_sweep.sh
```

Все задаётся через YAML (включая метод поиска: `grid_search`, `random_search`, `optuna`):

```yaml
output_root: artifacts/sweeps
run_name: ""

train:
  module: src.train_baseline
  args:
    data_root: data/processed_16k
    epochs: 16
    batch_size: 24
    eval_batch_size: 48
    max_parameters: 5000000
    save_all_checkpoints: true

sweep:
  method: grid_search
  fail_fast: true
  metric:
    key: dev_primary_hmean_cer
    fallback_key: dev_cer
    mode: min
  params:
    encoder_dim:
      type: int
      values: [128, 160, 192, 224, 256, 288]
    learning_rate:
      type: float
      values: [0.001, 0.0005, 0.0003, 0.0002]
  random_search:
    n_trials: 20
    seed: 42
  optuna:
    n_trials: 30
    seed: 42
    sampler: tpe
```

Принцип: чтобы добавить новый гиперпараметр в sweep, достаточно добавить его в `sweep.params` (и чтобы соответствующий `--arg` поддерживался `src.train_baseline`).

Что делает sweep:

- запускает серию обучений с выбранной стратегией (`grid_search` / `random_search` / `optuna`);
- всегда включает сохранение чекпоинта **каждой эпохи**;
- отфильтровывает конфиги с числом параметров больше `5_000_000`;
- сохраняет таблицы в `artifacts/sweeps/<run_name>/`:
  - `checkpoint_metrics_dev.csv` — строки по каждому чекпоинту (эпохе) с dev-метриками;
  - `run_best_dev.csv` — лучший чекпоинт в каждом запуске;
  - `skipped_by_param_limit.csv` — пропущенные конфиги из-за лимита параметров;
  - `failed_trials.csv` — неуспешные прогоны (если `fail_fast: false`);
  - `planned_trials.csv` — план запусков в режиме `--dry-run`;
  - `sweep_summary.json` — краткий итог sweep и путь к лучшему чекпоинту.

Выбирайте чекпоинт по `dev_primary_hmean_cer` (или `dev_cer`, если primary недоступна), затем запускайте сабмит:

```bash
uv run python -m src.make_submission \
  --checkpoint <path-to-best-checkpoint.pt> \
  --data-root data/processed_16k \
  --sample-submission data/competitions/asr-2026-spoken-numbers-recognition-challenge/sample_submission.csv \
  --output artifacts/submission.csv
```

## Inference И Разбор Ошибок

Чтобы прогнать лучший чекпоинт по `dev` и сохранить локально таргеты, предсказания и `CER` по каждому примеру:

```bash
uv run python -m src.infer_baseline \
  --checkpoint artifacts/baseline_ctc_words/best.pt \
  --data-root data/processed_16k \
  --split dev \
  --output-dir artifacts/baseline_ctc_words/inference_dev
```

Скрипт сохранит:

- `dev_predictions.csv` — `filename`, `speaker`, `target`, `prediction`, длины и `cer`;
- `dev_summary.json` — средний `CER` и метрики по спикерам;
- `dev_worst_examples.json` — 20 худших примеров для быстрого разбора.

## Сабмит на Kaggle

Сборка `submission.csv` по тесту (порядок строк выравнивается под `sample_submission.csv`, если размер совпадает):

```bash
uv run python -m src.make_submission \
  --checkpoint artifacts/baseline_ctc_words/best.pt \
  --data-root data/processed_16k \
  --sample-submission data/competitions/asr-2026-spoken-numbers-recognition-challenge/sample_submission.csv \
  --output artifacts/submission.csv
```
