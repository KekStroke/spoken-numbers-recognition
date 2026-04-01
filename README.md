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
