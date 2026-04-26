# Report

This repository is our solution for Kaggle competition
`asr-2026-spoken-numbers-recognition-challenge`. The task was to recognize
spoken numbers from short audio files and output digit strings for test set.

Kaggle team name: **Strijakov Iskhakov**.

We are from Russia, so for us it was comfortable to work with Russian number
words directly. The main idea was not to build very big ASR system, but to use
the fact that domain is small: only numbers are spoken. Because of this we used
CTC model with special tokenizer for Russian numbers.

We followed the main project rules. The model was trained from scratch, without
pretrained weights. We did not use `dev` data for training, only for validation
and choosing checkpoint. Audio input for model is 16 kHz. Final model has about
3.87M trainable parameters, so it is lower than 5M limit.

## What Was Done

First we added scripts for downloading data from Kaggle and preprocessing audio.
All audio is converted to mono WAV, 16 kHz. It makes training easier, because
all files have same format and sampling rate. CSV files are copied to processed
data folder, and filenames are changed to `.wav`.

Then we implemented baseline ASR pipeline:

- log-mel audio features;
- tokenizer for Russian number words / compact number tokens;
- small `Conv + BiGRU + CTC` model;
- training loop with CER metric on dev set;
- speaker-level CER and in-domain / out-of-domain CER;
- inference script for checking predictions;
- submission script which restores original test filenames and writes
  `submission.csv`.

After first baseline we added more experiment tools. There is a YAML config for
hyperparameter sweep, where we can try different `encoder_dim`, learning rate
and other parameters. Also we added saving checkpoints for every epoch, so it is
possible to select best epoch after training.

For final run we used a stronger configuration:

- tokenizer: `russian_number_compact`;
- model: `ConvBiGRUCTC`;
- encoder dimension: 224;
- encoder layers: 3;
- batch size: 256;
- eval batch size: 512;
- learning rate: 0.0008;
- weight decay: 0.01;
- scheduler: cosine with warmup;
- augmentations: speed perturbation and SpecAugment masks;
- training epochs: 70;
- device: CUDA.

The final model is saved in `artifacts/single_best/best.pt`.

## Experiments And Results

At the start the model was almost random. On epoch 1 dev CER was about 99.9%.
Then training became much better. Around epoch 14 dev CER was already near
26.5%, and after epoch 20 it was near 12-13%.

The best result by main metric in our local validation was:

- epoch: 63;
- dev CER: 7.2767%;
- dev primary harmonic mean CER: 6.2811%;
- in-domain CER: 5.1528%;
- out-of-domain CER: 8.0420%.

If looking only at usual dev CER, epoch 70 was very close and a little better:

- dev CER: 7.2737%;
- dev primary harmonic mean CER: 6.3043%;
- in-domain CER: 5.1917%;
- out-of-domain CER: 8.0240%.

So the model became stable after about 48-50 epochs. Later epochs changed result
only a little. Out-of-domain speakers were still harder than in-domain speakers,
which is expected because speaker variation is important for speech tasks.

The processed dataset sizes used locally were:

- train: 12553 rows;
- dev: 2265 rows.

We used dev only for validation. Test labels are not available, as usual in
Kaggle competition. Final `artifacts/submission.csv` was created by using
Kaggle sample submission order and has 2582 rows in our local artifact.
The best Kaggle score for our team was **5.410**.

## Submission History

We did not save full Kaggle leaderboard history inside the repository, so here is
the history from local work and artifacts.

1. First local baseline was simple CTC with Russian number words tokenizer. It
   was mainly for checking that data loading, training and decoding work.
2. Then we prepared sweep code to test model size and learning rate. This helped
   to understand that bigger GRU and better LR schedule gives better dev CER.
3. Final training was `artifacts/single_best`, 70 epochs, compact tokenizer,
   augmentation and cosine scheduler. Best local metric was around 6.28 by
   primary harmonic mean CER.
4. Final Kaggle file was generated as `artifacts/submission.csv` and also a
   public Kaggle notebook was prepared in `notebooks/public_submission_notebook.ipynb`.
   The notebook downloads repository code and `best.pt` checkpoint from GitHub
   release, preprocesses test audio, builds `submission.csv`, and validates its
   shape and columns.
5. Kaggle leaderboard scores were checked on the platform, but exact submission
   history was not saved in the repository. Our best Kaggle score was 5.410.
   This is not very good for history, and next time we should write down all
   submission scores after every upload.

For Kaggle submission the produced file has columns:

- `filename`;
- `transcription`.

Some preview predictions from the final file look like normal digit strings,
for example `461694`, `207730`, `79187`. There is also fallback to `0` if model
returns empty prediction.

## Problems And Possible Improvements

The main weak point is that this is still quite small ASR model. It works good
for this narrow number task, but probably can be improved. We think next steps
could be:

- use beam search or language rules for valid numbers instead of only argmax CTC;
- make more careful augmentations for different speakers;
- try ensembling several checkpoints;
- tune postprocessing for cases where model confuses tens and hundreds;

In general we think the repository has full reproducible pipeline: download data,
preprocess audio, train model, inspect dev predictions, and create Kaggle
submission. It also has public submission notebook which imports code and model
weights from GitHub release, as required for Kaggle. The final local validation
result is not perfect, but it is much better than first baseline and enough for
a normal working solution.
