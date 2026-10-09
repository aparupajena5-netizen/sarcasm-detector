# Contextual Sarcasm Detection: LSTM + Attention for Live-Stream Chat

Flags sarcastic / passive-aggressive comments in fast-moving chats. A comment is judged **together with the message it replies to** (in a live stream: the previous chat message), because "Great idea." means different things after "let's fix the bug" and after "let's delete production".

## Model
```
comment ─► Embedding ─► BiLSTM ─► h1..hT ─┐
                                          ├─► context-conditioned attention ─► s ─┐
context ─► Embedding ─► BiLSTM ─► mean ─► c ┘                                     ├─► [s, c, s*c, |s-c|] ─► MLP ─► sarcastic?
                                                                                   ┘
```
* Shared BiLSTM encoder for comment and context.
* Attention weights show *which words* triggered the flag (used in the live demo and heatmaps).
* `--no-context` trains the same model without context for an ablation, so you can show that context helps.

## Data (real, from Kaggle)
[**Sarcasm on Reddit** (`danofer/sarcasm`)](https://www.kaggle.com/datasets/danofer/sarcasm), file `train-balanced-sarcasm.csv`, about 1M balanced comments with `label`, `comment` and `parent_comment` (the context).
Text is cleaned for chat-style input: lowercasing, URLs and @mentions replaced by tokens, letter stretching (`sooooo` becomes `sooo`), emojis and `!!!` kept as tokens, and the explicit `/s` tag removed so the model cannot cheat.

**Passive-aggressive note:** no public Kaggle dataset labels "passive-aggressive" directly. The sarcasm labels are the supervision, and passive-aggressive remarks overlap heavily with them. If you have your own labelled chat CSV, pass it with `--csv` (columns `label`, `comment`, optional `parent_comment`).

## Setup (VS Code)
1. Open this folder in VS Code (`File > Open Folder`). Python 3.9 to 3.12 recommended.
2. Terminal:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Then choose the `.venv` interpreter (`Ctrl+Shift+P` > *Python: Select Interpreter*).
3. **Get the dataset**, either way:
   * *Automatic:* create a free Kaggle account, then *Settings > API > Create New Token*. Either save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`), or set the newer `KAGGLE_API_TOKEN` environment variable as Kaggle shows you. The first `python train.py` then downloads the data by itself.
   * *Manual:* download the dataset zip from the Kaggle page, extract `train-balanced-sarcasm.csv` into the `data/` folder.

## Run
```bash
python train.py                      # ~200k comments, 6 epochs (CPU: roughly 10-25 min; GPU: a few min)
python train.py --max-samples 50000 --epochs 4    # quick first run
python train.py --max-samples 0      # use all ~1M comments (best accuracy, needs time / GPU)
python train.py --no-context         # ablation
python evaluate.py                   # test metrics, confusion matrix, attention heatmap
python evaluate.py --name lstm_attn_noctx
python predict.py --comment "Oh great, another 3 hour queue." --context "server is down again"
python live_demo.py                  # replays sample_chat.txt like a live stream
python live_demo.py --interactive    # you type the chat
```
You can also use the Run and Debug panel (the ready-made configurations are in `.vscode/launch.json`).

Outputs land in `outputs/`: `*.pt` (model), `*_curves.png`, `*_confusion.png`, `*_attention.png`, `*_test_metrics.json`.

## Project layout
```
train.py  evaluate.py  predict.py  live_demo.py  sample_chat.txt
src/data.py       Kaggle download, chat-style cleaning, vocab, splits
src/model.py      BiLSTM + context attention
src/inference.py  single-message scoring with attention words
src/utils.py      seeding, metrics, threshold tuning, checkpoints
```

## Expectations
Sarcasm on Reddit is a hard task; published LSTM-type baselines on this dataset typically reach about 70-75% accuracy, and results here will depend on sample size. Compare `lstm_attn_ctx` with `lstm_attn_noctx` on your own run to quantify what context adds. Reddit sarcasm is longer than typical stream chat; for best results on real stream data, fine-tune on chat you have labelled.
