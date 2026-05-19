# Deep Learning Assignment-3 : Implementing a Transformer for Machine Translation

Author : G C V Sairam, DA25M012  
Github link : https://github.com/da25m012/DL_Assgn3  
Wandb Report link : https://wandb.ai/da25m012-indian-institute-of-technology-madras/da6401-a3/reports/DA6401-Assignment-3-Report--VmlldzoxNjkzMzUzNQ

---

## Project Structure

```
├── model.py          # Transformer architecture (MHA, Encoder, Decoder, PE)
├── train.py          # Training loop, label smoothing loss, BLEU evaluation
├── dataset.py        # Multi30k dataset loading and vocabulary building
├── lr_scheduler.py   # Noam learning rate scheduler
```

---

## Setup

```bash
pip install torch spacy datasets wandb tqdm
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

---

## Model Configuration

| Hyperparameter | Value |
|---|---|
| d_model | 512 |
| Encoder/Decoder Layers | 6 |
| Attention Heads | 8 |
| d_ff | 2048 |
| Dropout | 0.1 |
| Warmup Steps | 4000 |
| Label Smoothing | 0.1 |
| Batch Size | 128 |
| Max Epochs | 20 |

---

## Training

```bash
python train.py
```

Logs training loss, validation loss, learning rate, and test BLEU to W&B.

---

## Key Design Choices

- **Pre-LayerNorm** used in encoder and decoder layers for training stability.
- **Sinusoidal positional encoding** registered as a buffer (non-trainable), outperforming learned embeddings by 6 BLEU points on this dataset.
- **Greedy decoding** used at inference time.
- **Xavier uniform initialization** applied to all parameters with `dim > 1`.
