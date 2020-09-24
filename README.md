# Cross CRFAE

Code for the paper _Unsupervised Cross-Lingual Adaptation of Dependency Parsers Using CRF Autoencoders_ in the findings of EMNLP 2020.

## Requirements

- `python`: 3.7.0
- [`pytorch`](https://github.com/pytorch/pytorch): 1.3.0
- [`transformers`](https://github.com/huggingface/transformers): 2.1.1

## Usage

source:

```
python run.py train -p -d=$cuda -f=exp/source --feat=tag --crf --n_mlp_arc 50 --n_lstm_hidden 200 --n_lstm_layers 3 --n_embed 150 --output log/source --berts --seed 1
```

target (with W_Reg):

```
python run.py train -p -d=$cuda -f=exp/target --feat=tag --unsupervised --max_len 40 --lang ar  --n_mlp_arc 50 --n_lstm_hidden 200 --n_lstm_layers 3 --n_embed 150 --output log/target --epochs 1 --load "/path/to/cross-crfae/exp/source/model" --W_Reg --W_beta 1e8 --freeze_feat_emb --crf --bert --seed 1
```
