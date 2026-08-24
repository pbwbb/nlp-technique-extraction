"""
train_securebert.py
Lado SecureBERT do harness de comparacao CRF vs SecureBERT.

Usa EXATAMENTE o mesmo split que o CRF (le split.json gerado por
run_harness_crf.py) e a MESMA metrica (span com match exato de offsets de
caractere, so TECHNIQUE_EXP), para que o F1 seja diretamente comparavel.

IMPORTANTE: este script precisa de acesso ao Hugging Face Hub para baixar os
pesos do SecureBERT. Rode no Colab ou em maquina com internet para o HF.
(No sandbox onde o CRF rodou, huggingface.co nao esta acessivel.)

Instalacao:
    pip install "transformers>=4.40" "datasets" "torch" "seqeval"

Uso:
    python train_securebert.py \
        --data corpus_anotado_150_final.jsonl \
        --split split.json \
        --model ehsanaghaei/SecureBERT \
        --epochs 5

Documentos longos sao tratados por janelas deslizantes (stride), sem descartar
conteudo alem de 512 tokens. As predicoes de span sao unidas por documento.
"""
import argparse, json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer)

LABELS = ["O", "B-TECHNIQUE", "I-TECHNIQUE"]
L2I = {l: i for i, l in enumerate(LABELS)}
TREINAVEL = "TECHNIQUE_EXP"

# ---------- dados ----------
def load_corpus(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        l = l.strip()
        if not l: continue
        o = json.loads(l)
        exp = [(s, e) for (s, e, lab) in o.get("label", []) if lab == TREINAVEL]
        out[o["id"]] = {"text": o["text"], "exp_spans": exp}
    return out

def encode_doc(tok, text, exp_spans, max_len=512, stride=128, for_train=True):
    """Retorna janelas: cada uma com input_ids/attention_mask/offset_mapping (+ labels no treino)."""
    enc = tok(text, truncation=True, max_length=max_len, stride=stride,
              return_overflowing_tokens=True, return_offsets_mapping=True,
              padding="max_length")
    janelas = []
    for w in range(len(enc["input_ids"])):
        offsets = enc["offset_mapping"][w]
        item = {"input_ids": enc["input_ids"][w],
                "attention_mask": enc["attention_mask"][w],
                "offsets": offsets}
        if for_train:
            labels = []
            prev_in = None
            for (a, b) in offsets:
                if a == b:  # token especial / padding
                    labels.append(-100); prev_in = None; continue
                dentro = next(((s, e) for (s, e) in exp_spans if a >= s and b <= e), None)
                if dentro is None:
                    labels.append(L2I["O"]); prev_in = None
                elif dentro == prev_in:
                    labels.append(L2I["I-TECHNIQUE"])
                else:
                    labels.append(L2I["B-TECHNIQUE"]); prev_in = dentro
            item["labels"] = labels
        janelas.append(item)
    return janelas

class WinDS(Dataset):
    def __init__(self, janelas): self.j = janelas
    def __len__(self): return len(self.j)
    def __getitem__(self, i):
        w = self.j[i]
        d = {"input_ids": torch.tensor(w["input_ids"]),
             "attention_mask": torch.tensor(w["attention_mask"])}
        if "labels" in w: d["labels"] = torch.tensor(w["labels"])
        return d

# ---------- decodificacao BIO(subword) -> spans de caractere ----------
def bio_to_char_spans(pred_ids, offsets):
    out = set(); i = 0; n = len(pred_ids)
    while i < n:
        a, b = offsets[i]
        if a == b: i += 1; continue
        if LABELS[pred_ids[i]] == "B-TECHNIQUE":
            st = a; en = b; i += 1
            while i < n and (offsets[i][0] != offsets[i][1]) and LABELS[pred_ids[i]] == "I-TECHNIQUE":
                en = offsets[i][1]; i += 1
            out.add((st, en))
        else:
            i += 1
    return out

def score_exact(gold_sets, pred_sets):
    tp = fp = fn = 0
    for g, p in zip(gold_sets, pred_sets):
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"P": round(P, 3), "R": round(R, 3), "F1": round(F, 3), "TP": tp, "FP": fp, "FN": fn}

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True, help="split.json gerado pelo harness do CRF")
    ap.add_argument("--model", default="ehsanaghaei/SecureBERT")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--metrics-out", default="metrics_securebert.json")
    a = ap.parse_args()

    corpus = load_corpus(a.data)
    split = json.load(open(a.split))
    train_ids, test_ids = split["train_ids"], split["test_ids"]
    print(f"treino={len(train_ids)} (silver) | teste={len(test_ids)} (gold)")

    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)

    train_win = []
    for i in train_ids:
        d = corpus[i]
        train_win += encode_doc(tok, d["text"], d["exp_spans"], for_train=True)
    model = AutoModelForTokenClassification.from_pretrained(
        a.model, num_labels=len(LABELS), id2label={i: l for l, i in L2I.items()}, label2id=L2I)

    args = TrainingArguments(output_dir="sb_out", num_train_epochs=a.epochs,
                             per_device_train_batch_size=8, learning_rate=a.lr,
                             logging_steps=50, save_strategy="no", report_to=[])
    Trainer(model=model, args=args, train_dataset=WinDS(train_win)).train()

    # avaliacao: span exato no teste gold, unindo janelas por documento
    model.eval()
    gold_sets = []; pred_sets = []; dump = []
    for i in test_ids:
        d = corpus[i]
        janelas = encode_doc(tok, d["text"], d["exp_spans"], for_train=False)
        pred_spans = set()
        for w in janelas:
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([w["input_ids"]]),
                               attention_mask=torch.tensor([w["attention_mask"]])).logits[0]
            pred_ids = logits.argmax(-1).tolist()
            pred_spans |= bio_to_char_spans(pred_ids, w["offsets"])
        gold = set(d["exp_spans"])
        gold_sets.append(gold); pred_sets.append(pred_spans)
        dump.append({"id": i, "pred_exp_spans": sorted(pred_spans), "gold_exp_spans": sorted(gold)})
    m = score_exact(gold_sets, pred_sets)
    print("\n=== SecureBERT | treino=silver(135) teste=gold(15) | span EXATO, TECHNIQUE_EXP ===")
    print(m)
    json.dump({"modelo": "SecureBERT", "metrica": "span exato EXP", "split": "silver->gold", **m},
              open(a.metrics_out, "w"), ensure_ascii=False, indent=2)
    json.dump(dump, open("pred_securebert.json", "w"), ensure_ascii=False, indent=2)
    print(f"metricas em {a.metrics_out} | predicoes em pred_securebert.json")

if __name__ == "__main__":
    main()
