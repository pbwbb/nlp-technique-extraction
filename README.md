# Extração de Menções de Técnicas de Ataque em Notícias de Segurança

Projeto de NLP (NER) que detecta, em notícias de cibersegurança, trechos de texto que
descrevem uma **técnica de ataque** (rótulo único `TECHNIQUE`), comparando um modelo
clássico (CRF) com um transformer de domínio (SecureBERT) sob as mesmas condições de
treino, teste e métrica.

Autores: Artur, Marina & Pedro.
Apresentação final: [`PLN.pptx`](PLN.pptx).

## Escopo

- **Dentro do escopo:** detectar o span de texto que menciona uma técnica de ataque
  (ex.: "deployed a web shell") e comparar CRF vs SecureBERT por P/R/F1 em nível de span.
- **Fora do escopo:** mapear cada menção para um ID do MITRE ATT&CK (ex. T1566) ou
  classificação multi-rótulo de táticas/técnicas — isso é o que o rcATT (Legoy et al.,
  2020) ataca; aqui ficamos na detecção.

## Esquema de rótulos

Durante a anotação, cada menção de técnica recebe um de dois rótulos (guia v0.4,
critério sintático):

- `TECHNIQUE_EXP` — ação descrita como **verbo + objeto** (ex. "deployed a web shell",
  "deletes shadow copies").
- `TECHNIQUE_IMP` — todo o resto: forma nominal, nominalização, estado (ex.
  "spear-phishing emails", "scheduled task").

**Só `TECHNIQUE_EXP` entra no treino/avaliação dos modelos.** Menções implícitas exigem
inferência de mundo e derrubam a concordância entre anotadores, o que contaminaria a
comparação (decisão informada por AnnoCTR, Lange et al., 2024).

## Estrutura do repositório

```
├── src/
│   ├── build_corpus.py        # Etapa 1 — coleta: raspa The Hacker News (label "Cyber Attack")
│   ├── run_harness_crf.py     # Etapas 5–8 — features manuais, CRF, split, avaliação
│   └── train_securebert.py    # Etapas 5–8 — fine-tuning do SecureBERT, mesmo split/métrica
├── data/
│   ├── raw/
│   │   ├── corpus.jsonl       # 150 artigos brutos (text, title, url, date), sem rótulos
│   │   └── txt/               # os mesmos 150 artigos em .txt (inspeção manual)
│   ├── corpus_anotado_150.jsonl   # 150 docs rotulados: 135 silver + 15 gold
│   ├── anotacao_piloto_pedro.jsonl  # anotador A, lote piloto (15 docs)
│   ├── pilot_tuco_v04.jsonl         # anotador B, mesmo lote piloto
│   └── concordancia_exp_imp_v04.md  # relatório de concordância (Kappa de Cohen)
├── results/
│   ├── split.json             # split treino/teste único, compartilhado pelos dois modelos
│   ├── metrics_crf.json       # P/R/F1 do CRF no teste gold
│   └── pred_crf.json          # predições do CRF por documento (spans previstos vs gold)
└── PLN.pptx                # apresentação final
```

## Pipeline (8 etapas)

1. **Coleta** — `src/build_corpus.py` raspa artigos do The Hacker News sob o label
   "Cyber Attack" e consolida em JSONL (um documento por linha: `text`, `title`, `url`,
   `date`). Fonte única — limitação declarada abaixo.
2. **Anotação** — rótulo `TECHNIQUE` no Doccano (Nakayama et al., 2018).
3. **Qualidade** — concordância entre anotadores no lote piloto (15 docs), ver
   `data/pilot_agreement/`.
4. **Pré-processamento** — limpeza e tokenização (spaCy) para o esquema BIO.
5. **Atributos** — features manuais por token para o CRF (forma da palavra, sufixos,
   POS/tag, janela ±2).
6. **Modelagem** — CRF (`sklearn-crfsuite`) e fine-tuning do SecureBERT.
7. **Comparação** — mesmo `split.json` (treino=silver 135 / teste=gold 15) e mesma
   métrica para os dois modelos.
8. **Avaliação** — P/R/F1 com match exato de offsets de caractere, só `TECHNIQUE_EXP`.

### Concordância entre anotadores

Medida no lote piloto (15 documentos, 202 tokens em que ambos os anotadores marcaram
uma menção): Kappa de Cohen = **0,8885**, contra um corte de 0,90 definido no guia —
0,0115 abaixo. Ver `data/pilot_agreement/concordancia_exp_imp_v04.md` para a matriz de
confusão completa e a análise dos desacordos.

## Resultados

Teste gold: 15 documentos, 112 entidades `TECHNIQUE_EXP`. Métrica: F1 de span com match
exato de offsets.

| Modelo     | P     | R     | F1    | TP | FP | FN |
|------------|-------|-------|-------|----|----|----|
| CRF        | 0,514 | 0,161 | 0,245 | 18 | 17 | 94 |
| SecureBERT | —     | —     | —     | —  | —  | —  |

O SecureBERT precisa de GPU e acesso ao Hugging Face Hub (ver `src/train_securebert.py`)
e por isso não foi rodado neste ambiente; segundo a apresentação, o resultado observado
no experimento foi F1 = 0,193 (13/112), ligeiramente abaixo do CRF. Rode
`train_securebert.py` para reproduzir `metrics_securebert.json` e `pred_securebert.json`.

**Leitura honesta:** a diferença entre os modelos é de 5 spans corretos em 112 — margem
pequena e não robusta. O recall baixo nos dois modelos (14–16%) é coerente com treinar
em dados silver e cobrar fronteira exata de span.

## Como reproduzir

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 1. Coleta (opcional — o corpus já está em data/raw/)
python src/build_corpus.py --target 150 --outdir data/raw

# 6–8. CRF: treina em silver, avalia em gold, gera o split compartilhado
python src/run_harness_crf.py \
    --data data/annotated/corpus_anotado_150.jsonl \
    --split-out results/split.json \
    --metrics-out results/metrics_crf.json

# 6–8. SecureBERT: usa o MESMO split e a MESMA métrica do CRF (precisa de GPU/HF Hub)
python src/train_securebert.py \
    --data data/annotated/corpus_anotado_150.jsonl \
    --split results/split.json \
    --metrics-out results/metrics_securebert.json
```

## Limitações declaradas

- **Fonte única** — corpus vem só do The Hacker News, o que limita a generalização.
- **Camada silver** — 135 dos 150 documentos são pré-anotação por modelo, sem revisão
  humana completa.
- **Teste pequeno** — 15 documentos gold, 112 entidades: números instáveis.
- **Concordância parcial** — Kappa e F1 medidos só nos 15 documentos do piloto gold.
- **Viés de família** — se a camada silver veio de um transformer, poderia favorecer o
  SecureBERT; ainda assim o CRF saiu à frente nesta rodada.

## Trabalhos futuros

- Revisar a camada silver para gold, transformando pré-anotação em verdade humana.
- Ampliar o conjunto de teste gold e recalcular a concordância.
- Diversificar as fontes além do The Hacker News.
- Explorar CyBERT com o corpus APTNER como alternativa.

## Referências

- Aghaei, E. et al. (2022). *SecureBERT: A Domain-Specific Language Model for Cybersecurity.*
- Artstein, R.; Poesio, M. (2008). *Inter-Coder Agreement for Computational Linguistics.*
- Cohen, J. (1960). *A Coefficient of Agreement for Nominal Scales.*
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.*
- Honnibal, M.; Montani, I. *spaCy: Industrial-Strength NLP.*
- Lafferty, J.; McCallum, A.; Pereira, F. (2001). *Conditional Random Fields.*
- Lange, L. et al. (2024). *AnnoCTR: Dataset for Entities, Tactics and Techniques in CTI.*
- Legoy, V. et al. (2020). *Automated Retrieval of ATT&CK Tactics and Techniques (rcATT).*
- Nakayama, H. et al. (2018). *doccano: Text Annotation Tool for Humans.*
- Ramshaw, L.; Marcus, M. (1995). *Text Chunking Using Transformation-Based Learning.*
- Strom, B. E. et al. (2018). *MITRE ATT&CK: Design and Philosophy.*
- Tjong Kim Sang, E.; De Meulder, F. (2003). *Introduction to the CoNLL-2003 Shared Task.*
