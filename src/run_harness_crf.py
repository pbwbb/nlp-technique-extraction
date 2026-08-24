"""
run_harness_crf.py
Lado CRF do harness de comparacao CRF vs SecureBERT.

Split UNICO e deterministico, compartilhado pelos dois modelos:
  treino = documentos silver (135)  |  teste = documentos gold (15)
O split e definido pela procedencia (meta.camada) e salvo em split.json,
para que o SecureBERT use exatamente os mesmos documentos.

Metrica: span (entidade) com match EXATO de offsets de caractere, so TECHNIQUE_EXP.
E a mesma metrica da concordancia inter-anotadores e a que o passo 8 pede.
"""
import argparse, json
from collections import Counter
import spacy, sklearn_crfsuite

LABELS_TREINAVEIS = {"TECHNIQUE_EXP"}
nlp = spacy.load("en_core_web_sm")

def camada(d): return (d.get("meta") or {}).get("camada", "gold")

def load(path):
    arts=[]
    for l in open(path, encoding="utf-8"):
        l=l.strip()
        if not l: continue
        o=json.loads(l)
        arts.append({"id":o.get("id"),"text":o["text"],
                     "entities":[(e[0],e[1],e[2]) for e in o.get("label",[])],
                     "camada":camada(o)})
    return arts

def tok_bio(texto, ents):
    doc=nlp(texto)
    spans=[(s,e,l) for (s,e,l) in ents if l in LABELS_TREINAVEIS]
    bio=["O"]*len(doc); ativo=None
    offs=[(t.idx,t.idx+len(t)) for t in doc]
    for i,t in enumerate(doc):
        ts,te=offs[i]
        dentro=next(((s,e) for (s,e,l) in spans if ts>=s and te<=e), None)
        if dentro is None: bio[i]="O"; ativo=None
        elif dentro==ativo: bio[i]="I-TECHNIQUE"
        else: bio[i]="B-TECHNIQUE"; ativo=dentro
    toks=[(t.text,t.pos_,t.tag_) for t in doc]
    return toks, bio, offs

def feats(sent,i):
    w,p,tg=sent[i]
    f={"bias":1.0,"word.lower()":w.lower(),"word[-3:]":w[-3:],"word[-2:]":w[-2:],
       "word.isupper()":w.isupper(),"word.istitle()":w.istitle(),"word.isdigit()":w.isdigit(),
       "word.has_hyphen":"-" in w,"pos":p,"tag":tg}
    for off in (-2,-1,1,2):
        j=i+off
        if 0<=j<len(sent):
            w2,p2,_=sent[j]; pre=f"{off:+d}:"
            f.update({pre+"word.lower()":w2.lower(),pre+"pos":p2,pre+"istitle":w2.istitle()})
        else: f["BOS" if off<0 else "EOS"]=True
    return f
def s2f(s): return [feats(s,i) for i in range(len(s))]

def bio_to_spans(bio, offs):
    """decodifica runs B/I em spans de caractere (start,end)"""
    out=[]; i=0; n=len(bio)
    while i<n:
        if bio[i]=="B-TECHNIQUE":
            st=offs[i][0]; en=offs[i][1]; i+=1
            while i<n and bio[i]=="I-TECHNIQUE": en=offs[i][1]; i+=1
            out.append((st,en))
        else: i+=1
    return set(out)

def score_exact(gold_sets, pred_sets):
    tp=fp=fn=0
    for g,p in zip(gold_sets,pred_sets):
        tp+=len(g&p); fp+=len(p-g); fn+=len(g-p)
    P=tp/(tp+fp) if tp+fp else 0.0
    R=tp/(tp+fn) if tp+fn else 0.0
    F=2*P*R/(P+R) if P+R else 0.0
    return {"P":round(P,3),"R":round(R,3),"F1":round(F,3),"TP":tp,"FP":fp,"FN":fn}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--provenance",default=None,help="arquivo com meta.camada para reconstruir gold/silver por id")
    ap.add_argument("--split-out",default="split.json")
    ap.add_argument("--metrics-out",default="metrics_crf.json")
    a=ap.parse_args()

    arts=load(a.data)
    # se a procedencia foi apagada do --data, reconstroi a partir de --provenance por id
    if a.provenance:
        prov={}
        for l in open(a.provenance,encoding="utf-8"):
            l=l.strip()
            if not l: continue
            o=json.loads(l); prov[o.get("id")]=camada(o)
        for x in arts:
            x["camada"]=prov.get(x["id"], x["camada"])
    print("camadas:",dict(Counter(x["camada"] for x in arts)))
    train=[x for x in arts if x["camada"]!="gold"]     # 135 silver
    test =[x for x in arts if x["camada"]=="gold"]      # 15 gold
    if not train or not test:
        raise SystemExit(f"split invalido: treino={len(train)} teste={len(test)}. Cheque --provenance.")
    # salva split para o SecureBERT usar identico
    split={"train_ids":[x["id"] for x in train],"test_ids":[x["id"] for x in test],
           "regra":"train=silver(meta.camada=silver_pre_anotacao); test=gold(sem marca)"}
    json.dump(split,open(a.split_out,"w"),ensure_ascii=False,indent=2)
    print(f"split salvo em {a.split_out} | treino={len(train)} teste={len(test)}")

    Xtr=[]; ytr=[]
    for x in train:
        toks,bio,_=tok_bio(x["text"],x["entities"]); Xtr.append(s2f(toks)); ytr.append(bio)
    crf=sklearn_crfsuite.CRF(algorithm="lbfgs",c1=0.1,c2=0.1,max_iterations=100,all_possible_transitions=True)
    crf.fit(Xtr,ytr)

    gold_sets=[]; pred_sets=[]; preds_dump=[]
    for x in test:
        toks,bio,offs=tok_bio(x["text"],x["entities"])
        pred_bio=crf.predict([s2f(toks)])[0]
        g=set((s,e) for (s,e,l) in x["entities"] if l in LABELS_TREINAVEIS)
        p=bio_to_spans(pred_bio,offs)
        gold_sets.append(g); pred_sets.append(p)
        preds_dump.append({"id":x["id"],"pred_exp_spans":sorted(list(p)),"gold_exp_spans":sorted(list(g))})
    m=score_exact(gold_sets,pred_sets)
    print("\n=== CRF | treino=silver(135) teste=gold(15) | span EXATO, TECHNIQUE_EXP ===")
    print(m)
    json.dump({"modelo":"CRF","metrica":"span exato EXP","split":"silver->gold",**m},
              open(a.metrics_out,"w"),ensure_ascii=False,indent=2)
    json.dump(preds_dump,open("pred_crf.json","w"),ensure_ascii=False,indent=2)
    print(f"metricas em {a.metrics_out} | predicoes em pred_crf.json")

if __name__=="__main__": main()
