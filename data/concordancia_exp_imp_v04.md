# Concordância inter-anotadores no critério EXP vs IMP — lote piloto v0.4

**Objeto de avaliação.** Este relatório mede o critério de rotulação introduzido na v0.4
(`TECHNIQUE_EXP` = verbo + objeto; `TECHNIQUE_IMP` = todo o resto): dados dois anotadores
diante da *mesma* menção de técnica, eles atribuem o mesmo rótulo?

**Escopo.** Cálculo restrito aos **202 tokens em que ambos os anotadores identificaram
uma menção**. É o conjunto em que a decisão EXP/IMP efetivamente ocorre e, portanto,
o único em que ela pode ser medida.

- Anotadores: A (pedro) e B (tuco), independentes, mesmos 15 artigos do lote piloto
- Tokenização por regex de palavra (`\w+` com hífen e apóstrofo internos)
- Guia de anotação v0.4, seções 3, 5.4 e 6

---

## 1. Resultado

| Medida | Valor |
|---|---|
| Tokens avaliados (n) | 202 |
| Concordância observada (p_o) | 0,9455 |
| Concordância esperada por acaso (p_e) | 0,5117 |
| **Kappa de Cohen** | **0,8885** |
| Corte estabelecido (guia, seção 8.3) | 0,90 |

O acaso aqui está em 0,51 — a distribuição EXP/IMP é praticamente equilibrada nos dois
anotadores —, de modo que o kappa não sofre a inflação de p_e típica de tarefas com
classe majoritária dominante. O valor de 0,8885 fica **0,0115 abaixo do corte**.

### Distribuição marginal

| | EXP | IMP |
|---|---|---|
| A (pedro) | 115 | 87 |
| B (tuco) | 118 | 84 |

As marginais são quase idênticas: não há viés sistemático de um anotador em direção a
uma das classes.

### Matriz de confusão (tokens)

| A \ B | EXP | IMP |
|---|---|---|
| **EXP** | 111 | 4 |
| **IMP** | 7 | 80 |

---

## 2. Análise dos desacordos

Os 11 tokens divergentes vêm de **apenas 3 spans**, e todos os 3 caem em questões que o
próprio guia v0.4 já registrou como abertas (seção "Questões ainda abertas").

### 2.1 Reflexivo — 7 tokens (doc 15)

> EvilAI **disguises itself as productivity or AI-enhanced tools**

A anotou IMP (aplicando o default de dúvida da seção introdutória, por se tratar de forma
reflexiva não resolvida); B anotou EXP (lendo `itself` como objeto direto, o que satisfaz
literalmente a regra 5.4).

Ambas as leituras são compatíveis com o texto da v0.4. É a questão aberta "reflexivo
(`disguises itself`)" listada no guia.

### 2.2 Regra 5.5 (prioridade da forma verbo+objeto) — 4 tokens (docs 10 e 14)

> he installed OpenSSH Server and Tailscale, ... and **set up key-based SSH and a reverse tunnel**

> TE32, which is equipped to **execute commands directly via a PowerShell reverse shell**

Nos dois casos a mesma frase contém uma forma verbo+objeto e uma forma nominal da mesma
técnica. A aplicou a regra 5.5 e anotou o span verbal completo como EXP; B anotou o núcleo
nominal (`reverse tunnel`, `reverse shell`) como IMP.

A regra existe, mas não diz o que fazer quando a forma nominal é o **complemento** da
forma verbal, e não uma menção paralela. É a questão aberta "fronteira fina do
verbo+objeto".

---

## 3. Conclusão

O critério verbo+objeto da v0.4 é **reprodutível entre anotadores**: κ = 0,8885 sobre uma
linha de base de acaso de 0,51, com marginais simétricas e 94,6% de concordância bruta.
A substituição do critério antigo ("nomeia com terminologia reconhecível") pelo critério
sintático cumpriu o objetivo de tornar a decisão de rótulo operacionalizável.

O desacordo residual é **concentrado e diagnosticado**: 3 spans, 2 fenômenos, ambos já
identificados pela equipe como pendências na v0.4. Fixar exemplos-âncora para (i) reflexivo
e (ii) forma nominal como complemento de forma verbal deve ser suficiente para levar a
métrica acima do corte de 0,90 na próxima rodada, sem alteração do critério em si.

### Pauta sugerida para a reconciliação

1. `disguises itself as X` → EXP ou IMP? Fixar e adicionar à tabela da seção 7.
2. Quando a forma nominal é complemento da verbal (`set up ... a reverse tunnel`), a regra
   5.5 se aplica ou trata-se de menção única? Fixar exemplo-âncora.
3. Reanotar o piloto com a v0.5 e recalcular.
