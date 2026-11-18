# Identificação de Naipe em Cartas de Baralho

Projeto aplicado longitudinal — disciplina de Processamento de Imagens (2026-02), **etapa M3 (versão final)**.

Registros das entregas anteriores: [`docs/README_M2.md`](docs/README_M2.md) e [`docs/README_M1.md`](docs/README_M1.md).

## Integrantes

- Vinicius Lohn Ramires
- Guilherme Cassettari
- Guilherme Pires

As contribuições de cada integrante estão em [`docs/contribuicoes.md`](docs/contribuicoes.md).

## Problema

Dada uma foto de uma carta de baralho francês, identificar automaticamente o **naipe** (copas, ouros, espadas ou paus), o **valor** (A, 2–10, J, Q, K) e o **coringa**.

**Contexto:** automação de jogos de cartas, em que uma câmera sobre a mesa identifica as cartas conforme são reveladas.

## Objetivos

- **Obrigatório:** reconhecer o naipe em fotos reais a partir do índice do canto da carta, com um pipeline clássico de PDI, explicável e sem ajuste manual por imagem.
- **Extensões (M3):** reconhecer o valor da carta e detectar o coringa.
- Avaliar num conjunto de teste congelado, separado do conjunto de ajuste, com baseline, ablações, métricas por classe e análise de erros. Os 100% no conjunto original **não** provam robustez: a generalização foi investigada com condições simuladas (ruído, luz, rotação, escala, fundo), que mostram onde o método falha.

## Entradas e saídas

- **Entrada:** foto RGB (JPEG/PNG) de uma carta inteira vista de cima, sobre fundo mais escuro que o papel.
- **Saída:**
  - naipe e valor, ou "coringa";
  - cor, descritores de forma e caixas do valor e do naipe;
  - opcionalmente, as saídas intermediárias: máscara da carta, recorte do canto, componentes rotulados, caixas e máscara do símbolo.

## Pipeline final

```
foto RGB → máscara da carta (Otsu sobre min(R,G,B)) → recorte do canto
→ Otsu só na região útil + abertura → componentes conectados
→ coringa? (elemento > 55% da altura do canto)
→ valor (1 ou mais elementos) e símbolo do naipe
→ naipe: cor dominante + descritores de forma + regras calibradas
→ valor: 2 elementos → 10; 2 buracos parecidos → 8; senão template matching
```

A fundamentação, os parâmetros e os experimentos estão no [relatório final](docs/relatorio_final.md).

### Atributos extraídos

Todos são calculados sobre a máscara binária do símbolo do naipe (`src/features.py`), são adimensionais. O `fill_ratio` é sensível à resolução, porque o recorte do símbolo usa uma margem fixa de 5 px (relatório, Seção 7.5):

| Atributo | Cálculo | Uso |
|---|---|---|
| cor dominante | `R − (G+B)/2 > 15` nos pixels do símbolo → vermelho, senão preto | separa {copas, ouros} de {espadas, paus} |
| `fill_ratio` | pixels do símbolo ÷ área do recorte | k-NN |
| `vertical_symmetry` | fração de pixels iguais entre a metade de cima e a de baixo espelhada | k-NN |
| `width_profile_ratio` | largura média da metade de cima ÷ da metade de baixo | regra calibrada e k-NN |
| máscara 32×32 | máscara redimensionada | template matching do naipe e do valor |
| elementos e buracos do valor | componentes do valor e buracos fechados | regras do 10 e do 8 |

A tabela com os atributos de cada foto é gerada por `python -m src.visualize` em [`images/results/tabela_features.csv`](images/results/tabela_features.csv).

## Estágio atual (final)

- [x] 108 fotos reais (52 cartas × 2 + 4 coringas), divisão ajuste/teste congelada
- [x] Naipe calibrado no ajuste e avaliado no teste
- [x] Reconhecimento do valor da carta (M3)
- [x] Coringa como classe própria (M3)
- [x] Máscaras de referência do símbolo e Dice/IoU (M3)
- [x] Robustez a ruído e a iluminação fraca, com ablação escolhida no ajuste (M3)
- [x] Condições de captura simuladas: fundo claro, luz, rotação, escala (M3)
- [ ] Teste com fotos de outro baralho/ambiente (substituído por simulação; ver limitações)
- [x] Regras × template × k-NN repetidos nas condições simuladas (M3)
- [ ] Fundo claro e rotações fortes (limitações conhecidas)

## Resultados principais

Conjunto de teste: 40 fotos, 5 cartas por naipe que não aparecem no ajuste.

| Tarefa | Baseline (M2) | Final (M3) |
|---|---|---|
| Naipe, fotos limpas | 40/40 | **40/40** (F1 = 1,00) |
| Valor, fotos limpas | 32/40 (só template) | **40/40** (otimista; ver relatório, 3.4) |
| Coringa | rejeitado com erro | **4/4** (exploratório) |
| Segmentação do naipe (Dice, 20 máscaras), limpas | 0,998 | 0,998 |
| Naipe com ruído σ = 60 | 27,5% | **100%** |
| Naipe com iluminação fraca (×0,55) | 0% | **100%** |
| Naipe com rotação de 25° | 50% | 50% |
| Fundo claro | 0% | 0% |
| Regras × template × k-NN em 9 condições | — | nenhum método domina: rotação 25° 50% × 70% × 95%; escala 50% 100% × 100% × 75% |

Matrizes de confusão: [`avaliacao_reais.json`](images/results/avaliacao_reais.json) (naipe) e [`avaliacao_valor.json`](images/results/avaliacao_valor.json) (valor). Experimentos da M3: [`experimentos_m3.json`](images/results/experimentos_m3.json) e [`metodos_condicoes.json`](images/results/metodos_condicoes.json).

## Tecnologias

Testado com **Python 3.12.10**. As versões exatas estão em `requirements-lock.txt`.

| Biblioteca | Versão testada | Uso |
|---|---|---|
| Pillow | 13.0.0 | leitura, gravação, redimensionamento e rotação de imagens; desenho das figuras |
| NumPy | 2.5.4 | arrays |
| pytest | 9.1.1 | testes |

Otsu, componentes conectados, erosão/dilatação, mediana, descritores, calibração, template matching, k-NN e métricas estão implementados no próprio projeto, sem OpenCV ou scikit-learn.

## Estrutura

```
├── README.md
├── requirements.txt / requirements-lock.txt
├── AI_USAGE.md                       # declaração de uso de IA da M3
├── docs/
│   ├── relatorio_final.md            # relatório final (M3)
│   ├── relatorio_m2.md               # relatório parcial da M2
│   ├── proposta.md                   # proposta revisada
│   ├── contribuicoes.md              # contribuições dos integrantes
│   ├── README_M1.md / README_M2.md   # registros das entregas anteriores
│   └── AI_USAGE_M1.md / AI_USAGE_M2.md
├── images/
│   ├── input/reais/                  # 108 fotos reais (naipe_valor_nn.jpg)
│   ├── input/sinteticas/             # 20 imagens sintéticas da M1 (regressão)
│   ├── reference/                    # máscaras de referência do símbolo + folha de conferência
│   ├── split.json / split_reais.json # divisões ajuste/teste
│   └── results/
│       ├── limiares_reais.json       # regras do naipe calibradas no ajuste
│       ├── templates_valor.npz       # templates do valor (ajuste)
│       ├── avaliacao_reais.json      # naipe no teste: matriz + P/R/F1
│       ├── avaliacao_valor.json      # valor no teste: baseline × final, matriz 13×13 + P/R/F1
│       ├── experimentos.json         # experimentos 1–3 (M2)
│       ├── experimentos_m3.json      # experimentos 4–6 (M3)
│       ├── metodos_condicoes.json    # experimento 8: 3 métodos × 9 condições (M3)
│       ├── erros_rotacao_25.json     # análise de erros da rotação
│       ├── comparacao_metodos.json   # regras × template × k-NN (M2)
│       ├── tabela_features.csv       # features por imagem
│       ├── intermediarias/           # saídas intermediárias da M2
│       └── intermediarias_m3/        # sucessos e falhas da M3
├── src/
│   ├── preprocessing.py              # RGB, cinza, Otsu, mediana
│   ├── segmentation.py               # carta, canto, morfologia, componentes, valor/naipe, coringa
│   ├── features.py                   # descritores de forma
│   ├── classify.py                   # extração + decisão do naipe; configurações final e baseline
│   ├── rank.py                       # reconhecimento do valor
│   ├── split.py                      # divisão ajuste/teste
│   ├── calibrate.py                  # calibração das regras e avaliação
│   ├── metricas.py                   # precisão, revocação, F1
│   ├── reference.py                  # máscaras de referência e Dice/IoU
│   ├── compare.py                    # regras × template × k-NN (M2)
│   ├── experiments.py                # experimentos 1–3 (M2)
│   ├── robustez.py                   # experimentos 4–6 (M3)
│   ├── metodos_condicoes.py          # experimento 8 (M3)
│   ├── visualize.py / figuras_m3.py  # saídas intermediárias
│   ├── main.py                       # linha de comando
│   └── generate_synthetic_cards.py   # imagens sintéticas da M1
└── tests/                            # testes pytest
```

## Dados

As fotos foram tiradas pelo grupo com o próprio baralho, sem restrição de redistribuição, e estão no repositório. A origem, as condições de aquisição, as referências de avaliação (incluindo as máscaras do símbolo), a divisão e as condições simuladas estão na seção 3 do [relatório final](docs/relatorio_final.md).

## Instalação

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Execução

Exemplo mínimo, numa foto real (configuração final da M3):

```bash
python -m src.main images/input/reais/copas_Q_01.jpg --saida saida_exemplo
```

Saída esperada:

```
Imagem: images/input/reais/copas_Q_01.jpg
Tipo: foto (segmentação do canto)
Naipe identificado: copas
Valor identificado: Q
Cor detectada: vermelho
Threshold (Otsu): 134
Características: {'fill_ratio': 0.5321, 'vertical_symmetry': 0.5792, 'width_profile_ratio': 2.2097}
Caixa do valor (y0, y1, x0, x1): (201, 351, 193, 302)
Caixa do naipe (y0, y1, x0, x1): (378, 498, 195, 298)
Saídas intermediárias salvas em: saida_exemplo
```

`--baseline-m2` usa a configuração da M2. Num coringa a saída é `Carta identificada: coringa (sem naipe)`.

Para reproduzir todos os resultados, nesta ordem:

```bash
python -m src.split reais        # divisão ajuste/teste (já versionada)
python -m src.calibrate          # regras do naipe + avaliação no teste
python -m src.rank               # templates do valor + avaliação no teste
python -m src.compare            # regras × template matching × k-NN (M2)
python -m src.experiments        # experimentos 1, 2 e 3 (M2)
python -m src.reference gerar    # máscaras de referência (já versionadas)
python -m src.robustez           # experimentos 4 e 5 (M3)
python -m src.robustez carta     # experimento 6 (M3)
python -m src.robustez final     # experimento 4b: configuração final com ruído (M3)
python -m src.metodos_condicoes  # experimento 8: regras × template × k-NN nas condições (M3)
python -m src.visualize          # saídas intermediárias e tabela de features (M2)
python -m src.figuras_m3         # sucessos, falhas e erros da M3
```

Os experimentos 4–6 levam alguns minutos (a rotulação de componentes é em Python puro). Sementes: divisão 42; ruído `7 + índice`.

Testes:

```bash
pytest tests/ -v
```

## Documentação

- [Relatório final (M3)](docs/relatorio_final.md), com a resposta ao feedback da M2 na seção 10
- [Slides da M3](Slides_M3.pdf) ([pptx](Slides_M3.pptx)) e [roteiro por integrante](docs/roteiro_m3/)
- [Relatório parcial da M2](docs/relatorio_m2.md)
- [Proposta revisada](docs/proposta.md)
- [Registro da M1](docs/README_M1.md)
- [Declaração de uso de IA da M3](AI_USAGE.md)
- [Declaração de uso de IA da M2](docs/AI_USAGE_M2.md)
- [Declaração de uso de IA da M1](docs/AI_USAGE_M1.md)

## Vídeos

- M1: https://youtu.be/Elav2VKWgt4
- M2: https://www.youtube.com/watch?v=mVwQDrMyRRI
- M3: (link a adicionar)
