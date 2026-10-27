# Identificação de Naipe em Cartas de Baralho

Projeto aplicado longitudinal — disciplina de Processamento de Imagens (2026-02), **etapa M2**.

O registro da entrega M1 está em [`docs/README_M1.md`](docs/README_M1.md).

## Integrantes

- Vinicius Lohn Ramires
- Guilherme Cassettari
- Guilherme Pires

As contribuições de cada integrante estão em [`docs/contribuicoes.md`](docs/contribuicoes.md).

## Problema

Dada uma foto de uma carta de baralho francês, identificar automaticamente o naipe: **copas, ouros, espadas ou paus**. O valor da carta (A–K) é localizado no canto, mas ainda não é reconhecido. O reconhecimento do valor está planejado para a M3.

**Contexto:** automação de jogos de cartas, em que uma câmera sobre a mesa identifica as cartas conforme são reveladas.

## Objetivos

- Identificar o naipe em fotos reais a partir do índice do canto da carta (valor + símbolo), com um pipeline clássico de PDI, explicável e sem ajuste manual por imagem.
- Avaliar em um conjunto de teste congelado, separado do conjunto usado para ajustar os parâmetros, e publicar a matriz de confusão por naipe.

## Entradas e saídas

- **Entrada:** foto RGB (JPEG/PNG) de uma carta inteira vista de cima.
- **Saída:**
  - naipe e cor;
  - descritores de forma;
  - caixas do valor e do naipe;
  - opcionalmente, as saídas intermediárias: máscara da carta, recorte do canto, componentes rotulados, caixas e máscara do símbolo.

## Pipeline

```
foto RGB → máscara da carta → recorte do canto → Otsu sobre min(R,G,B) (+ abertura opcional)
→ componentes conectados → valor (1 ou mais elementos) e símbolo do naipe
→ cor dominante + descritores de forma → regras calibradas → naipe
```

O pipeline completo, a fundamentação e os parâmetros estão no [relatório da M2](docs/relatorio_m2.md).

## Estágio atual

- [x] 108 fotos reais do baralho físico (52 cartas × 2 + 4 coringas)
- [x] Divisão ajuste/teste congelada, estratificada por naipe e agrupada por carta
- [x] Segmentação do canto com valor, símbolo e múltiplos elementos
- [x] Limiares recalibrados só no conjunto de ajuste
- [x] Comparação entre regras, template matching e k-NN nas mesmas imagens
- [x] Experimento de morfologia (abertura) com ruído, medido por IoU
- [x] Saídas intermediárias e tabela de features por imagem
- [ ] Reconhecimento do valor da carta (M3)
- [ ] Teste externo com outro fundo e outra iluminação (M3)
- [ ] Robustez a ruído forte (M3)

## Resultados principais

Conjunto de teste: 40 fotos, com 5 cartas por naipe que não aparecem no ajuste.

| Experimento | Configuração | Resultado |
|---|---|---|
| Limiares | sintéticos (M1) × calibrados no ajuste | 23/40 × **40/40** |
| Métodos (fotos limpas) | regras × template matching × k-NN | 100% × 100% × 100% |
| Morfologia, ruído σ = 60 | sem × com abertura (naipes localizados com IoU ≥ 0,5) | 27,5% × **60%** |

Matriz de confusão por naipe: [`images/results/avaliacao_reais.json`](images/results/avaliacao_reais.json). A análise completa, com casos de sucesso e falha, está no [relatório](docs/relatorio_m2.md).

## Tecnologias

Testado com **Python 3.12.10**. As versões exatas estão em `requirements-lock.txt`.

| Biblioteca | Versão testada | Uso |
|---|---|---|
| Pillow | 13.0.0 | leitura, gravação e redimensionamento de imagens; desenho das figuras |
| NumPy | 2.5.4 | arrays |
| pytest | 9.1.1 | testes |

Otsu, componentes conectados, erosão/dilatação, descritores, calibração, template matching e k-NN estão implementados no próprio projeto, sem OpenCV ou scikit-learn.

## Estrutura

```
├── README.md
├── requirements.txt / requirements-lock.txt
├── AI_USAGE.md                       # declaração de uso de IA da M2
├── docs/
│   ├── relatorio_m2.md               # relatório parcial da M2
│   ├── proposta.md                   # proposta revisada
│   ├── contribuicoes.md              # contribuições dos integrantes
│   ├── README_M1.md                  # registro da entrega M1
│   └── AI_USAGE_M1.md                # declaração de uso de IA da M1
├── images/
│   ├── input/reais/                  # 108 fotos reais (naipe_valor_nn.jpg)
│   ├── input/sinteticas/             # 20 imagens sintéticas da M1 (regressão)
│   ├── split.json / split_reais.json # divisões ajuste/teste
│   └── results/
│       ├── limiares_reais.json       # regras calibradas no ajuste
│       ├── avaliacao_reais.json      # avaliação no teste + matriz de confusão
│       ├── comparacao_metodos.json   # regras × template × k-NN
│       ├── experimentos.json         # experimentos 1, 2 e 3
│       ├── tabela_features.csv       # features por imagem
│       └── intermediarias/           # saídas intermediárias de casos típicos e difíceis
├── src/
│   ├── preprocessing.py              # RGB, cinza, Otsu
│   ├── segmentation.py               # carta, canto, morfologia, componentes, valor/naipe
│   ├── features.py                   # descritores de forma
│   ├── classify.py                   # extração + decisão
│   ├── split.py                      # divisão ajuste/teste
│   ├── calibrate.py                  # calibração das regras e avaliação
│   ├── compare.py                    # regras × template matching × k-NN
│   ├── experiments.py                # experimentos controlados
│   ├── visualize.py                  # saídas intermediárias e tabela
│   ├── main.py                       # linha de comando
│   └── generate_synthetic_cards.py   # imagens sintéticas da M1
└── tests/                            # testes pytest
```

## Dados

As fotos foram tiradas pelo grupo com o próprio baralho, sem restrição de redistribuição, e estão no repositório. A origem, as condições de aquisição, a referência de avaliação e a divisão estão na seção 3 do [relatório](docs/relatorio_m2.md).

## Instalação

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Execução

Exemplo mínimo, numa foto real:

```bash
python -m src.main images/input/reais/copas_Q_01.jpg --saida saida_exemplo
```

Saída esperada:

```
Imagem: images/input/reais/copas_Q_01.jpg
Tipo: foto (segmentação do canto)
Naipe identificado: copas
Cor detectada: vermelho
Threshold (Otsu): 140
Características: {'fill_ratio': 0.5336, 'vertical_symmetry': 0.578, 'width_profile_ratio': 2.2068}
Caixa do valor (y0, y1, x0, x1): (201, 351, 193, 302)
Caixa do naipe (y0, y1, x0, x1): (378, 498, 195, 298)
Saídas intermediárias salvas em: saida_exemplo
```

A opção `--morfologia abertura` aplica a abertura na máscara do canto.

Para reproduzir todos os resultados, nesta ordem:

```bash
python -m src.split reais        # divisão ajuste/teste (já versionada)
python -m src.calibrate          # calibra no ajuste e avalia no teste
python -m src.compare            # regras × template matching × k-NN
python -m src.experiments        # experimentos 1, 2 e 3
python -m src.visualize          # saídas intermediárias e tabela de features
```

Testes:

```bash
pytest tests/ -v
```

## Documentação

- [Relatório parcial da M2](docs/relatorio_m2.md)
- [Proposta revisada](docs/proposta.md)
- [Registro da M1](docs/README_M1.md)
- [Declaração de uso de IA da M2](AI_USAGE.md)
- [Declaração de uso de IA da M1](docs/AI_USAGE_M1.md)

## Vídeos

- M1: https://youtu.be/Elav2VKWgt4
- M2: *(link a ser adicionado)*
