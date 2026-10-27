> Registro da entrega M1. Conteúdo do README no commit avaliado `83500d109a1c82d71ea5a8f40d5a1956ea1d9ea6`, preservado sem alterações. O README atual descreve a M2.

# Identificação de Naipe em Cartas de Baralho

Projeto aplicado longitudinal — disciplina de Processamento de Imagens (2026-02), etapa M1.

## Integrantes

- Vinicius Lohn Ramires
- Guilherme Cassettari
- Guilherme Pires

## Problema investigado

Dada uma foto de uma carta de baralho francês, identificar automaticamente a qual dos quatro naipes ela pertence: **copas, ouros, paus ou espadas**. O reconhecimento do valor da carta (A–K) fica fora do escopo desta etapa.

## Contexto de aplicação

Automação de jogos de cartas : uma câmera fixa sobre uma mesa identifica as cartas conforme são reveladas. 
## Objetivo geral

Implementar um pipeline clássico de PDI (sem aprendizado de máquina) que combine **cor dominante** e **descritores de forma** do símbolo do naipe para classificá-lo corretamente. 

## Visão resumida da solução

```
foto da carta → pré-processamento (RGB→cinza→Otsu) → segmentação (recorte + cor)
→ extração de forma (simetria, perfil de largura) → classificação por regras → naipe
```

## Conjunto de imagens

**Estágio atual:** pipeline validado com 20 imagens **sintéticas**, usadas apenas para desenvolvimento. As fotos reais do baralho físico do grupo ainda serão capturadas.

## Estágio atual do projeto

- [x] Definição do problema e escopo (apenas naipe, não valor)
- [x] Pipeline completo implementado (pré-processamento, segmentação, features, classificação)
- [x] Testes automatizados (pytest) — 9/9 passando
- [x] Validação preliminar com imagens sintéticas
- [ ] Captura de fotos reais do baralho físico
- [ ] Recalibração dos limiares com dados reais
- [ ] Gravação do vídeo da M1

## Organização do repositório

```
projeto-cartas-naipes/
├── README.md
├── docs/
│   └── proposta.md          # proposta detalhada (problema, objetivo, pipeline, viabilidade)
├── images/
│   ├── input/
│   │   └── sinteticas/      # imagens sintéticas de desenvolvimento (temporárias)
│   └── results/
│       └── resultados_sinteticos.json
├── src/
│   ├── preprocessing.py     # RGB, escala de cinza, binarização (Otsu manual)
│   ├── segmentation.py      # bounding box, recorte, cor dominante
│   ├── features.py          # descritores de forma
│   ├── classify.py          # regra de decisão (cor + forma)
│   ├── main.py               # CLI
│   └── generate_synthetic_cards.py  # gerador de dados sintéticos de dev
├── tests/
│   └── test_pipeline.py     # 9 testes pytest
├── AI_USAGE.md
├── requirements.txt
└── .gitignore
```

## Tecnologias

- **Python 3**
- **Pillow** — leitura/gravação de imagens
- **NumPy** — operações em array (infraestrutura; toda lógica de PDI — binarização, bounding box, descritores de forma — é implementada manualmente, sem funções prontas de reconhecimento)
- **pytest** — testes automatizados

## Como reproduzir

```bash
pip install -r requirements.txt

# gerar imagens sintéticas de desenvolvimento (se ainda não existirem)
python -m src.generate_synthetic_cards

# rodar o pipeline numa imagem
python -m src.main images/input/sinteticas/copas_00.png

# rodar os testes
pytest tests/ -v
```

## Vídeo da M1

https://youtu.be/Elav2VKWgt4
