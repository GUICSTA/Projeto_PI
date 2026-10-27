# Declaração de Uso de Inteligência Artificial Generativa

---

## Informações Gerais

* **Ferramenta:** Claude (Anthropic), via chat
* **Finalidade:** Apoio na estruturação do repositório, implementação inicial do pipeline de Processamento Digital de Imagens — PDI (pré-processamento, segmentação, extração de características e classificação por regras) e redação da documentação (`README.md` e proposta técnica).
* **Partes Afetadas:**
  * Estrutura de diretórios do repositório
  * Código-fonte em `src/` (`preprocessing.py`, `segmentation.py`, `features.py`, `classify.py`, `main.py`, `generate_synthetic_cards.py`)
  * Testes automatizados em `tests/test_pipeline.py`
  * Conjunto de dados sintéticos (`images/`)
  * Documentação geral (`README.md` e `docs/proposta.md`)

---

## Forma de Validação

* **Testes Automatizados:** Execução de suíte de testes via `pytest`, confirmando que o pipeline processou e classificou corretamente as 20 imagens sintéticas de desenvolvimento (Taxa de acerto: **20/20** — ver `images/results/resultados_sinteticos.json`).
* **Revisão de Código:** O grupo revisou detalhadamente a lógica de cada módulo (binarização por Otsu, critério de cor dominante, descritores de forma e regras de classificação), garantindo o alinhamento com a fundamentação matemática e geométrica dos descritores.
* **Validação Futura (Pendente):** Calibração de limiares com fotos reais do baralho físico. Os parâmetros atuais foram ajustados para imagens sintéticas e serão reajustados na etapa experimental com dados reais (conforme documentado em `docs/proposta.md`).

---

##  Modificações Realizadas

1. **Arquitetura do Projeto:** Definição da estrutura modular e organização de diretórios do repositório.
2. **Desenvolvimento de Módulos (`src/`):** Implementação dos algoritmos de tratamento de imagem, segmentação por limiarização e rotinas de extração de características geométricas e de cor.
3. **Geração de Dados de Teste:** Automação via script para desenhar programmaticamente os símbolos dos naipes, permitindo simular o pipeline antes do ensaio fotográfico.
4. **Qualidade de Software:** Elaboração de testes automatizados unitários e de integração (`tests/test_pipeline.py`).
5. **Documentação:** Redação e estruturação técnica do `README.md` e da proposta técnica do projeto.

---

## Registro de Prompts e Orientações

### 1. Arquitetura do Pipeline
> **Pergunta:** Como estruturar um pipeline modularizado de PDI em Python para classificação de objetos?
> 
> **Orientações recebidas:** Separe o fluxo em etapas independentes e bem definidas: pré-processamento (`preprocessing.py`), segmentação (`segmentation.py`), extração de características (`features.py`) e regras de decisão (`classify.py`).
> 
> **Uso no Projeto:** Aplicado na organização da pasta `src/` e modularização do código-fonte.

---

### 2. Geração de Imagens Sintéticas
> **Pergunta:** Como gerar imagens sintéticas programaticamente para testar o pipeline antes da etapa de captura real?
> 
> **Orientações recebidas:** Utilize `OpenCV` e `NumPy` para desenhar programmaticamente as formas geométricas dos símbolos dos naipes, criando variações de cor e posição para testes rigorosos.
> 
> **Uso no Projeto:** Implementado em `src/generate_synthetic_cards.py` para produzir as 20 imagens de validação inicial.

---

### 3. Testes Automatizados
> **Pergunta:** Como estruturar testes automatizados utilizando `pytest` para validar a acurácia do pipeline de PDI?
> 
> **Orientações recebidas:** Crie rotinas de teste prefixadas com `test_` no diretório `tests/` que carregam as imagens sintéticas, executam a classificação e comparam os rótulos preditos com os gabaritos esperados via `assert`.
> 
> **Uso no Projeto:** Estruturação e implementação do arquivo `tests/test_pipeline.py`.

---

### 4. Segmentação e Algoritmos de PDI
> **Pergunta:** Como aplicar binarização adaptativa/Otsu e extrair descritores de forma para segmentação?
> 
> **Orientações recebidas:** Converta a imagem para escala de cinza, aplique suavização gaussiana seguida por `cv2.threshold` com a flag `THRESH_OTSU`, e utilize `cv2.findContours` para obter descritores (área, perímetro, excentricidade, etc.).
> 
> **Uso no Projeto:** Implementado nos módulos `preprocessing.py`, `segmentation.py` e `features.py`.

---

## Declaração de Responsabilidade

O grupo declara-se integralmente responsável por todo o código, documentação e decisões arquiteturais presentes neste repositório. Declaramos ter pleno entendimento técnico das abordagens utilizadas, bem como das limitações atuais do sistema, responsabilizando-nos pela validação e calibração final com imagens reais do baralho físico.
