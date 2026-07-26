# Projeto 2 — Classificação CIFAR-10

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar imagens coloridas** em 10 categorias de objetos e animais (avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão), e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

Este projeto tem uma diferença importante em relação a uma classificação de dígitos: as imagens são **coloridas (RGB)** e visualmente mais complexas, o que torna a tarefa de classificação genuinamente mais difícil — por isso **data augmentation** é um requisito obrigatório aqui, não opcional.

## 🎯 Conjunto de Dados

Dataset **CIFAR-10**, disponível diretamente via `tf.keras.datasets.cifar10` (não é necessário download manual). 60.000 imagens 32x32 coloridas, 10 classes.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset CIFAR-10 via TensorFlow
- Split explícito treino/validação
- **Data augmentation** aplicada ao conjunto de treino, usando camadas do Keras
  (ex: `RandomFlip("horizontal")`, `RandomRotation`, `RandomZoom`) incorporadas ao
  modelo ou ao pipeline de treino
- Construção de uma CNN com 3-4 blocos convolucionais (`Conv2D` + `BatchNormalization`
  + `MaxPooling2D`) seguida de `Dropout`
- Treinamento com **early stopping** baseado na perda de validação
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

> 💡 Se você aplicar a augmentation de outra forma (ex: pré-processamento manual em
> `tf.data`), tudo bem — apenas descreva isso claramente no relatório, já que a
> correção automática busca primeiro por camadas de augmentation no próprio modelo.

> 💡 CIFAR-10 é mais difícil que MNIST/Fashion-MNIST para uma CNN simples treinada
> rapidamente em CPU — não se preocupe se a acurácia ficar bem abaixo de 90%. O
> importante é o pipeline completo funcionar corretamente.

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/2-classificacao-cifar/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 32x32, 3 canais (RGB), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 25-30, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Generalização** — uso adequado de data augmentation
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Eduardo Lustosa Ribeiro

### 1️⃣ Resumo da Arquitetura do Modelo
Implementei uma CNN com 3 blocos convolucionais. Cada bloco usa Conv2D, BatchNormalization (para estabilidade) e MaxPooling2D. Depois, uso Flatten, um Dropout de 0.5 (para segurar overfitting) e uma camada Dense final com ativação softmax para as 10 classes. O data augmentation (RandomFlip, RandomRotation de 0.1 e RandomZoom de 0.1) foi embutido direto como camada do Keras no modelo, o que ajuda a generalizar os dados de treino direto no pipeline.

### 2️⃣ Bibliotecas Utilizadas

TensorFlow / Keras (v2.21.0): construção da rede e conversão TFLite

NumPy (v2.4.6): manipulação de tensores e semente fixa para reprodutibilidade

OS, Sys e Argparse: nativas do Python, para controle de linha de comando e supressão de logs de C++ do sistema

### 3️⃣ Técnica de Otimização do Modelo

Usei Dynamic Range Quantization nativa do TFLite (tf.lite.Optimize.DEFAULT). Ela pega os pesos do modelo (float32) e converte para inteiros de 8 bits (int8). As ativações continuam sendo calculadas em float32 durante a inferência. A vantagem é reduzir drasticamente o tamanho do arquivo sem prejudicar a acurácia.

### 4️⃣ Resultados Obtidos
Acurácia de validação (treino): 63.92%
Acurácia .h5: 64.16%
Acurácia .tflite: 64.26%
Tamanho do model.h5: ~1.38 MB
Tamanho do model.tflite: ~0.12 MB (~120 KB)
Houve uma redução de mais de 10x no tamanho final do artefato.

### 5️⃣ Comentários Adicionais (Opcional)
O maior desafio não foi a modelagem, e sim a infraestrutura: o download do CIFAR-10 via keras.datasets.cifar10 travava por mais de 1h no GitHub Actions (servidor de origem lento), então troquei para tensorflow-datasets, que usa um mirror mais rápido. Também precisei recriar meu ambiente local usando Python 3.10 (igual ao do GitHub Actions), pois meu Codespace estava em Python 3.11 com uma versão do Keras (3.15) que não existe para 3.10 — o ambiente de validação só suporta até o Keras 3.12, o que causava erro de deserialização ao carregar o model.h5. Retreinar dentro do ambiente correto resolveu o problema de compatibilidade.

### 6️⃣ Exemplo de Inferência
Testando modelo em 5 amostras:
Amostra 6252 | Predito: sapo | Real: cervo
Amostra 4684 | Predito: cavalo | Real: cavalo
Amostra 1731 | Predito: sapo | Real: gato
Amostra 4742 | Predito: caminhão | Real: caminhão
Amostra 4521 | Predito: automóvel | Real: automóvel
Acertos: 3/5

Os dois erros (amostras 6252 e 1731) seguem o mesmo padrão: animais confundidos com "sapo". Isso é coerente com o tamanho reduzido da rede e a baixa resolução das imagens (32x32) — texturas de fundo semelhantes (vegetação, tons esverdeados/acinzentados) podem ativar os mesmos filtros convolucionais em classes visualmente distintas.
