
# Miau-net — Machine Learning para adoção de gatos

## 1. Sobre o projeto

O Miau-net é um projeto acadêmico que aplica técnicas de Machine Learning para identificar padrões associados à adoção de gatos, utilizando dados históricos de entradas e desfechos de um abrigo animal.

O projeto utiliza um modelo de classificação para distinguir registros com desfecho de adoção dos registros associados a outros desfechos conhecidos.

O modelo possui finalidade experimental e analítica. Seus resultados não garantem a adoção futura de um animal e não substituem a avaliação humana.

## 2. Objetivo

Desenvolver uma aplicação que permita explorar a classificação de desfechos de adoção de gatos a partir de características registradas no momento de entrada no abrigo.

## 3. Tecnologias utilizadas

- Python
- Pandas e NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib
- Jupyter Notebook

## 4. Modelo de Machine Learning

Foi utilizada a Regressão Logística com balanceamento de classes (`class_weight="balanced"`).

A variável-alvo foi definida da seguinte forma:

- `1`: adoção (`ADOPTION`);
- `0`: outros desfechos conhecidos.

Foram utilizadas sete variáveis preditoras: idade na entrada, sexo, condição na entrada, tipo e subtipo de entrada, cor predominante e jurisdição.

## 5. Resultados da avaliação

O modelo foi avaliado em um conjunto de teste separado dos dados usados no treinamento e na validação.

| Métrica | Resultado |
|---|---:|
| Acurácia | 79,94% |
| Precisão para adoção | 57,02% |
| Recall para adoção | 78,75% |
| F1-score para adoção | 66,15% |

Os resultados devem ser interpretados considerando as limitações da base histórica utilizada.

## 6. Estrutura do projeto

- `src/`: aplicação Streamlit;
- `modelos/`: modelo treinado e pré-processador;
- `notebooks/`: notebook de exploração;
- `documentacao/`: documentação técnica;
- `dados/`: dados utilizados no projeto;
- `assets/`: recursos visuais;
- `resultados/`: gráficos e métricas da avaliação.

## 7. Como executar

### Pré-requisitos

- Python compatível com as versões indicadas em `requirements.txt`;
- arquivos do modelo e do pré-processador na pasta `modelos/`;
- base de dados disponível no caminho esperado pela aplicação.

### Instalação

Crie e ative um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
```

No Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Executar a aplicação

Na raiz do projeto:

```bash
python -m streamlit run src/app_miau_net_interface.py
```

Se a aplicação não encontrar a base de dados, verifique o caminho do arquivo CSV ou utilize a opção de envio disponibilizada pela interface.

## 8. Limitações

- A base utilizada corresponde aos registros históricos de um abrigo específico.
- Os padrões identificados podem não representar a realidade de outros abrigos.
- O modelo pode cometer erros de classificação.
- A pontuação exibida não deve ser interpretada como uma probabilidade exata de adoção.
- O sistema tem finalidade acadêmica e exploratória.

## 9. Contexto acadêmico

Projeto desenvolvido no contexto acadêmico da UNIVESP, como parte do projeto integrador Miau-net.
