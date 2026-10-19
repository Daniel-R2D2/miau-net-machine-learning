# Documentação Técnica — Machine Learning | Miau-net

## 1. Objetivo do modelo

O objetivo desta etapa do projeto é desenvolver um modelo de classificação capaz de identificar padrões associados ao desfecho de adoção de gatos, utilizando dados históricos de entradas e desfechos de um abrigo animal.

Inicialmente, foi considerada a utilização de regressão linear para estimar o tempo de permanência dos animais no abrigo. Entretanto, após os testes realizados, os resultados não foram considerados suficientemente confiáveis para manter essa abordagem como foco principal. Por esse motivo, optou-se por reformular o problema como uma tarefa de classificação binária.

Nessa abordagem, os registros com desfecho `ADOPTION` foram classificados como adoção (1), enquanto os demais desfechos conhecidos foram agrupados na categoria outro desfecho (0).

O modelo tem finalidade experimental e analítica: identificar padrões presentes nos dados históricos, sem garantir o desfecho futuro de um animal individual.


## 2. Preparação dos dados

Foi utilizada uma base de dados pública de entradas e desfechos de animais de um abrigo, contendo informações sobre os animais e os registros de atendimento.

Na etapa inicial, a base completa possuía 57.562 registros e 29 colunas. Em seguida, foram selecionados apenas os registros referentes a gatos, resultando em 27.295 registros.

Para desenvolver o modelo de classificação, foram removidos os registros sem informação sobre o desfecho (`Outcome Type`), pois não seria possível determinar a classe de destino desses casos. Após essa filtragem, a base utilizada nessa etapa passou a conter 27.050 registros.

A variável-alvo foi criada a partir da coluna `Outcome Type`: os registros classificados como `ADOPTION` receberam o valor 1, enquanto os demais desfechos conhecidos receberam o valor 0, representando a categoria “Outro desfecho”.

## 3. Variáveis preditoras

Foram selecionadas sete variáveis preditoras para o desenvolvimento do modelo:

- `idade_na_entrada`: idade estimada do gato no momento da entrada no abrigo;
- `Sex`: sexo do animal;
- `Intake Condition`: condição do animal na entrada;
- `Intake Type`: tipo de entrada no abrigo;
- `Intake Subtype`: subtipo da entrada;
- `Primary Color`: cor predominante do animal;
- `Jurisdiction`: jurisdição associada ao registro.

A seleção buscou utilizar informações relacionadas ao animal e à sua entrada no abrigo. As variáveis diretamente relacionadas ao desfecho posterior, como `Outcome Type`, não foram utilizadas como preditoras, pois essa coluna define a própria variável-alvo do modelo.


## 4. Pré-processamento dos dados

Para preparar os dados para o treinamento, foram aplicadas técnicas de tratamento de valores ausentes e codificação de variáveis categóricas.

Na variável numérica `idade_na_entrada`, os valores ausentes foram preenchidos utilizando a mediana calculada a partir dos dados de treinamento. Nas variáveis categóricas, os valores ausentes foram substituídos pela categoria “Não informado”.

Em seguida, foi utilizado o método One-Hot Encoding para transformar as categorias textuais em representações numéricas que pudessem ser utilizadas pelo algoritmo.

O pré-processador foi ajustado exclusivamente com os dados de treinamento. As mesmas transformações foram aplicadas aos conjuntos de validação e teste, evitando que informações desses conjuntos influenciassem o ajuste do pré-processamento.

Após essa etapa, as sete variáveis originais foram transformadas em 142 características numéricas utilizadas pelo modelo.

## 5. Divisão dos dados

Após o pré-processamento, os dados foram divididos em três conjuntos: treinamento, validação e teste, utilizando a função `train_test_split` da biblioteca Scikit-Learn.

A divisão foi realizada da seguinte forma:

- **Treinamento (60%):** 16.230 registros, utilizados para ajustar os parâmetros dos modelos;
- **Validação (20%):** 5.410 registros, utilizados para comparar o desempenho dos modelos;
- **Teste (20%):** 5.410 registros, reservados para a avaliação final do modelo escolhido.

Foi utilizado o parâmetro `stratify` para preservar, aproximadamente, a proporção entre as classes de adoção e outro desfecho nos três conjuntos. O parâmetro `random_state=42` foi definido para permitir a reprodução da divisão dos dados.


## 6. Treinamento e comparação dos modelos

Foram treinados dois modelos de Regressão Logística: uma versão padrão e outra com balanceamento de classes, utilizando o parâmetro `class_weight="balanced"`.

O modelo padrão atribui pesos uniformes às classes durante o treinamento. Já a versão balanceada ajusta os pesos de acordo com a frequência das classes, buscando melhorar a identificação da classe menos frequente, correspondente à adoção.

Os modelos foram treinados exclusivamente com os dados de treinamento e comparados utilizando o conjunto de validação. Para essa comparação, foram consideradas as métricas acurácia, precisão, recall e F1-score, com atenção especial à classe de adoção.

A versão balanceada apresentou F1-score de 0,6739 para a classe de adoção, enquanto o modelo padrão apresentou 0,6592. Por esse motivo, a Regressão Logística Balanceada foi selecionada como candidata para a avaliação final.


## 7. Resultados da avaliação final

Após a comparação dos modelos no conjunto de validação, a Regressão Logística Balanceada foi selecionada para a avaliação final. O modelo foi avaliado utilizando o conjunto de teste, composto por 5.410 registros que não participaram do treinamento nem da escolha do modelo.

Os resultados obtidos foram:

- **Acurácia:** 79,94%;
- **Precisão para adoção:** 57,02%;
- **Recall para adoção:** 78,75%;
- **F1-score para adoção:** 66,15%.

A acurácia indica a proporção total de classificações corretas. A precisão mostra a proporção de registros classificados como adoção que realmente apresentaram esse desfecho. O recall representa a proporção dos registros de adoção identificados corretamente pelo modelo. O F1-score combina precisão e recall em uma única métrica.

Os resultados indicam que o modelo identificou uma parcela significativa dos registros de adoção, embora ainda apresente erros de classificação. Portanto, seu desempenho deve ser interpretado como experimental, considerando as características e limitações da base utilizada.


## 8. Análise da matriz de confusão

A matriz de confusão da avaliação final apresentou os seguintes resultados:

- **3.265 verdadeiros negativos:** registros de outro desfecho classificados corretamente;
- **799 falsos positivos:** registros de outro desfecho classificados incorretamente como adoção;
- **286 falsos negativos:** registros de adoção classificados incorretamente como outro desfecho;
- **1.060 verdadeiros positivos:** registros de adoção classificados corretamente.

Esses resultados mostram que o modelo identificou corretamente 1.060 dos 1.346 registros cujo desfecho real foi adoção, correspondendo a um recall de aproximadamente 78,75%.

Por outro lado, os 799 falsos positivos demonstram que o modelo também classificou incorretamente uma quantidade relevante de registros como adoção. Dessa forma, os resultados indicam potencial para análise de padrões históricos, mas não permitem garantir o desfecho futuro de um animal individual.


## 9. Limitações do modelo

O modelo apresenta algumas limitações que devem ser consideradas na interpretação dos resultados.

A primeira está relacionada à qualidade e à abrangência da base de dados, que contém registros históricos de um abrigo específico. Portanto, os padrões identificados podem não representar a realidade de outros abrigos.

Além disso, a precisão de 57,02% para a classe de adoção indica que uma parcela relevante dos registros classificados como adoção correspondeu, na realidade, a outros desfechos. Isso demonstra que o modelo ainda apresenta limitações na identificação dessa classe.

A variável-alvo representa o desfecho histórico registrado na base, e não uma garantia de adoção futura. O modelo também não estabelece relações de causa e efeito entre as características dos animais e seus desfechos.

Dessa forma, os resultados devem ser utilizados para fins acadêmicos e exploratórios, sem substituir a avaliação humana ou fundamentar decisões individuais sobre os animais.


## 10. Justificativa da mudança de abordagem

Inicialmente, o projeto previa a utilização de Regressão Linear para estimar o tempo de permanência dos gatos no abrigo. Entretanto, os experimentos preliminares apresentaram desempenho limitado, com erros elevados e baixa capacidade de explicar a variação do tempo de permanência.

Diante desses resultados, optou-se por reformular o problema como uma tarefa de classificação binária, com o objetivo de identificar padrões associados ao desfecho de adoção.

A mudança foi baseada nos resultados observados durante a experimentação, buscando uma abordagem mais adequada ao objetivo analítico definido para esta etapa do projeto. A Regressão Logística Balanceada foi selecionada após a comparação realizada na validação e apresentou F1-score de 66,15% na avaliação final.

Essa decisão representa um refinamento do escopo técnico do projeto a partir da análise empírica dos dados, mantendo o foco na aplicação de técnicas de Machine Learning ao contexto de adoção de animais.


