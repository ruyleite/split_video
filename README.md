# split_video
Cria clips de video conforme ocorre a mudança de cenas comparando a similaridade entre os frames.

## Sobre:
Esse é somente um script em python que faz analise de um grupo frames (step) e compara com o ultimo frame do grupo anterior e verifica se esse possui um percentual de similaridade (threshold), se esse valor for abaixo do esperados, cria um novo crip.

Uma fez que ele entende que ocorreu uma ruptura de similaridade, esse volta ao grupo anterior e analisa frame a frame ate contrar o conto exato (< threshold) para indicar um novo clip.

A efetividade vai depender de como é feito a mudança de cena e/ou mudanças drasticas no video.

Para videos cujo o conteúdo é uma apresentação, estilo powerpoint, com efeitos de transição, utilize um threshold maior que 90%.

Para coletânea ou compilado de vídeos, um threshold de 70% já é o suficiente.

Padrão o parametro step ira utilizar o FPS do vídeo para dividir em grupos de frames (chunks), quanto maior valor de step, menor tem que ser o threshold.

Um exemplo:
Para um vídeo de apresentação, com FPS de 30, o threshold de 75% ou maior já é o suficiente, mas havendo efeitos de transição, mude para 90 ou 95%.

Já para um video de filme ou que tenha troca de cenas, em especial plano de fundo distintos, valores entre 50% a 75%.


## Pre-requisitos:
python3

GPU(NVIDIA) - não obrigatório, mas deixa o processo mais rápido.

cuda - uso em combinação com a GPU

opencv

## Instalação:

Execute o script setup.sh, esse ira criar um "virtual environment" e instalar os pacotes necessários.

ou execute:

pip3 install -r requirements.txt

## Execução:

./split_video.py -f <<PATH E NOME DO ARQUIV DE VIDEO>> -s <NUMERO DE FRAMES PARA CHECAGEM> -t <THRESHOLD PARA CORTE DE CENA>

## Parâmetros:
usage: split_video.py -f FILENAME [-s STEP] [-t THRESHOLD] [-c]

-f ou --filename = path e nome do arquivo de vídeo a ser analizado
-s ou --step = Número de frames para comparação inicial, quando não informado ou "0" será assumido o FPS do vídeo.
-t ou --threshold = Percentual minimo para indicar se um frame é igual ao outro, o padrão é 90%
-c ou --clip_only = realiza somente o split dos clips, é necessário que o arquivo já tenha sido analizado, para confirmar verifique junto ao video se existe um arquivo json com o seguinte padrão NOME_ARQUIVO_VIDEO.json.


