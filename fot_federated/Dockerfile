FROM ubuntu:20.04

WORKDIR /fot-federated

# Copia todo o projeto para dentro da imagem
COPY . /fot-federated


# Troca os mirrors para evitar problemas com chaves GPG quebradas
RUN sed -i 's|http://.*.ubuntu.com|http://archive.ubuntu.com|g' /etc/apt/sources.list


# Instala dependências essenciais
RUN apt-get update --allow-insecure-repositories || true && \
    apt-get install -y --allow-unauthenticated \
        python3 \
        python3-pip \
        iputils-ping \
        net-tools \
        curl \
        jq \
        mosquitto-clients && \
    rm -rf /var/lib/apt/lists/*

# Instala dependências Python
#RUN pip3 install --upgrade pip && \
#    pip3 install tensorflow flwr numpy pandas matplotlib scikit-learn


RUN pip3 install --upgrade pip && \
    pip3 install protobuf==3.20.3 tensorflow-cpu==2.11.0 flwr numpy pandas matplotlib scikit-learn psutil tensorflow-model-optimization



# Permissões de execução para os scripts utilizados nos containers
# Comentado pois o servidor fl será executado no fot_gateway
#RUN chmod +x start_server.sh start_fl.sh

# Por padrão, o container fica interativo (mas o comando correto virá do docker-compose)
#ENTRYPOINT [ "/bin/bash" ]



