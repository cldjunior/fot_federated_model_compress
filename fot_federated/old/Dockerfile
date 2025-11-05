# Dockerfile otimizado para fot_federated
FROM ubuntu:20.04

# Evita prompts interativos durante o build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Bahia

WORKDIR /fot_federated

# Instala pacotes necessários e define o timezone corretamente
RUN apt-get update && apt-get install -y \
    tzdata \
    curl \
    python3-pip \
    python3-venv \
    build-essential \
    jq && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Instala dependências Python
RUN pip3 install --upgrade pip && \
    pip3 install tensorflow flwr numpy pandas matplotlib scikit-learn

# Copia os arquivos do projeto para o container
COPY . .

# Garante que o script possa ser executado
RUN chmod +x start_fl.sh

# Entrypoint padrão
ENTRYPOINT ["bash", "start_fl.sh"]




