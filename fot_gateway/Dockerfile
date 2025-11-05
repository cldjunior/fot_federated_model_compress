FROM ubuntu:20.04
WORKDIR /fot_gateway

# Copia os arquivos necessários
COPY . .

# Define o timezone
RUN ln -snf /usr/share/zoneinfo/$(curl https://ipapi.co/timezone) /etc/localtime

# Instala dependências do sistema
#RUN apt-get update && apt-get install -y \

RUN apt-get update && apt-get install -y \
    tzdata \
    curl \
    iproute2 \
    iptables \
    iputils-ping \
    net-tools \
    openvswitch-switch \
    openvswitch-testcontroller \
    tcpdump \
    mosquitto \
    mosquitto-clients \
    dnsutils \
    netcat \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Instala paho-mqtt via pip
#RUN pip3 install paho-mqtt==1.6.1
RUN pip3 install --upgrade pip && \
    pip3 install protobuf==3.20.3 tensorflow-cpu==2.11.0 flwr numpy pandas matplotlib scikit-learn paho-mqtt==1.6.1  tensorflow-model-optimization


# Comando padrão: apenas abre o bash (usado no docker-compose com entrypoint)
ENTRYPOINT ["/bin/bash"]

