FROM ubuntu:20.04

WORKDIR /fot_server

# Define o timezone
RUN apt-get update && apt-get install -y \
    tzdata \
    curl \
    ifupdown \
    iproute2 \
    iptables \
    iputils-ping \
    net-tools \
    openvswitch-switch \
    openvswitch-testcontroller \
    tcpdump \
    x11-xserver-utils \
    systemd \
    xterm \
    mosquitto-clients \
    dnsutils \
    netcat \
    mosquitto \
    openssh-server \
    python3-pip \
    python3-venv \
    build-essential \
    git && \
    ln -snf /usr/share/zoneinfo/$(curl -s https://ipapi.co/timezone) /etc/localtime

# Instala bibliotecas Python necessárias para federated learning
RUN pip3 install --upgrade pip && \
    pip3 install tensorflow flwr numpy pandas


COPY ./serverFL.py .

# Comente essa linha e use CMD se preferir passar via docker-compose
ENTRYPOINT ["/bin/bash"]

