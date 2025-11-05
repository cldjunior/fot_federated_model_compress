# =====================================================
# Dockerfile para o container sensors_compress (Mininet)
# Estrutura alinhada com docker-compose-sensors.yml
# =====================================================

FROM ubuntu:20.04

# Evita prompts interativos
ENV DEBIAN_FRONTEND=noninteractive

# Diretório de trabalho dentro do container
WORKDIR /fot_sensors/sim

# Copia tudo da pasta sim (Dockerfile, scripts, .py, .env, etc.)
COPY . /fot_sensors/sim

# ==============================
# Instalação de dependências base
# ==============================
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
    mosquitto \
    mininet \
    dnsutils \
    netcat \
    python3 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# ======================================
# Instala pacotes Python necessários
# ======================================
RUN pip3 install --no-cache-dir paho-mqtt numpy scikit-learn && \
    pip3 install --no-cache-dir --no-deps tatu


# ======================================
# Permissões e PATH
# ======================================
RUN chmod -R a+x /fot_sensors/sim && \
    echo 'export PATH=$PATH:/fot_sensors/sim' >> /etc/profile

# ======================================
# Configuração de timezone automática
# ======================================
RUN ln -snf /usr/share/zoneinfo/America/Bahia /etc/localtime && echo "America/Bahia" > /etc/timezone

# ======================================
# Comando padrão do container
# ======================================
CMD ["bash", "inicializa_mininet.sh"]

