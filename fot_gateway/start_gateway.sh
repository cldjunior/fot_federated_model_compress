#!/bin/bash

# Inicia o broker MQTT em background
mosquitto -c mosquitto-no-auth.conf &

# Aguarda o broker inicializar
sleep 5

# Inicia o cliente MQTT
python3 mqtt-client.py 127.0.0.1 227.227.227.9 &

# Aguarda um pouco
sleep 2

# Usa estratégia fornecida via variável de ambiente ou usa fedavg como padrão
STRATEGY="${STRATEGY:-fedavg}"
echo "Iniciando servidor federado com estratégia: $STRATEGY"
python3 server_quant.py --strategy "$STRATEGY" &

# Mantém o container ativo
tail -f /dev/null

