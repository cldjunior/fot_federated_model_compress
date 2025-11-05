#!/bin/bash

COMPOSE_FILE=docker-compose-federated-multi.yml
SENSORS=("sc01" "sc02" "sc03" "sc04" "sc05" "sc06" "sc07" "sc08" "sc09" "sc10")


echo "🚀 Iniciando servidor federado..."
docker compose -f $COMPOSE_FILE up -d sfl01
sleep 5

echo "🔍 Verificando status do servidor..."
docker inspect -f '{{.State.Status}}' sfl01

echo "🧠 Iniciando gateways..."
docker compose -f $COMPOSE_FILE up -d g01
docker compose -f $COMPOSE_FILE up -d g03
sleep 3

echo "🔍 Verificando status dos gateways..."
docker inspect -f '{{.State.Status}}' g01
docker inspect -f '{{.State.Status}}' g03

echo "📁 Criando diretório de logs..."
mkdir -p logs

echo "📡 Iniciando sensores federados..."
for SENSOR in "${SENSORS[@]}"; do
  echo "🛰️  Iniciando $SENSOR..."
  docker compose -f $COMPOSE_FILE up -d "$SENSOR"
  sleep 10
  echo "🔍 Verificando status de $SENSOR..."
  docker inspect -f '{{.State.Status}}' "$SENSOR"
  echo "💾 Salvando logs de $SENSOR..."
  docker logs "$SENSOR" > "logs/$SENSOR.log" 2>&1
done

echo "📊 Monitoramento dos containers:"
docker stats --no-stream

echo "✅ Todos os serviços foram iniciados com sucesso!"

