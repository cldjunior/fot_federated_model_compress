#!/bin/bash

echo "📁 Criando pasta de logs..."
mkdir -p logs

echo "📦 Salvando logs de sensores..."
for i in $(seq -w 1 14); do
    docker logs sc$i > logs/sc$i.log 2>&1
done

echo "📦 Salvando logs dos gateways..."
docker logs g01 > logs/g01.log 2>&1
docker logs g03 > logs/g03.log 2>&1

echo "📦 Salvando logs do servidor..."
docker logs sfl01 > logs/sfl01.log 2>&1

echo "✅ Todos os logs foram salvos na pasta ./logs/"
