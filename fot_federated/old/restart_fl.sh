#!/bin/bash

echo "♻️ Reiniciando todos os containers do sistema federado..."

docker-compose -f docker-compose-federated-multi.yml down --remove-orphans
sleep 2

docker-compose -f docker-compose-federated-multi.yml up --build -d
sleep 5

echo "✅ Reinicialização concluída. Containers ativos:"
docker ps
