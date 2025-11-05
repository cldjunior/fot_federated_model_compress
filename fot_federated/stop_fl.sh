#!/bin/bash

echo "🛑 Parando e removendo todos os containers..."
docker-compose -f docker-compose-federated-multi.yml down --remove-orphans

echo "✅ Todos os containers foram parados e removidos."
