#!/bin/bash

echo "📋 STATUS dos containers do Federated Learning:"
echo

docker ps --filter "name=sfl01" --format "🔵 Servidor: {{.Names}} - {{.Status}}"
docker ps --filter "name=g01" --format "🟠 Gateway: {{.Names}} - {{.Status}}"
docker ps --filter "name=g03" --format "🟠 Gateway: {{.Names}} - {{.Status}}"

for i in $(seq -w 1 5); do
    docker ps --filter "name=dc$i" --format "🟢 Device: {{.Names}} - {{.Status}}"
done

echo
echo "🌐 IPs dos containers (rede fot-net):"
echo
docker inspect $(docker ps -q --filter network=fot-net) | grep -E 'Name|IPAddress' | grep -v null | grep -v 127.0.0.1
