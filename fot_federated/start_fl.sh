

#!/bin/bash

if [ -z "$1" ]; then
  echo "❌ Device ID não fornecido!"
  exit 1
fi

DEVICE=$1
echo "🛰️  Iniciando cliente federado para $DEVICE"

# Garante que o diretório existe
mkdir -p models/$DEVICE
sleep 5  # Garante que o servidor já iniciou


python3 client_fl_quant.py --device_id $DEVICE



