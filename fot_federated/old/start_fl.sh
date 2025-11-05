

#!/bin/bash

if [ -z "$1" ]; then
  echo "❌ Sensor ID não fornecido!"
  exit 1
fi

SENSOR=$1
echo "🛰️  Iniciando cliente federado para $SENSOR"
python3 client_fl.py --sensor_id $SENSOR



