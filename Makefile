# Makefile - FOT Federated Simulator (Centralizado)

# Variáveis de rede
REDE=fot-net
SUBNET=172.27.27.0/24
GATEWAY_IP=172.27.27.1

# Caminhos para os arquivos docker-compose
GATEWAY_COMPOSE=fot_gateway/docker-compose-gateway.yml
FEDERATED_COMPOSE=fot_federated/docker-compose-federated.yml
SENSORS_COMPOSE=docker-compose-sensors.yml  # <-- Corrigido

.PHONY: ajuda rede build subir subir-gateway subir-servidor subir-sensors parar limpar logs

ajuda:
	@echo "Comandos disponíveis:"
	@echo "  make rede             -> Cria a rede fot-net"
	@echo "  make build            -> Constrói todas as imagens"
	@echo "  make subir            -> Cria a rede e sobe todos os containers"
	@echo "  make subir-gateway    -> Sobe apenas o gateway"
	@echo "  make subir-servidor   -> Sobe apenas os clientes federados"
	@echo "  make subir-sensors    -> Sobe apenas os sensores (Mininet)"
	@echo "  make parar            -> Para todos os containers"
	@echo "  make limpar           -> Remove os containers"
	@echo "  make logs             -> Mostra logs ativos"

rede:
	@if docker network ls | grep -q $(REDE); then \
		echo "Rede $(REDE) já existe."; \
	else \
		echo "Criando rede $(REDE)..."; \
		docker network create --driver bridge --subnet=$(SUBNET) --gateway=$(GATEWAY_IP) $(REDE); \
		echo "Rede $(REDE) criada com sucesso."; \
	fi

build:
	@echo "Construindo imagens..."
	docker build -t cldjunior/fot_gateway:1.2 -f fot_gateway/Dockerfile ./fot_gateway
	docker build -t cldjunior/fot_federated:1.2 -f fot_federated/Dockerfile ./fot_federated
	docker build -t cldjunior/fot_sensors:1.2 -f devops/Dockerfile .  # <-- Corrigido

subir-gateway: rede
	docker compose -f $(GATEWAY_COMPOSE) up -d

subir-servidor: rede
	docker compose -f $(FEDERATED_COMPOSE) up -d

subir-sensors: rede
	docker compose -f $(SENSORS_COMPOSE) up -d

subir: rede subir-gateway subir-servidor subir-sensors

parar:
	docker compose -f $(GATEWAY_COMPOSE) down
	docker compose -f $(FEDERATED_COMPOSE) down
	docker compose -f $(SENSORS_COMPOSE) down

limpar:
	docker compose -f $(GATEWAY_COMPOSE) down --remove-orphans
	docker compose -f $(FEDERATED_COMPOSE) down --remove-orphans
	docker compose -f $(SENSORS_COMPOSE) down --remove-orphans

logs:
	@docker ps
