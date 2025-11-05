# Federated Learning Project for IoT

This repository contains the full structure of a simulated federated learning system using Docker, divided into three main components:

- **`sim/`**: acts as a sensor simulator and data generator.
- **`fot_gateway/`**: acts as a gateway and FL coordination server, including MQTT support.
- **`fot_federated/`**: represents the clients/sensors that train models locally.
- **`fot_server/`**: represents the servers.

---


# Federated Learning with Devices, Virtual Sensors, and Aggregator Gateway


This project simulates a Federated Learning (FL) environment with multiple sensors and devices connected to an aggregator gateway, using Docker + Mininet architecture.

---

## Device Structure

- **Sensors**: different types, replicated for each device, totaling 65 sensors.
- **Devices (`dc01` a `dc05`)**: act as FL clients.
- **Gateways (`g01`, `g03`)**: receive data from sensors and can act as FL servers.
- **FL Server (`g01`)**: performs aggregation using Flower.

---

## Technologies and Tools

- **Docker & Docker Compose** – containerization and orchestration of the main services (*gateway*, *sensors*, and *federated clients*).  
- **Python 3** – primary programming language for simulation, sensor logic, and federated training scripts.  
- **Mininet** – used for network topology simulation between sensors and gateway (in specific scenarios).  
- **Flower (FL framework)** – communication and coordination layer for federated learning experiments.  
- **TensorFlow / TensorFlow Lite** – model development and lightweight deployment on constrained IoT devices.  
- **MQTT (via Mosquitto)** – lightweight message protocol for data exchange between sensors and gateway.  
- **Simulation with virtual sensors (`sensors.py`)** – generation and publication of synthetic IoT data for FL experiments.  

---

This stack enables the complete simulation of a **Federated Learning environment**, covering all layers — from data generation at virtual IoT sensors, through MQTT communication and local model training, up to global aggregation and evaluation at the gateway.

---


## Container Structure

| Compose Group     | Container Name                 | Function                                   | Image / Version                | IP (Docker `fot-net`)      |
|-------------------|--------------------------------|--------------------------------------------|--------------------------------|----------------------------|
| **fot_gateway**   | g01_compress                   | Gateway + FL Server (Flower) + MQTT Broker | cldjunior/fot_gateway:1.5      | 172.27.27.4                |
| **fot_federated** | dc01_compress – dc05_compress  | Federated Clients (local FL nodes)         | cldjunior/fot_federated:1.5    | 172.27.27.10–172.27.27.14  |
| **sim**           | sensors_compress               | Mininet + Virtual Sensors Orchestrator     | cldjunior/fot-fed:1.5          | 172.27.27.3                |
| *(virtual)*       | sc01–sc65                      | Virtual Sensors (Mininet hosts)            | internal Mininet nodes         | 10.0.0.1–10.0.0.65         |

**Notes**
- Each container group corresponds to a separate `docker-compose` project (`fot_gateway`, `fot_federated`, and `sim`).  
- The **gateway** (`g01_compress`) is both the **MQTT broker** and **Flower-based aggregator**.  
- **Federated clients** (`dc01–dc05_compress`) run local training rounds using the same base image (`fot_federated:1.5`).  
- The **simulation container** (`sensors_compress`) manages the Mininet network and launches the virtual sensors (`sc01–sc65`), defined in `data_hosts.json`.  
- All containers are connected through the external Docker network **`fot-net`** with fixed IP assignments for reliable communication.


## How to Use

Prerequisites

* Ubuntu 20.04  or later

* [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)


* [Docker desktop (opcional)](https://docs.docker.com/desktop/install/linux-install/)





> **TL;DR (Ubuntu 22.04 / 24.04)**

```bash
# Install Docker + Compose v2 and add your user to the docker group (log out/in after this)

sudo apt-get update && sudo apt-get install -y ca-certificates curl gnupg lsb-release

sudo install -m 0755 -d /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker $USER


# Clone and enter the repository
git clone https://github.com/cldjunior/fot_federated_model_compress.git
cd fot_federated_model_compress

You can either run pre-built images (recommended for class) or build them locally (for development and customization).

This is the fastest way to run the environment during class.
All required images are hosted on Docker Hub under cldjunior/.
# Gateway (MQTT + FL Server)
docker compose -f fot_gateway/docker-compose-gateway.yml up -d

# Sensors (Mininet + Virtual Sensors)
docker compose -f sim/docker-compose-sensors.yml up -d

# Federated Clients
docker compose -f fot_federated/docker-compose-federated.yml up -d




The Docker engine will automatically pull the following images:

cldjunior/fot_gateway:1.5
cldjunior/fot_federated:1.5
cldjunior/fot-fed:1.5


Verify containers:

docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"


Expected:

g01_compress           cldjunior/fot_gateway:1.5      Up ...
sensors_compress       cldjunior/fot-fed:1.5          Up ...
dc01_compress          cldjunior/fot_federated:1.5    Up ...
...
dc05_compress          cldjunior/fot_federated:1.5    Up ...


If you want to modify the code or build images manually, use the following commands:

# Gateway
cd fot_gateway
docker build -t cldjunior/fot_gateway:1.5 -f Dockerfile .
cd ..

# Federated Clients
cd fot_federated
docker build -t cldjunior/fot_federated:1.5 -f Dockerfile .
cd ..

# Sensors + Mininet
cd sim
docker build -t cldjunior/fot-fed:1.5 -f Dockerfile .
cd ..


Once built locally, the same launch commands above apply.

To stop all containers and remove associated volumes:

docker compose -f fot_federated/docker-compose-federated.yml down -v
docker compose -f sim/docker-compose-sensors.yml down -v
docker compose -f fot_gateway/docker-compose-gateway.yml down -v


You can also remove unused networks or images (optional):

docker network prune -f
docker image prune -f

Notes

Always start services in the order: Gateway → Sensors → Federated.
Shut down in reverse: Federated → Sensors → Gateway.

All components communicate over the external Docker network fot-net with fixed IPs.

Ensure the MQTT port 1883 is free on the host before launching.

Check running containers and images:

docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
docker images | egrep 'fot_gateway|fot_federated|fot-fed'


On some distros, you may need to add your user to the docker group and re-login:

sudo usermod -aG docker $USER


For logs and debugging:

docker compose -f fot_gateway/docker-compose-gateway.yml logs -f
docker compose -f sim/docker-compose-sensors.yml logs -f
docker compose -f fot_federated/docker-compose-federated.yml logs -f

---

## Main Files

- client_fl.py: FL client script executed on devices
- server.py: FL server script (runs on gateway g01)
- sensors.py: simulates sensors with different types of readings
- data_hosts.json: defines all sensors, gateways, and devices
- association_hosts.json: maps sensors → gateways → devices
- config.json: MQTT configuration parameters and sensor names
---

## Expected Results

- Federated training rounds between devices
- Centralized aggregation at g01
- Data simulated by sensors is published via MQTT
- Generated models:
  - model_final.h5
  - model_final.tflite
- Results are logged per device and per round
---




## Dataset Used

- The project currently uses the AirQualityUCI dataset as its data source. Values are simulated by virtual sensors based on attributes such as temperature, humidity, gases, and various chemical compounds.
- For training, the MNIST dataset was used, divided into 14 parts to simulate 14 clients.

---

## Example of MQTT Publishing

Each sensor publishes messages to the topic dev/scXX with payloads like:

json
{
  "method": "flow",
  "sensor": "temperatureSensor",
  "time": {"collect": 10000, "publish": 10000},
  "location": {"lat": -12.999999, "long": -38.507299},
  "user_id": "12345"
}


- In total, the environment simulates **65 sensor**, distributed as **13 different sensor**  per **5 devices (dc01 to dc05)**.

---



# Project Structure: fot_federated

## Main Scripts

| File                 | Description                                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `client_fl.py`       | Runs the FL client. Receives the device ID via `--device_id` and performs local training, sending weights to the server. |
| `analyze_metrics.py` | Consolidates metrics from all clients into plots and statistical analysis (training time, accuracy, etc.).               |
| `plot_comparison.py` | Generates comparison plots between strategies (FedAvg, FedProx, FedAdam, etc.).                                          |

---

## Automação via Shell Scripts

| Script          | Purpose                                                                |
| --------------- | ---------------------------------------------------------------------- |
| `start_fl.sh`   | Starts an FL client using its `DEVICE_ID`.                             |
| `restart_fl.sh` | Stops and restarts all FL containers.                                  |
| `stop_fl.sh`    | Terminates and removes federation containers.                          |
| `status_fl.sh`  | Displays the status and IPs of containers.                             |
| `logs_fl.sh`    | Saves logs from sensors, gateways, and server into the `logs/` folder. |
| `menu_fl.sh`    | Simple interface for running the above scripts in a menu format.       |

---

## Docker & Orchestration

| File                           | Description                                                                               |
| ------------------------------ | ----------------------------------------------------------------------------------------- |
| `Dockerfile`                   | Base image for `fot_federated` devices, includes dependencies like TensorFlow and Flower. |
| `docker-compose-federated.yml` | Defines the client containers. Used with helper scripts.                                  |

---

## Key Directories

| Path           | Content                                                                                       |
| -------------- | --------------------------------------------------------------------------------------------- |
| `models/dcXX/` | Each client stores its files here: metrics (`.csv`), plots (`acc_*.png`, `loss_*.png`), data. |
| `logs/`        | After running `logs_fl.sh`, container logs are stored here for analysis.                      |

---

# Project Structure: fot_gateway

This directory contains the scripts responsible for:
- Acting as a **gateway** between sensors and the FL server.
- Running a **local MQTT broker**.
- Starting the federated learning server (server.py).
---

## Main Scripts

| File             | Description                                                                                               |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| `server.py`      | Central FL coordinator. Receives client weights and applies strategies like FedAvg, FedProx, and FedAdam. |
| `mqtt-client.py` | Simulates a gateway that listens to MQTT topics and processes messages from sensors.                      |

---

## Shell Scripts

| Script             | Purpose                                                             |
| ------------------ | ------------------------------------------------------------------- |
| `start_gateway.sh` | Starts the MQTT broker, then runs `mqtt-client.py` and `server.py`. |
| `start_server.sh`  | Alternative to manually start the FL server only.                   |

---

## Docker & Orchestration

| File                         | Description                                                         |
| ---------------------------- | ------------------------------------------------------------------- |
| `Dockerfile`                 | Builds Docker image with FL server, MQTT, and Mininet dependencies. |
| `docker-compose-gateway.yml` | Orchestrates gateway containers, setting static IP and entrypoint.  |
| `mosquitto-no-auth.conf`     | Mosquitto configuration for no-authentication mode.                 |

---

## Notes

- **(IMPORTANT) The gateway must be the first service to be initialized.**
- It acts as the bridge between sensors and the FL server
- You can monitor MQTT and server logs directly in the container.

---

# Project Structure: fot_sensors

This directory contains the simulated sensors. They publish data to the gateway and may also simulate data generation for federated learning training.

---

## Main Files

| File                     | Description                                                                 |
| ------------------------ | --------------------------------------------------------------------------- |
| `main.py`                | Simulator entry point. Executes main logic and initializes the environment. |
| `sim.py`                 | Defines the simulation behavior using the Mininet network structure.        |
| `tatu.py`                | Utility module (e.g., link configuration, topologies).                      |
| `sensors.py`             | Contains logic for creating and assigning sensors to virtual devices.       |
| `config.json`            | General configuration file to control simulation behavior.                  |
| `association_hosts.json` | Defines which sensors are assigned to which devices in the topology.        |
| `data_hosts.json`        | Specifies each device’s data, such as IPs and logical names.                |



---

## Docker & Orchestration

| File                         | Description                                            |
| ---------------------------- | ------------------------------------------------------ |
| `Dockerfile`                 | Base image for sensors with FL and TensorFlow support. |
| `docker-compose-sensors.yml` | Orchestrates sensor containers.                        |
| `local_docker.env`           | Sets environment variables like Docker subnet.         |

---

## Notes

- Each sensor has its own folder in fot_federated/models/dcXX containing its local metrics and data.
- Execution is modular and parallel, simulating multiple federated devices in a networked environment.


# 📬 Contact

Claudio Junior Nascimento da Silva
Ph.D. Candidate in Computer Science – Federal University of Bahia (UFBA)
Member of the Wiser Research Group – IC/UFBA
Lattes CV: http://lattes.cnpq.br/1148018071335071

Advisor: Prof. Dr. Cássio Vinícius Serafim Prazeres

GitHub / LinkedIn: @cldjunior

# 🎓 Acknowledgments
This research was partially funded by:
- CAPES (Coordenação de Aperfeiçoamento de Pessoal de Nível Superior) – Finance Code 001
- FAPESB – Scholarship Program and Grant INCITE PIE0002/2022
- CNPq (Conselho Nacional de Desenvolvimento Científico e Tecnológico) – Grant No. 403231/2023-0

---

