import paho.mqtt.client as pub
import sensors
import json
import multiprocessing
import logging
from datetime import datetime
import os
import pytz

# You don't need to change this file. Just change sensors.py and config.json

from time import sleep


# Definindo o fuso horário local
local_timezone = pytz.timezone('America/Sao_Paulo')

procs = []
pub_client = pub.Client(client_id='', clean_session=True, userdata=None, protocol=pub.MQTTv31)
mqttBroker = ""


class sensorProcess(multiprocessing.Process):
    def __init__(self, idP, deviceName, sensorName, met, topic, topicError, pub_client, collectTime, publishTime, datetime_pub, location, user_id):
        multiprocessing.Process.__init__(self)
        self.processID = idP
        self.deviceName = deviceName
        self.sensorName = sensorName
        self.met = met
        self.topic = topic
        self.topicError = topicError
        self.pub_client = pub_client
        self.publishTime = publishTime
        self.collectTime = collectTime
        self.datetime_pub = datetime_pub
        self.location = location
        self.user_id = user_id

    def run(self):
        print("Starting process " + self.processID)

        if (self.met == "EVENT"):
            buildEventAnwserDevice(self.deviceName, self.sensorName, self.topic, self.topicError, self.pub_client,
                                   self.collectTime, self.publishTime)
        elif (self.met == "GET"):
            buildGetAnwserDevice(self.deviceName, self.sensorName, self.topic, self.topicError, self.pub_client)
        elif (self.met == "flow"):
            buildFlowAnwserDevice(self.deviceName, self.sensorName, self.topic, self.topicError, self.pub_client,
                                  self.collectTime, self.publishTime, self.datetime_pub, self.location, self.user_id)

        print("Stopping process " + self.processID)


class actuatorProcess(multiprocessing.Process):
    def __init__(self, idP, deviceName, sensorName, met, topic, topicError, pub_client, value):
        multiprocessing.Process.__init__(self)
        self.processID = idP
        self.deviceName = deviceName
        self.sensorName = sensorName
        self.met = met
        self.topic = topic
        self.topicError = topicError
        self.pub_client = pub_client
        self.value = value

    def run(self):
        print("Starting process " + self.processID)

        if (self.met == "POST"):
            buildPostAnwserDevice(self.deviceName, self.sensorName, self.topic, self.topicError, self.pub_client,
                                  self.value)

        print("Stopping process " + self.processID)


def buildFlowAnwserDevice(deviceName, sensorName, topic, topicError, pub_client, collectTime, publishTime, datetime_pub, location, user_id):
    value = ""
    t = 0
    try:
        methodFLOW = getattr(sensors, sensorName)
        listValues = []
        while True:
            listValues.append(str(methodFLOW()))
            t = t + collectTime + 1000
            datetime_pub =  datetime.now(local_timezone).strftime('%Y-%m-%d %H:%M:%S')
            # Request: {"method":"FLOW", "sensor":"sensorName", "time":{"collect":collectTime,"publish":publishTime}}
            responseModel = {"code": "post", "method": "flow", "header": {"sensor": sensorName, "device": deviceName,
                                                                          "time": {"collect": collectTime,
                                                                                   "publish": publishTime},
                                                                          "location": {"lat": location["lat"],
                                                                                       "long": location["long"]},
                                                                          "user_id": user_id},
                             "data": [listValues[0], listValues[0]], "datetime_pub": datetime_pub}
            # responseModel = {"CODE":"POST","METHOD":"FLOW","HEADER":{"NAME":deviceName},"BODY":{sensorName:listValues,"FLOW":{"collect":(collectTime),"publish":(publishTime)}}}
            response = json.dumps(responseModel)
            # print(topic)
            # print(response)
            pub_client.publish(topic, response)
            t = 0
            listValues = []
            sleep(int(publishTime / 1000))
    except:
        print("erro")
        errorMessage = "There is no " + sensorName + " sensor in device " + deviceName
        errorNumber = 1
        responseModel = {"code": "ERROR", "number": errorNumber, "message": errorMessage}
        response = json.dumps(responseModel)
        pub_client.publish(topicError, response)


def buildEventAnwserDevice(deviceName, sensorName, topic, topicError, pub_client, collectTime, publishTime):
    try:
        methodEvent = getattr(sensors, sensorName)
        value = methodEvent()
        # Request: {"method":"EVENT", "sensor":"sensorName", "time":{"collect":collectTime}}
        responseModel = {"CODE": "POST", "METHOD": "EVENT", "HEADER": {"NAME": deviceName},
                         "BODY": {sensorName: value, "EVENT": {"collect": (collectTime), "publish": (publishTime)}}}
        response = json.dumps(responseModel)
        pub_client.publish(topic, response)

        while True:
            sleep(collectTime)
            publishTime = publishTime + collectTime
            aux = methodEvent()
            if aux != value:
                value = aux
                # Request: {"method":"EVENT", "sensor":"deviceName", "time":{"collect":collectTime}}
                responseModel = {"CODE": "POST", "METHOD": "EVENT", "HEADER": {"NAME": deviceName},
                                 "BODY": {sensorName: value,
                                          "EVENT": {"collect": (collectTime), "publish": (publishTime)}}}
                response = json.dumps(responseModel)
                pub_client.publish(topic, response)
    except:
        errorMessage = "There is no " + sensorName + " sensor in device " + deviceName
        errorNumber = 1
        responseModel = {"code": "ERROR", "number": errorNumber, "message": errorMessage}
        response = json.dumps(responseModel)
        pub_client.publish(topicError, response)


def buildGetAnwserDevice(deviceName, sensorName, topic, topicError, pub_client):
    try:
        methodGet = getattr(sensors, sensorName)
        value = methodGet()

        # Request: {"method":"GET", "sensor":"sensorName"}
        responseModel = {'CODE': 'POST', 'METHOD': 'GET', 'HEADER': {'NAME': deviceName}, 'BODY': {sensorName: value}}
        response = json.dumps(responseModel)

        pub_client.publish(topic, response)
    except:
        errorMessage = "There is no " + sensorName + " sensor in device " + deviceName
        errorNumber = 1
        responseModel = {"code": "ERROR", "number": errorNumber, "message": errorMessage}
        response = json.dumps(responseModel)
        pub_client.publish(topicError, response)


def buildPostAnwserDevice(deviceName, sensorName, topic, topicError, pub_client, value):
    try:
        # p = "__" + sensorName + "__"
        methodPost = getattr(sensors, sensorName)
        methodPost(value)

        # Request: {"method":"POST", "sensor":"sensorName", "value":value}
        responseModel = {"code": "POST", "method": "POST", "sensor": sensorName, "value": value}
        response = json.dumps(responseModel)
        pub_client.publish(topic, response)
    except:
        errorMessage = "There is no " + sensorName + " sensor in device " + deviceName
        errorNumber = 1
        responseModel = {"code": "ERROR", "number": errorNumber, "message": errorMessage}
        response = json.dumps(responseModel)
        pub_client.publish(topicError, response)


# def status ():
# {"method":"EVENT", "sensor":"soundSensor", "status":False}
# {"method":"STOP", "target": "EVENT", "sensor":"soundSensor"}

def on_disconnect(mqttc, obj, rc):
    print("disconnected tatu!")  


def main(data, msg):
    logging.basicConfig(filename='./python.log', level=logging.INFO)
    global mqttBroker
    # print("oi")
    mqttBroker = data["mqttBroker"]
    mqttPort = data["mqttPort"]
    mqttUsername = data["mqttUsername"]
    mqttPassword = data["mqttPassword"]
    deviceName = data["deviceName"]
    topic = data["topicPrefix"] + deviceName + data["topicRes"]
    topicError = data["topicPrefix"] + deviceName + data["topicErr"]

    msgJson = json.loads(msg.payload)
    sensorName = msgJson["sensor"]
    met = msgJson["method"]

    print("-------------------------------------------------")
    print("| Topic:" + str(msg.topic))
    print("| Message: " + str(msg.payload))
    print("-------------------------------------------------")
    idP = met + "_" + deviceName + "_" + sensorName
    print("Metodo " + idP)
    # pub_client = pub.Client(idP,protocol=pub.MQTTv31)
    # pub_client.username_pw_set(mqttUsername, mqttPassword)
    # pub_client.user_data_set(data)
    pub_client.on_disconnect = on_disconnect
    pub_client.connect(mqttBroker, int(mqttPort), 1800)
    print("chamad")

    if (met == "STOP"):
        # {"method":"STOP", "target": "EVENT", "sensor":"soundSensor"}
        idP = msgJson["target"] + "_" + deviceName + "_" + sensorName
        stopped = False
        for proc in procs:
            if (proc.processID == idP):
                print("Stopping process " + proc.processID)
                procs.remove(proc)
                proc.terminate()
                stopped = True
                break
        if not stopped:
            # error in json to /ERR
            errorMessage = "There is no running process named " + idP
            errorNumber = 2
            responseModel = {"code": "ERROR", "number": errorNumber, "message": errorMessage}
            response = json.dumps(responseModel)
            pub_client.publish(topicError, response)
    elif (met == "POST"):
        # {"method":"POST", "sensor":"sensorName", "value":value}
        value = msgJson["value"]
        proc = actuatorProcess(idP, deviceName, sensorName, met, topic, topicError, pub_client, value)
        procs.append(proc)
        proc.start()
    else:
        if (met == "GET"):
            collectTime = 0
            publishTime = 0
        elif (met == "flow"):
            time = msgJson["time"]
            collectTime = time["collect"]
            publishTime = time["publish"]
            datetime_pub =  datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            location = msgJson["location"]
            user_id = msgJson["user_id"]

        elif (met == "EVENT"):
            time = msgJson["time"]
            collectTime = time["collect"]
            publishTime = 0

        proc = sensorProcess(idP, deviceName, sensorName, met, topic, topicError, pub_client, collectTime, publishTime, datetime_pub, location, user_id)
        procs.append(proc)
        proc.start()
