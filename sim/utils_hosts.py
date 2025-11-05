# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import io
import json

# Raiz = pasta onde este arquivo está
SIM_ROOT = os.path.dirname(os.path.abspath(__file__))

class to_object(object):
    def __init__(self, j):
        # no Py2 json.loads devolve unicode; tudo bem
        self.__dict__ = json.loads(j)

def _join(*parts):
    return os.path.join(SIM_ROOT, *parts)

def _read_jsonl(abspath):
    if not os.path.exists(abspath):
        raise IOError("Arquivo nao encontrado: {}".format(abspath))
    lines = []
    with io.open(abspath, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                lines.append(ln)
    return lines

def return_hosts():
    path = _join("data_hosts.json")
    lines = _read_jsonl(path)
    return [to_object(ln) for ln in lines]

def return_association():
    path = _join("association_hosts.json")
    lines = _read_jsonl(path)
    devices = []
    for ln in lines:
        obj = to_object(ln)
        if getattr(obj, "name_gateway", None) != "cloud":
            devices.append(obj)
    return devices

def return_hosts_per_type(type_host):
    hosts = return_hosts()
    return [h for h in hosts if getattr(h, "type", None) == type_host]

def return_user(user_id):
    path = _join("user.json")
    if not os.path.exists(path):
        return None
    with io.open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i in data:
        if i.get("user_id") == user_id:
            return i
    return None

def write_host(st):
    path = _join("data_hosts.json")
    with io.open(path, "a", encoding="utf-8") as x:
        x.write((u"{}".format(st)).rstrip() + u"\n")

def write_hosts(h):
    for item in h:
        write_host(json.dumps(item))

def return_host_per_name(name_host):
    for h in return_hosts():
        if (str(getattr(h, "name", "")) == name_host) or (str(getattr(h, "name_iot", "")) == name_host):
            return h

def update_flow(value):
    cfg = _join("config.json")
    if not os.path.exists(cfg):
        return
    with io.open(cfg, "r", encoding="utf-8") as a_file:
        json_object = json.load(a_file)
    value = int(value)
    if int(json_object.get("publish", 0)) != value:
        json_object["publish"] = value
        json_object["collect"] = value
        with io.open(cfg, "w", encoding="utf-8") as a_file:
            json.dump(json_object, a_file)

def get_pub():
    cfg = _join("config.json")
    with io.open(cfg, "r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("publish", 0))


