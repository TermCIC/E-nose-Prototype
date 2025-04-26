import serial
import json
import serial.tools.list_ports
import os


# Settings
PORT_FILE = 'port.json'
sensor_list = [f"MQ{k}" for k in range(2, 10)] + ["MQ135"] + ["GM102B", "GM302B", "GM502B", "GM702B"]
headings = ["Treatment", "Timestamp"] + sensor_list


# Function to find the working COM port
def find_working_port():
    choose = None
    saved_ports = read_port_from_file()
    ports = [port[0] for port in serial.tools.list_ports.comports()]
    ports_info_1 = [port[1] for port in serial.tools.list_ports.comports()]
    ports_info_2 = [port[2] for port in serial.tools.list_ports.comports()]
    for p in range(len(ports)):
        available = False
        if ports_info_1[p][:38] == "Silicon Labs CP210x USB to UART Bridge":
            available = True
            choose = ports[p]
        saved_ports[ports[p]] = {
            "info_1": ports_info_1[p],
            "info_2": ports_info_2[p],
            "available": available
        }
    save_port_to_file(saved_ports)
    return choose
    

# Function to read port from JSON file
def read_port_from_file():
    if os.path.exists(PORT_FILE):
        with open(PORT_FILE, 'r') as file:
            data = json.load(file)
            return data.get('port')
    return {"port": {}}


# Function to save port to JSON file
def save_port_to_file(port):
    with open(PORT_FILE, 'w') as file:
        json.dump({'port': port}, file, indent=4)
