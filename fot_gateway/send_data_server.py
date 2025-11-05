import socket
import time
import os
import sys

# Usage
#file_path = 'data.json'
#new_file_path = 'data-send.json'
#server_ip = sys.argv[1]
server_port = 1234

# O server_ip é recebido via argmento pelo mqtt_client.py
# file_path recebido no mqqt_client.py é o "send-data.json"
def send_file(server_ip, file_path):
    #while True:
        #time.sleep(20)
        # os.system("clear")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((server_ip, server_port))

            remove_invalid_data(file_path)
            try:
                with open(file_path, 'rb') as file:
                    data = file.read()
                    s.sendall(data)
                print("File sent successfully!")
            except Exception as err:
                print("Error sending file \n", err)
        except Exception as err:
            print("Waiting server \n", err)


def remove_invalid_data(file_path):
    text_to_remove = '"method":"flow"'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Filter out lines containing the text to remove
    remaining_lines = [line for line in lines if text_to_remove not in line]

    with open(file_path, 'w') as file:
        file.writelines(remaining_lines)


if __name__ == '__main__':
    send_file("", "")
