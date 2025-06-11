import socket
import torch
import pickle
from multiprocessing import Process, Queue
import time

class TensorServer:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.queue = Queue()
        self.tensor = None
        
    def receive_tensor(self, host = 'localhost', port = 12345):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host, port))
        s.listen(1)
        conn, addr = s.accept()
        data = b""
        while True:
            packet= conn.recv(40960000)
            data += packet
            if len(packet) == 0:
                break
        conn.close()
        tensor = torch.tensor(pickle.loads(data), device="cuda")
        return tensor

    def start(self):
        self.process = Process(target=self.run)
        self.process.start()

    def run(self):
        while True:
            tensor = self.receive_tensor(self.host, self.port)
            self.queue.put(tensor)

    def get_tensor(self):
        if not self.queue.empty():
            self.tensor = self.queue.get()
            return self.tensor
        return self.tensor
    
# Start the tensor server
tensor_server = TensorServer()
tensor_server.start()

while True:
    tensor=tensor_server.get_tensor()
    if tensor is not None:
        print(tensor.shape)
    else:
        print("None")
    time.sleep(0.1)


# Example usage
for i in range(100):
    if i==100:
        received_tensor = tensor_server.get_tensor()
        print(received_tensor.shape)
    else:
        print("1")
    time.sleep(0.1)
    # print(received_tensor)