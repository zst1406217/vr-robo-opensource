import torch
import socket
import pickle
import time

# Function to receive tensor
def receive_tensor(host, port):
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
    start_time = time.time()
    tensor = torch.tensor(pickle.loads(data), device="cuda")
    B=tensor.shape[0]
    tensor=tensor.reshape(B,-1)
    print(f"Elapsed time: {time.time() - start_time} seconds")
    return tensor

# Example usage
host = 'localhost'
port = 12345

from PIL import Image
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

received_tensor = receive_tensor(host, port)
print(received_tensor.shape)
B=received_tensor.shape[0]
received_tensor=received_tensor.reshape(B,3,240,320)

received_tensor = received_tensor.repeat(10,1,1,1)
print(received_tensor.shape)
received_tensor=preprocess(received_tensor/255)
# print(received_tensor[0])
