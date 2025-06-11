import rpyc
import torch
import copy
import time
import torchvision
from PIL import Image

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

pos=torch.tensor([-1.5, 0, 0.3], device='cuda')
ori=torch.tensor([0.5, -0.5, 0.5, -0.5], device='cuda')

pos=pos.repeat(100, 1)
ori=ori.repeat(100, 1)

conn = rpyc.connect('localhost', 18861, config = {"allow_public_attrs" : True})
conn._config['timeout'] = 240

start_time = time.time()

# image = torch.zeros([5, 3, 480, 640], device='cuda')
image = conn.root.render(pos, ori)

start_time = time.time()
image = copy.deepcopy(image)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# start_time = time.time()
# image=torch.load("/home/zhust/tmp.pt")
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time} seconds")
ndarr = image[0].permute(1, 2, 0).to("cpu", torch.uint8).numpy()
im = Image.fromarray(ndarr)
im.save('./test.png')
