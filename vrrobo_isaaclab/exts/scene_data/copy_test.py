import rpyc
import torch
import copy
import time

a=torch.rand([20,3,480,640], device='cuda')
while True:
    pass

# start_time = time.time()
# b=copy.deepcopy(a).cuda()
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time} seconds")