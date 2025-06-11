import torch

# Load the JIT script
jit_script_path = '/home/zhust/codes/go2-loco-isaaclab/logs/rsl_rl/unitree_go2_gsfix/2025-01-02_21-09-00/exported/policy.pt'
model = torch.jit.load(jit_script_path)
print(model)

# Test the loaded model
for i in range(10):
    input_tensor = torch.zeros([1,609])  # Example input tensor
    output = model(input_tensor)

    print(output)
    model.reset()