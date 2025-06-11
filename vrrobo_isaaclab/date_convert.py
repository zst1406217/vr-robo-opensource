import numpy as np
np.set_printoptions(precision=5,suppress=True)

data=np.load("./exts/mpc_data/mpc_data_no_command_go2.npy")
print(data[0])
data=np.concatenate([data[:,:,:12],data[:,:,13:15],data[:,:,18:19]],axis=2)
# data=np.concatenate([data[:,:,:12],data[:,:,13:]],axis=2)
# data=data[:,:,:12]
new_data=data.copy()
print(data[0])
print(data.shape)

new_names=['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
old_names=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']

for i in range(12):
    name=new_names[i]
    new_data[:,:,i]=data[:,:,old_names.index(name)]

print(new_data[0])
print(new_data.shape)
np.save("./exts/mpc_data/mpc_data_go2.npy",new_data)

# data=np.load("./exts/mpc_data/mpc_data_go2.npy")
# print(data[0])