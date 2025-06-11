#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories

import torchvision.models as models
from torchvision import transforms

class ActorCriticCNNRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        cnn_type="mobilenet_v3_small",
        pretrain=True,
        image_size=[3, 180, 320],
        num_cnn_features=576,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=rnn_hidden_size,
            num_critic_obs=rnn_hidden_size,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )
        self.preprocess = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        activation = get_activation(activation)
        
        # self.cnn_model_a = getattr(models, cnn_type)(pretrained=pretrain)
        # self.num_cnn_features = num_cnn_features
        # if cnn_type == "mobilenet_v3_small":
        #     self.cnn_model_a.classifier = nn.Identity()
        # elif cnn_type == "resnet18":
        #     self.cnn_model_a.fc = nn.Identity()
            
        self.cnn_model_a = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.num_cnn_features = 192
            
        print(f"Actor CNN: {self.cnn_model_a}")
        
        # self.cnn_model_c = getattr(models, cnn_type)(pretrained=True)
        # self.cnn_model_c.fc = nn.Identity()  # Remove the final fully connected layer
        
        self.image_size = image_size
        self.image_length = image_size[0] * image_size[1] * image_size[2]

        self.memory_a = Memory(num_actor_obs-self.image_length+self.num_cnn_features, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        if len(observations.shape)==2:
            input_image = observations[:, :self.image_length].reshape(-1, *self.image_size)
            # image_features = self.cnn_model_a(self.preprocess(input_image/255))
            image_features = self.cnn_model_a(input_image)
            input_backbone = torch.cat((image_features, observations[:, self.image_length:]), dim=1)
        else:
            shape = observations.shape
            input_image = observations[..., :self.image_length].reshape(-1, *self.image_size)
            # image_features = self.cnn_model_a(self.preprocess(input_image/255))
            image_features = self.cnn_model_a(input_image)
            image_features = image_features.reshape(shape[0], shape[1], -1)
            input_backbone = torch.cat((image_features, observations[..., self.image_length:]), dim=-1)
        input_a = self.memory_a(input_backbone, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        input_image = observations[:, :self.image_length].reshape(-1, *self.image_size)
        # image_features = self.cnn_model_a(self.preprocess(input_image/255))
        image_features = self.cnn_model_a(input_image)
        input_backbone = torch.cat((image_features, observations[:, self.image_length:]), dim=1)
        input_a = self.memory_a(input_backbone)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states
    
    @torch.no_grad()
    def clip_std(self, min= None, max= None):
        self.std.copy_(self.std.clip(min= min, max= max))


class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0
