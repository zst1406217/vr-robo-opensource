#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

import torch.utils.data as data
import numpy as np
import random

class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        discriminator,
        discriminator_paras,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        clip_min_std= 1e-15,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_min_std = torch.tensor(clip_min_std, device= self.device) if isinstance(clip_min_std, (tuple, list)) else clip_min_std
        
        if discriminator is not None:
            self.discriminator=discriminator.to(self.device)
            self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=discriminator_paras["learning_rate"])
            self.discriminator.train()
            self.gp_weight=discriminator_paras["gradient_penalty"]
            
            self.dataset=MPC_dataset(file_path=discriminator_paras["file_path"])
            self.MSELoss=nn.MSELoss()
            self.train_with_amp=True
        else:
            self.train_with_amp=False
        self.train_with_image=False

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, transitions_shape, image_shape):
        if image_shape[0]:
            self.train_with_image=True
        else:
            self.train_with_image=False
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, transitions_shape, image_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        if self.train_with_image:
            self.transition.observations = obs[:, self.actor_critic.image_length:]
            self.transition.images = obs[:, :self.actor_critic.image_length].to(torch.uint8)
            self.transition.critic_observations = critic_obs
        else:
            self.transition.observations = obs
            self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def update_transitions(self, transitions):
        self.transition.transitions = transitions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_real_loss = 0
        mean_fake_loss = 0
        mean_gp_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            transitions_batch,
            image_batch,
        ) in generator:
            if self.train_with_image:
                obs_batch=torch.cat([image_batch, obs_batch], dim=-1)
                
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            
            if self.train_with_amp:
                # AMP reward
                fake=transitions_batch.to(self.device)
                if fake.shape[0]:
                    index=torch.LongTensor(random.sample(range(len(self.dataset)), fake.shape[0]))
                    real=self.dataset[index,...].to(self.device)
                    real.requires_grad_()
                    real_target=torch.ones([real.shape[0], 1], device=self.device)
                    fake_target=-torch.ones([fake.shape[0], 1], device=self.device)
                    real_score=self.discriminator(real)
                    fake_score=self.discriminator(fake)
                    gradient = torch.autograd.grad(
                        inputs=real,
                        outputs=real_score,
                        grad_outputs=torch.ones_like(real_score),
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    gradient=gradient.view(len(gradient), -1)
                    gradient_norm=gradient.norm(2, dim=1)
                    gp_loss=self.gp_weight*torch.mean(gradient_norm)
                    real_loss=self.MSELoss(real_score, real_target)
                    fake_loss=self.MSELoss(fake_score, fake_target)
            else:
                gp_loss=torch.tensor([0])
                real_loss=torch.tensor([0])
                fake_loss=torch.tensor([0])
            
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            discriminator_loss = real_loss + fake_loss + gp_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            if self.train_with_amp:
                self.discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                self.discriminator_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_fake_loss += fake_loss.item()
            mean_real_loss += real_loss.item()
            mean_gp_loss += gp_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_fake_loss /= num_updates
        mean_real_loss /= num_updates
        mean_gp_loss /= num_updates
        self.storage.clear()
        
        if hasattr(self.actor_critic, "clip_std"):
            self.actor_critic.clip_std(min= self.clip_min_std)

        return mean_value_loss, mean_surrogate_loss, mean_real_loss, mean_fake_loss, mean_gp_loss

class MPC_dataset(data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data=np.load(self.file_path)
        self.data=torch.tensor(self.data).flatten(1,2).to(torch.float).cuda()
        assert self.data.shape[0]>1024*24, "MPC dataset is too small!"
        print("Loaded MPC dataset", self.data.shape)

    def __getitem__(self, index):
        mpc_data=self.data[index]
        return mpc_data

    def __len__(self):
        return self.data.shape[0]