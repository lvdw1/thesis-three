import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from collections import deque
import socket
import time

from utils import *
from processor import *
from transformer import *

class UnityCarEnv:
    """
    Environment for communicating with the Unity simulator.
    
    Observation:
      Unity sends a 7-dimensional sensor vector:
          [time, x_pos, z_pos, yaw, v_x, v_y, v_yaw]
      This raw data is processed (e.g., combined with track information and PCA-transformed)
    
    Action:
      A 3-dimensional vector: [steering, throttle, brake]
    """
    def __init__(self, processor, transformer, json_path = 'sim/tracks/track17.json', host='127.0.0.1', port=65432):
        # Dimensions: raw sensor, processed state
        self.raw_state_dim = 7
        self.action_dim = 3  # [steering, throttle, brake]
        self.max_episode_steps = 1000
        self.current_step = 0
        self.state = None  # Actual state from Unity will be stored here
        
        self.host = host
        self.port = port

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(5)

        self.connection, addr = self.server_socket.accept()
        print("Connected to Unity at", addr)

        self.processor = processor
        self.transformer = transformer

        self.transformer.load()
        print("tf loaded")
        self.json_path = json_path

        print(f"[UnityCarEnv] Connected to Unity at {host}:{port}")

    def send_reset_to_unity(self):
        """
        Sends a reset command to Unity to restart the simulation.
        Replace this stub with your actual communication logic.
        """
        message = "reset\n"
        self.connection.sendall(message.encode())
        print("[UnityCarEnv] Sent reset command to Unity.")

    def send_action_to_unity(self, action):
        """
        Sends the action command to Unity.
        Action is expected to be a 3-dimensional vector: [steering, throttle, brake].
        """
        message = f"{action[0]},{action[1]},{action[2]}\n"
        self.connection.sendall(message.encode())
        print(f"[UnityCarEnv] Sent action: {message.strip()}")

    def receive_observation_from_unity(self):
        buffer = ""
        while "\n" not in buffer:
            buffer += self.connection.recv(1024).decode('utf-8')
        # Split by newline and take the first complete message.
        line, remainder = buffer.split('\n', 1)
        # Optionally, save 'remainder' for the next call.
        raw_data = line.strip()
        fields = raw_data.split(',')
        sensor_data = {
            "time": time.time(),
            "x_pos": float(fields[0]),
            "z_pos": float(fields[1]),
            "yaw_angle": -float(fields[2]) + 90,
            "long_vel": float(fields[3]),
            "lat_vel": float(fields[4]),
            "yaw_rate": float(fields[5]),
            "steering": None,
            "throttle": None,
            "brake": None
        }
        print(f"[UnityCarEnv] Received sensor data: {sensor_data}")
        return sensor_data

    def process_unity_observation(self, sensor_data):
        """
        Processes raw sensor data along with track data to produce a processed state vector.
        """
        track_data = self.processor.build_track_data(self.json_path)
        frame = self.processor.process_frame(sensor_data, track_data)

        df_single = pd.DataFrame([frame])
        df_features = df_single.drop(columns=["time","x_pos", "z_pos", "yaw_angle"])

        df_trans = self.transformer.transform(df_features)
        return df_trans.values

    def calculate_reward(self, sensor_data):
        """
        Computes reward based on sensor data.
        Dummy function for the moment
        """
        reward = abs(sensor_data["v_x"])
        return reward

    def reset(self):
        """
        Resets the Unity simulation and returns the initial processed state.
        """
        self.current_step = 0
        self.send_reset_to_unity()
        time.sleep(0.1)  # Allow time for Unity to process the reset.
        sensor_data = self.receive_observation_from_unity()
        self.state = self.process_unity_observation(sensor_data)
        return self.state

    def step(self, action):
        """
        Sends an action to Unity, receives the next observation, computes the reward,
        and returns the new state along with the done flag and diagnostic info.
        """
        self.current_step += 1
        self.send_action_to_unity(action)
        sensor_data = self.receive_observation_from_unity()
        self.state = self.process_unity_observation(sensor_data)
        reward = self.calculate_reward(sensor_data)
        done = self.current_step >= self.max_episode_steps
        return self.state, reward, done

    def close(self):
        """
        Closes the connection to Unity.
        """
        self.connection.close()
        print("[UnityCarEnv] Connection closed.")

class PPOTransformerPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=192, nhead=4, num_layers=4, mlp_ratio=2.0, context_length=5):
        super(PPOTransformerPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_length = context_length
        
        # Embed the state into a d_model-dimensional space.
        self.input_fc = nn.Linear(state_dim, d_model)
        # Learnable positional embeddings.
        self.pos_embedding = nn.Parameter(torch.zeros(context_length, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        feedforward_dim = int(d_model * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=feedforward_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Actor head: produces the mean of the Gaussian for the action.
        self.actor_head = nn.Linear(d_model, action_dim)
        # Critic head: outputs a scalar value for the state.
        self.critic_head = nn.Linear(d_model, 1)
        # Learnable log standard deviation for the action distribution.
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, x):
        """
        x: [batch_size, state_dim]
        We replicate the state to form a sequence of fixed length (context_length)
        and then process it through the transformer.
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # [1, state_dim]
            x = x.repeat(self.context_length, 1)  # [context_length, state_dim]
        elif len(x.shape) == 2:
            batch_size = x.size(0)
            x = x.unsqueeze(1)  # [batch_size, 1, state_dim]
            x = x.repeat(1, self.context_length, 1)  # [batch_size, context_length, state_dim]
            x = x.transpose(0, 1)  # [context_length, batch_size, state_dim]
        else:
            raise ValueError("Unsupported input shape")
        
        x = self.input_fc(x)  # [context_length, batch_size, d_model]
        x = x + self.pos_embedding.unsqueeze(1)  # add positional embedding
        x = self.transformer_encoder(x)  # [context_length, batch_size, d_model]
        x = x[-1, :, :]  # use the last token's representation; shape: [batch_size, d_model]
        
        action_mean = self.actor_head(x)  # [batch_size, action_dim]
        value = self.critic_head(x)       # [batch_size, 1]
        std = self.log_std.exp().expand_as(action_mean)
        return action_mean, value.squeeze(-1), std

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4, entropy_coef=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma              # Discount factor
        self.eps_clip = eps_clip        # Clipping epsilon for PPO
        self.K_epochs = K_epochs        # Number of epochs per update
        self.entropy_coef = entropy_coef

        self.policy = PPOTransformerPolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()
        
        # Memory to store transitions (on-policy)
        # Each transition: (state, action, log_prob, reward, done)
        self.memory = []

    def select_action(self, state):
        """
        Given a state, sample an action from the policy's distribution.
        Returns the action, its log probability, and the value estimate.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # [1, state_dim]
        with torch.no_grad():
            action_mean, value, std = self.policy(state_tensor)
        dist = distributions.Normal(action_mean, std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach().numpy()[0], action_logprob.detach(), value.detach()

    def store_transition(self, transition):
        """
        Store a transition tuple:
        (state, action, log_prob, reward, done)
        """
        self.memory.append(transition)

    def update(self):
        """
        Update the policy using the collected on-policy data.
        This includes computing discounted returns and advantages,
        then optimizing the PPO clipped objective with an entropy bonus.
        """
        # Convert lists of transitions into tensors.
        states = torch.FloatTensor([t[0] for t in self.memory])
        actions = torch.FloatTensor([t[1] for t in self.memory])
        old_logprobs = torch.FloatTensor([t[2] for t in self.memory])
        rewards = [t[3] for t in self.memory]
        dones = [t[4] for t in self.memory]
        
        # Compute discounted rewards.
        discounted_rewards = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        
        # Evaluate the current policy for the stored states.
        action_means, state_values, std = self.policy(states)
        dist = distributions.Normal(action_means, std)
        logprobs = dist.log_prob(actions).sum(dim=-1)
        
        # Calculate advantages.
        advantages = discounted_rewards - state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize policy for K epochs.
        for _ in range(self.K_epochs):
            action_means, state_values, std = self.policy(states)
            dist = distributions.Normal(action_means, std)
            logprobs = dist.log_prob(actions).sum(dim=-1)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Clipped surrogate objective.
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss.
            value_loss = self.MseLoss(state_values, discounted_rewards)
            
            # Entropy bonus to encourage exploration.
            entropy_loss = -self.entropy_coef * dist.entropy().mean()
            
            # Total loss.
            loss = policy_loss + 0.5 * value_loss + entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Clear memory after the update.
        self.memory = []


def train_rl(num_episodes=10, max_steps=1000):
    """
    Training loop that interacts with Unity through the environment,
    collects transitions, and updates the PPO agent.
    
    Note: Assumes that a valid json_path is defined and that the processor
    and transformer modules are properly implemented.
    """
    # Assume processor and transformer are imported from your modules.
    processor = Processor()
    transformer = FeatureTransformer()
    
    # Create the environment.
    env = UnityCarEnv(processor, transformer)
    
    # For processed states, assume the dimension is 121.
    state_dim = 121
    action_dim = env.action_dim  # 3
    agent = PPOAgent(state_dim, action_dim)
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done and steps < max_steps:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition((state, action, log_prob, reward, done))
            state = next_state
            ep_reward += reward
            steps += 1
        
        # Update policy after each episode.
        agent.update()
        episode_rewards.append(ep_reward)
        print(f"Episode {ep+1}/{num_episodes}, Reward: {ep_reward}")
    
    env.close()
    return episode_rewards

if __name__ == "__main__":
    # To train the agent:
    rewards_df = train_rl(num_episodes=10, max_steps=1000)
    print(rewards_df)
