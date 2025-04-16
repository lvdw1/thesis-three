import numpy as np
import math as m
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
import socket
import time
import subprocess

from utils import *
from processor import *
from transformer import *
from nndriver_v2 import NNModel

def compute_reward(state, next_state, action):

    steering = action[0].item()
    throttle = action[1].item()
    brake = action[2].item()

    # factors for the reward function
    speed_reward = 0.5

    steering_punishment = 0.2

    # Check for exceptions - termination conditions
    dist_yel = next_state["yr12"].item()
    dist_bl = next_state["br12"].item()

    # Check if car is on the track, to prevent hitting cones
    if dist_yel < 0.75 or dist_bl < 0.75:
        done = 0
        return -100, done

    # Avoid 180s
    elif dist_yel == 20.0 and dist_bl == 20.0:
        done = 0
        return -300, done

    # No exception happened, so we can just return an actual reward value
    else:
        done = 0
        penalty = 0
        reward = 0

        # Reward speed
        reward += next_state["long_vel"].item()**2*speed_reward
        reward += next_state["ax"].item()**2 + next_state["ay"].item()**2

        # # Crafting the brake
        if brake == 0.0:
            reward += 100
        # else:
        #     # reward += (m.exp(10*brake)-1)/(m.exp(10)-1)
        #     reward += 0.0

        # Penalize understeer
        penalty += abs(steering)*steering_punishment

        # Penalize jitter
        # penalty += abs(next_state["steering"].item() - state["steering"])*jitter_punishment

        return reward-penalty, done

def launch_sim():
    return subprocess.Popen(["open", "unity/Simulator_WithTrackGeneration/sim_normal.app"])

class UnityEnv:
    """
    Handles communication with Unity.
    """
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"[UnityEnv] Server listening on {self.host}:{self.port}...")
        self.sim_process = subprocess.Popen(["open", "unity/Simulator_WithTrackGeneration/sim_normal.app"])
        self.client_socket, self.addr = self.server_socket.accept()
        print(f"[UnityEnv] Connection from {self.addr}")

    def receive_state(self):
        raw_data = self.client_socket.recv(1024).decode('utf-8').strip()
        messages = raw_data.splitlines()
        # Iterate over messages in reverse order
        for msg in reversed(messages):
            msg = msg.strip()
            fields = msg.split(',')
            if len(fields) >= 6:
                state = {
                    "time": time.time(),
                    "x_pos": float(fields[0]),
                    "z_pos": float(fields[1]),
                    "yaw_angle": -float(fields[2]) + 90,
                    "long_vel": float(fields[3]),
                    "lat_vel": float(fields[4]),
                    "yaw_rate": float(fields[5]),
                    "steering": 0.0,
                    "throttle": 0.0,
                    "brake": 0.0
                }
                return state
        raise ValueError("Incomplete state received, not enough fields in any message.")

    def send_command(self, steering, throttle, brake):
        message = f"{steering},{throttle},{0.0}\n"
        try:
            self.client_socket.sendall(message.encode())
        except Exception as e:
            print("[UnityEnv] Error sending command:", e)
            self.reconnect()

    def send_reset(self):
        message = "reset\n"
        try:
            self.client_socket.sendall(message.encode())
            print("[UnityEnv] Reset command sent.")
            self.client_socket.close()
            # Wait for Unity to reconnect (after scene reload)
            print("[UnityEnv] Waiting for Unity to reconnect after reset...")
            self.client_socket, self.addr = self.server_socket.accept()
            print(f"[UnityEnv] Reconnected: {self.addr}")
        except Exception as e:
            print("[UnityEnv] Error sending reset command:", e)
            self.reconnect()

    def reconnect(self):
        try:
            if self.client_socket:
                self.client_socket.close()
        except Exception as e:
            print("[UnityEnv] Error during reconnect cleanup:", e)
        print("[UnityEnv] Waiting for Unity to reconnect...")
        self.client_socket, self.addr = self.server_socket.accept()
        print(f"[UnityEnv] Reconnected: {self.addr}")

    def close(self):
        try:
            if self.client_socket:
                self.client_socket.close()
        except:
            pass
        self.server_socket.close()
        print("[UnityEnv] Connection closed.")

class Actor:
    """
    Wraps a pretrained NNModel and its preprocessing to produce control outputs.
    """
    def __init__(self, processor, transformer, model_path=None, track_data=None, output_csv=None):
        self.output_csv = output_csv
        self.processor = processor
        self.transformer = transformer
        self.transformer.load()
        self.nn_model = NNModel()
        self.nn_model.load(model_path)
        self.nn_model.eval()
        self.track_data = track_data

    def process(self, state):
        frame = self.processor.process_frame(state, self.track_data)
        df_single = pd.DataFrame([frame])
        df_features = df_single.drop(columns=["time", "x_pos", "z_pos", "yaw_angle", "steering", "throttle", "brake"], errors='ignore')
        df_trans = self.transformer.transform(df_features)
        return df_features.astype(float), df_trans.astype(float)

    def act(self, state, preprocessed=None):
        if preprocessed is None:
            preprocessed = self.process(state)
        prediction = self.nn_model.predict(preprocessed)[0]
        st_pred, th_pred, br_pred = prediction
        if br_pred < 0.05:
            br_pred = 0.0
        return st_pred, th_pred, br_pred

class Critic(nn.Module):
    """
    A simple feedforward network to estimate state value.
    """
    def __init__(self, input_size, hidden_layer_sizes=(20, 20)):
        super(Critic, self).__init__()
        layers = []
        last_size = input_size
        for hidden in hidden_layer_sizes:
            layers.append(nn.Linear(last_size, hidden))
            layers.append(nn.ReLU())
            last_size = hidden
        layers.append(nn.Linear(last_size, 1))
        self.value_network = nn.Sequential(*layers)

    def forward(self, x):
        return self.value_network(x)

class PPO:
    def __init__(self, actor, critic, actor_optimizer, critic_optimizer,
                 clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, num_epochs=10, batch_size=64,
                 gamma=0.99, lam=0.95, steering_std=0.01, throttle_std=0.01, brake_std=0.01, device='mps'):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.device = device

        self.steering_std = steering_std
        self.throttle_std = throttle_std
        self.brake_std = brake_std

    def evaluate_actions(self, states, actions):
        mean_actions = self.actor.nn_model(states)
        # Create a per-dimension standard deviation tensor.
        std_vector = torch.tensor(
            [self.steering_std, self.throttle_std, self.brake_std],
            device=mean_actions.device,
            dtype=mean_actions.dtype
        )
        # Expand std_vector to match the batch size of mean_actions.
        std_vector = std_vector.unsqueeze(0).expand_as(mean_actions)
        
        # Create a Normal distribution with the per-dimension standard deviation.
        dist = torch.distributions.Normal(mean_actions, std_vector)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(states).squeeze(-1)
        return log_probs, entropy, values

    def compute_gae(self, rewards, values, dones, next_value):
        gae = 0
        returns = []
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, rollouts):
        states = torch.FloatTensor(np.array(rollouts['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(rollouts['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(rollouts['log_probs'])).to(self.device)
        returns = torch.FloatTensor(np.array(rollouts['returns'])).to(self.device)
        advantages = torch.FloatTensor(np.array(rollouts['advantages'])).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.num_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                mb_inds = indices[start:end]
                mb_states = states[mb_inds]
                mb_actions = actions[mb_inds]
                mb_old_log_probs = old_log_probs[mb_inds]
                mb_returns = returns[mb_inds]
                mb_advantages = advantages[mb_inds]

                log_probs, entropy, values = self.evaluate_actions(mb_states, mb_actions)
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values, mb_returns.squeeze(-1))
                actor_loss = policy_loss - self.entropy_coef * entropy.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.nn_model.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

    def train(self, env, rollout_length=2048):
        rollouts = {'states': [], 'actions': [], 'log_probs': [],
                    'rewards': [], 'dones': [], 'values': []}
        state = env.receive_state()
        features, preprocessed = self.actor.process(state)
        state_tensor = torch.FloatTensor(preprocessed.values).to(self.device)
        
        episode_return = 0.0
        for step in range(rollout_length):
            with torch.no_grad():
                mean_action = self.actor.nn_model(state_tensor)
                
                # Create a per-dimension standard deviation tensor.
                std_vector = torch.tensor(
                    [self.steering_std, self.throttle_std, self.brake_std],
                    device=mean_action.device, 
                    dtype=mean_action.dtype
                ).unsqueeze(0).expand_as(mean_action)  # Expand to match the batch shape.

                # Create a Normal distribution.
                dist = torch.distributions.Normal(mean_action, std_vector)
                
                # Sample the action from the distribution.
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = self.critic(state_tensor).squeeze(-1)
            
            action = action.squeeze(0)
            env.send_command(action[0].item(), action[1].item(), action[2].item())
            next_state = env.receive_state()
            next_features, next_preprocessed = self.actor.process(next_state)
            
            # Compute reward and done flag
            reward, done = compute_reward(features, next_features, action)
            
            rollouts['states'].append(state_tensor.cpu().squeeze(0).numpy())
            rollouts['actions'].append(action.cpu().numpy())
            rollouts['log_probs'].append(log_prob.cpu().numpy())
            rollouts['rewards'].append(reward)
            rollouts['dones'].append(done)
            rollouts['values'].append(value.item())

            episode_return += reward
            # If early termination, break out of the loop.
            if done:
                state = next_state
                break

            state_tensor = torch.FloatTensor(next_preprocessed.values.astype(np.float32)).to(self.device)
        
        print(f"[Episode] Early termination at step {step}, total reward: {episode_return}")
        if done:
            next_value = 0.0
        else:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(self.actor.process(state)[1].values.astype(np.float32)).to(self.device)
                next_value = self.critic(next_state_tensor).item()
        
        returns = self.compute_gae(rollouts['rewards'], rollouts['values'], rollouts['dones'], next_value)
        advantages = [ret - val for ret, val in zip(returns, rollouts['values'])]

        rollouts['advantages'] = advantages
        rollouts['returns'] = returns

        # if len(returns) > 0:
        #     total_discounted_reward = returns[0]
        #     print("Total Discounted Reward for this run:", total_discounted_reward)

        self.update(rollouts)
        env.send_reset()

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    unity = UnityEnv(host='127.0.0.1', port=65432)
    processor = Processor()
    transformer = FeatureTransformer()
    track_data = processor.build_track_data("sim/tracks/validation/normal.json")
    
    # Initialize the Actor without loading imitation learning weights
    actor = Actor(processor, transformer, model_path="models/networks/nn_model_corrected_validation_double_005.pt", track_data=track_data)
    # Load PPO-trained actor checkpoint
    # actor_checkpoint = torch.load("models/rlmodels/second_gen/ppo_actor_350.pth", map_location=device)
    # actor.nn_model.load_state_dict(actor_checkpoint)
    # actor.nn_model.to(device)
    # actor.nn_model.train()  # Ensure the model is in training mode
    
    # Process a sample state to get input feature dimensions
    sample_state = unity.receive_state()
    frame_sample = processor.process_frame(sample_state, track_data)
    df_sample = pd.DataFrame([frame_sample])
    df_features_sample = df_sample.drop(columns=["time", "x_pos", "z_pos", "yaw_angle", "steering", "throttle", "brake"], errors='ignore')
    df_trans_sample = transformer.transform(df_features_sample).astype(float)
    input_size = df_trans_sample.shape[1]
    print("Input feature size:", input_size)
    
    # Initialize the Critic and load its PPO checkpoint if available
    critic = Critic(input_size=input_size, hidden_layer_sizes=(20,20)).to(device)
    # critic_checkpoint = torch.load("models/rlmodels/second_gen/ppo_critic_350.pth", map_location=device)
    # critic.load_state_dict(critic_checkpoint)
    # critic.train()  # Set to training mode
    
    # Initialize optimizers and load their states if saved
    actor_optimizer = optim.Adam(actor.nn_model.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)

    # Optionally, load the optimizer states if you have saved them:
    # actor_optimizer.load_state_dict(torch.load("models/rlmodels/actor_optimizer_350.pth", map_location=device))
    # critic_optimizer.load_state_dict(torch.load("models/rlmodels/critic_optimizer_350.pth", map_location=device))
    
    ppo_agent = PPO(actor, critic, actor_optimizer, critic_optimizer, 
                    clip_param=0.1, value_loss_coef=0.5, entropy_coef=0.01,
                    max_grad_norm=0.5, num_epochs=50, batch_size=64,
                    gamma=0.99, lam=0.95, steering_std=0.01, throttle_std=0.001, brake_std=0.00001, device=device)
    
    episode = 0
    try:
        while True:
            print("Collecting rollouts and updating policy...")
            ppo_agent.train(unity, rollout_length=4096)
            episode += 1
            if episode % 50 == 0:
                # Save both actor and critic checkpoints along with optimizer states if desired.
                torch.save(ppo_agent.actor.nn_model.state_dict(), f"models/rlmodels/ppo_actor_{episode}.pth")
                torch.save(ppo_agent.critic.state_dict(), f"models/rlmodels/ppo_critic_{episode}.pth")
                # Optionally, save optimizer states:
                torch.save(actor_optimizer.state_dict(), f"models/rlmodels/actor_optimizer_{episode}.pth")
                torch.save(critic_optimizer.state_dict(), f"models/rlmodels/critic_optimizer_{episode}.pth")
                print(f"Checkpoint saved at episode {episode}")
    except Exception as e:
        print(f"[Main] Error: {e}")
    finally:
        unity.close()
