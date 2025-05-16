import torch
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from models.nn import CNN, SmallCNN 
from torch.cuda.amp import GradScaler, autocast


KEYS = ("obs", "action", "reward", "next_obs", "done")

class Player:
    def __init__(self, state_dims, action_dims, model , learning_rate, gamma, epsilon, epsilon_decay, epsilon_min, buffer_size, batch_size, sync_rate, loss_fn, device):
        
        # size of input and output
        self.input = state_dims
        self.output = action_dims
        self.device = device

        nn = CNN if model == "CNN" else SmallCNN
        self.online = nn(self.input, self.output).to(self.device)
        self.target = nn(self.input, self.output).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        # model parameters
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_rate

        # optimizer and loss function
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.lr)
        self.loss = loss_fn
        self.scaler = GradScaler() if torch.cuda.is_available() else None

        # buffer parameters
        self.buffer_size = buffer_size
        self.sync_rate = sync_rate

        self.buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(self.buffer_size)
        )
        
        self.env_steps = 0
        self.learn_steps = 0

    @torch.no_grad()
    def select_action(self, obs):
        
        if self.epsilon > np.random.rand():
            return np.random.randint(self.output)
        obs = torch.as_tensor(np.squeeze(obs,-1), dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.online(obs)
        return int(q.argmax().item())
    
    def store_experience(self, obs, action, reward, next_obs, done):
        self.buffer.add(
                TensorDict(
                    {
                        "obs": torch.as_tensor(np.array(obs), dtype=torch.uint8),
                        "action": torch.tensor(action, dtype=torch.int64),
                        "reward": torch.tensor(reward, dtype=torch.float32),
                        "next_obs": torch.as_tensor(np.array(next_obs), dtype=torch.uint8),
                        "done": torch.tensor(done, dtype=torch.bool)
                    
                    }, 
                    batch_size=[]))

    def save_model(self, path):
        torch.save(self.online.state_dict(), path)

    def load_model(self, path):
        self.online.load_state_dict(torch.load(path), map_location=self.device)
        self.target.load_state_dict(torch.load(path), map_location=self.device)

    def learn(self, steps = 10):
        self.env_steps += 1
        if len(self.buffer) >= self.batch_size:
            self._learn()

        if self.env_steps % self.sync_rate == 0:
            self.target.load_state_dict(self.online.state_dict())

    def _learn(self):
        self.learn_steps += 1
        samples = self.buffer.sample(self.batch_size).to(self.device)
        obs, actions, reward, next_obs, done = (
            samples["obs"],
            samples["action"],
            samples["reward"],
            samples["next_obs"],
            samples["done"]
        )
        obs = obs.squeeze(-1)
        next_obs = next_obs.squeeze(-1)
        batch_indices = torch.arange(obs.shape[0], device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.device == "cuda":
            with autocast():
                q = self.online(obs)[batch_indices, actions.squeeze()]
                best_actions = self.online(next_obs).argmax(dim=1)
                next_q_target = self.target(next_obs).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                target_q = reward + self.gamma * next_q_target * (1 - done.float())
                loss = self.loss(q, target_q)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            q = self.online(obs)
            q =q[batch_indices, actions.squeeze()]
            with torch.no_grad():
                best_actions = self.online(next_obs).argmax(dim=1)
                next_q_target = self.target(next_obs).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                target_q = reward + self.gamma * next_q_target * (1 - done.float())
            loss = self.loss(q, target_q)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
            self.optimizer.step()
        
        
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

