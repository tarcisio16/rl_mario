import torch
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from models.nn import CNN, SmallCNN 


KEYS = ("obs", "action", "reward", "next_obs", "done")

class Player:
    def __init__(self, state_dims, action_dims, model , learning_rate, gamma, epsilon, epsilon_decay, epsilon_min, buffer_size, batch_size, sync_rate, loss_fn, device):
        
        # size of input and output
        self.input = state_dims
        self.output = action_dims
        self.device = device
        if model == "CNN":
            nn = CNN
        elif model == "SCNN":
            nn = SmallCNN
        self.online = nn(self.input, self.output).to(self.device)
        self.target = nn(self.input, self.output).to(self.device)

        # model parameters
        self.learning_rate = learning_rate
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.ls = 0
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_rate

        # optimizer and loss function
        self.optimizer = torch.optim.Adam(self.online.network.parameters(), lr=self.learning_rate)
        self.loss = loss_fn

        # buffer parameters
        self.buffer_size = buffer_size
        self.sync_rate = sync_rate

        self.buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(self.buffer_size)
        )

    def select_action(self, obs):
        
        if torch.rand(1).item() < self.epsilon:
            return np.random.randint(self.output)
        sample = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            q = self.online(sample).argmax().item()
        return q
    
    def store_experience(self, obs, action, reward, next_obs, done):
        self.buffer.add(TensorDict({
            "obs": torch.as_tensor(np.array(obs), dtype=torch.uint8),
            "action": torch.tensor(action),
            "reward": torch.tensor(reward), 
            "next_obs": torch.as_tensor(np.array(next_obs), dtype=torch.uint8),
            "done": torch.tensor(done)
        }, batch_size=[]))

    def save_model(self, path):
        torch.save(self.online.state_dict(), path)

    def sync_networks(self):
        if self.ls % self.sync_network_rate == 0 and self.ls >0:
            self.target.load_state_dict(self.online.state_dict())

    def load_model(self, path):
        self.online.load_state_dict(torch.load(path))
        self.target.load_state_dict(torch.load(path))

    def learn(self, steps = 10):
        if len(self.buffer) >= self.batch_size and self.ls % steps == 0:
            self._learn()
    

    def _learn(self):

        self.sync_networks()
        self.optimizer.zero_grad()

        samples = self.buffer.sample(self.batch_size).to(self.device)

        obs, actions, reward, next_obs, done = samples["obs"], samples["action"], samples["reward"], samples["next_obs"], samples["done"]
        obs = obs.to(torch.float32).to(self.device) / 255.0
        next_obs = next_obs.to(torch.float32).to(self.device) / 255.0
        q = self.online(obs)
        q = q[torch.arange(self.batch_size), actions.squeeze()]

        with torch.no_grad():
            best_actions = self.online(next_obs).argmax(dim=1)
            next_q_target = self.target(next_obs)
            target_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

        target_q_values = reward + self.gamma * target_q * (1 - done.float())

        loss = self.loss(q, target_q_values)
        loss.backward()
        self.optimizer.step()
        self.ls += 1
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

