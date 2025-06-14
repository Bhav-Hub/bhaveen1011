import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self, num_layers, width, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(num_layers):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        
        self._model = NeuralNetwork(num_layers, width, input_dim, output_dim)
        self._criterion = nn.MSELoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)

    def predict_one(self, state):
        state = torch.tensor(np.reshape(state, (1, self._input_dim)), dtype=torch.float32)
        with torch.no_grad():
            return self._model(state).numpy()

    def predict_batch(self, states):
        states = torch.tensor(states, dtype=torch.float32)
        with torch.no_grad():
            return self._model(states).numpy()

    def train_batch(self, states, q_sa):
        states = torch.tensor(states, dtype=torch.float32)
        q_sa = torch.tensor(q_sa, dtype=torch.float32)
        
        self._optimizer.zero_grad()
        outputs = self._model(states)
        loss = self._criterion(outputs, q_sa)
        loss.backward()
        self._optimizer.step()

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self._model.state_dict(), os.path.join(path, 'trained_model.pth'))

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = NeuralNetwork(0, 0, input_dim, output_dim=1)  # Dummy values for hidden layers
        self._load_my_model(model_path)

    def _load_my_model(self, model_folder_path):
        model_file_path = os.path.join(model_folder_path, 'trained_model.pth')
        if os.path.isfile(model_file_path):
            self._model.load_state_dict(torch.load(model_file_path))
            self._model.eval()
        else:
            raise FileNotFoundError("Model file not found")

    def predict_one(self, state):
        state = torch.tensor(np.reshape(state, (1, self._input_dim)), dtype=torch.float32)
        with torch.no_grad():
            return self._model(state).numpy()

    @property
    def input_dim(self):
        return self._input_dim
