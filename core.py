import torch
from torch import nn


class RecurrentNetwork(nn.Module):
    def __init__(self, *args,num_neurons=10,timestep=0.001,oscillator_frequency=100,change_rate =0.01, **kwargs): 
        super().__init__(*args, **kwargs)
        self.connection_matrix = torch.randn(num_neurons, num_neurons)
        self.state = torch.rand(num_neurons).reshape([-1,1])
        self.activation = nn.Tanh()
        self.time = torch.tensor(0)
        self.timestep = timestep
        self.oscillator_pos = (1)
        self.oscillator_frequency = torch.tensor(oscillator_frequency)
        self.change_rate = change_rate

    def oscillator(self):
        return torch.sin(self.oscillator_frequency * self.time)
    
    def step(self):
        self.state = self.activation(self.state + ((self.connection_matrix @ self.state)*self.change_rate))
        self.state[self.oscillator_pos] = self.oscillator()
        self.time = self.time + self.timestep
        return self.state
    

