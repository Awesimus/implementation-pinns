# import torch
# import torch.nn as nn

# class PINN(nn.Module):
#     def __init__(self, layers=[2, 50, 50, 50, 1]):
#         """
#         layers: list of layer sizes, input is (x,t), output is W
#         """
#         super(PINN, self).__init__()
#         self.layers = nn.ModuleList()
#         for i in range(len(layers)-1):
#             self.layers.append(nn.Linear(layers[i], layers[i+1]))
#         self.activation = nn.Tanh()

#     def forward(self, x, t):
#         X = torch.cat([x, t], dim=1)
#         for layer in self.layers[:-1]:
#             X = self.activation(layer(X))
#         X = self.layers[-1](X)
#         return X

import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    Frequency-domain PINN:
      input:  (x, omega)
      output: (W_real, W_imag)
    """
    def __init__(self, layers=[2, 64, 64, 64, 2]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()

    def forward(self, x, omega):
        X = torch.cat([x, omega], dim=1)
        for layer in self.layers[:-1]:
            X = self.activation(layer(X))
        X = self.layers[-1](X)  # shape (N, 2)
        W_real = X[:, 0:1]
        W_imag = X[:, 1:2]
        return W_real, W_imag
