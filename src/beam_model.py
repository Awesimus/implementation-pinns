# # beam_model.py is a physics definition file, containing:
# #     beam constants
# #     governing equation form
# #     boundary-condition expressions

# import torch

# class BeamModel:
#     """
#     Euler-Bernoulli beam model for a cantilever beam with a tip mass and rotational inertia.
#     """
#     def __init__(self, L=1.0, E=210e9, I=1e-6, rho=7800, A=0.01):
#         """
#         L: beam length
#         E: Young's modulus
#         I: second moment of area
#         rho: density
#         A: cross-sectional area
#         """
#         self.L = L
#         self.E = E
#         self.I = I
#         self.rho = rho
#         self.A = A

#     def pde_residual(self, x, t, W, W_t, W_x, W_xx, W_xxx, W_xxxx):
#         """
#         Computes the Euler-Bernoulli PDE residual:
#         rho*A*W_tt + E*I*W_xxxx = 0 (assuming no distributed load)
#         """
#         return self.rho * self.A * W_t - self.E * self.I * W_xxxx

class BeamModel:
    """
    Euler-Bernoulli cantilever beam parameters.
    Frequency-domain equation uses:
      EI * W'''' - rho*A*omega^2*W = 0
    """
    def __init__(self, L=1.0, E=210e9, I=1e-6, rho=7800.0, A=0.01):
        self.L = float(L)
        self.E = float(E)
        self.I = float(I)
        self.rho = float(rho)
        self.A = float(A)
