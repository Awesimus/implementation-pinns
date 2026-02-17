# src/beam_model.py

class BeamModel:
    """
    Euler–Bernoulli beam constants (geometry + density only).

    The inverse PINN learns the viscoelastic modulus via:
        E*(omega) = E0 * (1 + i*eta)

    Governing equation in frequency domain:
        E*I * W''''(x,omega) - rho*A*omega^2*W(x,omega) = 0
    """
    def __init__(self, L=200/1000, b=10/1000, h=1/1000, rho=2700.0):
        self.L = float(L)
        self.b = float(b)
        self.h = float(h)
        self.rho = float(rho)

        self.A = self.b * self.h
        self.I = self.b * (self.h ** 3) / 12.0



# class BeamModel:
#     """
#     Euler-Bernoulli cantilever beam parameters.
#     Frequency-domain equation uses:
#       EI * W'''' - rho*A*omega^2*W = 0
#     """
#     def __init__(self, L=1.0, E=210e9, I=1e-6, rho=7800.0, A=0.01):
#         self.L = float(L)
#         self.E = float(E)
#         self.I = float(I)
#         self.rho = float(rho)
#         self.A = float(A)
