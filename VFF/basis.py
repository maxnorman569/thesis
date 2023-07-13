import torch

class FourierBasis:
    
    """ Constructs a Fourier Basis for Variational Fourier Features """

    def __init__( self, M : int, a : float, b : float ):
        self.a = a
        self.b = b
        self.M = M
        self.omegas = torch.tensor([(2 * torch.pi * m) / (b - a) for m in range(M+1)])

    def __call__( self, x : float ) -> torch.Tensor:
        
        """ Evaluates the Fourier Basis at a point x """

        cosines = torch.cos(self.omegas * (x - self.a)) # includes omega = 0 frequency (cos(0) = 1))
        sines = torch.sin(self.omegas[1:] * (x - self.a)) # exclues omega = 0 frequency

        return torch.cat((cosines, sines))   