import torch

import math

def matern_12( 
        tau : torch.Tensor,
        sigma : float,
        lengthscale : float,
        ) -> torch.Tensor:

        """
        Matern 1/2 kernel

        Arguments:
            tau (torch.Tensor): distance between points (i.e. |x - x'|)
            sigma (float)           : amplitude hyperparameter
            lengthscale (float)     : lengthscale hyperparameter

        Returns:
            torch.Tensor: Kernel weighting for distance tau
        """
        # compute the kernel
        kernel = (sigma ** 2) * torch.exp(-tau / lengthscale)

        return kernel


def matern_32(
        tau : torch.Tensor,
        sigma : float,
        legnthscale : float,
        ) -> torch.Tensor:
        """
        Matern 3/2 kernel

        Arguments:
            tau (torch.Tensor)      : distance between points (i.e. |x - x'|)
            sigma (float)           : amplitude hyperparameter
            lengthscale (float)     : lengthscale hyperparameter 

        Returns:
            torch.Tensor: Kernel weighting for distance tau
        """
        # compute the kernel
        kernel = (sigma ** 2) * (1 + ((math.sqrt(3) * tau) / legnthscale)) * torch.exp(- (math.sqrt(3)) / legnthscale)

        return kernel

 
def matern_52(
        tau : torch.Tensor,
        sigma : float,
        legnthscale : float,
        ) -> torch.Tensor:
        """
        Matern 5/2 kernel

        Arguments:
            tau (torch.Tensor)      : distance between points (i.e. |x - x'|)
            sigma (float)           : amplitude hyperparameter
            lengthscale (float)     : lengthscale hyperparameter 

        Returns:
            torch.Tensor: Kernel weighting for distance tau
        """
        # compute the kernel
        kernel = (sigma ** 2) * (1 + ((math.sqrt(5) * tau) / legnthscale) + ( (5 * (tau ** 2)) / (3 * (legnthscale ** 2)))) * torch.exp(- (math.sqrt(5) * tau) / legnthscale)

        return kernel