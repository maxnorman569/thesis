import torch

import math

def matern_12( 
        omega : torch.Tensor, 
        sigma : float, 
        lengthscale : float 
        ) -> torch.Tensor:
    """
    Computes the spectra corersponding to the Matérn 1/2 covariances

    Arguments:
        omega (torch.Tensor)    : frequency
        sigma (float)           : amplitude hyperparameter
        lengthscale (float)     : lengthscale hyperparameter

    Returns:
        (torch.Tensor)          : spectral density
    """
    # get lamnda
    lmbda = 1 / lengthscale

    # compute spectral density
    numerator = 2 * (sigma ** 2) * lmbda
    denominator = (lmbda ** 2) + (omega ** 2)
    spectral_density = numerator / denominator

    return spectral_density


def matern_32(
        omega : torch.Tensor, 
        sigma : float, 
        lengthscale : float
        ) -> torch.Tensor:
    """
    Computes the spectra corersponding to the Matérn 3/2 covariances

    Arguments:
        omega (torch.Tensor)    : frequency
        sigma (float)           : amplitude hyperparameter
        lengthscale (float)     : lengthscale hyperparameter (lmbda = sqrt(3) / original lengthscale)

    Returns:
        (torch.Tensor)          : spectral density
    """
    # get lmbda
    lmbda = math.sqrt(3) / lengthscale

    # compute spectral density
    numerator = 4 * (sigma ** 2) * (lmbda ** 3)
    denominator = (lmbda ** 2) + (omega ** 2)
    spectral_density = numerator / (denominator ** 2)

    return spectral_density


def matern_52(
        omega : torch.Tensor, 
        sigma : float, 
        lengthscale : float
        ) -> torch.Tensor:
    """
    Computes the spectra corersponding to the Matérn 5/2 covariances

    Arguments:
        omega (torch.Tensor)    : frequency
        sigma (float)           : amplitude hyperparameter
        lengthscale (float)     : lengthscale hyperparameter (lmbda = sqrt(5) / original lengthscale)

    Returns:
        (torch.Tensor)          : spectral density
    """
    # get lmbda
    lmbda = math.sqrt(5) / lengthscale

    # compute spectral density
    numerator = (16 / 3) * (sigma ** 2) * (lmbda ** 5)
    denominator = (lmbda ** 2) + (omega ** 2)
    spectral_density = numerator / (denominator ** 3)

    return spectral_density