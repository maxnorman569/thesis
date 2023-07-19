import torch    

from VFF import spectra
from VFF.basis import FourierBasis
    

def _alpha(
        omegas: torch.Tensor,
        scalesigma: float,
        lengthscale: float,
        a: float,
        b: float,
) -> torch.Tensor:
    """
    Computes alpha half of the Kuu representation for the Matérn 1/2 covarainces

    Arguments:
        omegas (torch.Tensor)   : frequency (! omegas[0] = 0 !)
        scalesigma (float)      : amplitude hyperparameter
        lengthscale (float)     : lengthscale hyperparameter
        a (float)               : lower bound of the input space
        b (float)               : upper bound of the input space

    Returns:
        (torch.Tensor)          : alpha
    """
    # check that omegas[0] = 0
    assert omegas[0] == 0, "The first element of omegas must be 0"

    # compute the inverse spectral density
    S_inv = 1 / spectra.matern_12(omegas, scalesigma, lengthscale)

    # compute the alpha half
    alpha = ((b - a) / 2) * torch.cat([2 * S_inv[0][None], S_inv[1:], S_inv[1:]])

    return alpha


def _beta(
        omegas: torch.Tensor,
        scalesigma: float,
) -> torch.Tensor:
    """
    Computes the beta half of the Kuu representation for the Matérn 1/2 covarainces

    Arguments:
        omega (torch.Tensor)    : frequency
        sigma (float)           : amplitude hyperparameter

    Returns:
        (torch.Tensor)          : beta
    """

    # compute the sigma half
    sigma_half = torch.ones(len(omegas))/scalesigma

    # compute the zero half
    zero_half = torch.zeros(len(omegas) - 1)

    # compute beta
    beta = torch.cat((sigma_half, zero_half))

    return beta


def matern_12_Kuu(
        omegas: torch.Tensor,
        scalesigma: float,
        lengthscale: float,
        a: float,
        b: float,
) -> torch.Tensor:
    """
    Computes the Kuu using the representation given by (62) in the VFF paper for the Matérn 1/2 covarainces

    Arguments:
        omegas (torch.Tensor)   : frequency (! omegas[0] = 0 !)
        scalesigma (float)      : amplitude hyperparameter
        lengthscale (float)     : lengthscale hyperparameter
        a (float)               : lower bound of the input space
        b (float)               : upper bound of the input space

    Returns:
        (torch.Tensor)          : Kuu
    """

    # compute the alphas
    alphas = _alpha(omegas, scalesigma, lengthscale, a, b)

    # compute the betas
    betas = _beta(omegas, scalesigma)

    # compute Kuu
    Kuu = torch.diag(alphas) + torch.outer(_beta(omegas, scalesigma), _beta(omegas, scalesigma))

    return Kuu


def matern_12_Kuf( 
        fourier_basis : 'FourierBasis', 
        x : float,
        ) -> torch.Tensor:
    """ 
    Returns the cross-covariance between the domains 

    Arguments:
        fourier_basis (FourierBasis)    : Fourier Basis
        x (float)                       : point in the domain to evaluate the cross-covariance at

    Returns:
        (torch.Tensor)                  : cross-covariance
    """

    return fourier_basis(x)