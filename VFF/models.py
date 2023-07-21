import torch
import gpytorch

from VFF.basis import FourierBasisMatern12
from VFF.covariances import matern_12_Kuf, matern_12_Kuu


class Matern12VFF(gpytorch.Module):

    """ 
    Variational Fourier Feature (VFF) Gaussian Process (GP) model with Matern 1/2 kernel.
    The approximate posterior is given by 

    q(f) = \int p(f|u)q(u)du = GP(mu(), sigma()) 

    where
    * mu(x_star) = Ku(x_star)^T Kuu^{-1} m
    * sigma(x_star) = k(x_star , x_star) + Ku(x_star)^T Kuu^{-1} (S - Kuu) Kuu^{-1} Ku(x_star)

    where x_star are the locations of the test points.

    Given we have a Guassian Likelihood, this class uses the closed form solution for the optimal moments of the variational distribution q(u) = N(m_hat, sigma_hat)

    where
    * m_hat = noisesigma^{-2} Kuu @ sigma @ Kuf @ y
    * sigma_hat = Kuu^{-1} @ sigma @ Kuu


    For the value of these optimal parameters, the ELBO is given by 

    ELBO = log(N(y|0, Kuf^T @ Kuu^{-1} @ Kuf + noisesigma^2 * I)) - 1/2 * noisesigma^{-2} tr(Kff - Kuf^T @ Kuu^{-1} @ Kuf)

    here we define the code approx_prior = Kuf^T @ Kuu^{-1} @ Kuf.


    """

    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor, 
                 nfrequencies : int, 
                 alim : float, 
                 blim : float) -> 'Matern12VFF':
        """ 
        Class inherits from gpytorch.Module and implements the VFF GP model with a Matern 1/2 kernel, zero mean and Gaussian likelihood

        Arguments:
            X (torch.Tensor)        : training inputs (n,) 
            y (torch.Tensor)        : training targets
            nfrequencies (int)      : number of frequencies to use in the Fourier basis
            alim (float)            : lower bound of the input space
            blim (float)            : upper bound of the input space
        """
        super().__init__()
        # parameters
        self.nfrequencies = nfrequencies
        self.alim = alim
        self.blim = blim
        # data
        self.train_inputs = (X,)
        self.train_targets = y
        # model components
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.mean = gpytorch.means.ZeroMean()
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1/2))

    def _Kuu(self,) -> torch.Tensor:
        """ 
        Computes the covariance matrix of the inducing variables u, Kuu

        Arguments:
            None

        returns:
            (torch.Tensor)          : Kuu (covariance matrix)
        """
        # Fourier Basis
        scale_sigma = self.kernel.outputscale.sqrt()
        lengthscale = self.kernel.base_kernel.lengthscale[0] # REPEATED
        basis = FourierBasisMatern12(self.nfrequencies, self.alim, self.blim, lengthscale) # REPEATED
        # compute Kuu
        Kuu = matern_12_Kuu(basis.omegas, scale_sigma, lengthscale, self.alim, self.blim)
        return Kuu

    def _Kuf(self, 
             x : torch.Tensor) -> torch.Tensor:
        """ 
        Computes the cross covariance matrix across inducing variables u and training inputs x, Kuf

        Arguments:
            x (torch.Tensor)        : training inputs (n,)

        returns:
            (torch.Tensor)          : Kuf (covariance matrix)
        """
        # Fourier Basis
        lengthscale = self.kernel.base_kernel.lengthscale[0] # REPEATED
        basis = FourierBasisMatern12(self.nfrequencies, self.alim, self.blim, lengthscale) # REPEATED
        # compute Kuu
        Kuf = matern_12_Kuf(basis, x)
        return Kuf

    def _sigma_inv(self, ) -> torch.Tensor:
        """ 
        Compute the inverse of Sigma for the variational Gaussian distribution, sigma^{-1} = [Kuu + noisesigma^{-2} Kuf @ Kuf^T]

        Arguments:
            None

        returns:
            sigma_inv (torch.Tensor)    : inverse of sigma (m, m)
        """
        # get noise sigma
        noise_sigma = self.likelihood.noise[0]
        # compute matrices
        Kuu = self._Kuu()
        Kuf = self._Kuf(self.train_inputs[0])
        return Kuu + (Kuf @ Kuf.T) / noise_sigma
    
    def _variational_mu(self, ) -> torch.Tensor:
        """ 
        Computes the optimal mean for the variational Gaussian distribution, m_hat = noisesigma^{-2} Kuu @ sigma @ Kuf @ y

        Arguments:
            None

        returns:
            mu (torch.Tensor)       : optimal mean over (m,)
        """
        # get noise sigma
        noise_sigma = self.likelihood.noise[0]
        # compute matrices
        Kuu = self._Kuu()
        Kuf = self._Kuf(self.train_inputs[0])
        sigma = self._sigma_inv()
        # compute optimal variational mean
        chol = torch.linalg.cholesky(sigma)
        mu = (1 / noise_sigma) * Kuu @ torch.cholesky_solve(Kuf, chol) @ self.train_targets
        return mu
    
    def _variational_cov(self, ) -> torch.Tensor:
        """ 
        Computes the optima; covariance for the variational Gaussian distribution, sigma_hat = Kuu^{-1} @ sigma @ Kuu

        Arguments:
            None

        returns:
            cov (torch.Tensor)      : optimal covariance (m, m)
        """
        # compute matrices
        Kuu = self._Kuu()
        sigma = self._sigma_inv()
        # compute optimal variational covariance
        chol = torch.linalg.cholesky(sigma)
        cov = Kuu @ torch.cholesky_solve(Kuu, chol)
        return cov

    def _conditional_mu(self, 
                        x_star : torch.Tensor) -> torch.Tensor:
        """ 
        Computes the conditional mean for the variational Gaussian distribution, mu(x_star) = Ku(x_star)^T Kuu^{-1} m

        Arguments:
            x_star (torch.Tensor)   : test inputs (n_star,)

        returns:
            cond_mu (torch.Tensor)  : conditional mean (n_star,)
        """
        # compute matrices
        Kuu = self._Kuu()
        Kuf_star = self._Kuf(x_star)
        optimal_mu = self._variational_mu()
        # compute the conditional mean
        chol = torch.linalg.cholesky(Kuu)
        cond_mu = Kuf_star.T @ (torch.cholesky_solve(optimal_mu.unsqueeze(-1), chol))
        return cond_mu

    def _conditional_cov(self, 
                         x_star : torch.Tensor) -> torch.Tensor:
        """ 
        Computes the conditional covariance for the variational Gaussian distribution, sigma(x_star) = k(x_star , x_star) + Ku(x_star)^T Kuu^{-1} (S - Kuu) Kuu^{-1} Ku(x_star) 

        Arguments:
            x_star (torch.Tensor)   : test inputs (n_star,)

        returns:
            cond_cov (torch.Tensor) : conditional covariance (n_star, n_star)
        """
        # compute matrices
        Kuu = self._Kuu()
        Kuf_star = self._Kuf(x_star)
        Kff_star = self.kernel(x_star).evaluate()
        optimal_cov = self._variational_cov()
        # compute the conditional covariance
        chol = torch.linalg.cholesky(Kuu)
        cond_cov = Kff_star + Kuf_star.T @ (torch.cholesky_solve(optimal_cov - Kuu, chol)) @ (torch.cholesky_solve(Kuf_star, chol))
        return cond_cov
    
    def _elbo(self,) -> torch.Tensor:
        """ 
        Computes the evidence lower bound (ELBO) for the variational Gaussian distribution 

        ELBO = log(N(y|0, Kuf^T @ Kuu^{-1} @ Kuf + noisesigma^2 * I)) - 1/2 * noisesigma^{-2} tr(Kff - Kuf^T @ Kuu^{-1} @ Kuf)

        Arguments:
            None

        returns:
            elbo (torch.Tensor)     : evidence lower bound (scalar)
        """
        # get noise sigma
        noise_sigma = self.likelihood.noise[0]
        # compute matrices
        Kuu = self._Kuu()
        Kuf = self._Kuf(self.train_inputs[0])
        Kff = self.kernel(self.train_inputs[0]).evaluate()
        # compute approximate prior structure
        chol = torch.linalg.cholesky(Kuu)
        approx_prior = Kuf.T @ (torch.cholesky_solve(Kuf, chol))
        # evidence term 
        mean = self.mean(self.train_inputs[0])
        evidence_term = gpytorch.distributions.MultivariateNormal(mean, approx_prior + noise_sigma * torch.eye(len(self.train_inputs[0]))).log_prob(self.train_targets)
        # trace term
        trace_term = torch.trace(Kff - approx_prior) / (2 * noise_sigma)
        return evidence_term - trace_term
    
    def fit(self, 
            niter : int, 
            learningrate : float,
            trace : bool = False) -> None:
        """ 
        Optimizes the model hyper-parameters (scale, noise, lengthscale) by maximizing the ELBO

        Arguments:
            niter (int)             : number of iterations
            learningrate (float)    : learning rate for the optimizer
            trace (bool)            : whether to return the ELBO trace

        returns:
            None
        """
        # initialize optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learningrate)
        # perform gradient descent
        history = torch.empty(niter)
        for i in range(niter):
            optimizer.zero_grad()
            elbow = -self._elbo()
            history[i] = elbow.item()
            elbow.backward()
            optimizer.step()

        if trace:
            return history
        else:
            return None
        
    def posterior(self, x_star):
        """ 
        Computes the posterior distribution of the Gaussian Process for the provided test points

        Arguments:
            x_star (torch.Tensor)                                   : test inputs (n_star,)

        returns:
            posterior (gpytorch.distributions.MultivariateNormal)   : posterior distribution
        """
        # compute the conditional mean and covariance
        cond_mu = self._conditional_mu(x_star).flatten()
        cond_cov = self._conditional_cov(x_star)
        # sample from the conditional distribution
        posterior = gpytorch.distributions.MultivariateNormal(cond_mu, cond_cov)
        return posterior

    def posterior_predictive(self, x_star):
        """ 
        Computes the posterior predictive distribution  of the Gaussian Process for the provided test points 
        
        Arguments:
            x_star (torch.Tensor)                                   : test inputs (n_star,)

        returns:
            posterior (gpytorch.distributions.MultivariateNormal)   : posterior predictive distribution
        """
        # compute the conditional mean and covariance
        cond_mu = self._conditional_mu(x_star).flatten()
        cond_cov = self._conditional_cov(x_star)
        # sample from the conditional distribution
        posterior = gpytorch.distributions.MultivariateNormal(cond_mu, cond_cov)
        # pass it through the likelihood
        posterior_predictive = self.likelihood(posterior)
        return posterior_predictive