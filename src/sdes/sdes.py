"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.
Taken and adapted from 
https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""
import abc
import math
import warnings

import numpy as np
import torch
from utils.registry import Registry

SDERegistry = Registry("SDE")


def sum_dif_matrix(a, b):
    s = (a + b) / 2.0
    d = (a - b) / 2.0
    return torch.stack(
        (torch.stack((s, d), dim=-1), torch.stack((d, s), dim=-1)), dim=-1
    )


def batch_broadcast(a, x):
    """Broadcasts a over all dimensions of x, except the batch dimension, which must match."""

    if len(a.shape) != 1:
        a = a.squeeze()
        if len(a.shape) != 1:
            raise ValueError(
                f"Don't know how to batch-broadcast tensor `a` with more than one effective dimension (shape {a.shape})"
            )

    if a.shape[0] != x.shape[0] and a.shape[0] != 1:
        raise ValueError(
            f"Don't know how to batch-broadcast shape {a.shape} over {x.shape} as the batch dimension is not matching"
        )

    out = a.view((x.shape[0], *(1 for _ in range(len(x.shape) - 1))))
    return out


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.
        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, *args):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.
        Useful for computing the log-likelihood via probability flow ODE.
        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
        """
        pass

    def discretize(self, x, t, *args, **kwargs):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.
        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)
        Returns:
            f, G
        """
        dt = getattr(kwargs, "dt", 1 / self.N)
        drift, diffusion = self.sde(x, t, *args)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(oself, score_model, probability_flow=False):
        """Create the reverse-time SDE/ODE.
        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                rsde_parts = self.rsde_parts(x, t, *args)
                total_drift, diffusion = (
                    rsde_parts["total_drift"],
                    rsde_parts["diffusion"],
                )
                return total_drift, diffusion

            def rsde_parts(self, x, t, *args):
                sde_drift, sde_diffusion = sde_fn(x, t, *args)
                pad_dim = (...,) + (None,) * (x.ndim - sde_diffusion.ndim)
                warnings.warn(f"x.shape: {x.shape}; sde_diffusion.shape: {sde_diffusion.shape}; t.shape: {t.shape}; args: {args[0].shape}")
                score = score_model(x, t, *args)
                score_drift = (
                    -sde_diffusion[pad_dim] ** 2
                    * score
                    * (0.5 if self.probability_flow else 1.0)
                )
                diffusion = (
                    torch.zeros_like(sde_diffusion)
                    if self.probability_flow
                    else sde_diffusion
                )
                total_drift = sde_drift + score_drift
                return {
                    "total_drift": total_drift,
                    "diffusion": diffusion,
                    "sde_drift": sde_drift,
                    "sde_diffusion": sde_diffusion,
                    "score_drift": score_drift,
                    "score": score,
                }

            def discretize(self, x, t, *args, **kwargs):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t, *args, **kwargs)
                pad_dim = (...,) + (None,) * (f.ndim - G.ndim)
                rev_f = f - G[pad_dim] ** 2 * score_model(x, t, *args) * (
                    0.5 if self.probability_flow else 1.0
                )
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

    @abc.abstractmethod
    def copy(self):
        pass


@SDERegistry.register("mix")
class MixSDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--sde-n",
            type=int,
            default=1000,
            help="The number of timesteps in the SDE discretization. 30 by default",
        )
        parser.add_argument(
            "--theta",
            type=float,
            default=1.5,
            help="The constant stiffness of the Ornstein-Uhlenbeck process. 1.5 by default.",
        )
        parser.add_argument(
            "--sigma-min",
            type=float,
            default=0.05,
            help="The minimum sigma to use. 0.05 by default.",
        )
        parser.add_argument(
            "--sigma-max",
            type=float,
            default=0.5,
            help="The maximum sigma to use. 0.5 by default.",
        )
        return parser

    def __init__(self, ndim, d_lambda, sigma_min, sigma_max, N=1000):
        """Construct a Variance Exploding SDE for source separation.
        Note that the "noise mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).
        dx = -A (y-x) dt + sigma(t) dw
        with
        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))
        Args:
            theta: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.ndim = ndim
        self.d_lambda = d_lambda
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.ratiosig = self.sigma_max / self.sigma_min
        self.logsig = math.log(self.ratiosig)
        self.N = N

        self.A, self.Pn = self.get_mix_mat()

        # prepare some things for sampling time

    def send_to(self, device):
        if self.A.device != device:
            self.A = self.A.to(device)
        if self.Pn.device != device:
            self.Pn = self.Pn.to(device)

    def get_mix_mat(self):
        ones = torch.ones((self.ndim, 1))
        # averaging matrix
        avg_mat = ones @ ones.T / self.ndim
        # projection onto null space of average matrix
        P_null = torch.eye(self.ndim) - avg_mat
        return avg_mat[None, ...], P_null[None, ...]

    def copy(self):
        return MixSDE(
            self.ndim, self.d_lambda, self.sigma_min, self.sigma_max, N=self.N
        )

    @property
    def T(self):
        return 1.0

    def sample_time_varprop(self, n, t_eps=0.0, device=None):
        """rejection sampler to sample time proportionally to the variance of the noise"""
        L_max = float(self._var(torch.tensor([self.T], device=device)).sqrt())
        n_acc = 0
        stack = []
        while n_acc < n:
            t = torch.zeros(3 * (n - n_acc), device=device).uniform_(t_eps, self.T)
            u = torch.zeros(3 * (n - n_acc), device=device).uniform_(0, L_max)
            std = self._var(t).sqrt()
            acc = u < std
            t_acc = t[acc]
            n_acc += t_acc.shape[0]
            stack.append(t_acc)
        t = torch.cat(stack)[:n]
        return t

    def sde(self, x, t, mix):
        self.send_to(x.device)
        drift = -self.d_lambda * self.Pn @ x
        # the sqrt(2*logsig) factor is required here so that logsig does not in the end affect the perturbation kernel
        # standard deviation. this can be understood from solving the integral of [exp(2s) * g(s)^2] from s=0 to t
        # with g(t) = sigma(t) as defined here, and seeing that `logsig` remains in the integral solution
        # unless this sqrt(2*logsig) factor is included.
        sigma = self.sigma_min * self.ratiosig ** t
        diffusion = sigma * np.sqrt(2 * self.logsig)
        return drift, diffusion

    def _mean_mix_mat(self, t):
        decay = torch.exp(-t[:, None, None] * self.d_lambda)
        mat = self.A + decay * self.Pn
        return mat

    def _mean(self, x0, t):
        mat = self._mean_mix_mat(t)
        mean = mat @ x0
        return mean

    def _cov_eigval(self, t):
        # for the covariance matrix
        mult = self.sigma_min ** 2
        s_ratio_power = self.ratiosig ** (2 * t)

        # eigenvalue corresponding to averaging matrix
        ev1 = mult * (s_ratio_power - 1)

        # other eigenvalues
        exponential = torch.exp(-2.0 * self.d_lambda * t)  # (time, 2)
        denom = 1.0 + self.d_lambda / self.logsig
        ev2 = mult * (s_ratio_power - exponential) / denom

        return ev1, ev2

    def _var(self, t):
        ev1, ev2 = self._cov_eigval(t)
        return 0.5 * (ev1 + ev2)

    def _std(self, t):
        # construct the matrix from its eigenvalues
        ev1, ev2 = self._cov_eigval(t)
        ev1, ev2 = ev1[:, None, None], ev2[:, None, None]
        L = ev1.sqrt() * self.A + ev2.sqrt() * self.Pn
        return L  # matrix square root of covariance matrix

    def marginal_prob(self, x0, t, *args):
        self.send_to(x0.device)
        return self._mean(x0, t), self._std(t)

    @staticmethod
    def mult_std(std, x):
        return std @ x

    @staticmethod
    def mult_std_inv(std, x):
        return torch.linalg.solve(std, x)

    def prior_sampling(self, shape, y):
        """input is the mixture signal"""
        self.send_to(y.device)
        """y is the mixture vector (possibly noisy)"""
        if shape != y.shape:
            warnings.warn(
                f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape."
            )
        t = torch.ones((y.shape[0],), device=y.device) * self.T
        std = self._std(t)
        mean = torch.broadcast_to(0.5 * y, (y.shape[0], 2, y.shape[2]))
        x_T = mean + std @ torch.randn_like(mean)
        return x_T

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")


@SDERegistry.register("priormix")
class PriorMixSDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--sde-n",
            type=int,
            default=1000,
            help="The number of timesteps in the SDE discretization. 30 by default",
        )
        parser.add_argument(
            "--theta",
            type=float,
            default=1.5,
            help="The constant stiffness of the Ornstein-Uhlenbeck process. 1.5 by default.",
        )
        parser.add_argument(
            "--sigma-min",
            type=float,
            default=0.05,
            help="The minimum sigma to use. 0.05 by default.",
        )
        parser.add_argument(
            "--sigma-max",
            type=float,
            default=0.5,
            help="The maximum sigma to use. 0.5 by default.",
        )
        return parser

    def __init__(self, ndim, d_lambda, sigma_min, sigma_max, N=1000, avg_len=510):
        """Construct a Variance Exploding SDE for source separation.
        Note that the "noise mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).
        dx = -A (y-x) dt + sigma(t) dw
        with
        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))
        Args:
            theta: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.ndim = ndim
        self.d_lambda = d_lambda
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.ratiosig = self.sigma_max / self.sigma_min
        self.logsig = math.log(self.ratiosig)
        self.N = N

        self.A, self.Pn = self.get_mix_mat()
        self.avg_len = avg_len

    def send_to(self, device):
        if self.A.device != device:
            self.A = self.A.to(device)
        if self.Pn.device != device:
            self.Pn = self.Pn.to(device)

    def get_mix_mat(self):
        ones = torch.ones((self.ndim, 1))
        # averaging matrix
        avg_mat = ones @ ones.T / self.ndim
        # projection onto null space of average matrix
        P_null = torch.eye(self.ndim) - avg_mat
        return avg_mat[None, ...], P_null[None, ...]

    def copy(self):
        return PriorMixSDE(
            self.ndim,
            self.d_lambda,
            self.sigma_min,
            self.sigma_max,
            N=self.N,
            avg_len=self.avg_len,
        )

    @property
    def T(self):
        return 1.0

    def sample_time_varprop(self, n, t_eps=0.0, device=None):
        """rejection sampler to sample time proportionally to the variance of the noise"""
        L_max = float(self._var(torch.tensor([self.T], device=device)).sqrt())
        n_acc = 0
        stack = []
        while n_acc < n:
            t = torch.zeros(3 * (n - n_acc), device=device).uniform_(t_eps, self.T)
            u = torch.zeros(3 * (n - n_acc), device=device).uniform_(0, L_max)
            std = self._var(t).sqrt()
            acc = u < std
            t_acc = t[acc]
            n_acc += t_acc.shape[0]
            stack.append(t_acc)
        t = torch.cat(stack)[:n]
        return t

    def sde(self, x, t, mix):
        self.send_to(x.device)
        drift = -self.d_lambda * self.Pn @ x

        # the sqrt(2*logsig) factor is required here so that logsig does not in the end affect the perturbation kernel
        # standard deviation. this can be understood from solving the integral of [exp(2s) * g(s)^2] from s=0 to t
        # with g(t) = sigma(t) as defined here, and seeing that `logsig` remains in the integral solution
        # unless this sqrt(2*logsig) factor is included.

        # per sample std dev.
        sigma_mix = self._std_sigma_mix(mix)
        sigma_mix = torch.broadcast_to(
            sigma_mix, (sigma_mix.shape[0], self.ndim, sigma_mix.shape[2])
        )

        # global scaling
        sigma = self.sigma_min * self.ratiosig ** t
        pad = (...,) + (None,) * (sigma_mix.ndim - sigma.ndim)
        diffusion = sigma[pad] * np.sqrt(2 * self.logsig) * sigma_mix
        return drift, diffusion

    def _mean_mix_mat(self, t):
        decay = torch.exp(-t[:, None, None] * self.d_lambda)
        mat = self.A + decay * self.Pn
        return mat

    def _std_sigma_mix(self, mix):
        self.send_to(mix.device)

        sigma_mix = torch.nn.functional.avg_pool1d(
            mix ** 2, kernel_size=self.avg_len, stride=1, padding=self.avg_len // 2
        )
        sigma_mix = sigma_mix.clamp(min=1e-4).sqrt()
        if self.avg_len % 2 == 0:
            sigma_mix = sigma_mix[..., :-1]  # adjust to match size of mix

        sigma_mix = 0.5 * sigma_mix

        return sigma_mix

    def _mean(self, x0, t):
        mat = self._mean_mix_mat(t)
        mean = mat @ x0
        return mean

    def _cov_eigval(self, t):
        # for the covariance matrix
        mult = self.sigma_min ** 2
        s_ratio_power = self.ratiosig ** (2 * t)

        # eigenvalue corresponding to averaging matrix
        ev1 = mult * (s_ratio_power - 1)

        # other eigenvalues
        exponential = torch.exp(-2.0 * self.d_lambda * t)  # (time, 2)
        denom = 1.0 + self.d_lambda / self.logsig
        ev2 = mult * (s_ratio_power - exponential) / denom

        return ev1, ev2

    def _var(self, t):
        ev1, ev2 = self._cov_eigval(t)
        return 0.5 * (ev1 + ev2)

    def _std(self, t, mix):
        # per sample standard deviation
        sigma_mix = self._std_sigma_mix(mix)

        ev1, ev2 = self._cov_eigval(t)  # (batch,)

        # build matrix
        # L.shape == (batch, nspkr, nspkr, nsamples)
        ev1 = ev1[:, None, None, None]
        ev2 = ev2[:, None, None, None]
        L = ev1.sqrt() * self.A[..., None] + ev2.sqrt() * self.Pn[..., None]
        L = L * sigma_mix[:, None, :, :]

        return L  # matrix square root of covariance matrix

    @staticmethod
    def mult_std(std, x):
        return torch.einsum("bcdt,bdt->bct", std, x)

    @staticmethod
    def mult_std_inv(std, x):
        ndim = x.shape[1]
        if ndim > 2:
            std = std.permute([0, 3, 1, 2])
            x = x.permute([0, 2, 1])
            sol = torch.linalg.solve(std, x)
            sol = sol.permute([0, 2, 1]).contiguous()
        elif ndim == 2:
            # manual 2x2 system solution
            # | a b | | y1 |   | x_1 |
            # | c d | | y2 | = | x_2 |
            a = std[:, 0, 0, :]
            b = std[:, 0, 1, :]
            c = std[:, 1, 0, :]
            d = std[:, 1, 1, :]
            x1 = x[:, 0, :]
            x2 = x[:, 1, :]
            div = torch.reciprocal(a * d - c * b)
            y1 = div * (d * x1 - b * x2)
            y2 = div * (a * x2 - c * x1)
            sol = torch.stack((y1, y2), dim=1)
        else:
            raise ValueError(f"ndim={ndim}, it should be >= 2")
        return sol

    def marginal_prob(self, x0, t, mix):
        self.send_to(x0.device)
        return self._mean(x0, t), self._std(t, mix)

    def prior_sampling(self, shape, mix):
        """input is the mixture signal"""
        self.send_to(mix.device)
        """mix is the mixture vector (possibly noisy)"""
        if shape != mix.shape:
            warnings.warn(
                f"Target shape {shape} does not match shape of mix {mix.shape}! Ignoring target shape."
            )
        t = torch.ones((mix.shape[0],), device=mix.device) * self.T
        std = self._std(t, mix)
        if mix.shape[1] == self.ndim:
            mean = mix
        elif mix.shape[1] == 1:
            mean = torch.broadcast_to(
                0.5 * mix, (mix.shape[0], self.ndim, mix.shape[2])
            )
        else:
            raise ValueError(
                "The input provided to prior_sampling should have 1 channel,"
                f" or the same as the number of speakers. Found {mix.shape[1]} "
                "channels instead."
            )
        x_T = mean + self.mult_std(std, torch.randn_like(mean))
        return x_T

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")


@SDERegistry.register("ouve")
class OUVESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--sde-n",
            type=int,
            default=1000,
            help="The number of timesteps in the SDE discretization. 30 by default",
        )
        parser.add_argument(
            "--theta",
            type=float,
            default=1.5,
            help="The constant stiffness of the Ornstein-Uhlenbeck process. 1.5 by default.",
        )
        parser.add_argument(
            "--sigma-min",
            type=float,
            default=0.05,
            help="The minimum sigma to use. 0.05 by default.",
        )
        parser.add_argument(
            "--sigma-max",
            type=float,
            default=0.5,
            help="The maximum sigma to use. 0.5 by default.",
        )
        return parser

    def __init__(self, theta, sigma_min, sigma_max, N=1000, **ignored_kwargs):
        """Construct an Ornstein-Uhlenbeck Variance Exploding SDE.
        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).
        dx = -theta (y-x) dt + sigma(t) dw
        with
        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))
        Args:
            theta: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.theta = theta
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.N = N

    def copy(self):
        return OUVESDE(self.theta, self.sigma_min, self.sigma_max, N=self.N)

    @property
    def T(self):
        return 1

    def sde(self, x, t, y):
        drift = self.theta * (y - x)
        # the sqrt(2*logsig) factor is required here so that logsig does not in the end affect the perturbation kernel
        # standard deviation. this can be understood from solving the integral of [exp(2s) * g(s)^2] from s=0 to t
        # with g(t) = sigma(t) as defined here, and seeing that `logsig` remains in the integral solution
        # unless this sqrt(2*logsig) factor is included.
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(2 * self.logsig)
        return drift, diffusion

    def _mean(self, x0, t, y):
        theta = self.theta
        
        exp_interp = torch.exp(-theta * t) #[:, None, None, None] 
        pad = (...,) + (None,) * (x0.ndim - exp_interp.ndim)
        exp_interp = exp_interp[pad]
        return exp_interp * x0 + (1 - exp_interp) * y

    def _std(self, t):
        # This is a full solution to the ODE for P(t) in our derivations, after choosing g(s) as in self.sde()
        sigma_min, theta, logsig = self.sigma_min, self.theta, self.logsig
        # could maybe replace the two torch.exp(... * t) terms here by cached values **t
        return torch.sqrt(
            (
                sigma_min ** 2
                * torch.exp(-2 * theta * t)
                * (torch.exp(2 * (theta + logsig) * t) - 1)
                * logsig
            )
            / (theta + logsig)
        )

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(
                f"Target shape {shape} does not match shape of y {y.shape}! Brodcasting mix shape."
            )
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        pad = (...,) + (None,) * (y.ndim - std.ndim)
        x_T = y + torch.randn(shape, device=y.device) * std[pad]#[:, None, None, None]
        return x_T

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")


@SDERegistry.register("sbve")
class SBVESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--N", type=int, default=50, help="The number of timesteps in the SDE discretization. 50 by default")
        parser.add_argument("--k", type=float, default=2.6, help="Parameter of the diffusion coefficient. 2.6 by default.")
        parser.add_argument("--c", type=float, default=0.4, help="Parameter of the diffusion coefficient. 0.4 by default.")
        parser.add_argument("--eps", type=float, default=1e-8, help="Small constant to avoid numerical instability. 1e-8 by default.")
        parser.add_argument("--sampler_type", type=str, default="ode")
        return parser

    def __init__(self, k, c, N=50, eps=1e-8, sampler_type="ode", **ignored_kwargs):
        """Construct a Schrodinger Bridge with Variance Exploding SDE.

        As described in Jukić et al., „Schrödinger Bridge for Generative Speech Enhancement“, 2024.

        Args:
            k: stiffness parameter.
            c: diffusion parameter.
            N: number of discretization steps
        """
        super().__init__(N)
        self.k = k
        self.c = c
        self.N = N
        self.eps = eps
        self.sampler_type = sampler_type

    def copy(self):
        return SBVESDE(self.k, self.c, N=self.N)

    @property
    def T(self):
        return 1

    def sde(self, x, y, t):
        f = 0.0                                                                 # Table 1
        g = torch.sqrt(torch.tensor(self.c)) * self.k**(t)                      # Table 1
        return f, g

    def _sigmas_alphas(self, t):
        alpha_t = torch.ones_like(t)
        alpha_T = torch.ones_like(t)
        sigma_t = torch.sqrt((self.c*(self.k**(2*t)-1.0)) \
            / (2*torch.log(torch.tensor(self.k))))                              # Table 1
        sigma_T = torch.sqrt((self.c*(self.k**(2*self.T)-1.0)) \
            / (2*torch.log(torch.tensor(self.k))))                              # Table 1   
        
        alpha_bart = alpha_t / (alpha_T + self.eps)                             # below Eq. (9)
        sigma_bart = torch.sqrt(sigma_T**2 - sigma_t**2 + self.eps)             # below Eq. (9)

        return sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart

    def _mean(self, x0, y, t):
        sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = self._sigmas_alphas(t)

        w_xt = alpha_t * sigma_bart**2 / (sigma_T**2 + self.eps)                # below Eq. (11)
        w_yt = alpha_bart * sigma_t**2 / (sigma_T**2 + self.eps)                # below Eq. (11)
        pad_dim = (...,) + (None,) * (x0.ndim - t.ndim)
        mu = w_xt[pad_dim] * x0 + w_yt[pad_dim] * y     # Eq. (11)
        return mu

    def _std(self, t):
        sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = self._sigmas_alphas(t)

        sigma_xt = (alpha_t * sigma_bart * sigma_t) / (sigma_T + self.eps) 
        return sigma_xt

    def marginal_prob(self, x0, y, t):  
        return self._mean(x0, y, t), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        x_T = y
        return x_T

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for SBVE SDE not yet implemented!")  
