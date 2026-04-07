from abc import ABC, abstractmethod
from typing import Optional
import logging
import time

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Complex, Key, jaxtyped, Bool
from numpy import loadtxt
import requests
from beartype import beartype as typechecker

from jimgw.core.constants import (
    C_SI,
    DEG_TO_RAD,
)

from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.utils import inner_product, complex_inner_product
from jimgw.core.single_event.detector import Detector

import lisaorbits
from jaxgb.jaxgb import JaxGB
from jaxgb.params import GBObject

class SpaceBasedGB(Detector):
    """Object representing a space-based detector, specifically for galactic binaries.

    Contains information about the orbit and detector parameters in the solar system,
    as well as actual strain data and the PSD of the associated noise.

    Attributes:
        name (str): Name of the detector.
        data (Data): Array of Fourier-domain strain data.
        psd (PowerSpectrum): Power spectral density object.
    """

    polarization_mode: list[Polarization]
    data: Data
    psd: PowerSpectrum
    channel: str
    t_init: Float = 1.0e3
    t0: Float = 1.0e4
    n_freqs: int = 2000

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __init__(
        self,
        name: str,
        orbit: str,
        tdi_channel: str,
        t_obs: float,
        num_freqs: int,
        t_init: float,
        t0: float,
        P_noise: Float[Array],
        A_noise: Float[Array],
        **kwargs
    ):
        """Initialize a ground-based detector.

        Args:
            name (str): Name of the detector.
            
        """
        super().__init__()

        self.name = name
        self.orbit_name = orbit
        self.tdi_channel = tdi_channel
        
        self.num_freqs = num_freqs
        self.t_obs = t_obs
        self.t_init = t_init
        self.t0 = t0
        self.data = Data()
        self.psd = PowerSpectrum()

        self.P_noise_params = P_noise
        self.A_noise_params = A_noise

        if self.orbit_name == 'equal':
            self.orbits = lisaorbits.EqualArmlengthOrbits()
        elif self.orbit_name == 'kepler':
            self.orbits = lisaorbits.KeplerianOrbits()
        else:
            raise NotImplementedError
        
        self.myjaxgb = myjaxgb = JaxGB(self.orbits, t_obs=self.t_obs, t0=self.t0, n=self.num_freqs)
    
    def get_params(
        self,
        params: dict[str, Float]
    ) -> Float[Array]:
        f0,fdot,A,ra,dec,psi,iota,phi0, = params['f0'],params['fdot'],params['A'],params['ra'],params['dec'],params['psi'],params['iota'],params['phi0'],                
        f0_t0 = f0 - fdot * (self.t_init - self.t0)
        # shift `phi0`, to account for difference `t_init` (catalogue epoch) and `t0`
        _shift = jnp.pi * (
            2 * f0 * (self.t_init - self.t0) - fdot * (self.t_init - self.t0) ** 2
            )
        phi0_t0 = (phi0 + _shift) % (2 * jnp.pi)

        return jnp.array(
            [
                f0_t0,  # shifted f0
                fdot,
                A,
                ra,
                dec,
                psi,
                iota,
                phi0_t0,  # shifted phi0
            ]
        ).T

    def fd_response(
        self,
        frequency: Float[Array, " n_sample"],
        params: dict[str, Float],
        **kwargs,
    ) -> Complex[Array, " n_sample"]:
        """Modulate the waveform in the sky frame by the detector response in the frequency domain.

        Args:
            frequency (Float[Array, "n_sample"]): Array of frequency samples.
            h_sky (dict[str, Float[Array, "n_sample"]]): Dictionary mapping polarization names
                to frequency-domain waveforms. Keys are polarization names (e.g., 'plus', 'cross')
                and values are complex strain arrays.
            params (dict[str, Float]): Dictionary of source parameters containing:
                - ra (Float): Right ascension in radians
                - dec (Float): Declination in radians
                - psi (Float): Polarization angle in radians
            **kwargs: Additional keyword arguments.

        Returns:
            Array: Complex strain measured by the detector in frequency domain, obtained by
                  combining the antenna patterns for each polarization mode.
        """

        wave_params = self.get_params(params)

        if self.tdi_channel == 'AE':
            response = jnp.asarray(myjaxgb.get_tdi(wave_params, tdi_generation = self.tdi_gen, tdi_combination = self.tdi_channel )[:-1])
        else:
            response = jnp.asarray(myjaxgb.get_tdi(wave_params, tdi_generation = self.tdi_gen, tdi_combination = self.tdi_channel ))
        
        return response

    def td_response(
        self,
        time: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict,
        **kwargs,
    ) -> Array:
        """Modulate the waveform in the sky frame by the detector response in the time domain.

        Args:
            time: Array of time samples.
            h_sky: Dictionary mapping polarization names to time-domain waveforms.
            params: Dictionary of source parameters.
            **kwargs: Additional keyword arguments.

        Returns:
            Array of detector response in time domain.
        """
        raise NotImplementedError

    @jaxtyped(typechecker=typechecker)
    def load_and_set_psd(self, freqs,) -> PowerSpectrum:
        """Load power spectral density (PSD) from file or default GWTC-2 catalog,
            and set it to the detector.

        Args:
            psd_file (str, optional): Path to file containing PSD data. If empty, uses GWTC-2 PSD.

        Returns:
            Float[Array, "n_sample"]: Array of PSD values of the detector.
        """
        return self.psd

    def _equal_data_psd_frequencies(self) -> Bool:
        """Check if the frequencies of the data and PSD match.
        A helper function for `set_data` and `set_psd`.

        Return:
            Bool: True if the frequencies match, False otherwise.
        """
        if self.psd.is_empty or self.data.is_empty:
            # In this case, we simply skip the check
            return True
        if self.psd.n_freq != self.data.n_freq:
            # Cannot proceed comparison, needs interpolation
            return False
        if (self.psd.frequencies == self.data.frequencies).all():
            # Frequencies match
            return True
        # This case means the frequencies are different
        return False

    def set_data(self, data: Data | Array, **kws) -> None:
        """Add data to the detector.

        Args:
            data (Data | Array): Data to be added to the detector, either as a `Data` object
                or as a timeseries array.
            **kws (dict): Additional keyword arguments to pass to `Data` constructor.

        Returns:
            None
        """
        if isinstance(data, Data):
            self.data = data
        else:
            self.data = Data(data, **kws)
        # Assert PSD frequencies agree with data
        if not ((self.psd is None) or self._equal_data_psd_frequencies()):
            self.psd = self.psd.interpolate(self.data.frequencies)

    def set_psd(self, psd: PowerSpectrum | Array, **kws) -> None:
        """Add PSD to the detector.

        Args:
            psd (PowerSpectrum | Array): PSD to be added to the detector, either as a `PowerSpectrum`
                object or as a timeseries array.
            **kws (dict): Additional keyword arguments to pass to `PowerSpectrum` constructor.

        Returns:
            None
        """
        if isinstance(psd, PowerSpectrum):
            self.psd = psd
        else:
            # not clear if we want to support this
            self.psd = PowerSpectrum(psd, **kws)
        # Assert PSD frequencies agree with data frequencies
        if not ((self.data is None) or self._equal_data_psd_frequencies()):
            self.psd = self.psd.interpolate(self.data.frequencies)

    def inject_signal(
        self,
        parameters: dict[str, float],
        is_zero_noise: bool = False,
        rng_key: Optional[Key] = None,
    ) -> None:
        """Inject a signal into the detector data.

        Note: The power spectral density must be set beforehand.

        Args:
            waveform_model: The waveform model to be injected.
            parameters (dict): Dictionary of parameters for the waveform model.

        Returns:
            None
        """
        # 1. Set empty data to initialise the detector
        # n_times = int(duration * sampling_frequency)
        n_times = int(jnp.round(duration * sampling_frequency))
        self.set_data(
            Data(
                name=f"{self.name}_empty",
                td=jnp.zeros(n_times),
                delta_t=1 / sampling_frequency,
                epoch=epoch,
            )
        )

        # 2. Compute the projected strain from parameters
        polarisations = waveform_model(self.frequencies, parameters)
        projected_strain = self.fd_response(self.frequencies, polarisations, parameters)

        # 3. Set the new data
        strain_data = jnp.where(self.frequency_mask, projected_strain, 0.0 + 0.0j)
        if not is_zero_noise:
            if rng_key is None:
                seed = int(time.time())
                rng_key = jax.random.key(seed)
                logger.info(
                    "No rng_key provided for noise simulation. Using time-based key with seed=%d (key=%s).",
                    seed,
                    rng_key,
                )
            strain_data += jnp.where(
                self.frequency_mask, self.psd.simulate_data(rng_key), 0.0 + 0.0j
            )

        self.set_data(
            Data.from_fd(
                name=f"{self.name}_injected",
                fd_strain=strain_data,
                frequencies=self.frequencies,
                epoch=self.data.epoch,
            )
        )

        # 4. Update the sliced data and psd with the (potentially) new frequency bounds
        self.set_frequency_bounds()
        masked_signal = projected_strain[self.frequency_mask]

        df = self.sliced_frequencies[1] - self.sliced_frequencies[0]
        _optimal_snr_sq = inner_product(
            masked_signal, masked_signal, self.sliced_psd, df
        )
        optimal_snr = _optimal_snr_sq**0.5
        match_filtered_snr = complex_inner_product(
            masked_signal, self.sliced_fd_data, self.sliced_psd, df
        )
        match_filtered_snr /= optimal_snr

        logger.info(f"For detector {self.name}, the injected signal has:")
        logger.info(f"  - Optimal SNR: {optimal_snr:.4f}")
        logger.info(f"  - Match filtered SNR: {match_filtered_snr:.4f}")

        
        
def S_ij_TM(f, A = 1):
        
    # write everything in SI units
    c = 299792458 #m/s
    pi = jnp.pi
    
    S = A**2 * 1e-30 * (1 + ( 4e-4 / f )**2) * (1 + (f/8e-3)**4) / (2*pi*c*f)**2

    return S

# Optical Metrology System (OMS) noise PSD
def S_ij_OMS(f, P = 1):
        
    # write everything in SI units
    c = 299792458 #m/s
    pi = jnp.pi
    
    S = P**2 *1e-24 * (1 + ( 2e-3 / f )**4) * (2*pi*f/c)**2

    return S

