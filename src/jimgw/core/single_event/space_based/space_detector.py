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

class SpaceBased(Detector):
    """Object representing a ground-based detector.

    Contains information about the location and orientation of the detector on Earth,
    as well as actual strain data and the PSD of the associated noise.

    Attributes:
        name (str): Name of the detector.
        latitude (Float): Latitude of the detector in radians.
        longitude (Float): Longitude of the detector in radians.
        xarm_azimuth (Float): Azimuth of the x-arm in radians.
        yarm_azimuth (Float): Azimuth of the y-arm in radians.
        xarm_tilt (Float): Tilt of the x-arm in radians.
        yarm_tilt (Float): Tilt of the y-arm in radians.
        elevation (Float): Elevation of the detector in meters.
        polarization_mode (list[Polarization]): List of polarization modes (`pc` for plus and cross) to be used in
            computing antenna patterns; in the future, this could be expanded to
            include non-GR modes.
        data (Data): Array of Fourier-domain strain data.
        psd (PowerSpectrum): Power spectral density object.
    """

    polarization_mode: list[Polarization]
    data: Data
    psd: PowerSpectrum

    latitude: Float = 0
    longitude: Float = 0
    xarm_azimuth: Float = 0
    yarm_azimuth: Float = 0
    xarm_tilt: Float = 0
    yarm_tilt: Float = 0
    elevation: Float = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __init__(
        self,
        name: str,
        latitude: float = 0,
        longitude: float = 0,
        elevation: float = 0,
        xarm_azimuth: float = 0,
        yarm_azimuth: float = 0,
        xarm_tilt: float = 0,
        yarm_tilt: float = 0,
        modes: str = "pc",
    ):
        """Initialize a ground-based detector.

        Args:
            name (str): Name of the detector.
            latitude (float, optional): Latitude of the detector in radians. Defaults to 0.
            longitude (float, optional): Longitude of the detector in radians. Defaults to 0.
            elevation (float, optional): Elevation of the detector in meters. Defaults to 0.
            xarm_azimuth (float, optional): Azimuth of the x-arm in radians. Defaults to 0.
            yarm_azimuth (float, optional): Azimuth of the y-arm in radians. Defaults to 0.
            xarm_tilt (float, optional): Tilt of the x-arm in radians. Defaults to 0.
            yarm_tilt (float, optional): Tilt of the y-arm in radians. Defaults to 0.
            modes (str, optional): Polarization modes. Defaults to "pc".
        """
        super().__init__()

        self.name = name

        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.xarm_azimuth = xarm_azimuth
        self.yarm_azimuth = yarm_azimuth
        self.xarm_tilt = xarm_tilt
        self.yarm_tilt = yarm_tilt

        self.polarization_mode = [Polarization(m) for m in modes]
        self.data = Data()
        self.psd = PowerSpectrum()

    @staticmethod
    def _get_arm(
        lat: Float, lon: Float, tilt: Float, azimuth: Float
    ) -> Float[Array, "3"]:
        """Construct detector-arm vectors in geocentric Cartesian coordinates.

        Args:
            lat (Float): Vertex latitude in radians.
            lon (Float): Vertex longitude in radians.
            tilt (Float): Arm tilt in radians.
            azimuth (Float): Arm azimuth in radians.

        Returns:
            Float[Array, "3"]: Detector arm vector in geocentric Cartesian coordinates.
        """

    @property
    def arms(self) -> tuple[Float[Array, "3"], Float[Array, "3"]]:
        """Get the detector arm vectors.

        Returns:
            tuple[Float[Array, "3"], Float[Array, "3"]]: A tuple containing:
                - x: X-arm vector in geocentric Cartesian coordinates
                - y: Y-arm vector in geocentric Cartesian coordinates
        """


    @property
    def tensor(self) -> Float[Array, "3 3"]:
        """Get the detector tensor defining the strain measurement.

        For a 2-arm differential-length detector, this is given by:

        .. math::

            D_{ij} = \\left(x_i x_j - y_i y_j\\right)/2

        for unit vectors :math:`x` and :math:`y` along the x and y arms.

        Returns:
            Float[Array, "3 3"]: The 3x3 detector tensor in geocentric coordinates.
        """
      raise NotImplemented

    def fd_response(
        self,
        frequency: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
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
                - trigger_time (Float): The trigger time in sec
                - t_c (Float): The difference between peak time and trigger time in sec
                - gmst (Float): The greenwich mean sidereal time at the trigger time in radian
            **kwargs: Additional keyword arguments.

        Returns:
            Array: Complex strain measured by the detector in frequency domain, obtained by
                  combining the antenna patterns for each polarization mode.
        """
        ra, dec, psi, gmst = params["ra"], params["dec"], params["psi"], params["gmst"]
        antenna_pattern = self.antenna_pattern(ra, dec, psi, gmst)
        time_shift = self.delay_from_geocenter(ra, dec, gmst)
        time_shift += params["trigger_time"] - self.epoch + params["t_c"]

        h_detector = jax.tree_util.tree_map(
            lambda h, antenna: h * antenna,
            h_sky,
            antenna_pattern,
        )
        projected_strain = jnp.sum(
            jnp.stack(jax.tree_util.tree_leaves(h_detector)), axis=0
        )

        phase_shift = jnp.exp(-2j * jnp.pi * frequency * time_shift)
        return projected_strain * phase_shift

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
    def load_and_set_psd(self, psd_file: str = "", asd_file: str = "") -> PowerSpectrum:
        """Load power spectral density (PSD) from file or default GWTC-2 catalog,
            and set it to the detector.

        Args:
            psd_file (str, optional): Path to file containing PSD data. If empty, uses GWTC-2 PSD.

        Returns:
            Float[Array, "n_sample"]: Array of PSD values of the detector.
        """
        if psd_file != "":
            f, psd_vals = loadtxt(psd_file, unpack=True)
        elif asd_file != "":
            f, asd_vals = loadtxt(asd_file, unpack=True)
            psd_vals = asd_vals**2
        else:
            logger.info("Grabbing GWTC-2 PSD for " + self.name)
            url = asd_file_dict[self.name]
            data = requests.get(url)
            tmp_file_name = f"fetched_default_asd_{self.name}.txt"
            open(tmp_file_name, "wb").write(data.content)
            f, asd_vals = loadtxt(tmp_file_name, unpack=True)
            psd_vals = asd_vals**2

        _loaded_psd = PowerSpectrum(psd_vals, f, name=f"{self.name}_psd")
        self.set_psd(_loaded_psd)
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
        duration: float,
        sampling_frequency: float,
        epoch: float,
        waveform_model,
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

    def get_whitened_frequency_domain_strain(
        self, frequency_series: Complex[Array, " n_freq"]
    ) -> Complex[Array, " n_freq"]:
        """Get the whitened frequency-domain strain.
        Args:
            frequency_series (Complex[Array, "n_freq"]): Array of frequency domain data/signal.
        Returns:
            Complex[Array, "n_freq"]: Whitened frequency-domain strain.
        """
        scaled_asd = jnp.sqrt(self.psd.values * self.duration / 4)
        return (frequency_series / scaled_asd) * self.frequency_mask

    def whitened_frequency_to_time_domain_strain(
        self, whitened_frequency_series: Complex[Array, " n_time // 2 + 1"]
    ) -> Float[Array, " n_time"]:
        """Get the whitened frequency-domain strain.
        Args:
            whitened_frequency_series (Complex[Array, "n_time // 2 + 1"]):
                Array of whitened frequency domain data/signal.
        Returns:
            Float[Array, "n_time"]: Whitened time-domain strain/signal.
        """
        freq_mask_ratio = len(self.frequency_mask) / jnp.sqrt(
            jnp.sum(self.frequency_mask)
        )
        return jnp.fft.irfft(whitened_frequency_series) * freq_mask_ratio

    @property
    def whitened_frequency_domain_data(self) -> Complex[Array, " n_sample"]:
        """Get the whitened frequency-domain data.

        Args:
            frequency (Float[Array, "n_sample"]): Array of frequency samples.

        Returns:
            Float[Array, "n_sample"]: Whitened frequency-domain data.
        """

        return self.get_whitened_frequency_domain_strain(self.data.fd)

    @property
    def whitened_time_domain_data(self) -> Float[Array, " n_sample"]:
        """Get the whitened time-domain data.

        Args:
            time (Float[Array, "n_sample"]): Array of time samples.

        Returns:
            Float[Array, "n_sample"]: Whitened time-domain data.
        """
        return self.whitened_frequency_to_time_domain_strain(
            self.whitened_frequency_domain_data
        )
