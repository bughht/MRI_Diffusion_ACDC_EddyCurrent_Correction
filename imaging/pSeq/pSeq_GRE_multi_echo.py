import numpy as np
import pypulseq as pp

from .pSeq_Base import pSeq_Base


class pSeq_GRE_multi_echo(pSeq_Base):
    """Cartesian multi-echo FLASH/GRE helper for coil sensitivity calibration."""

    def __init__(
        self,
        pseq=None,
        fov=220e-3,
        nx=128,
        ny=128,
        flip_angle=np.deg2rad(12.0),
        slice_thickness=5e-3,
        rf_duration=1.5e-3,
        readout_time=0.64e-3,
        te1=4.0e-3,
        echo_spacing=2.3e-3,
        tr=35e-3,
        n_echoes=6,
        readout_mode="bipolar",
        rf_spoil_inc_deg=117.0,
        spoiler_area_ro=6.0 / 1e-3,
        spoiler_area_pe=6.0 / 1e-3,
        spoiler_area_sl=8.0 / 1e-3,
        *args,
        **kwargs,
    ):
        super().__init__(pseq=pseq, *args, **kwargs)

        self.fov = float(fov)
        self.nx = int(nx)
        self.ny = int(ny)
        self.flip_angle = float(flip_angle)
        self.slice_thickness = float(slice_thickness)
        self.rf_duration = float(rf_duration)
        self.readout_time = float(readout_time)
        self.te1 = float(te1)
        self.echo_spacing = float(echo_spacing)
        self.tr = float(tr)
        self.n_echoes = int(n_echoes)
        self.readout_mode = str(readout_mode).lower()
        self.rf_spoil_inc_deg = float(rf_spoil_inc_deg)

        if self.readout_mode not in {"bipolar", "monopolar"}:
            raise ValueError("readout_mode must be either 'bipolar' or 'monopolar'.")

        self.spoiler_area_ro = float(spoiler_area_ro)
        self.spoiler_area_pe = float(spoiler_area_pe)
        self.spoiler_area_sl = float(spoiler_area_sl)

        self.rf = None
        self.gz = None
        self.gz_reph = None
        self.gx_ro = None
        self.adc = None
        self.gx_pre = None
        self.gx_rew = None
        self.gx_echo_rew = None
        self.gx_spoil = None
        self.gy_spoil = None
        self.gz_spoil = None
        self.delay_te1 = None
        self.delay_echo = None
        self.delay_tr = None
        self.k_delta = None

    @staticmethod
    def _round_up_to_raster(value, raster):
        return float(np.ceil(max(0.0, value) / raster) * raster)

    def _ky_area(self, line_index):
        return (float(line_index) - self.ny / 2.0) * self.k_delta

    def _next_rf_spoil_phase(self):
        if not hasattr(self, "rf_spoil_phase"):
            self.rf_spoil_phase = 0.0
        if not hasattr(self, "rf_spoil_idx"):
            self.rf_spoil_idx = 0

        self.rf_spoil_phase = (self.rf_spoil_phase + self.rf_spoil_inc_deg) % 360.0
        self.rf_spoil_idx += 1
        return np.deg2rad(self.rf_spoil_phase)

    def prep(self, apodization=0.5, tbw=4.0):
        self.k_delta = 1.0 / self.fov

        self.rf, self.gz, self.gz_reph = pp.make_sinc_pulse(
            flip_angle=self.flip_angle,
            system=self.system,
            duration=self.rf_duration,
            slice_thickness=self.slice_thickness,
            apodization=apodization,
            time_bw_product=tbw,
            use="excitation",
            return_gz=True,
        )

        ro_area = self.nx * self.k_delta
        self.gx_ro = pp.make_trapezoid(
            channel="x",
            system=self.system,
            flat_area=ro_area,
            flat_time=self.readout_time,
        )

        adc_raster = self.system.adc_raster_time
        adc_dwell_raw = self.gx_ro.flat_time / self.nx
        adc_dwell = np.floor(adc_dwell_raw / adc_raster) * adc_raster
        if adc_dwell <= 0:
            raise RuntimeError("Computed ADC dwell is non-positive after raster quantization.")

        adc_duration = adc_dwell * self.nx
        adc_delay = self.gx_ro.rise_time + 0.5 * (self.gx_ro.flat_time - adc_duration)
        adc_delay = self._round_up_to_raster(adc_delay, self.system.grad_raster_time)

        self.adc = pp.make_adc(
            num_samples=self.nx,
            dwell=adc_dwell,
            delay=adc_delay,
            system=self.system,
        )

        self.gx_pre = pp.make_trapezoid(
            channel="x",
            system=self.system,
            area=-self.gx_ro.area / 2.0,
        )
        self.gx_rew = pp.make_trapezoid(
            channel="x",
            system=self.system,
            area=-self.gx_pre.area,
        )

        self.gx_spoil = pp.make_trapezoid(
            channel="x", system=self.system, area=self.spoiler_area_ro
        )
        self.gy_spoil = pp.make_trapezoid(
            channel="y", system=self.system, area=self.spoiler_area_pe
        )
        self.gz_spoil = pp.make_trapezoid(
            channel="z", system=self.system, area=self.spoiler_area_sl
        )

        rf_block = pp.calc_duration(self.rf, self.gz)
        rf_center = self.rf.delay + pp.calc_rf_center(self.rf)[0]
        rf_to_end = rf_block - rf_center

        ro_center = self.adc.delay + self.adc.dwell * (self.adc.num_samples / 2.0)
        pre_block = pp.calc_duration(self.gz_reph, self.gx_pre)

        raster = self.system.grad_raster_time

        te1_delay = self.te1 - (rf_to_end + pre_block + ro_center)
        self.delay_te1 = self._round_up_to_raster(te1_delay, raster)

        ro_block = pp.calc_duration(self.gx_ro, self.adc)

        if self.readout_mode == "monopolar" and self.n_echoes > 1:
            # Monopolar mode: constrain rewinder duration to fill the inter-echo gap.
            # No additional delay block is inserted before the next echo.
            target_rew_duration = self.echo_spacing - ro_block
            if target_rew_duration <= 0:
                raise RuntimeError(
                    "Requested echo spacing is too short for constrained monopolar rewind. "
                    f"Need > {ro_block*1e3:.3f} ms, got {self.echo_spacing*1e3:.3f} ms."
                )

            try:
                self.gx_echo_rew = pp.make_trapezoid(
                    channel="x",
                    system=self.system,
                    area=-self.gx_ro.area,
                    duration=target_rew_duration,
                )
            except Exception as exc:
                gx_echo_rew_min = pp.make_trapezoid(
                    channel="x",
                    system=self.system,
                    area=-self.gx_ro.area,
                )
                min_rew = pp.calc_duration(gx_echo_rew_min)
                raise RuntimeError(
                    "Constrained monopolar rewind cannot meet gradient limits at requested echo spacing. "
                    f"Need at least {(ro_block + min_rew)*1e3:.3f} ms, got {self.echo_spacing*1e3:.3f} ms."
                ) from exc

            inter_echo_min = ro_block + pp.calc_duration(self.gx_echo_rew)
            self.delay_echo = 0.0
        else:
            self.gx_echo_rew = None
            inter_echo_min = ro_block
            echo_delay = self.echo_spacing - inter_echo_min
            self.delay_echo = self._round_up_to_raster(echo_delay, raster)

        if self.readout_mode == "monopolar" and self.gx_echo_rew is not None:
            inter_echo_time = (self.n_echoes - 1) * (pp.calc_duration(self.gx_echo_rew) + self.delay_echo)
        else:
            inter_echo_time = (self.n_echoes - 1) * self.delay_echo

        tr_fixed = (
            rf_block
            + pre_block
            + self.delay_te1
            + self.n_echoes * ro_block
            + inter_echo_time
            + pp.calc_duration(self.gx_rew)
            + pp.calc_duration(self.gx_spoil, self.gy_spoil, self.gz_spoil)
        )
        self.delay_tr = self._round_up_to_raster(self.tr - tr_fixed, raster)

        if tr_fixed > self.tr + 1e-9:
            raise RuntimeError(
                "Requested TR is too short for configured multi-echo GRE timing. "
                f"Need at least {tr_fixed*1e3:.3f} ms, got {self.tr*1e3:.3f} ms."
            )

        if self.delay_te1 <= 0:
            raise RuntimeError(
                "Requested TE1 is too short for configured RF/readout timing."
                f"Need at least {(rf_to_end + pre_block + ro_center)*1e3:.3f} ms, got {self.te1*1e3:.3f} ms."
            )

        if self.readout_mode == "bipolar" and self.delay_echo <= 0 and self.n_echoes > 1:
            raise RuntimeError(
                "Requested echo spacing is too short for the readout duration."
                f"Need at least {inter_echo_min*1e3:.3f} ms, got {self.echo_spacing*1e3:.3f} ms."
            )

        print("Prepared multi-echo FLASH blocks:")
        print(f"  Matrix: {self.nx}x{self.ny}, FOV: {self.fov*1e3:.1f} mm")
        print(f"  Readout mode: {self.readout_mode}")
        if self.readout_mode == "monopolar" and self.gx_echo_rew is not None:
            print(f"  Inter-echo rewind (constrained): {pp.calc_duration(self.gx_echo_rew)*1e3:.3f} ms")
            print("  Inter-echo delay block: 0.000 ms")
        print(f"  Echoes: {self.n_echoes}, TE1: {self.te1*1e3:.2f} ms, dTE: {self.echo_spacing*1e3:.2f} ms")
        print(f"  TR: {self.tr*1e3:.2f} ms, TR fill: {self.delay_tr*1e3:.2f} ms")

    def add_line_to_seq(self, pseq0, line_index, slice_offset=0.0, rep_idx=0, slc_idx=0, label=True):
        if self.rf is None:
            raise RuntimeError("Call prep() before add_line_to_seq().")

        ky_area = self._ky_area(line_index)
        gy_pre = pp.make_trapezoid(channel="y", system=self.system, area=ky_area)
        gy_rew = pp.make_trapezoid(channel="y", system=self.system, area=-ky_area)

        self.rf.freq_offset = self.gz.amplitude * float(slice_offset)
        rf_center = pp.calc_rf_center(self.rf)[0]
        self.rf.phase_offset = -2.0 * np.pi * self.rf.freq_offset * rf_center

        spoil_phase = self._next_rf_spoil_phase()
        self.rf.phase_offset += spoil_phase
        self.adc.phase_offset = spoil_phase

        if label:
            pseq0.seq.add_block(
                pp.make_label(type="SET", label="REP", value=int(rep_idx)),
                pp.make_label(type="SET", label="SLC", value=int(slc_idx)),
                pp.make_label(type="SET", label="LIN", value=int(line_index)),
            )

        pseq0.seq.add_block(self.rf, self.gz)
        pseq0.seq.add_block(self.gz_reph, self.gx_pre, gy_pre)

        if self.delay_te1 > 0:
            pseq0.seq.add_block(pp.make_delay(self.delay_te1))

        for echo_idx in range(self.n_echoes):
            if self.readout_mode == "bipolar":
                gx_sign = 1.0 if (echo_idx % 2 == 0) else -1.0
            else:
                gx_sign = 1.0

            gx_echo = pp.scale_grad(self.gx_ro, gx_sign)
            lrev = pp.make_label(type="SET", label="REV", value=int(gx_sign < 0)) if label else None
            leco = pp.make_label(type="SET", label="ECO", value=int(echo_idx)) if label else None

            if label:
                pseq0.seq.add_block(gx_echo, self.adc, lrev, leco)
            else:
                pseq0.seq.add_block(gx_echo, self.adc)

            if echo_idx < (self.n_echoes - 1):
                if self.readout_mode == "monopolar" and self.gx_echo_rew is not None:
                    pseq0.seq.add_block(self.gx_echo_rew)
                if self.delay_echo > 0:
                    pseq0.seq.add_block(pp.make_delay(self.delay_echo))

        # pseq0.seq.add_block(self.gx_rew, gy_rew)
        pseq0.seq.add_block(self.gx_spoil, self.gy_spoil, self.gz_spoil)

        if self.delay_tr > 0:
            pseq0.seq.add_block(pp.make_delay(self.delay_tr))

    def get_timing(self):
        if self.rf is None:
            raise RuntimeError("Call prep() before get_timing().")

        ro_block = pp.calc_duration(self.gx_ro, self.adc)

        if self.readout_mode == "monopolar" and self.gx_echo_rew is not None:
            inter_echo_time = (self.n_echoes - 1) * (pp.calc_duration(self.gx_echo_rew) + self.delay_echo)
        else:
            inter_echo_time = (self.n_echoes - 1) * self.delay_echo

        tr_nominal = (
            pp.calc_duration(self.rf, self.gz)
            + pp.calc_duration(self.gz_reph, self.gx_pre)
            + self.delay_te1
            + self.n_echoes * ro_block
            + inter_echo_time
            + pp.calc_duration(self.gx_rew)
            + pp.calc_duration(self.gx_spoil, self.gy_spoil, self.gz_spoil)
            + self.delay_tr
        )

        return {
            "te1": self.te1,
            "echo_spacing": self.echo_spacing,
            "readout_mode": self.readout_mode,
            "inter_echo_delay": self.delay_echo,
            "inter_echo_rewind_duration": (
                0.0 if self.gx_echo_rew is None else pp.calc_duration(self.gx_echo_rew)
            ),
            "tr_nominal": tr_nominal,
            "n_echoes": self.n_echoes,
        }
