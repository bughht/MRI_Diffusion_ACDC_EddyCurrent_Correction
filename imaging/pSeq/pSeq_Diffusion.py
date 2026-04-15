import numpy as np
import pypulseq as pp


class pSeq_Diffusion:
    """
    Monopolar diffusion gradient designer for spin-echo diffusion EPI.

    This class designs a pair of identical monopolar trapezoidal lobes that
    match a target b-value under system gradient/slew limits while minimizing
    TE and enforcing spin-echo alignment at the EPI k-space center.
    """

    def __init__(
        self,
        system,
        channel="x",
        target_b_s_mm2=1000.0,
        max_lobe_duration=40e-3,
        min_lobe_duration=None,
    ):
        self.system = system
        self.channel = channel
        self.target_b_s_mm2 = float(target_b_s_mm2)
        self.max_lobe_duration = float(max_lobe_duration)
        self.min_lobe_duration = min_lobe_duration

        self.lobe = None
        self.lobe_duration = None
        self.amplitude = None
        self.delay1 = None
        self.delay2 = None
        self.pre_lobe_delay = None
        self.inter_lobe_delay = None
        self.te = None
        self.b_actual_s_mm2 = None
        self.spin_echo_mismatch = None

    @staticmethod
    def _ceil_to_raster(value, raster):
        return float(np.ceil(value / raster) * raster)

    @staticmethod
    def _sample_trapezoid(grad, dt):
        """Sample a trapezoid gradient object at midpoint times."""
        total = pp.calc_duration(grad)
        n = int(np.round(total / dt))
        if n <= 0:
            return np.zeros(0, dtype=float)

        t = (np.arange(n) + 0.5) * dt

        amp = float(grad.amplitude)
        rise = float(getattr(grad, "rise_time", 0.0))
        flat = float(getattr(grad, "flat_time", 0.0))
        fall = float(getattr(grad, "fall_time", 0.0))

        g = np.zeros_like(t)

        # Rising ramp
        if rise > 0:
            m = t < rise
            g[m] = amp * (t[m] / rise)

        # Flat
        m = (t >= rise) & (t < rise + flat)
        g[m] = amp

        # Falling ramp
        if fall > 0:
            m = (t >= rise + flat) & (t < rise + flat + fall)
            g[m] = amp * (1.0 - (t[m] - rise - flat) / fall)

        return g

    def _compute_spin_echo_timing(self, rf90, rf180, epi, diff_dur):
        """Compute delays that force EPI center to coincide with spin echo."""
        rf90_after = rf90.get_duration() - rf90.get_timing()[1]["center_incl_delay"]
        rf180_center = rf180.get_timing()[1]["center_incl_delay"]
        rf180_after = rf180.get_duration() - rf180_center
        epi_to_te = epi.get_timeToTE()["timeToTE"]

        left = rf90_after + diff_dur + rf180_center
        right = rf180_after + diff_dur + epi_to_te
        half_te = max(left, right)

        d1 = max(0.0, half_te - left)
        d2 = max(0.0, half_te - right)

        raster = self.system.grad_raster_time
        d1 = self._ceil_to_raster(d1, raster)
        d2 = self._ceil_to_raster(d2, raster)

        # Residual mismatch after raster alignment (should be tiny)
        left_aligned = left + d1
        right_aligned = right + d2
        mismatch = left_aligned - right_aligned

        return {
            "rf90_after": rf90_after,
            "rf180_duration": rf180.get_duration(),
            "delay1": d1,
            "delay2": d2,
            "pre_lobe_delay": rf90_after,
            "inter_lobe_delay": d1 + rf180.get_duration() + d2,
            "te": 2.0 * max(left_aligned, right_aligned),
            "spin_echo_mismatch": mismatch,
        }

    def _b_from_lobe_timing(self, lobe, amp_scale, pre_lobe_delay, inter_lobe_delay):
        """Compute b-value (s/mm^2) using discrete integration with full timing."""
        dt = float(self.system.grad_raster_time)

        g0 = self._sample_trapezoid(lobe, dt) * float(amp_scale)
        # Post-180 sign inversion for effective dephasing trajectory
        g1 = -self._sample_trapezoid(lobe, dt) * float(amp_scale)

        n_pre = int(np.round(pre_lobe_delay / dt))
        n_gap = int(np.round(inter_lobe_delay / dt))

        g = np.concatenate(
            [
                np.zeros(max(0, n_pre), dtype=float),
                g0,
                np.zeros(max(0, n_gap), dtype=float),
                g1,
            ]
        )

        if g.size == 0:
            return 0.0

        q = 2.0 * np.pi * np.cumsum(g * dt)  # rad/m
        b_s_m2 = np.sum((q ** 2) * dt)
        return float(b_s_m2 * 1e-6)

    def prep_monopolar(self, rf90, rf180, epi, target_b_s_mm2=None, verbose=True):
        """
        Design minimum-TE monopolar diffusion lobes for target b-value.

        Returns a dict of final design parameters.
        """
        if target_b_s_mm2 is not None:
            self.target_b_s_mm2 = float(target_b_s_mm2)

        raster = float(self.system.grad_raster_time)
        gmax = float(self.system.max_grad)

        # Minimum feasible duration at max amplitude under slew limit
        # (triangular lobe: flat=0, total duration=2*rise).
        rise_min = self._ceil_to_raster(gmax / float(self.system.max_slew), raster)
        min_dur = 2.0 * rise_min
        if self.min_lobe_duration is not None:
            min_dur = max(min_dur, float(self.min_lobe_duration))

        min_dur = self._ceil_to_raster(min_dur, raster)
        max_dur = self._ceil_to_raster(self.max_lobe_duration, raster)

        best = None
        n_steps = int(np.round((max_dur - min_dur) / raster)) + 1

        for i in range(max(1, n_steps)):
            dur = min_dur + i * raster
            try:
                lobe_max = pp.make_trapezoid(
                    channel=self.channel,
                    system=self.system,
                    amplitude=gmax,
                    duration=dur,
                )
            except ValueError:
                # Duration may be too short for the requested amplitude/slew.
                continue
            lobe_dur = pp.calc_duration(lobe_max)

            t = self._compute_spin_echo_timing(rf90, rf180, epi, lobe_dur)
            b_max = self._b_from_lobe_timing(
                lobe=lobe_max,
                amp_scale=1.0,
                pre_lobe_delay=t["pre_lobe_delay"],
                inter_lobe_delay=t["inter_lobe_delay"],
            )

            if b_max <= 0:
                continue

            amp_needed = gmax * np.sqrt(self.target_b_s_mm2 / b_max)
            if amp_needed > gmax + 1e-9:
                continue

            lobe_final = pp.make_trapezoid(
                channel=self.channel,
                system=self.system,
                amplitude=amp_needed,
                duration=lobe_dur,
            )

            b_actual = self._b_from_lobe_timing(
                lobe=lobe_final,
                amp_scale=1.0,
                pre_lobe_delay=t["pre_lobe_delay"],
                inter_lobe_delay=t["inter_lobe_delay"],
            )

            candidate = {
                "lobe": lobe_final,
                "lobe_duration": pp.calc_duration(lobe_final),
                "amplitude": float(lobe_final.amplitude),
                "delay1": t["delay1"],
                "delay2": t["delay2"],
                "pre_lobe_delay": t["pre_lobe_delay"],
                "inter_lobe_delay": t["inter_lobe_delay"],
                "te": t["te"],
                "b_actual_s_mm2": b_actual,
                "spin_echo_mismatch": t["spin_echo_mismatch"],
            }

            if best is None or candidate["te"] < best["te"]:
                best = candidate

        if best is None:
            raise RuntimeError(
                "Could not meet target b-value within system limits and search range. "
                "Try lowering target_b_s_mm2 or increasing max_lobe_duration."
            )

        self.lobe = best["lobe"]
        self.lobe_duration = best["lobe_duration"]
        self.amplitude = best["amplitude"]
        self.delay1 = best["delay1"]
        self.delay2 = best["delay2"]
        self.pre_lobe_delay = best["pre_lobe_delay"]
        self.inter_lobe_delay = best["inter_lobe_delay"]
        self.te = best["te"]
        self.b_actual_s_mm2 = best["b_actual_s_mm2"]
        self.spin_echo_mismatch = best["spin_echo_mismatch"]

        if verbose:
            print("Diffusion designer summary:")
            print(f"  target b: {self.target_b_s_mm2:.2f} s/mm^2")
            print(f"  actual b: {self.b_actual_s_mm2:.2f} s/mm^2")
            print(f"  lobe amplitude: {self.amplitude:.3f} Hz/m")
            print(f"  lobe duration: {self.lobe_duration*1e3:.3f} ms")
            print(f"  delay1: {self.delay1*1e3:.3f} ms")
            print(f"  delay2: {self.delay2*1e3:.3f} ms")
            print(f"  pre-lobe delay: {self.pre_lobe_delay*1e3:.3f} ms")
            print(f"  inter-lobe delay: {self.inter_lobe_delay*1e3:.3f} ms")
            print(f"  TE(min): {self.te*1e3:.3f} ms")
            print(f"  spin-echo mismatch: {self.spin_echo_mismatch*1e6:.3f} us")

        return best

    def _make_rotated_trapezoid(self, channel, direction_component, amp_scale):
        return pp.make_trapezoid(
            channel=channel,
            system=self.system,
            amplitude=self.amplitude * float(amp_scale) * float(direction_component),
            duration=self.lobe_duration,
        )

    def get_rotated_lobes(self, direction, amp_scale=1.0):
        """Return rotated first/second monopolar lobe blocks for a 3D direction."""
        if self.lobe is None:
            raise RuntimeError("Call prep_monopolar(...) before requesting lobes.")

        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm == 0:
            d = np.zeros(3, dtype=float)
        else:
            d = direction / norm

        g0 = [
            self._make_rotated_trapezoid("x", d[0], amp_scale),
            self._make_rotated_trapezoid("y", d[1], amp_scale),
            self._make_rotated_trapezoid("z", d[2], amp_scale),
        ]
        g1 = [
            self._make_rotated_trapezoid("x", d[0], amp_scale),
            self._make_rotated_trapezoid("y", d[1], amp_scale),
            self._make_rotated_trapezoid("z", d[2], amp_scale),
        ]
        return g0, g1
