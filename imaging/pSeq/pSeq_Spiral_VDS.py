import numpy as np
import pypulseq as pp

from .pSeq_Base import pSeq_Base


class pSeq_Spiral_VDS(pSeq_Base):
    """Variable-density spiral readout block with multi-interleaf support.

    The trajectory generator follows a constrained integration strategy using
    radial/azimuthal dynamics with explicit gradient-amplitude and slew limits,
    similar in spirit to classic VDS formulations.

    Notes
    -----
    The ``accel`` parameter controls in-plane spiral undersampling.
    ``accel=1`` is nominal sampling. Larger values both relax the FOV-driven
    Nyquist cap and reduce spiral turn-density, which shortens readout duration
    at the cost of increased aliasing burden on reconstruction.

    ADC dwell can be set independently from gradient raster in ``prep``.
    The trajectory design uses ADC dwell for the FOV/Nyquist cap so higher
    readout bandwidth (smaller dwell) can reduce readout duration.
    """

    def __init__(
        self,
        pseq=None,
        fov=220e-3,
        nx=128,
        ny=None,
        n_interleaves=1,
        fov_coeff=None,
        accel=1.0,
        rmax=None,
        oversamp=4,
        max_points=2_000_000,
        adc_divisor=4,
        *args,
        **kwargs,
    ):
        super().__init__(pseq, *args, **kwargs)

        self.fov = float(fov)
        self.Nx = int(nx)
        self.Ny = int(nx if ny is None else ny)
        self.n_interleaves = int(n_interleaves)
        self.fov_coeff = np.array([1.0, 0.0] if fov_coeff is None else fov_coeff, dtype=float)
        self.accel = float(accel)
        self.accel_effective = float(accel)
        self.oversamp = int(oversamp)
        self.max_points = int(max_points)
        self.adc_divisor = int(adc_divisor)

        # Nominal k-space radius for target in-plane resolution.
        self.kmax = float(self.Nx) / (2.0 * self.fov) if rmax is None else float(rmax)

        self._validate_inputs()

        self.dt = float(self.system.grad_raster_time)
        self.k_traj = None
        self.g_traj_hz = None
        self.slew_hz = None
        self.t = None

        self.readout_duration = None
        self.rampdown_duration = None
        self.rewinder_duration = None
        self.shot_duration = None

        self.adc = None
        self.adc_samples = None
        self.adc_dwell = None
        self.adc_time = None
        self.time_to_center = None

        self._interleaf_events = []
        self._interleaf_ktraj = []
        self._interleaf_adc_ktraj = []
        self._interleaf_endk = []
        self._rampdowns = []
        self._rewinders = []

    def _validate_inputs(self):
        if self.fov <= 0:
            raise ValueError("fov must be positive.")
        if self.Nx <= 0 or self.Ny <= 0:
            raise ValueError("nx and ny must be positive integers.")
        if self.n_interleaves <= 0:
            raise ValueError("n_interleaves must be >= 1.")
        if self.accel < 1.0:
            raise ValueError("accel must be >= 1.0.")
        if self.kmax <= 0:
            raise ValueError("rmax/kmax must be positive.")
        if self.oversamp < 2 or (self.oversamp % 2) != 0:
            raise ValueError("oversamp must be an even integer >= 2.")
        if self.adc_divisor <= 0:
            raise ValueError("adc_divisor must be positive.")
        if self.fov_coeff.size == 0:
            raise ValueError("fov_coeff cannot be empty.")

    def _fov_poly(self, r):
        x = np.clip(r / self.kmax, 0.0, 1.0)
        powers = np.array([x ** i for i in range(len(self.fov_coeff))], dtype=float)
        fscale = float(np.dot(self.fov_coeff, powers))

        if fscale <= 0:
            raise ValueError("fov_coeff produces non-positive FOV profile.")

        fov_r = self.fov * fscale

        if len(self.fov_coeff) == 1:
            dscale_dr = 0.0
        else:
            dscale_dr = 0.0
            for i in range(1, len(self.fov_coeff)):
                dscale_dr += i * self.fov_coeff[i] * (x ** (i - 1)) / self.kmax

        dfov_dr = self.fov * dscale_dr
        return fov_r, dfov_dr

    @staticmethod
    def _quadratic_roots(a, b, c):
        disc = b * b - 4.0 * a * c
        if disc < 0:
            disc = 0.0
        d = np.sqrt(disc)
        return ((-b + d) / (2.0 * a), (-b - d) / (2.0 * a))

    def _design_vds_single_interleaf_once(self, adc_dwell, accel_design):
        gamma_hz_per_t = float(self.system.gamma)

        # Internal limits in Hz/m and Hz/m/s.
        # Use a small design margin so downstream event construction remains
        # within hard scanner constraints after discretization/rounding.
        gmax_hz = 0.999 * float(self.system.max_grad)
        smax_hz = 0.995 * float(self.system.max_slew)

        dt_int = self.dt / float(self.oversamp)

        theta = 0.0
        theta_dot = 0.0
        r = 0.0
        r_dot = 0.0

        r_hist = [r]
        th_hist = [theta]

        n = 1
        while r < self.kmax and n < self.max_points:
            fov_r, dfov_dr = self._fov_poly(r)

            # Acceleration is modeled as reduced effective sampling FOV.
            # This impacts both Nyquist gradient cap and turn density.
            fov_eff = fov_r / accel_design
            dfov_eff_dr = dfov_dr / accel_design

            # FOV-limited gradient from sampling theorem using ADC dwell.
            gmax_fov_hz = 1.0 / (fov_eff * adc_dwell)
            gmax_eff_hz = min(gmax_hz, gmax_fov_hz)

            twopi_over_n = 2.0 * np.pi / float(self.n_interleaves)
            w = twopi_over_n * fov_eff
            max_r_dot = np.sqrt((gmax_eff_hz ** 2) / (1.0 + (w * r) ** 2))

            if r_dot > max_r_dot:
                r_ddot = (max_r_dot - r_dot) / dt_int
            else:
                a = 1.0 + (w ** 2) * (r ** 2)
                b = (
                    2.0 * (w ** 2) * r * (r_dot ** 2)
                    + 2.0 * (w ** 2) * (r ** 2) * (r_dot ** 2) * (dfov_eff_dr / fov_eff)
                )
                c = (
                    (w ** 4) * (r ** 2) * (r_dot ** 4)
                    + 4.0 * (w ** 2) * (r_dot ** 4)
                    + ((twopi_over_n * dfov_eff_dr) ** 2) * (r ** 2) * (r_dot ** 4)
                    + 4.0 * (w ** 2) * (dfov_eff_dr / fov_eff) * r * (r_dot ** 4)
                    - (smax_hz ** 2)
                )

                rt1, rt2 = self._quadratic_roots(a, b, c)
                r_ddot = max(rt1, rt2)

            theta_ddot = (
                twopi_over_n * dfov_eff_dr * (r_dot ** 2)
                + w * r_ddot
            )

            theta_dot = max(0.0, theta_dot + theta_ddot * dt_int)
            theta = theta + theta_dot * dt_int

            r_dot = max(0.0, r_dot + r_ddot * dt_int)
            r = r + r_dot * dt_int

            r_hist.append(r)
            th_hist.append(theta)
            n += 1

        if n >= self.max_points:
            raise RuntimeError("VDS trajectory generation reached max_points before kmax.")

        r_hist = np.asarray(r_hist, dtype=float)
        th_hist = np.asarray(th_hist, dtype=float)

        # Decimate to hardware raster.
        dec0 = self.oversamp // 2
        r_dec = r_hist[dec0:: self.oversamp]
        th_dec = th_hist[dec0:: self.oversamp]

        # Keep ADC/sample count scanner-friendly.
        n_keep = (len(r_dec) // self.adc_divisor) * self.adc_divisor
        if n_keep < self.adc_divisor:
            raise RuntimeError("Designed spiral is too short after decimation.")
        r_dec = r_dec[:n_keep]
        th_dec = th_dec[:n_keep]

        k = r_dec * np.exp(1j * th_dec)  # cycles/m
        g_hz = np.diff(np.concatenate(([0.0 + 0.0j], k))) / self.dt  # Hz/m
        s_hz = np.diff(np.concatenate(([0.0 + 0.0j], g_hz))) / self.dt  # Hz/m/s

        # Hardware validation in SI units.
        g_t_per_m = np.abs(g_hz) / gamma_hz_per_t
        s_t_per_m_s = np.abs(s_hz) / gamma_hz_per_t

        if np.max(g_t_per_m) > (self.system.max_grad / gamma_hz_per_t) * 1.005:
            raise RuntimeError("Spiral gradient exceeds max_grad after design.")
        if np.max(s_t_per_m_s) > (self.system.max_slew / gamma_hz_per_t) * 1.005:
            raise RuntimeError("Spiral slew exceeds max_slew after design.")

        t = np.arange(len(k), dtype=float) * self.dt
        return k, g_hz, s_hz, t

    def _design_vds_single_interleaf(self, adc_dwell):
        """Design one interleaf, backing off acceleration if needed for slew safety."""
        self.accel_effective = float(self.accel)

        try:
            return self._design_vds_single_interleaf_once(adc_dwell=adc_dwell, accel_design=self.accel_effective)
        except RuntimeError as exc_req:
            if self.accel <= 1.0:
                raise exc_req

            # Requested acceleration can be infeasible under hardware/slew limits.
            # Find the largest feasible acceleration in [1, accel] by binary search.
            lo = 1.0
            hi = float(self.accel)

            try:
                best = (lo, self._design_vds_single_interleaf_once(adc_dwell=adc_dwell, accel_design=lo))
            except RuntimeError:
                # If even nominal acceleration is infeasible, propagate original error.
                raise exc_req

            for _ in range(12):
                mid = 0.5 * (lo + hi)
                try:
                    out_mid = self._design_vds_single_interleaf_once(adc_dwell=adc_dwell, accel_design=mid)
                    lo = mid
                    best = (mid, out_mid)
                except RuntimeError:
                    hi = mid

            self.accel_effective = float(best[0])
            return best[1]

    def _make_rewinder_pair(self, ax, ay):
        """Create a rewinder gradient pair with exact area cancellation.

        Uses iterative correction to compensate for gradient raster discretization
        so that the rewinder trapezoid area exactly matches the requested value.
        """
        gx = None
        gy = None
        if abs(ax) > 1e-9:
            gx = self._make_trapezoid_exact_area(channel="x", area_target=ax)
        if abs(ay) > 1e-9:
            gy = self._make_trapezoid_exact_area(channel="y", area_target=ay)

        if gx is None and gy is None:
            return None, None
        if gx is None:
            gx = pp.make_trapezoid(channel="x", system=self.system, area=0.0, duration=pp.calc_duration(gy))
        if gy is None:
            gy = pp.make_trapezoid(channel="y", system=self.system, area=0.0, duration=pp.calc_duration(gx))
        gx, gy = pp.align(right=gx, left=gy)
        return gx, gy

    def _make_trapezoid_exact_area(self, channel, area_target, max_iter=5, tol=1e-12):
        """Create a trapezoid whose actual area matches the target within tolerance.

        Iteratively corrects for gradient raster discretization errors.
        """
        area_req = float(area_target)
        g = pp.make_trapezoid(channel=channel, system=self.system, area=area_req)
        for _ in range(max_iter):
            err = area_target - g.area
            if abs(err) < tol:
                break
            area_req += err
            g = pp.make_trapezoid(channel=channel, system=self.system, area=area_req)
        return g

    def prep(self, adc_dwell=None):
        """Prepare spiral trajectory, ADC, and rotated interleaf gradient events."""
        self.adc_dwell = float(self.dt if adc_dwell is None else adc_dwell)
        if self.adc_dwell <= 0:
            raise ValueError("adc_dwell must be positive.")

        k, g_hz, s_hz, t = self._design_vds_single_interleaf(adc_dwell=self.adc_dwell)

        adc_delay = max(float(self.system.adc_dead_time), 0.0)
        # Hold k=0 long enough so the first ADC sample is guaranteed at center.
        t_first_adc = adc_delay + 0.5 * self.adc_dwell
        n_center_hold = int(np.ceil(t_first_adc / self.dt))
        if n_center_hold < 1:
            n_center_hold = 1

        if n_center_hold > 0:
            k = np.concatenate((np.zeros(n_center_hold, dtype=complex), k))
            g_hz = np.diff(np.concatenate(([0.0 + 0.0j], k))) / self.dt
            s_hz = np.diff(np.concatenate(([0.0 + 0.0j], g_hz))) / self.dt
            t = np.arange(len(k), dtype=float) * self.dt

        self.k_traj = k
        self.g_traj_hz = g_hz
        self.slew_hz = s_hz
        self.t = t

        acq_grad_dur = float(len(k) * self.dt)
        n_adc_max = int(np.floor((acq_grad_dur - adc_delay) / self.adc_dwell))
        if n_adc_max <= 0:
            raise RuntimeError("ADC window does not fit readout duration. Increase readout duration or reduce adc_dwell.")

        if n_adc_max >= self.adc_divisor:
            self.adc_samples = (n_adc_max // self.adc_divisor) * self.adc_divisor
        else:
            self.adc_samples = n_adc_max

        if self.adc_samples <= 0:
            raise RuntimeError("ADC sample count is non-positive after rounding.")

        self.adc = pp.make_adc(
            num_samples=self.adc_samples,
            dwell=self.adc_dwell,
            delay=adc_delay,
            system=self.system,
        )
        self.adc_time = self.adc.delay + (np.arange(self.adc_samples, dtype=float) + 0.5) * self.adc_dwell
        if self.adc_time[-1] > (acq_grad_dur + 1e-12):
            self.adc_samples -= 1
            if self.adc_samples <= 0:
                raise RuntimeError("ADC timing does not overlap with readout.")
            self.adc = pp.make_adc(
                num_samples=self.adc_samples,
                dwell=self.adc_dwell,
                delay=adc_delay,
                system=self.system,
            )
            self.adc_time = self.adc.delay + (np.arange(self.adc_samples, dtype=float) + 0.5) * self.adc_dwell

        self.time_to_center = float(self.adc_time[0])

        self._interleaf_events = []
        self._interleaf_ktraj = []
        self._interleaf_adc_ktraj = []
        self._interleaf_endk = []
        self._rampdowns = []
        self._rewinders = []

        t_grad = np.arange(1, len(k) + 1, dtype=float) * self.dt
        t_interp = np.concatenate(([0.0], t_grad))
        smax_hz = float(self.system.max_slew)

        for interleaf in range(self.n_interleaves):
            phi = 2.0 * np.pi * float(interleaf) / float(self.n_interleaves)
            rot = np.exp(1j * phi)

            g_rot = self.g_traj_hz * rot
            k_rot_nominal = k * rot

            # Use raster-point node amplitudes that preserve slew checks from
            # interval-centered gradients and define extended trapezoids exactly.
            g_nodes_base = np.concatenate(([0.0 + 0.0j], g_rot))

            # Slew-limited in-event ramp to zero so the block ends cleanly.
            g_last = g_nodes_base[-1]
            n_ramp_x = int(np.ceil(abs(np.real(g_last)) / (smax_hz * self.dt))) if abs(np.real(g_last)) > 0 else 0
            n_ramp_y = int(np.ceil(abs(np.imag(g_last)) / (smax_hz * self.dt))) if abs(np.imag(g_last)) > 0 else 0
            n_ramp = max(n_ramp_x, n_ramp_y)

            if n_ramp > 0:
                g_tail = np.linspace(g_last, 0.0 + 0.0j, n_ramp + 1, dtype=complex)[1:]
                g_nodes = np.concatenate((g_nodes_base, g_tail))
            else:
                g_nodes = g_nodes_base
                if abs(g_nodes[-1]) > 1e-12:
                    g_nodes = np.concatenate((g_nodes, np.array([0.0 + 0.0j], dtype=complex)))

            # Guard against tiny floating-point overshoots at the hard slew limit
            # when converting to per-axis extended trapezoids.
            base_len = len(g_nodes_base)
            if len(g_nodes) > 1:
                slew_x = np.diff(np.real(g_nodes)) / self.dt
                slew_y = np.diff(np.imag(g_nodes)) / self.dt
                slew_peak = max(float(np.max(np.abs(slew_x))), float(np.max(np.abs(slew_y))))
            else:
                slew_peak = 0.0

            if slew_peak > smax_hz:
                # Auto-correct only small exceedances; larger violations indicate
                # a real design issue and should fail loudly.
                if slew_peak <= 1.02 * smax_hz:
                    scale = (smax_hz * (1.0 - 1e-6)) / slew_peak
                    g_nodes = g_nodes * scale
                    g_nodes_base = g_nodes[:base_len]
                else:
                    raise RuntimeError(
                        f"Spiral slew exceeds max_slew by {(slew_peak / smax_hz - 1.0) * 100.0:.2f}% before event creation."
                    )

            t_nodes = np.arange(len(g_nodes), dtype=float) * self.dt

            # k-space for acquisition portion (before ramp tail).
            k_nodes_base = np.zeros(len(g_nodes_base), dtype=complex)
            k_nodes_base[1:] = np.cumsum(0.5 * (g_nodes_base[:-1] + g_nodes_base[1:]) * self.dt)
            k_rot = k_nodes_base[1:]

            # k-space at raster nodes for the full played event.
            k_nodes_full = np.zeros(len(g_nodes), dtype=complex)
            k_nodes_full[1:] = np.cumsum(0.5 * (g_nodes[:-1] + g_nodes[1:]) * self.dt)

            gx = pp.make_extended_trapezoid(
                channel="x",
                amplitudes=np.real(g_nodes),
                times=t_nodes,
                system=self.system,
            )
            gy = pp.make_extended_trapezoid(
                channel="y",
                amplitudes=np.imag(g_nodes),
                times=t_nodes,
                system=self.system,
            )

            # Exact k(t) sampling for ADC times under piecewise-linear g(t).
            seg_idx = np.floor(self.adc_time / self.dt).astype(int)
            seg_idx = np.clip(seg_idx, 0, len(g_nodes) - 2)
            t0 = seg_idx.astype(float) * self.dt
            dtau = self.adc_time - t0
            g0 = g_nodes[seg_idx]
            g1 = g_nodes[seg_idx + 1]
            s0 = (g1 - g0) / self.dt
            adc_ktraj = k_nodes_full[seg_idx] + g0 * dtau + 0.5 * s0 * (dtau ** 2)

            end_k = k_nodes_full[-1]
            # Rewinder is created dynamically in add_to_seq() to ensure
            # exact cancellation of the actual played gradient area (which
            # may differ slightly after pypulseq's block discretization).

            self._interleaf_events.append((gx, gy))
            self._interleaf_ktraj.append(k_rot)
            self._interleaf_adc_ktraj.append(adc_ktraj)
            self._interleaf_endk.append(end_k)
            self._rampdowns.append((None, None))
            self._rewinders.append(None)  # Will be populated in add_to_seq()

        block_main_dur = pp.calc_duration(*self._interleaf_events[0], self.adc)
        ro_dur = acq_grad_dur
        rd_dur = max(0.0, block_main_dur - acq_grad_dur)

        # Estimate rewinder duration from a temporary rewinder for the first interleaf.
        gx0, gy0 = self._interleaf_events[0]
        rew_dur = 0.0
        if abs(gx0.area) > 1e-9 or abs(gy0.area) > 1e-9:
            gx_tmp = self._make_trapezoid_exact_area(channel="x", area_target=-gx0.area)
            gy_tmp = self._make_trapezoid_exact_area(channel="y", area_target=-gy0.area)
            gx_tmp, gy_tmp = pp.align(right=gx_tmp, left=gy_tmp)
            rew_dur = pp.calc_duration(gx_tmp)

        self.readout_duration = float(ro_dur)
        self.rampdown_duration = float(rd_dur)
        self.rewinder_duration = float(rew_dur)
        self.shot_duration = float(block_main_dur + rew_dur)

        return {
            "adc_samples": self.adc_samples,
            "adc_dwell": self.adc_dwell,
            "adc_bandwidth_hz": 1.0 / self.adc_dwell,
            "center_hold_duration": n_center_hold * self.dt,
            "accel_requested": self.accel,
            "accel_effective": self.accel_effective,
            "readout_duration": self.readout_duration,
            "rampdown_duration": self.rampdown_duration,
            "rewinder_duration": self.rewinder_duration,
            "shot_duration": self.shot_duration,
            "kmax": self.kmax,
        }

    def get_timeToTE(self, save=True):
        timings = {
            "durationToCenter": self.time_to_center,
            "pre_duration": 0.0,
            "timeToTE": self.time_to_center,
            "ghost_nav": 0.0,
            "readout": self.readout_duration,
            "rewinder": self.rewinder_duration,
        }
        if save:
            return timings
        print("Time to TE is {} s".format(timings["timeToTE"]))

    def get_timing(self):
        return {
            "readout_duration": self.readout_duration,
            "rampdown_duration": self.rampdown_duration,
            "rewinder_duration": self.rewinder_duration,
            "shot_duration": self.shot_duration,
            "time_to_k0": self.time_to_center,
        }

    def get_ktraj(self, interleaf=0):
        if len(self._interleaf_ktraj) == 0:
            raise RuntimeError("Call prep() before get_ktraj().")
        idx = int(interleaf) % int(self.n_interleaves)
        return self._interleaf_ktraj[idx]

    def get_adc_ktraj(self, interleaf=0):
        if len(self._interleaf_adc_ktraj) == 0:
            raise RuntimeError("Call prep() before get_adc_ktraj().")
        idx = int(interleaf) % int(self.n_interleaves)
        return self._interleaf_adc_ktraj[idx]

    def add_to_seq(self, pseq0, interleaf=0, label=True, add_rewinder=True):
        if len(self._interleaf_events) == 0:
            raise RuntimeError("Call prep() before add_to_seq().")

        idx = int(interleaf) % int(self.n_interleaves)
        gx, gy = self._interleaf_events[idx]

        if label:
            lseg = pp.make_label(type="SET", label="SEG", value=idx)
            llin = pp.make_label(type="SET", label="LIN", value=idx)
            lrev = pp.make_label(type="SET", label="REV", value=0)
            pseq0.seq.add_block(gx, gy, self.adc, lseg, llin, lrev)
        else:
            pseq0.seq.add_block(gx, gy, self.adc)

        gx_rd, gy_rd = self._rampdowns[idx]
        if gx_rd is not None and gy_rd is not None:
            pseq0.seq.add_block(gx_rd, gy_rd)

        if add_rewinder:
            # Retrieve the ACTUAL gradient events from the sequence block to
            # get their true areas after pypulseq's block discretization.
            # This ensures the rewinder exactly cancels what was actually played.
            last_block_idx = len(pseq0.seq.block_events)
            played_block = pseq0.seq.get_block(last_block_idx)
            played_gx = getattr(played_block, 'gx', None)
            played_gy = getattr(played_block, 'gy', None)

            if played_gx is not None and played_gy is not None:
                ax = -played_gx.area
                ay = -played_gy.area
                gx_rw, gy_rw = self._make_rewinder_pair(ax, ay)
                if gx_rw is not None and gy_rw is not None:
                    pseq0.seq.add_block(gx_rw, gy_rw)
