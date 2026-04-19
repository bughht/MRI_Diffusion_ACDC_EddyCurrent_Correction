import os
import numpy as np
import pypulseq as pp

from .pSeq_Base import pSeq_Base


class pSeq_FieldCam_Calib(pSeq_Base):
    """Skope field-camera delay calibration sequence.

    Sequence structure:
    1. `calib_rep` repetitions of [ADC + trigger + TR delay]
    2. One prescan delay block
    3. `scan_rep` repetitions of [ADC + trigger + TR delay]
    """

    def __init__(
        self,
        pseq=None,
        tr=500e-3,
        trigger_delay=0e-3,
        prescan_delay=10.0,
        calib_rep=40,
        scan_rep=10,
        adc_dwell=1e-6,
        adc_duration=8e-3,
        trigger_duration=20e-6,
        trigger_channel="ext1",
        *args,
        **kwargs,
    ):
        super().__init__(pseq=pseq, *args, **kwargs)

        self.tr = float(tr)
        self.trigger_delay = float(trigger_delay)
        self.prescan_delay = float(prescan_delay)
        self.calib_rep = int(calib_rep)
        self.scan_rep = int(scan_rep)
        self.adc_dwell = float(adc_dwell)
        self.adc_duration = float(adc_duration)
        self.trigger_duration = float(trigger_duration)
        self.trigger_channel = str(trigger_channel)

        self.adc = None
        self.trig = None
        self.adc_samples = None
        self.adc_actual_duration = None

    @staticmethod
    def _round_up_to_raster(value, raster):
        return float(np.ceil(max(0.0, value) / raster) * raster)

    def prep(self):
        if self.calib_rep < 0 or self.scan_rep < 0:
            raise ValueError("calib_rep and scan_rep must be non-negative.")
        if self.tr <= 0:
            raise ValueError("tr must be positive.")
        if self.adc_dwell <= 0 or self.adc_duration <= 0:
            raise ValueError("adc_dwell and adc_duration must be positive.")
        if self.trigger_duration <= 0:
            raise ValueError("trigger_duration must be positive.")

        grad_raster = self.system.grad_raster_time
        adc_raster = self.system.adc_raster_time

        dwell_q = np.floor(self.adc_dwell / adc_raster) * adc_raster
        if dwell_q <= 0:
            raise RuntimeError("adc_dwell became non-positive after raster quantization.")

        adc_samples = int(np.round(self.adc_duration / dwell_q))
        if adc_samples < 1:
            raise RuntimeError("adc_duration and adc_dwell produce zero ADC samples.")

        delay_q = self._round_up_to_raster(self.trigger_delay, grad_raster)

        self.adc = pp.make_adc(
            num_samples=adc_samples,
            dwell=dwell_q,
            delay=delay_q,
            system=self.system,
        )
        self.trig = pp.make_digital_output_pulse(
            self.trigger_channel,
            duration=self.trigger_duration,
            delay=delay_q,
        )

        self.adc_samples = adc_samples
        self.adc_actual_duration = float(adc_samples * dwell_q)

        print("Prepared FieldCam calibration blocks:")
        print(f"  ADC samples: {self.adc_samples}")
        print(f"  ADC dwell: {self.adc.dwell * 1e6:.3f} us")
        print(f"  ADC duration: {self.adc_actual_duration * 1e3:.3f} ms")
        print(f"  Trigger delay: {delay_q * 1e3:.3f} ms")
        print(f"  Trigger channel: {self.trigger_channel}")

    def add_to_seq(self, pseq0=None, label_start=0):
        if self.adc is None or self.trig is None:
            self.prep()

        target = self if pseq0 is None else pseq0

        tr_delay = pp.make_delay(self._round_up_to_raster(self.tr, self.system.grad_raster_time))

        for i in range(self.calib_rep):
            llin = pp.make_label(type="SET", label="LIN", value=int(label_start + i))
            target.seq.add_block(self.adc, self.trig, llin, tr_delay)

        if self.prescan_delay > 0:
            target.seq.add_block(
                pp.make_delay(self._round_up_to_raster(self.prescan_delay, self.system.grad_raster_time))
            )

        for i in range(self.scan_rep):
            idx = int(label_start + self.calib_rep + i)
            llin = pp.make_label(type="SET", label="LIN", value=idx)
            target.seq.add_block(self.adc, self.trig, llin, tr_delay)

    def make_default_seq(self):
        self.init_seq()
        self.add_to_seq(self)
        return self.seq

    def get_timing(self):
        if self.adc is None:
            self.prep()

        tr_block = self._round_up_to_raster(self.tr, self.system.grad_raster_time)
        sequence_duration = (
            (self.calib_rep + self.scan_rep) * tr_block
            + self._round_up_to_raster(self.prescan_delay, self.system.grad_raster_time)
        )

        return {
            "tr": tr_block,
            "calib_rep": self.calib_rep,
            "scan_rep": self.scan_rep,
            "prescan_delay": self.prescan_delay,
            "adc_samples": self.adc_samples,
            "adc_dwell": self.adc.dwell,
            "adc_duration": self.adc_actual_duration,
            "sequence_duration": sequence_duration,
        }

    def write_seq(self, seq_name=None, out_dir="scan_data/seq"):
        if len(self.seq.block_events) == 0:
            self.make_default_seq()

        if seq_name is None:
            seq_name = (
                f"fieldcam_calib_cal{self.calib_rep:d}_"
                f"scan{self.scan_rep:d}.seq"
            )

        out_path = os.path.join(out_dir, seq_name)
        out_parent = os.path.dirname(out_path)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)

        self.seq.set_definition("Name", "fieldcam_delay_calibration")
        self.seq.set_definition("CalibRep", int(self.calib_rep))
        self.seq.set_definition("ScanRep", int(self.scan_rep))
        self.seq.set_definition("TR", float(self.tr))

        self.seq.write(out_path)
        return out_path

    def check_timing(self, verbose=True):
        ok, error_report = self.seq.check_timing()
        if verbose:
            if ok:
                print("Timing check passed successfully")
            else:
                print("Timing check failed. Error listing follows:")
                for err in error_report:
                    print(err)
        return ok, error_report
