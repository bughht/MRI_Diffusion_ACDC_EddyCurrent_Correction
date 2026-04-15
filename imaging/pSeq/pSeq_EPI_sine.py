import numpy as np
import pypulseq as pp
from .pSeq_EPI_base import pSeq_EPI_base

class pSeq_EPI_sine(pSeq_EPI_base):
    def __init__(self, pseq=None, 
                fov = 192e-3,
                nx = 140,
                ny = 140,
                readoutTime = 5e-4,
                RSegment = 1,
                partFourierFactor = 0.75,
                R=2,
                Nmulti  = 1,
                ghost_lines = 3,
                *args, **kwargs,
                
                 ):
        
        super().__init__(pseq, *args, **kwargs)

        self.fov = fov
        self.Nx = nx
        self.Ny = ny
        self.readoutTime = readoutTime
        self.RSegment = RSegment
        self.partFourierFactor = partFourierFactor
        self.R = R
        self.Nmulti = Nmulti
        self.ghost_lines = ghost_lines
        self.ghost_flag = False

        if pseq:
            self.epi_system = pseq.epi_system
        else: 
            self.epi_system = self.system

    def prep(self,ro_os=1,pe_enable=1, ref_polarity=False):
        self.readoutBW = 1/self.readoutTime

        deltak = 1/self.fov # 1/m
        deltaky = self.RSegment * self.R * deltak
        kWidth = self.Nx * deltak

        blip_dur = np.ceil(2 * np.sqrt(deltaky / self.epi_system.max_slew) / 10e-6 / 2) * 10e-6 * 2
        gy = pp.make_trapezoid(channel='y', system=self.epi_system, area=-deltaky, duration=blip_dur)

        # --- NEW SINE WAVEFORM LOGIC ---
        T = self.readoutTime + blip_dur # s
        T_ro = self.readoutTime # s
        A = (kWidth * np.pi) / (2 * T * np.sin(np.pi * T_ro / (2 * T)))
        # kWidth*np.pi / (T_ro * 2 * np.cos(t0))

        print('Grad amplitude:', A)
        
        t_arr = np.linspace(0, T, int(np.round(T / self.epi_system.grad_raster_time)+1))
        waveform = A* np.sin(np.pi * t_arr / T)
        
        gx = pp.make_extended_trapezoid(channel='x', system=self.epi_system, amplitudes=waveform, times=t_arr)
        # Give it properties used by later parts of base script
        gx.amplitude = A
        gx.area = np.sum(waveform) * self.epi_system.grad_raster_time
        print('gx area:', gx.area)
        print('kWidth:' , kWidth)
        
        adc_dwell_nyquist = deltak / A / ro_os
        adc_dwell = np.floor(adc_dwell_nyquist * 1e7) * 1e-7

        adc_samples = np.floor(self.readoutTime / adc_dwell / 4) * 4
        adc = pp.make_adc(int(adc_samples), dwell=adc_dwell, delay=blip_dur / 2)
        self.adc_samples = adc_samples

        time_to_center = adc.dwell * ((adc_samples - 1) / 2 + 0.5) 
        # Center of sine gradient is at T / 2
        adc.delay = np.round((T / 2 - time_to_center) * 1e6) * 1e-6

        gy_parts = pp.split_gradient_at(gy, blip_dur / 2, self.epi_system)
        gy_blipup, gy_blipdown, _ = pp.align(right=gy_parts[0], left=[gy_parts[1], gx])
        gy_blipdownup = pp.add_gradients([gy_blipdown, gy_blipup], system=self.epi_system)

        gy_blipup.waveform *= pe_enable
        gy_blipdown.waveform *= pe_enable
        gy_blipdownup.waveform *= pe_enable

        Ny_pre = round((self.partFourierFactor - 1 / 2) * self.Ny - 1)
        Ny_pre = round(Ny_pre / self.RSegment / self.R)
        Ny_post = round(self.Ny / 2 + 1)
        Ny_post = round(Ny_post / self.RSegment / self.R)
        Ny_meas = Ny_pre + Ny_post

        gx_pre = pp.make_trapezoid(channel='x', system=self.epi_system, area=-gx.area / 2)
        gy_pre = pp.make_trapezoid(channel='y', system=self.epi_system, area=(Ny_pre * deltaky - (self.Nmulti - 1) * deltak * self.R))

        gx_pre, gy_pre = pp.align(right=gx_pre, left=gy_pre)

        gy_pre = pp.make_trapezoid(channel='y', system=self.epi_system, area=gy_pre.area, duration=pp.calc_duration(gx_pre, gy_pre))
        gy_pre.amplitude *= pe_enable

        gx_pre_post = pp.make_trapezoid(channel='x', system=self.epi_system, area=-gx.area / 2)
        gy_pre_post = pp.make_trapezoid(channel='y', system=self.epi_system, area=(Ny_post - 1) * deltaky)

        gx_pre_post, gy_pre_post = pp.align(right=gx_pre_post, left=gy_pre_post)

        gy_pre_post = pp.make_trapezoid(channel='y', system=self.epi_system, area=gy_pre_post.area, duration=pp.calc_duration(gx_pre_post, gy_pre_post))
        gy_pre_post.amplitude *= pe_enable

        if Ny_meas % 2 == 0:
            gx_pre_post.amplitude = -gx_pre_post.amplitude

        if ref_polarity:
            gy_pre = pp.scale_grad(gy_pre, -1)
            gy_blipdownup = pp.scale_grad(gy_blipdownup, -1)
            gy_blipdown = pp.scale_grad(gy_blipdown, -1)
            gy_blipup = pp.scale_grad(gy_blipup, -1)
            gy_pre_post = pp.scale_grad(gy_pre_post, -1)

        if pe_enable:
            if hasattr(gy_blipup, 'last'): gy_blipup.last = 0
            if hasattr(gy_blipdown, 'first'): gy_blipdown.first = 0
            if hasattr(gy_blipdownup, 'last'): gy_blipdownup.last = 0
            if hasattr(gy_blipdownup, 'first'): gy_blipdownup.first = 0

        self.ESP = 1e3 * pp.calc_duration(gx)
        self.Ny_pre = Ny_pre
        self.gxPre = gx_pre
        self.gyPre = gy_pre
        self.gx = gx

        self.gx_pre_post = gx_pre_post
        self.gy_pre_post = gy_pre_post

        self.gy_blipup = gy_blipup
        self.gy_blipdown = gy_blipdown
        self.gy_blipdownup = gy_blipdownup
        self.Ny_meas = Ny_meas
        self.adc = adc
        self.adc_samples = adc_samples

        self.ROpolarity = np.sign(self.gx.amplitude)
        self.blip_dur = blip_dur

    def get_echoTime_shift(self):
        # Calculate actual echo spacing for the sine arbitrary gradient
        actual_esp = pp.calc_duration(self.gx)
        TEShift = actual_esp / self.RSegment
        TEShift = round(TEShift, 5)
        TEShift_before_echo = (self.Nmulti - 1) * TEShift

        if TEShift_before_echo == 0:
            TEShift_before_echo = 0.00001  # Apply the minimum duration for the no delay case

        TEShift_after_echo = (self.RSegment - (self.Nmulti - 1)) * TEShift
        dETS_before = pp.make_delay(TEShift_before_echo)
        dETS_after = pp.make_delay(TEShift_after_echo)

        return dETS_before, dETS_after

    def add_ghost_nav(self, pseq0):
        center_line = np.floor(self.Ny / 2) 
        if self.partFourierFactor < 1:
            center_line -= (1 - self.partFourierFactor) * self.Ny

        pseq0.seq.add_block(self.gxPre, pp.make_label(type='SET', label='NAV', value=1),
                            pp.make_label(type='SET', label='LIN', value=center_line))
        
        gx_fwd = self.gx
        gx_rev = pp.scale_grad(self.gx, -1)
        gx_curr = gx_fwd
        
        self.nav_sign_track = []
        for i in range(1, self.ghost_lines + 1):
            resultREV = int(np.sign(gx_curr.amplitude) == self.ROpolarity)
            resultSEG = int(np.sign(gx_curr.amplitude) == self.ROpolarity)
            resultAVG = int(i == self.ghost_lines)

            self.nav_sign_track.append(resultREV)

            pseq0.seq.add_block(pp.make_label(type='SET', label='REV', value=resultREV),
                                pp.make_label(type='SET', label='SEG', value=resultSEG), 
                                pp.make_label(type='SET', label='AVG', value=resultAVG)) 

            pseq0.seq.add_block(gx_curr, self.adc)
            gx_curr = gx_fwd if gx_curr == gx_rev else gx_rev
            
        pseq0.seq.add_block(self.gxPre, pp.make_label(type='SET', label='NAV', value=0))
        
        self.gx = gx_curr

    def add_calib(self, pseq0, ghost_scan=False, EchoTimeShift=0):
        dETS_before, dETS_after = self.get_echoTime_shift()

        if EchoTimeShift:
            pseq0.seq.add_block(dETS_before)

        start_sign_y = np.sign(self.calib_gyPre.amplitude)
        start_sign_x = np.sign(self.calib_gxPre.amplitude)

        pseq0.seq.add_block(self.calib_gyPre, self.calib_gxPre)

        gx_fwd = self.gx
        gx_rev = pp.scale_grad(self.gx, -1)
        gx_curr = gx_fwd

        for i in range(1, int(self.Ny_meas + 1)):
            if i == 1:
                pseq0.seq.add_block(gx_curr, self.adc)
            elif i == self.Ny_meas:
                pseq0.seq.add_block(self.calib_gy_blipdown, gx_curr, self.adc)
            else:
                pseq0.seq.add_block(self.calib_gy_blipdownup, gx_curr, self.adc)

            gx_curr = gx_fwd if gx_curr == gx_rev else gx_rev

        if EchoTimeShift:
            pseq0.seq.add_block(dETS_after)

        if start_sign_y != np.sign(self.calib_gyPre.amplitude):
            self.calib_gyPre = pp.scale_grad(self.calib_gyPre, -1)

        if start_sign_x != np.sign(self.calib_gxPre.amplitude):
            self.calib_gxPre = pp.scale_grad(self.calib_gxPre, -1)

        self.gx = gx_curr

    def add_epi_v1(self, pseq0, ghost_scan=False, EchoTimeShift=0):   
        dETS_before, dETS_after = self.get_echoTime_shift()

        if EchoTimeShift:
            pseq0.seq.add_block(dETS_before)

        # Check trig
        if hasattr(self, 'trig'):
            pseq0.seq.add_block(self.gyPre, self.gxPre, self.trig)
        else:
            pseq0.seq.add_block(self.gxPre, self.gyPre)

        gx_fwd = self.gx
        gx_rev = pp.scale_grad(self.gx, -1)
        gx_curr = gx_fwd

        for i in range(1, int(self.Ny_meas + 1)):
            lrev = pp.make_label(type ='SET', label = 'REV', value = int(np.sign(self.gx.amplitude) != self.ROpolarity ))
            llin = pp.make_label(type = 'SET', label = 'LIN', value = i-1)
            if i ==1:
                pseq0.seq.add_block(gx_curr,self.gy_blipup,self.adc, lrev, llin)
            elif i == self.Ny_meas:
                pseq0.seq.add_block(gx_curr,self.gy_blipdown,self.adc, lrev, llin)
            else:
                pseq0.seq.add_block(gx_curr,self.gy_blipdownup,self.adc, lrev, llin)
            
            gx_curr = gx_fwd if np.equal(gx_curr.waveform, gx_rev.waveform).all() else gx_rev

        if EchoTimeShift:
            pseq0.seq.add_block(dETS_after)
            
        self.gx = gx_curr
