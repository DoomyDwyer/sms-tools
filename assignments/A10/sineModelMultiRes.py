import numpy as np
from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft
import sys, os, math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models'))
import dftModel as DFT
import utilFunctions as UF


def sineModelMultiRes(x, fs, w, N, t, B):
    """
    Analysis/synthesis of a sound using the sinusoidal model, without sine tracking
    x: input array sound,
    w: list of analysis windows,
    N: list of sizes of complex spectrum for each window,
    t: threshold in negative dB,
    B: list of frequency bands for each window
    returns y: output array sound
    """

    if len(w) != len(N) != len(B):
        raise ValueError('w, N and B must all have the same size')

    # Lower band limit (at position -1)
    B.append(0)

    # Variables for analysis
    hM1 = []
    hM2 = []
    for i in range(len(w)):
        hM1.append(int(math.floor((w[i].size + 1) / 2)))             # half analysis window size by rounding
        hM2.append(int(math.floor(w[i].size / 2)))                   # half analysis window size by floor
        w[i] = w[i] / sum(w[i])                                      # normalize analysis window

    # Variables for synthesis
    Ns = 512                                                         # FFT size for synthesis (even)
    H = Ns // 4                                                      # Hop size used for analysis and synthesis
    hNs = Ns // 2                                                    # half of synthesis FFT size
    pin = max(hNs, max(hM1))                                         # init sound pointer in middle of anal window
    pend = x.size - max(hNs, max(hM1))                               # last sample to start a frame
    yw = np.zeros(Ns)                                                # initialize output sound frame
    y = np.zeros(x.size)                                             # initialize output array
    sw = np.zeros(Ns)                                                # initialize synthesis window
    ow = triang(2 * H)                                               # triangular window
    sw[hNs - H:hNs + H] = ow                                         # add triangular window
    bh = blackmanharris(Ns)                                          # blackmanharris window
    bh = bh / sum(bh)                                                # normalized blackmanharris window
    sw[hNs - H:hNs + H] = sw[hNs - H:hNs + H] / bh[hNs - H:hNs + H]  # normalized synthesis window

    # -----analysis-----
    while pin < pend:                                                # while input sound pointer is within sound
        # Initialise the sine peaks frequencies, magnitudes and phases used for the synthesis
        xpfreq = np.empty(0)
        xpmag = np.empty(0)
        xpphase = np.empty(0)

        # Perform a DFT and Peak Detection for each set of parameters passed
        for i in range(len(w)):
            x1 = x[pin - hM1[i]:pin + hM2[i]]                            # select frame
            mX, pX = DFT.dftAnal(x1, w[i], N[i])                         # compute dft
            ploc = UF.peakDetection(mX, t)                               # detect locations of peaks
            iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)          # refine peak values by interpolation
            ipfreq = fs * iploc / float(N[i])                            # convert peak locations to Hertz

            # Extract the frequencies for the band currently under analysis
            band_elements = (ipfreq >= B[i-1]) & (ipfreq < B[i])
            zeros = np.zeros(ipfreq.size)

            pfreq = np.where(band_elements, ipfreq, zeros)
            pmag = np.where(band_elements, ipmag, zeros)
            pphase = np.where(band_elements, ipphase, zeros)

            # Append non-zero frequencies to the
            # sine peaks frequencies, magnitudes and phases used for the synthesis
            non_zeros = np.flatnonzero(pfreq)
            xpfreq = np.append(xpfreq, np.take(pfreq, non_zeros))
            xpmag = np.append(xpmag, np.take(pmag, non_zeros))
            xpphase = np.append(xpphase, np.take(pphase, non_zeros))

        # -----synthesis-----
        Y = UF.genSpecSines(xpfreq, xpmag, xpphase, Ns, fs)            # generate sines in the spectrum
        # Alternative if utilFunctions_C hasn't been successfully compiled
        #Y = UF.genSpecSines_p(xpfreq, xpmag, xpphase, Ns, fs)          # generate sines in the spectrum
        fftbuffer = np.real(ifft(Y))                                   # compute inverse FFT
        yw[:hNs - 1] = fftbuffer[hNs + 1:]                             # undo zero-phase window
        yw[hNs - 1:] = fftbuffer[:hNs + 1]
        y[pin - hNs:pin + hNs] += sw * yw                              # overlap-add and apply a synthesis window

        pin += H                                                       # advance sound pointer
    return y
