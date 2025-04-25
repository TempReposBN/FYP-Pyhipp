from scipy import signal
import numpy as np
import pandas as pd
from fooof import FOOOF
import fooof
from fooof.utils.data import trim_spectrum

# Helper Functions:
def _new_fit_peaks(self, flat_iter):
    """Iteratively fit peaks to flattened spectrum.

    Parameters
    ----------
    flat_iter : 1d array
        Flattened power spectrum values.

    Returns
    -------
    gaussian_params : 2d array
        Parameters that define the gaussian fit(s).
        Each row is a gaussian, as [mean, height, standard deviation].
    """

    # Initialize matrix of guess parameters for gaussian fitting
    guess = np.empty([0, 3])
    stds = []
    means = []
    peak_gauss_list = []
    peaks, peak_freqs = [], []
    flat_iter_list = [flat_iter,]


    # Find peak: Loop through, finding a candidate peak, and fitting with a guess gaussian
    #   Stopping procedures: limit on # of peaks, or relative or absolute height thresholds
    while len(guess)<self.max_n_peaks:
        std_flat_iter = np.std(flat_iter)
        mean_flat_iter = np.mean(flat_iter)
        # Find candidate peak - the maximum point of the flattened spectrum
        max_ind = np.argmax(flat_iter[1:-1]) + 1
        max_height = flat_iter[max_ind]

        # Stop searching for peaks once height drops below height threshold
        threshold = self.peak_threshold * std_flat_iter
        if max_height <= threshold:
            break

        # print(f'peak_threshold: {threshold}, the peak is {max_height}')

        # Set the guess parameters for gaussian fitting, specifying the mean and height
        guess_freq = self.freqs[max_ind]
        guess_height = max_height

        # Halt fitting process if candidate peak drops below minimum height
        if not guess_height > self.min_peak_height:
            break

        stds.append(std_flat_iter)
        means.append(np.mean(flat_iter))
        peaks.append(guess_height)
        peak_freqs.append(self.freqs[max_ind])
        
        # Data-driven first guess at standard deviation
        #   Find half height index on each side of the center frequency
        half_height = 0.5 * max_height
        le_ind = next((val for val in range(max_ind - 1, 0, -1)
                        if flat_iter[val] <= half_height), None)
        ri_ind = next((val for val in range(max_ind + 1, len(flat_iter), 1)
                        if flat_iter[val] <= half_height), None)

        # Guess bandwidth procedure: estimate the width of the peak
        try:
            # Get an estimated width from the shortest side of the peak
            #   We grab shortest to avoid estimating very large values from overlapping peaks
            # Grab the shortest side, ignoring a side if the half max was not found
            short_side = min([abs(ind - max_ind) \
                for ind in [le_ind, ri_ind] if ind is not None])

            # Use the shortest side to estimate full-width, half max (converted to Hz)
            #   and use this to estimate that guess for gaussian standard deviation
            fwhm = short_side * 2 * self.freq_res
            guess_std = fooof.utils.params.compute_gauss_std(fwhm)

        except ValueError:
            # This procedure can fail (very rarely), if both left & right inds end up as None
            #   In this case, default the guess to the average of the peak width limits
            guess_std = np.mean(self.peak_width_limits)

        # Check that guess value isn't outside preset limits - restrict if so
        #   Note: without this, curve_fitting fails if given guess > or < bounds
        if guess_std < self._gauss_std_limits[0]:
            guess_std = self._gauss_std_limits[0]
        if guess_std > self._gauss_std_limits[1]:
            guess_std = self._gauss_std_limits[1]

        # Collect guess parameters and subtract this guess gaussian from the data
        guess = np.vstack((guess, (guess_freq, guess_height, guess_std)))
        peak_gauss = fooof.core.funcs.gaussian_function(self.freqs, guess_freq, guess_height, guess_std)
        flat_iter = flat_iter - peak_gauss

        peak_gauss_list.append(peak_gauss)
        flat_iter_list.append(flat_iter)

    return stds, means, peak_gauss_list, flat_iter_list, peaks, peak_freqs

def get_power_spec(epoch, sampling_frequency, nperseg=1000, method='fft', freq=None):

    if method == 'welch':
        # Compute power spectrum using Welch's method
        freq, xk = signal.welch(epoch, fs=sampling_frequency, nperseg=nperseg)
    elif method == 'fft':
        fft_result = np.fft.fft(epoch)
        n = len(epoch)  # Number of sample points
        xk = (1/n**2) * np.abs(fft_result) ** 2
        # xk = 1.0 / (sampling_frequency * (np.abs(fft_result)**2).sum()) * np.abs(fft_result)

        xk = xk[:n//2]  # Keep only positive frequencies
        if len(xk) % 2:
            xk[..., 1:] *=2
        else:
            xk[..., 1:-1] *= 2
        freq = np.fft.fftfreq(n, 1/sampling_frequency)[:n // 2]  # Frequency values for positive frequencies
    else:
        raise ValueError("Invalid method. Choose 'welch' or 'fft'.")

    return freq, xk

# Created this because jagged matrix cannot be transposed normally.
def manual_transpose(matrix):
    transposed_matrix = [[None] * len(matrix) for _ in range(len(matrix[0]))]

    # Manually transpose the matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            transposed_matrix[j][i] = matrix[i][j]
    return transposed_matrix

def get_cue_segments(lfp_data, markers, timeStamps, window_size = 1000):
    cue_onsets =[11,12,13,14,15,16]
    end_trial_cues = [31,32,33,34,35,36]
    annotations = []
    timeStamps = np.array(timeStamps).flatten()

    for i, m in enumerate(np.array(markers).flatten()):
        annotations.append([int(m), timeStamps[i]])  # get [[marker, timestamp], ....]

    annotations = sorted(annotations, key=lambda x: x[1])  # sort by ascending timestamps
    
    lfp_segmented = pd.DataFrame(columns=['segment', 'channel', 'start_position', 'cue_onset'])

    for i, [marker, timeStamps] in enumerate(annotations):

        # Check if the marker is an End Trial Marker
        if marker in end_trial_cues:
            if i + 1 < len(annotations):
                next_cue = annotations[i + 1]

                # Check if the next cue is Cue Onset Marker
                if next_cue[0] in cue_onsets:
                    start_idx = int(next_cue[1] * 1000)
                    end_idx = start_idx + window_size

                    # Get a list of Windowed Segments from All Channel for this Cue Onset
                    segments = [
                        ch[start_idx:end_idx]
                        for ch in lfp_data['lfp_data']
                    ]

                    new_rows = [
                        {
                            'segment': seg,
                            'channel': lfp_data['channel'][ch_idx],
                            'start_position': int(marker),
                            'cue_onset': int(next_cue[0])
                        }
                        for ch_idx, seg in enumerate(segments)
                    ]

                    # Append new rows to the DataFrame all at once for efficiency
                    lfp_segmented = pd.concat([lfp_segmented, pd.DataFrame(new_rows)], ignore_index=True)
    
    return lfp_segmented


def get_control_segments(lfp_data, markers, timeStamps, window_size = 1000):
    cue_onsets =[11,12,13,14,15,16]
    end_trial_cues = [31,32,33,34,35,36]
    annotations = []
    timeStamps = np.array(timeStamps).flatten()

    for i, m in enumerate(np.array(markers).flatten()):
        annotations.append([int(m), timeStamps[i]])  # get [[marker, timestamp], ....]

    annotations = sorted(annotations, key=lambda x: x[1])  # sort by ascending timestamps
    
    lfp_segmented = pd.DataFrame(columns=['segment', 'channel', 'start_position', 'cue_onset'])

    for i, [marker, timeStamps] in enumerate(annotations):

        # Check if the marker is an End Trial Marker
        if marker in end_trial_cues:
            if i + 1 < len(annotations):
                next_cue = annotations[i + 1]

                # Check if the next cue is Cue Onset Marker
                if next_cue[0] in cue_onsets:
                    start_idx = int(next_cue[1] * 1000) - window_size
                    end_idx = start_idx + window_size

                    # Get a list of Windowed Segments from All Channel for this Cue Onset
                    segments = [
                        ch[start_idx:end_idx]
                        for ch in lfp_data['lfp_data']
                    ]

                    new_rows = [
                        {
                            'segment': seg,
                            'channel': lfp_data['channel'][ch_idx],
                            'start_position': int(marker),
                            'cue_onset': int(next_cue[0])
                        }
                        for ch_idx, seg in enumerate(segments)
                    ]

                    # Append new rows to the DataFrame all at once for efficiency
                    lfp_segmented = pd.concat([lfp_segmented, pd.DataFrame(new_rows)], ignore_index=True)
    
    return lfp_segmented

def get_navigation_segments(lfp_data, markers, timeStamps, window_size = 7000):
    cue_offsets =[21,22,23,24,25,26]
    end_trial = [31,32,33,34,35,36]
    annotations = []
    timeStamps = np.array(timeStamps).flatten()

    for i, m in enumerate(np.array(markers).flatten()):
        annotations.append([int(m), timeStamps[i]])  # get [[marker, timestamp], ....]

    annotations = sorted(annotations, key=lambda x: x[1])  # sort by ascending timestamps
    lfp_segmented = pd.DataFrame(columns=['segment', 'channel', 'start_position', 'goal_poster'])

    for i, [marker, timeStamps] in enumerate(annotations):

        # Check if the marker is an End Trial Marker
        if marker in cue_offsets:
            if i + 1 < len(annotations):
                next_cue = annotations[i + 1]
                # Check if the next cue is Cue Onset Marker
                if next_cue[0] in end_trial:
                    start_idx = int(next_cue[1] * 1000) - window_size
                    end_idx = start_idx + window_size

                    # Get a list of Windowed Segments from All Channel for this Cue Onset
                    segments = [
                        ch[start_idx:end_idx]
                        for ch in lfp_data['lfp_data']
                    ]

                    new_rows = [
                        {
                            'segment': seg,
                            'channel': lfp_data['channel'][ch_idx],
                            'start_position': annotations[i-1][0],
                            'goal_poster': int(next_cue[0])
                        }
                        for ch_idx, seg in enumerate(segments)
                    ]

                    # Append new rows to the DataFrame all at once for efficiency
                    lfp_segmented = pd.concat([lfp_segmented, pd.DataFrame(new_rows)], ignore_index=True)
    
    return lfp_segmented

# TODO Finish this function to return fm object (maybe not fm object cuz they might lag?)
def get_peak_fits(psd, freq, freq_range, max_n_peaks=3, peak_threshold=2):
    """
    Performs spectral fitting and peak analysis.

    Parameters:
        psd (array-like): Power spectral density values.
        freq (array-like): Frequency values corresponding to the PSD.
        freq_range (list): Range of frequencies to analyze [low, high].

    Returns:
        tuple: A collection of computed parameters:
            - psd_fm: The original power spectrum.
            - ap_fit_og: The aperiodic fit of the original spectrum.
            - flat_spec_og: The original flattened spectrum.
            - p_stds: Standard deviations of the peaks.
            - p_means: Means of the peaks.
            - peak_gauss: Gaussian fits for the peaks.
            - flat_specs: Flattened spectra.
            - peak_list: List of identified peaks.
            - peak_index_list: Indexes of identified peaks in the spectrum.
    """
    fm = FOOOF(max_n_peaks=max_n_peaks,peak_threshold=peak_threshold)

    # Add data and calculate original PSD and aperiodic fits across whole spectrum range
    fm.add_data(freq, psd, freq_range=[1, 150])  # Extended range for initial processing
    psd_fm = fm.power_spectrum
    fm._aperiodic_params = fm._robust_ap_fit(fm.freqs, fm.power_spectrum)
    fm._ap_fit = fooof.sim.gen.gen_aperiodic(fm.freqs, fm._aperiodic_params)
    fm._spec_flat = fm.power_spectrum - fm._ap_fit

    # Store original flattened spectrum and aperiodic fit
    flat_spec_og = fm._spec_flat
    ap_fit_og = fm._ap_fit

    # Trim into theta range we want for find peaks
    freqs, flat_spec = trim_spectrum(fm.freqs, fm._spec_flat, freq_range)

    # Update FOOOF object for the trimmed range
    fm.add_data(freq, psd, freq_range=freq_range)
    fm._spec_flat = flat_spec
    fm.freqs = freqs

    # Perform peak fitting
    p_stds, p_means, peak_gauss, flat_specs, peak_list, peak_index_list = _new_fit_peaks(fm, fm._spec_flat)

    return psd_fm, ap_fit_og, flat_spec_og, p_stds, p_means, peak_gauss, flat_specs, peak_list, peak_index_list