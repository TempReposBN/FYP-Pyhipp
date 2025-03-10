import os
from pathlib import Path
from scipy.io import loadmat
import pandas as pd
import mne
import h5py
import numpy as np
import pickle as pkl

event_dict = {
    11: "Cue Onset - 1",
    12: "Cue Onset - 2",
    13: "Cue Onset - 3",
    14: "Cue Onset - 4",
    15: "Cue Onset - 5",
    16: "Cue Onset - 6",
    21: "Cue Offset - 1",
    22: "Cue Offset - 2",
    23: "Cue Offset - 3",
    24: "Cue Offset - 4",
    25: "Cue Offset - 5",
    26: "Cue Offset - 6",
    31: "Reward - 1",
    32: "Reward - 2",
    33: "Reward - 3",
    34: "Reward - 4",
    35: "Reward - 5",
    36: "Reward - 6",
    41: "Error/Timeout - 1",
    42: "Error/Timeout - 2",
    43: "Error/Timeout - 3",
    44: "Error/Timeout - 4",
    45: "Error/Timeout - 5",
    46: "Error/Timeout - 6"
}

def get_data(day):
    current_dir = Path(__file__).parent
    script_path = current_dir / "get_channels.sh"

    if not script_path.exists():
        print(f"Error: Shell script '{script_path}' not found!")
        return None, None

    os.system(f"sh {script_path} {day}")

    ch_list = []
    
    with open("channel_list.txt", "r") as file:
        for line in file.readlines():
            ch_list.append(line.strip())
    os.system("rm channel_list.txt")

    ch_names = [ch[-3:] for ch in ch_list]

    # Extract LFP Data
    lfp = []
    for ch in ch_list:
        ch_data = loadmat(f"/Volumes/Hippocampus/Data/picasso-misc/{day}/session01/{ch}/rpllfp.mat")
        lfp.append(ch_data.get('df')[0, 0][0][0, 0][0].flatten())

    lfp_df = pd.DataFrame({'channel': ch_names, 'lfp_data': lfp})

    # Get Session details
    rp_file = h5py.File(f"/Volumes/Hippocampus/Data/picasso-misc/{day}/session01/rplparallel.mat", 'r')
    rp = rp_file.get('rp').get('data')
    session_start_time = rp.get('session_start_sec')[0,0]
    markers = rp.get('markers')
    timeStamps = rp.get('timeStamps')
    sampling_frequency = 1000

    # Create MNE object to plot scrollable
    info = mne.create_info(ch_names=[ch.replace('annel','') for ch in ch_list], sfreq=sampling_frequency)
    lfp_mne = mne.io.RawArray(lfp,info)
    annotations_mne = mne.Annotations(onset=np.array(timeStamps).flatten(), duration=[0] * len(np.array(timeStamps).flatten()), description=[event_dict[i] for i in np.array(markers).flatten()])
    lfp_mne.set_annotations(annotations_mne)

    return lfp_df, ch_list, lfp_mne, session_start_time, markers, timeStamps, sampling_frequency


def load_pkl(file_path):
    segments_df = pd.DataFrame()
    try:
        with open(file_path, 'rb') as file:
            segments_df = pkl.load(file)
        print(f"Data successfully loaded from {file_path}")
        print(f"Loaded data has {segments_df.shape[0]} rows and {segments_df.shape[1]} columns.")
        print(segments_df.head())
        return segments_df
    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        return segments_df  # or raise error


def save_pkl(file_path, segments_df):
    """
    Save the current segments DataFrame to a Pickle file.
    """
    if segments_df.empty:
        print("No data available to export.")
        return

    try:
        if not file_path.endswith('.pkl'):
            file_path += '.pkl'

        with open(file_path, 'wb') as file:
            pkl.dump(segments_df, file)
        print(f"Data successfully exported to {file_path}")
    except Exception as e:
        print(f"Failed to export data: {str(e)}")