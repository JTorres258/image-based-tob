import os
import re
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.signal.windows import boxcar

import matplotlib.pyplot as plt

def load_raw(filename: str, 
             reshape_size: tuple,
             dtype: type) -> np.ndarray:
    """
    Read data contained in RAW file.

    Parameters
    ----------
    filename : str
        Path of RAW file.
    reshape_size : tuple
        Reshape the RAW data.
    dtype : type
        Numerical type of the original RAW data.

    Returns
    -------
    raw_data : np.array
        Array with RAW data .

    """
    
    # Open
    with open(filename, "rb") as f:
        
        raw_data = np.fromfile(f, dtype)
        
        # MATLAB uses Fortran order to reshape by default
        raw_data = raw_data.reshape(reshape_size, order="F").transpose()
    
    return raw_data
    
    
def decode_image(x):
    """ Convert binary string paths to Python strings """
    
    input_path = x.numpy().decode('utf-8')  # Assuming file_paths are in utf-8 encoding
    np_img = load_raw(input_path, (336, 252), dtype = np.int16)
    
    return np_img
    


def get_GMM_params(path_video,
                   n_components):
    list_frames = sorted(os.listdir(path_video))

    list_frame_ms = [int(x.split('ms')[0].split('_')[1]) \
          for x in list_frames]
                       
    # Linear time-spaced index
    idx = []
    interval_ms = 30*1000
    current_ms = list_frame_ms[0]
    for i, ms in enumerate(list_frame_ms):
        if ms - current_ms >= interval_ms:
            idx.append(i)
            current_ms = ms

    data = []
    for i in idx:
        input_path = os.path.join(path_video, list_frames[i])
        raw_data = load_raw(input_path, (336, 252), dtype = np.int16)
        converted_data = raw_data.astype('float32')*0.04 - 273.15
        data.append(converted_data) 
    
    # Define model
    gmm = GaussianMixture(n_components=n_components, max_iter=1000, random_state=10, covariance_type='full')

    # Fit model
    X = np.stack(data).flatten()[:, np.newaxis]

    gmm.fit(X)
    
    # Get params
    means = gmm.means_[:, 0]
    covs  = gmm.covariances_[:, 0, 0]
    weights = gmm.weights_
    data_info = (np.mean(X), np.var(X), np.min(X), np.max(X))
    
    return means, covs, weights, data_info
    
    
def fix_thermal_video(old_frames, old_timestamps, fix_timestamps=False):
    
    # Add extra frames when autocalibration
    diff_time = [0] + [old_timestamps[i+1] - old_timestamps[i] - 120 \
                for i in range(len(old_timestamps)-1)]

    new_frames = []
    for i, diff in enumerate(diff_time):
        if abs(diff) < 120:
            new_frames.append(old_frames[i])
        else:
            reps = int(np.ceil(diff/120)) + 1
            rep_list = [old_frames[i-1]]*reps
            
            if fix_timestamps:
                # Extract parts from the original list -> 00001_0ms.raw
                f_num = [r.split('_')[0] for r in rep_list]
                f_ms = [int(r.split('_')[1].split('ms')[0]) for r in rep_list]

                # Create a new list with cumulative increase in f_ms by 120ms
                c = 120
                new_list = [f'{x}_{y + c*(i+1)}ms.raw' for i, (x, y) in 
                    enumerate(zip(f_num, f_ms))]
                
                new_frames.extend(new_list)
            else:
                new_frames.extend(rep_list)
    
    if fix_timestamps:
        new_timestamps = [int(x.split('ms')[0].split('_')[1]) for x in new_frames]
    else:
        new_timestamps = old_timestamps.copy()
                        
    return new_frames, new_timestamps


def create_timeline(output_path, anns_path, save_path):
    
    # Read anns for ToB
    df_anns = pd.read_csv(anns_path, sep='\t', header=None).dropna(axis=1)
    df_anns.columns=['Class','Start','Stop','Duration']
    
    ToB = df_anns[df_anns['Class']=='Time of birth']['Start'].values[0]
    ToB_s = round(ToB / 1000, 1)
    
    # Read output
    with open(output_path, 'r') as f:
        preds = f.read().splitlines()
    
    duration = len(preds)
        
    # Split in label and logits
    pred_label = []
    pred_score = []
    for p in preds: # pred_key
        _p = p.split('\t')
        pred_label.append(_p[0])
        pred_score.append(np.array(eval(re.sub(r'\s+', ',', _p[1]))))
        
    timeline_pred_score = np.vstack(pred_score).T[:,:duration]
    
    # Possible ToB
    filt = boxcar(25)
    timeline_filt_score = np.convolve(timeline_pred_score[1], filt/sum(filt), mode='same')
        
    # Define lines for plotting
    x_axis = np.linspace(0, (duration-1)/8.33, duration)
    
    lines = np.vstack((x_axis, timeline_pred_score[1])).T
    lines_filt = np.vstack((x_axis, timeline_filt_score)).T
    
    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(8,6), dpi=300) 

    # Raw scores
    ax[0].plot(x_axis, timeline_pred_score[1], color='b', linewidth=2)
    ax[0].set_xlim((0, 52))
    ax[0].set_ylim((-0.1, 1.1))
    ax[0].set_yticks([0, 0.5, 1.0])
    ax[0].grid(which='major', alpha=1, axis='y')
    ax[0].set_title('Raw scores', fontsize=10)


    # Filtered scores
    binary_tob = (np.asarray(timeline_filt_score) >= 0.9).astype(int)
    idx_frame = np.where(binary_tob == 1)[0][0]
    tob_found = round(idx_frame/8.33, 1)

    ax[1].plot(x_axis, timeline_filt_score, color='b', linewidth=2)
    ax[1].axhline(y=.9, color='cyan', linewidth=2, linestyle='--')
    ax[1].axvline(x=ToB_s, color='r', linewidth=3)
    ax[1].axvline(x=tob_found, color='lime', linewidth=3)
    ax[1].set_xlim((0, 52))
    ax[1].set_ylim((-0.1, 1.1))
    ax[1].set_yticks([0, 0.5, 1.0])
    ax[1].grid(which='major', alpha=1, axis='y')
    ax[1].set_title('Filtered scores', fontsize=10)
    ax[1].legend(ax[1].get_lines()[2:], 
              ['ToB annotated (%.1f s)' % ToB_s,
               'ToB estimated (%.1f s)' % tob_found], 
              loc='best', fontsize=10)
    ax[1].set_xlabel('Duration (sec)', fontsize=10)
    
    # Save figure
    plt.suptitle('Timeline of Newborn Detection on Simulated Data', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path)
