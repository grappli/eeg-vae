__author__ = 'Steffen'
import matplotlib
matplotlib.use('Agg')

from mne.io import *
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt


nchannels = 19
seg_len = 128

def normalize_segments(segments, seg_len):
    segments_norm = np.subtract(segments, np.repeat(np.min(segments, axis=2)[:, :, np.newaxis], seg_len, axis=2))
    segments_norm = np.divide(segments_norm,
                              np.repeat(np.max(segments_norm, axis=2)[:, :, np.newaxis], seg_len, axis=2))

    return segments_norm

def extract_combined_channels(files):
    recording_idx = 0
    all_segments = np.empty((0,nchannels,128))
    all_segments_norm = np.empty((0,nchannels,128))
    all_segments_overlap_norm = np.empty((0,nchannels,128))
    all_segments_quant = np.empty((0,nchannels,128))
    all_indices = np.empty((0,))
    all_data = np.empty((0,nchannels))
    for path in files:
        print('Processing {}'.format(path))
        data = read_raw_eeglab(path, verbose='CRITICAL')
        valid_ch_idx = np.array([i for i in range(len(data.info['ch_names']))
                        if not data.info['ch_names'][i] in ['EMG', 'LEOG', 'REOG', 'STI 014']])
        read_data = data[valid_ch_idx,:][0]
        seq_len = data.n_times
        n_segments = int(seq_len/seg_len)
        segments = np.reshape(read_data[:,:n_segments * seg_len],
                              (len(read_data), n_segments, seg_len))
        segments = np.moveaxis(segments, 0, 1)

        segments_overlap = np.array([read_data[:,start:start+seg_len]
                                     for start in [int(seg_len/2) * i for i in range(n_segments * 2 - 1)]])
        segments_overlap = np.moveaxis(segments_overlap, 0, 1)
        read_data = np.moveaxis(read_data, 0, 1)

        segments_norm = normalize_segments(segments, seg_len)
        segments_overlap_norm = normalize_segments(segments, seg_len)

        # segments_quant is not supported atm!!
        segments_quant = quantize(segments_norm)
        segment_index = (all_segments.shape[0],)
        indices = np.full((read_data.shape[0],), recording_idx)

        all_data = np.concatenate((all_data, read_data), axis=0)
        all_segments = np.concatenate((all_segments, segments), axis=0)
        all_segments_norm = np.concatenate((all_segments_norm, segments_norm), axis=0)
        all_segments_overlap_norm = np.concatenate((all_segments_overlap_norm, segments_overlap_norm), axis=0)
        all_segments_quant = np.concatenate((all_segments_quant, segments_quant), axis=0)
        all_indices = np.concatenate((all_indices, indices), axis=0)
        recording_idx += 1
    return all_data, all_segments, all_segments_norm, all_segments_quant, all_segments_overlap_norm, all_indices

def to_hdf5(raw_array, one_sec_array, one_sec_array_norm, one_sec_array_quant, one_sec_overlap_norm, all_indices, filename):
    with h5py.File(filename, 'w') as h5f:
        h5f.create_dataset('raw', data=raw_array)
        h5f.create_dataset('one_sec', data=one_sec_array)
        h5f.create_dataset('one_sec_norm', data=one_sec_array_norm)
        h5f.create_dataset('one_sec_overlap_norm', data=one_sec_overlap_norm)
        h5f.create_dataset('one_sec_quant', data=one_sec_array_quant)
        h5f.create_dataset('indices', data=all_indices)

def quantize(x, mu=255, mu_law=False):
    if mu_law:
        x_mu = np.sign(x) * np.log(1.0 + mu * np.abs(x)) / np.log(1.0 + mu)
        return ((x_mu + 1.0)/2.0 * mu).astype('int16')
    else:
        return(((x + 1.0) / 2.0) * mu).astype('int16')

def dequantize(y, mu=255.0, mu_law=False):
    if mu_law:
        scaled = 2.0 * (y / mu) - 1.0
        magnitude = (1.0 / mu) * (np.power(1.0 + mu, np.abs(scaled)) - 1.0)
        return np.sign(scaled) * magnitude
    else:
        return ((y / mu) - 0.5) * 2.0

def get_files(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".mat"):
                yield(os.path.join(root, file))

def get_hist():
    all_data = get_eeg_data()
    print(all_data)
    min = np.min(all_data)
    max = np.max(all_data)
    print(min)
    print(max)
    vals = [quantize(x) for x in all_data.flatten()]
    plt.hist(vals, bins=256)
    plt.title('min: {} max: {}'.format(min, max))
    plt.savefig('hist.png')


import h5py


def get_eeg_data(file='/data/eeg_1s.h5', train_test_val_tuple=(0.9,0.5,0.5)):

    with h5py.File(file, 'r') as h5f:
        one_sec = h5f['one_sec']
        one_sec_raw = h5f['raw']
        one_sec_norm = h5f['one_sec_norm']
        one_sec_overlap_norm = h5f['one_sec_overlap_norm']
        segment_indices = h5f['indices']
        data_len = one_sec.shape[0]
        test_start_idx = int(train_test_val_tuple[0]*data_len)
        val_start_idx = int(sum(train_test_val_tuple[0:2])*data_len)
        
        raw_data_len = one_sec_raw.shape[0]
        raw_test_start = int(train_test_val_tuple[0]*raw_data_len)
        raw_val_start = int(sum(train_test_val_tuple[0:2])*raw_data_len)

        
        return one_sec_raw[:raw_test_start,:], \
               one_sec_norm[:test_start_idx,:,:], \
               one_sec_overlap_norm[:test_start_idx,:,:], \
               segment_indices[:raw_test_start], \
               one_sec_raw[raw_test_start:raw_val_start,:], \
               one_sec_norm[test_start_idx:val_start_idx,:,:], \
               one_sec_overlap_norm[test_start_idx:val_start_idx,:,:], \
               segment_indices[raw_test_start:raw_val_start], \
               one_sec_raw[raw_val_start:,:], \
               one_sec_norm[val_start_idx:,:,:], \
               one_sec_overlap_norm[val_start_idx:,:,:], \
               segment_indices[raw_val_start:]

if __name__ == '__main__':
    # get_hist()
    files = list(get_files('/data/sleep/data.import/'))
    all_data, all_segments, all_segments_norm, all_segments_quant, all_segments_overlap_norm, all_indices = extract_combined_channels(files)
    to_hdf5(all_data, all_segments, all_segments_norm, all_segments_quant, all_segments_overlap_norm, all_indices, '/data/eeg_1s.h5')


