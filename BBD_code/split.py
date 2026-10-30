import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from collections import defaultdict

def split(split_time, predict_result, dataset_name):
    data = np.array([row[2] for row in predict_result])
    hist, bin_edges = np.histogram(data, bins=100)

    if (dataset_name == 'cifar10'):
        valid_bins = hist > len(predict_result) * 0.001


    bin_indices = np.digitize(data, bin_edges) - 1
    bin_indices[bin_indices >= len(valid_bins)] = len(valid_bins) - 1
    filtered_data = data[valid_bins[bin_indices]]

    kde = gaussian_kde(filtered_data)
    x = np.linspace(filtered_data.min(), filtered_data.max(), 1000)
    pdf = kde(x)
    peaks_max, _ = find_peaks(pdf)
    peaks_min, _ = find_peaks(-pdf)

    if 'CTM' in split_time:
        clean_pool_image_paths = [row[3] for row in predict_result if row[2] < x[peaks_min[0]]]
        poison_pool_image_paths = [row[3] for row in predict_result if row[2] >= x[peaks_min[0]]]
    elif 'PTM' in split_time :
        clean_pool_image_paths = [row[3] for row in predict_result if row[2] > x[peaks_min[0]]]
        poison_pool_image_paths = [row[3] for row in predict_result if row[2] <= x[peaks_min[0]]]
    elif 'no_defense' in split_time:
        class_dict = defaultdict(list)
        for row in predict_result:
            class_dict[row[1]].append(row)
        clean_pool_image_paths = []
        for label, rows in class_dict.items():
            sorted_rows = sorted(rows, key=lambda x: x[2])
            num_samples = len(sorted_rows)
            middle_start = (num_samples - 1000) * 10 // 10
            middle_end = middle_start + 1000
            selected_samples = sorted_rows[middle_start:middle_end]
            clean_pool_image_paths.extend([row[3] for row in selected_samples])
        clean_pool_set = set(clean_pool_image_paths)
        poison_pool_image_paths = [row[3] for row in predict_result if row[3] not in clean_pool_set]
    else :
        clean_pool_image_paths = [row[3] for row in predict_result if row[2] < x[peaks_min[0]]]
        poison_pool_image_paths = [row[3] for row in predict_result if row[2] > x[peaks_min[0]]]
    return clean_pool_image_paths, poison_pool_image_paths
