import numpy as np
import numpy.fft as fft

import pandas as pd

from queue import PriorityQueue
from numba import njit


# the sliding windows for a time series and a window size
def sliding_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


#  the sliding dot product between a query subsequence and a time series
def slidingDotProduct(query, ts):
    m = len(query)
    n = len(ts)

    ts_add = 0
    if n % 2 == 1:
        ts = np.insert(ts, 0, 0)
        ts_add = 1

    q_add = 0
    if m % 2 == 1:
        query = np.insert(query, 0, 0)
        q_add = 1

    query = query[::-1]
    query = np.pad(query, (0, n - m + ts_add - q_add), 'constant')
    trim = m - 1 + ts_add
    dot_product = fft.irfft(fft.rfft(ts) * fft.rfft(query))
    return dot_product[trim:]


# the sliding mean and std for a time series and a window size
def sliding_mean_std(TS, m):
    s = np.insert(np.cumsum(TS), 0, 0)
    sSq = np.insert(np.cumsum(TS ** 2), 0, 0)
    segSum = s[m:] - s[:-m]
    segSumSq = sSq[m:] - sSq[:-m]
    movmean = segSum / m
    movstd = np.sqrt(segSumSq / m - (segSum / m) ** 2)
    return [movmean, movstd]


# kNN indices with dot-product / no-loops for a time series, a window size and k neighbours
def compute_distances_iterative(TS, m, k):
    l = len(TS) - m + 1
    knns = np.zeros(shape=(l, k), dtype=np.int64)

    dot_prev = None
    means, stds = sliding_mean_std(TS, m)

    for order in range(0, l):
        # first iteration O(n log n)
        if order == 0:
            dot_first = slidingDotProduct(TS[:m], TS)
            # dot_first = np.dot(X[order,:], X.T)
            dot_rolled = dot_first
        # O(1) further operations
        else:
            dot_rolled = np.roll(dot_prev, 1) + TS[order + m - 1] * TS[m - 1:l + m] - TS[order - 1] * np.roll(TS[:l], 1)
            dot_rolled[0] = dot_first[order]

        x_mean = means[order]
        x_std = stds[order]

        dist = 2 * m * (1 - (dot_rolled - m * means * x_mean) / (m * stds * x_std))

        # self-join: exclusion zone
        trivialMatchRange = (int(max(0, order - np.round(m / 2, 0))), int(min(order + np.round(m / 2 + 1, 0), l)))
        dist[trivialMatchRange[0]:trivialMatchRange[1]] = np.inf

        idx = np.argpartition(dist, k)

        knns[order, :] = idx[:k]
        dot_prev = dot_rolled

    return knns


# kNN indices relabeling at a given split index
@njit(fastmath=True, cache=True)
def _calc_knn_labels(knn_mask, split_idx, window_size):
    k_neighbours, n_timepoints = knn_mask.shape

    # create hypothetical labels
    y_true = np.concatenate((
        np.zeros(split_idx, dtype=np.int64),
        np.ones(n_timepoints - split_idx, dtype=np.int64)
    ))

    knn_mask_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=np.int64)

    # relabel the kNN indices
    for i_neighbor in range(k_neighbours):
        neighbours = knn_mask[i_neighbor]
        knn_mask_labels[i_neighbor] = y_true[neighbours]

    # compute kNN prediction
    ones = np.sum(knn_mask_labels, axis=0)
    zeros = k_neighbours - ones
    y_pred = np.asarray(ones > zeros, dtype=np.int64)

    # apply exclusion zone at split point
    exclusion_zone = np.arange(split_idx - window_size, split_idx)
    y_pred[exclusion_zone] = np.ones(window_size, dtype=np.int64)

    return y_true, y_pred


# roc-auc score calculation
@njit(fastmath=True, cache=True)
def _roc_auc_score(y_score, y_true):
    # make y_true a boolean vector
    y_true = (y_true == 1)

    # sort scores and corresponding truth values (y_true is sorted by design)
    desc_score_indices = np.arange(y_score.shape[0])[::-1]

    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.concatenate((
        distinct_value_indices,
        np.array([y_true.size - 1])
    ))

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.concatenate((np.array([0]), tps))
    fps = np.concatenate((np.array([0]), fps))

    if fps[-1] <= 0 or tps[-1] <= 0:
        return np.nan

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    if fpr.shape[0] < 2:
        return np.nan

    direction = 1
    dx = np.diff(fpr)

    if np.any(dx < 0):
        if np.all(dx <= 0): direction = -1
        else: return np.nan

    area = direction * np.trapz(tpr, fpr)
    return area


# clasp profile calculation for the kNN indices and a score
@njit(fastmath=True)
def _calc_profile(window_size, knn_mask, score, offset):
    n_timepoints = knn_mask.shape[1]
    profile = np.full(shape=n_timepoints, fill_value=np.nan, dtype=np.float64)

    for split_idx in range(offset, n_timepoints - offset):
        y_true, y_pred = _calc_knn_labels(knn_mask, split_idx, window_size)

        try:
            profile[split_idx] = score(y_true, y_pred)
        except:
            # roc_auc_score fails if y_true only has one class, can (and does) happen in principal
            pass

    return profile


# clasp calculation for a time series and a window size
def calc_clasp(time_series, window_size, k_neighbours=3, score=_roc_auc_score, interpolate=True, offset=.05):
    knn_mask = compute_distances_iterative(time_series, window_size, k_neighbours).T

    n_timepoints = knn_mask.shape[1]
    offset = np.int64(n_timepoints * offset)

    profile = _calc_profile(window_size, knn_mask, score, offset)

    if interpolate is True: profile = pd.Series(profile).interpolate(limit_direction='both').to_numpy()
    return profile, knn_mask


# checks if a candidate change point is in close proximity to other change points
def is_trivial_match(candidate, change_points, n_timepoints, exclusion_radius=.05):
    change_points = [0] + change_points + [n_timepoints]
    exclusion_radius = np.int64(n_timepoints * exclusion_radius)

    for change_point in change_points:
        left_begin = max(0, change_point - exclusion_radius)
        right_end = min(n_timepoints, change_point + exclusion_radius)
        # print(exclusion_radius,range(left_begin, right_end))
        if candidate in range(left_begin, right_end): return True

    return False


# change point extraction procedure
def extract_clasp_cps(time_series, window_size, n_change_points=None, offset=.05):
    queue = PriorityQueue()

    # compute global clasp
    profile, knn_mask = calc_clasp(time_series=time_series, window_size=window_size, offset=offset)
    queue.put((-np.max(profile), (np.arange(time_series.shape[0]).tolist(), np.argmax(profile))))

    change_points = []
    scores = []

    for idx in range(n_change_points):
        # should not happen ... safety first
        if queue.empty() is True: break

        # get profile with highest change point score
        priority, (profile_range, change_point) = queue.get()

        change_points.append(change_point)
        scores.append(-priority)

        if idx == n_change_points-1:
            break

        # create left and right local range
        left_range = np.arange(profile_range[0], change_point).tolist()
        right_range = np.arange(change_point, profile_range[-1]).tolist()

        # create and enqueue left local profile
        if len(left_range) > window_size:
            left_profile, _ = calc_clasp(time_series=time_series[left_range], window_size=window_size, offset=offset)
            left_change_point = np.argmax(left_profile)
            left_score = left_profile[left_change_point]

            global_change_point = left_range[0] + left_change_point

            if not is_trivial_match(global_change_point, change_points, time_series.shape[0], exclusion_radius=offset):
                queue.put((-left_score, [left_range, global_change_point]))

        # create and enqueue right local profile
        if len(right_range) > window_size:
            right_profile, _ = calc_clasp(time_series=time_series[right_range], window_size=window_size, offset=offset)
            right_change_point =  np.argmax(right_profile)
            right_score = right_profile[right_change_point]

            global_change_point = right_range[0] + right_change_point

            if not is_trivial_match(global_change_point, change_points, time_series.shape[0], exclusion_radius=offset):
                queue.put((-right_score, [right_range, global_change_point]))

    # print(profile.shape)
    return profile, np.array(change_points), np.array(scores)


def extract_clasp_cps_from_multivariate_ts(time_series, window_size, n_change_points=None, offset=.05):
    queue = PriorityQueue()

    # compute global clasp
    if len(time_series.shape) == 1:
        dim = 1
    else:
        _, dim = time_series.shape

    if dim == 1:
        profile, knn_mask = calc_clasp(time_series=time_series.flatten(), window_size=window_size, offset=offset)
    else:
        profile_list = []
        for i in range(dim):
            profile, knn_mask = calc_clasp(time_series=time_series[:,i].flatten(), window_size=window_size, offset=offset)
            profile_list.append(profile)
        profile = np.mean(profile_list, axis=0).flatten()

    queue.put((-np.max(profile), (np.arange(time_series.shape[0]).tolist(), np.argmax(profile))))

    change_points = []
    scores = []

    for idx in range(n_change_points):
        # should not happen ... safety first
        if queue.empty() is True: break

        # get profile with highest change point score
        priority, (profile_range, change_point) = queue.get()

        change_points.append(change_point)
        scores.append(-priority)

        if idx == n_change_points-1:
            break

        # create left and right local range
        left_range = np.arange(profile_range[0], change_point).tolist()
        right_range = np.arange(change_point, profile_range[-1]).tolist()

        # create and enqueue left local profile
        if len(left_range) > window_size:
            if dim == 1:
                left_profile, _ = calc_clasp(time_series=time_series[left_range].flatten(), window_size=window_size, offset=offset)
            else:
                left_profile_list = []
                for i in range(dim):
                    left_profile, _ = calc_clasp(time_series=time_series[left_range,i].flatten(), window_size=window_size, offset=offset)
                    left_profile_list.append(left_profile)
                left_profile = np.mean(left_profile_list, axis=0)
            left_change_point = np.argmax(left_profile)
            left_score = left_profile[left_change_point]

            global_change_point = left_range[0] + left_change_point

            if not is_trivial_match(global_change_point, change_points, time_series.shape[0], exclusion_radius=offset):
                queue.put((-left_score, [left_range, global_change_point]))

        # create and enqueue right local profile
        if len(right_range) > window_size:
            if dim == 1:
                right_profile, _ = calc_clasp(time_series=time_series[right_range].flatten(), window_size=window_size, offset=offset)
            else:
                right_profile_list = []
                for i in range(dim):
                    right_profile, _ = calc_clasp(time_series=time_series[right_range,i].flatten(), window_size=window_size, offset=offset)
                    right_profile_list.append(right_profile)
                right_profile = np.mean(right_profile_list, axis=0)
            right_change_point =  np.argmax(right_profile)
            right_score = right_profile[right_change_point]

            global_change_point = right_range[0] + right_change_point

            if not is_trivial_match(global_change_point, change_points, time_series.shape[0], exclusion_radius=offset):
                queue.put((-right_score, [right_range, global_change_point]))

    return profile, np.array(change_points), np.array(scores)