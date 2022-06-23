import os, sys, time
sys.path.append("../")

import numpy as np
np.random.seed(1379)

import pandas as pd

from sklearn.metrics.pairwise import paired_euclidean_distances

from src.clasp import extract_clasp_cps


def load_floss_dataset(n_change_points=2):
    desc_filename = "../datasets/FLOSS/desc.txt"
    desc_file = np.genfromtxt(fname=desc_filename, delimiter=',', filling_values=[None], dtype=None, encoding='utf8')

    df = []

    for ts_name, window_size, floss_score, cp_1, cp_2 in desc_file:
        if n_change_points == 1 and cp_2 != -1: continue

        change_points = [cp_1]
        if cp_2 != -1: change_points.append(cp_2)

        ts = np.loadtxt(fname=os.path.join('../datasets/FLOSS/', ts_name + '.txt'), dtype=np.float64)
        df.append((ts_name, window_size, np.array(change_points), ts))

    return pd.DataFrame.from_records(df, columns=["name", "window_size", "change points", "time_series"])


def load_ucrcp_dataset():
    desc_filename = "../datasets/UCRCP/desc.txt"
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines(): desc_file.append(line.split(","))

    df = []

    for row in desc_file:
        (ts_name, window_size), change_points = row[:2], row[2:]

        ts = np.loadtxt(fname=os.path.join('../datasets/UCRCP/', ts_name + '.txt'), dtype=np.float64)
        df.append((ts_name, int(window_size), np.array([int(_) for _ in change_points]), ts))

    return pd.DataFrame.from_records(df, columns=["name", "window_size", "change points", "time_series"])


def load_combined_dataset():
    return pd.concat([load_floss_dataset(), load_ucrcp_dataset()])


def floss_score(cps_true, cps_pred, ts_len):
    assert len(cps_true) == len(cps_pred), "true/predicted cps must have the same length."
    differences = 0

    for cp_pred in cps_pred:
        distances = paired_euclidean_distances(
            np.array([cp_pred]*len(cps_true)).reshape(-1,1),
            cps_true.reshape(-1,1)
        )
        cp_true_idx = np.argmin(distances, axis=0)
        cp_true = cps_true[cp_true_idx]
        differences += np.abs(cp_pred-cp_true)

    return np.round(differences / (len(cps_true) * ts_len), 6)


if __name__ == '__main__':
    df_comb = load_combined_dataset()

    for idx, (name, window_size, cps, ts) in df_comb.iterrows():
        runtime = time.process_time()
        _, found_cps, _ = extract_clasp_cps(ts, window_size, len(cps))
        runtime = np.round(time.process_time() - runtime, 6)

        score = floss_score(cps, found_cps, ts.shape[0])
        print(f"Time Series: {name}: True Change Points: {cps}, Found Change Points: {found_cps}, Score: {score}, Runtime: {runtime}")