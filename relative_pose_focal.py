import h5py
import numpy as np
from utils.geometry import *
from utils.misc import *
import poselib
import pycolmap
import datetime
import posebench
import cv2
from tqdm import tqdm
import argparse


# Compute metrics for relative pose estimation
# AUC for max(err_R,err_t) and avg/med for runtime
def compute_metrics(results, thresholds=[5.0, 10.0, 20.0]):
    methods = results.keys()
    metrics = {}
    for m in methods:
        max_err = [np.max((a, b)) for (a, b, _) in results[m]['errs']]
        metrics[m] = {}
        aucs = compute_auc(max_err, thresholds)
        for auc, t in zip(aucs, thresholds):
            metrics[m][f'AUC{int(t)}'] = auc
        aucs = compute_auc([c for (_, _, c) in results[m]['errs']], thresholds)
        for auc, t in zip(aucs, thresholds):
            metrics[m][f'f AUC{int(t)}'] = auc
        metrics[m]['avg_rt'] = np.mean(results[m]['runtime'])
        metrics[m]['med_rt'] = np.median(results[m]['runtime'])
    return metrics


def eval_focal_relative_pose_estimator(instance, estimator=None):
    opt = instance['opt']
    K = camera_dict_to_calib_matrix(instance['cam1'])
    pp = K[:2, 2]
    f_gt = (K[0, 0] + K[1, 1]) / 2

    tt1 = datetime.datetime.now()
    image_pair, info = poselib.estimate_shared_focal_relative_pose(instance['x1'], instance['x2'], pp, opt, {})
    tt2 = datetime.datetime.now()
    inl = info['inliers']

    if np.sum(inl) < 5:
        return [180.0, 180.0, 100.0], (tt2 - tt1).total_seconds()

    R = image_pair.pose.R
    t = image_pair.pose.t
    err_f = np.abs(f_gt - image_pair.camera1.focal()) / f_gt
    err_R = rotation_angle(instance['R'] @ R.T)
    err_t = angle(instance['t'], t)

    return [err_R, err_t, 100.0 * err_f], (tt2 - tt1).total_seconds()


def main(dataset_path='data/relative', force_opt={}, dataset_filter=[], method_filter=[]):
    datasets = [
        ('scannet1500_sift', 1.5),
        ('scannet1500_spsg', 1.5),
    ]
    if len(dataset_filter) > 0:
        datasets = [(n, t) for (n, t) in datasets if substr_in_list(n, dataset_filter)]

    evaluators = {
        '6pt pose + shared f (poselib)': lambda i: eval_focal_relative_pose_estimator(i, estimator='poselib'),
        }
    if len(method_filter) > 0:
        evaluators = {k: v for (k, v) in evaluators.items() if substr_in_list(k, method_filter)}

    metrics = {}
    full_results = {}
    for (dataset, threshold) in datasets:
        f = h5py.File(f'{dataset_path}/{dataset}.h5', 'r')

        opt = {
            'max_reproj_error': threshold,
            'max_epipolar_error': threshold,
            'max_iterations': 1000,
            'min_iterations': 100,
            'success_prob': 0.9999
        }

        for k, v in force_opt.items():
            opt[k] = v

        results = {}
        for k in evaluators.keys():
            results[k] = {
                'errs': [],
                'runtime': []
            }

        for k, v in tqdm(list(f.items()), desc=dataset):
            instance = {
                'x1': v['x1'][:],
                'x2': v['x2'][:],
                'cam1': h5_to_camera_dict(v['camera1']),
                'cam2': h5_to_camera_dict(v['camera2']),
                'R': v['R'][:],
                't': v['t'][:],
                'threshold': threshold,
                'opt': opt
            }

            for name, fcn in evaluators.items():
                errs, runtime = fcn(instance)
                results[name]['errs'].append(np.array(errs))
                results[name]['runtime'].append(runtime)
        metrics[dataset] = compute_metrics(results)
        full_results[dataset] = results
    return metrics, full_results


if __name__ == '__main__':
    force_opt, method_filter, dataset_filter = posebench.parse_args()
    metrics, _ = main(force_opt=force_opt, method_filter=method_filter, dataset_filter=dataset_filter)
    posebench.print_metrics_per_dataset(metrics)
