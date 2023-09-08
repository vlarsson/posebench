import h5py
import numpy as np
from utils.geometry import *
from utils.misc import *
import poselib
import datetime
import posebench
from tqdm import tqdm

# Compute metrics for homography estimation
# AUC for max(err_R,err_t) and avg/med for runtime
def compute_metrics(results, thresholds = [5.0, 10.0, 20.0]):
    methods = results.keys()
    metrics = {}
    for m in methods:
        max_err = [np.max((a,b)) for (a,b) in results[m]['errs']]
        metrics[m] = {}
        aucs = compute_auc(max_err, thresholds)
        for auc, t in zip(aucs, thresholds):
            metrics[m][f'AUC{int(t)}'] = auc
        metrics[m]['avg_rt'] = np.mean(results[m]['runtime'])
        metrics[m]['med_rt'] = np.median(results[m]['runtime'])

    return metrics

def eval_homography_estimator(instance):
    x1 = instance['x1']
    x2 = instance['x2']
    cam1 = instance['cam1']
    cam2 = instance['cam2']

    threshold = instance['threshold']

    opt = {
        'max_epipolar_error': threshold,
        'max_iterations': 1000
    }

    # TODO Update to homography estimation + factorization
    tt1 = datetime.datetime.now()
    pose, info = poselib.estimate_relative_pose(x1, x2, cam1, cam2, opt, {})
    tt2 = datetime.datetime.now()

    R_gt = instance['R']
    t_gt = instance['t']

    err_R = rotation_angle(R_gt @ pose.R.T)
    err_t = angle(t_gt, pose.t)

    return [err_R, err_t], (tt2-tt1).total_seconds()

def main(dataset_path='data/homography', datasets=None):
    if datasets is None:
        datasets = [
            ('barath_Alamo', 1.0),
            ('barath_NYC_Library', 1.0),
        ]

    evaluators = {
        'H': eval_homography_estimator,
    }
    
    metrics = {}
    full_results = {}
    for (dataset, threshold) in datasets:
        f = h5py.File(f'{dataset_path}/{dataset}.h5', 'r')

        results = {}
        for k in evaluators.keys():
            results[k] = {
                'errs': [],
                'runtime': []
            }

        for k, v in tqdm(f.items(), desc=dataset):
            instance = {
                'x1': v['x1'][:],
                'x2': v['x2'][:],
                'cam1': h5_to_camera_dict(v['camera1']),
                'cam2': h5_to_camera_dict(v['camera2']),
                'R': v['R'][:],
                't': v['t'][:],
                'threshold': threshold   
            }

            for name, fcn in evaluators.items():
                errs, runtime = fcn(instance)
                results[name]['errs'].append(np.array(errs))
                results[name]['runtime'].append(runtime)
        metrics[dataset] = compute_metrics(results)
        full_results[dataset] = results
    return metrics, full_results

if __name__ == '__main__':
    metrics, _ = main()
    posebench.print_metrics_per_dataset(metrics)
    # TODO print results