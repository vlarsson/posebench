import h5py
import numpy as np
from utils.geometry import *
from utils.misc import *
import poselib
import datetime
import posebench
from tqdm import tqdm

# Compute metrics for absolute pose estimation
# AUC for camera center and avg/med for runtime
def compute_metrics(results, thresholds = [0.1, 1.0, 5.0]):
    methods = results.keys()
    metrics = {}
    for m in methods:
        rot_err = [a for (a,b) in results[m]['errs']]
        cc_err = [b for (a,b) in results[m]['errs']]
        metrics[m] = {}
        aucs = compute_auc(cc_err, thresholds)
        for auc, t in zip(aucs, thresholds):
            metrics[m][f'AUC{int(t)}'] = auc
        metrics[m]['med_rot'] = np.median(rot_err)
        metrics[m]['med_pos'] = np.median(cc_err)
        metrics[m]['avg_rt'] = np.mean(results[m]['runtime'])
        metrics[m]['med_rt'] = np.median(results[m]['runtime'])

    return metrics


def eval_pnp_estimator(instance):
    points2D = instance['p2d']
    points3D = instance['p3d']
    cam = instance['cam']
    threshold = instance['threshold']

    opt = {
        'max_reproj_error': threshold,
        'max_iterations': 1000
    }

    tt1 = datetime.datetime.now()
    pose, info = poselib.estimate_absolute_pose(points2D, points3D, cam, opt, {})
    tt2 = datetime.datetime.now()

    R_gt = instance['R']
    t_gt = instance['t']

    err_R = rotation_angle(R_gt @ pose.R.T)
    err_c = np.linalg.norm(R_gt.T @ t_gt - pose.R.T @ pose.t)
    return [err_R, err_c], (tt2-tt1).total_seconds()


def eval_pnpl_estimator(instance):
    points2D = instance['p2d']
    points3D = instance['p3d']
    lines2D = instance['l2d']
    lines3D = instance['l3d']
    
    cam = instance['cam']
    threshold = instance['threshold']

    opt = {
        'max_reproj_error': threshold,
        'max_iterations': 1000
    }
    tt1 = datetime.datetime.now()
    pose, info = poselib.estimate_absolute_pose_pnpl(points2D, points3D, lines2D[:,0:2], lines2D[:,2:4], lines3D[:,0:3], lines3D[:,3:6], cam, opt, {})
    tt2 = datetime.datetime.now()

    R_gt = instance['R']
    t_gt = instance['t']

    err_R = rotation_angle(R_gt @ pose.R.T)
    err_c = np.linalg.norm(R_gt.T @ t_gt - pose.R.T @ pose.t)
    return [err_R, err_c], (tt2-tt1).total_seconds()




def main(dataset_path='data/absolute', datasets=None):
    if datasets is None:
        datasets = [
            'eth3d_130_dusmanu',
            '7scenes_heads',
            '7scenes_stairs',
            'cambridge_landmarks_GreatCourt',
            'cambridge_landmarks_ShopFacade',
            'cambridge_landmarks_KingsCollege',
            'cambridge_landmarks_StMarysChurch',
            'cambridge_landmarks_OldHospital'
        ]

    evaluators = {
        'PnP': eval_pnp_estimator,
        'PnPL': eval_pnpl_estimator,
    }

    metrics = {}
    full_results = {}
    for dataset in datasets:
        f = h5py.File(f'{dataset_path}/{dataset}.h5', 'r')

        results = {}
        for k in evaluators.keys():
            results[k] = {
                'errs': [],
                'runtime': []
            }

        for k, v in tqdm(f.items(), desc=dataset):
            instance = {
                'p2d': v['p2d'][:],
                'p3d': v['p3d'][:],
                'cam': h5_to_camera_dict(v['camera']),
                'R': v['R'][:],
                't': v['t'][:],
                'threshold': 12.0    
            }

            # Check if we have 2D-3D line correspondences
            if 'l2d' in v:
                instance['l2d'] = v['l2d'][:]
                instance['l3d'] = v['l3d'][:]
            else:
                instance['l2d'] = np.zeros((0,4))
                instance['l3d'] = np.zeros((0,6))
                
            # Run each of the evaluators 
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
    import ipdb
    ipdb.set_trace()