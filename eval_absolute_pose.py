import h5py
import numpy as np
from utils.geometry import *
from utils.misc import *
import poselib
import datetime
from tqdm import tqdm

def eval_abspose_estimator(instance):
    points2D = instance['x']
    points3D = instance['X']
    cam = instance['camera']
    threshold = instance['threshold']

    opt = {
        'max_reproj_error': threshold
    }

    tt1 = datetime.datetime.now()
    pose, info = poselib.estimate_absolute_pose(points2D, points3D, cam, opt, {})
    tt2 = datetime.datetime.now()

    R_gt = instance['R']
    t_gt = instance['t']

    err_R = rotation_angle(R_gt @ pose.R.T)
    err_c = np.linalg.norm(R_gt.T @ t_gt - pose.R.T @ pose.t)
    return [err_R, err_c], (tt2-tt1).total_seconds()


def main(dataset_path='data/absolute', datasets=None):
    if datasets is None:
        datasets = [
            'eth3d_130_dusmanu'
        ]

    evaluators = {
        'Abspose': eval_abspose_estimator,
        
    }
    
    for dataset in datasets:
        f = h5py.File(f'{dataset_path}/{dataset}.h5', 'r')

        results = {}
        for k in evaluators.keys():
            results[k] = {
                'errs': [],
                'runtime': []
            }

        for k, v in tqdm(f.items()):
            instance = {
                'x': v['x'][:],
                'X': v['X'][:],
                'camera': h5_to_camera_dict(v['camera']),
                'R': v['R'][:],
                't': v['t'][:],
                'threshold': 12.0    
            }

            for name, fcn in evaluators.items():
                errs, runtime = fcn(instance)
                results[name]['errs'].append(np.array(errs))
                results[name]['runtime'].append(runtime)

        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    main()