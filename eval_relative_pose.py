import h5py
import numpy as np
from utils.geometry import *
from utils.misc import *
import poselib
import datetime
from tqdm import tqdm

def eval_essential_estimator(instance):
    x1 = instance['x1']
    x2 = instance['x2']
    cam1 = calib_matrix_to_camera_dict(instance['K1'])
    cam2 = calib_matrix_to_camera_dict(instance['K2'])
    threshold = instance['threshold']

    opt = {
        'max_epipolar_error': threshold
    }

    tt1 = datetime.datetime.now()
    pose, info = poselib.estimate_relative_pose(x1, x2, cam1, cam2, opt, {})
    tt2 = datetime.datetime.now()

    R_gt = instance['R']
    t_gt = instance['t']

    err_R = rotation_angle(R_gt @ pose.R.T)
    err_t = angle(t_gt, pose.t)

    return [err_R, err_t], (tt2-tt1).total_seconds()

def eval_essential_refinement(instance):
    x1 = instance['x1']
    x2 = instance['x2']
    K1 = instance['K1']
    K2 = instance['K2']
    cam1 = calib_matrix_to_camera_dict(K1)
    cam2 = calib_matrix_to_camera_dict(K2)
    
    R_gt = instance['R']
    t_gt = instance['t']
    E_gt = essential_from_pose(R_gt, t_gt)
    F_gt = np.linalg.inv(K2.T) @ E_gt @ np.linalg.inv(K1)
    samp_err = sampson_error(F_gt, x1, x2)

    threshold = instance['threshold']
    inl = samp_err < threshold

    init_pose = poselib.CameraPose()
    init_pose.R = R_gt
    init_pose.t = t_gt

    tt1 = datetime.datetime.now()
    pose, info = poselib.refine_relative_pose(x1[inl], x2[inl], init_pose, cam1, cam2, {})
    tt2 = datetime.datetime.now()

    err_R = rotation_angle(R_gt @ pose.R.T)
    err_t = angle(t_gt, pose.t)

    return [err_R, err_t], (tt2-tt1).total_seconds()


def eval_fundamental_estimator(instance):
    x1 = instance['x1']
    x2 = instance['x2']
    K1 = instance['K1']
    K2 = instance['K2']
    threshold = instance['threshold']
    opt = {
        'max_epipolar_error': threshold
    }

    tt1 = datetime.datetime.now()
    F, info = poselib.estimate_fundamental_matrix(x1, x2, opt, {})
    tt2 = datetime.datetime.now()

    R_gt = instance['R']
    t_gt = instance['t']

    # TODO factorize essential matrix

    err_R = rotation_angle(R_gt @ pose.R.T)
    err_t = angle(t_gt, pose.t)

    return [err_R, err_t], (tt2-tt1).total_seconds()

def eval_fundamental_refinement(instance):
    return [0.0], 0.0


def main(dataset_path='data/relative', datasets=None):
    if datasets is None:
        datasets = [
            'scannet1500_sift',
            'scannet1500_spsg',
            'imc_british_museum',
            'imc_london_bridge',
            'imc_piazza_san_marco',
            'imc_florence_cathedral_side',
            'imc_milan_cathedral',
            'imc_sagrada_familia',
            'imc_lincoln_memorial_statue',
            'imc_mount_rushmore',
            'imc_st_pauls_cathedral'
        ]

    evaluators = {
        'Essential': eval_essential_estimator,
        'Essential (Ref)': eval_essential_refinement,
        #'Fundamental': eval_fundamental_estimator,
        #'Fundamental (Ref)': eval_fundamental_refinement,
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
                'x1': v['x1'][:],
                'x2': v['x2'][:],
                'K1': v['K1'][:],
                'K2': v['K2'][:],
                'R': v['R'][:],
                't': v['t'][:],
                'threshold': 1.0    
            }

            for name, fcn in evaluators.items():
                errs, runtime = fcn(instance)
                results[name]['errs'].append(np.array(errs))
                results[name]['runtime'].append(runtime)


if __name__ == '__main__':
    main()