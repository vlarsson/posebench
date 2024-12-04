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
def compute_metrics(results, thresholds_px = [1.0, 2.0, 5.0]):
    methods = results.keys()
    metrics = {}
    for m in methods:
        rot_err = [a for (a,b,c) in results[m]['errs']]
        cc_err = [b for (a,b,c) in results[m]['errs']]
        rms_err = [c for (a,b,c) in results[m]['errs']]

        metrics[m] = {}
        aucs = compute_auc(rms_err, thresholds_px)
        for auc, t in zip(aucs, thresholds_px):
            metrics[m][f'AUC{int(t)}px'] = auc
        metrics[m]['med_proj'] = np.median(rms_err)
        metrics[m]['med_rot'] = np.median(rot_err)
        metrics[m]['med_pos'] = np.median(cc_err)
        metrics[m]['avg_rt'] = np.mean(results[m]['runtime'])
        metrics[m]['med_rt'] = np.median(results[m]['runtime'])

    return metrics


def eval_pnp_estimator(instance, estimator='poselib_pnp', estimate_focal=False, estimate_extra=False, gt_pp=False):
    opt = instance['opt']
    cam = instance['cam']
    opt['estimate_focal_length'] = estimate_focal
    opt['estimate_extra_params'] = estimate_extra
    opt['bundle'] = {'refine_principal_point': estimate_extra}

    if estimate_focal and not estimate_extra:
        cam_pl = poselib.Camera('SIMPLE_PINHOLE', [], cam['width'], cam['height'])
        cam = cam_pl.todict()
    if estimate_focal and estimate_extra:
        cam_pl = poselib.Camera('OPENCV_FISHEYE', [], cam['width'], cam['height'])
        cam = cam_pl.todict()

    if gt_pp:
        cam_gt = poselib.Camera(instance['cam'])
        cam_pl = poselib.Camera(cam)
        pp_gt = cam_gt.principal_point()
        cam_pl.set_principal_point(pp_gt[0], pp_gt[1])
        cam = cam_pl.todict()


    if estimator == 'poselib_pnp':
        tt1 = datetime.datetime.now()
        image, info = poselib.estimate_absolute_pose(instance['p2d'], instance['p3d'], cam, opt)
        tt2 = datetime.datetime.now()
        (R,t) = (image.pose.R, image.pose.t)
        cam = image.camera
         
    elif estimator == 'pycolmap':
        opt = poselib_opt_to_pycolmap_opt(opt)
        tt1 = datetime.datetime.now()
        result = pycolmap.absolute_pose_estimation(instance['p2d'], instance['p3d'], instance['cam'],
                opt.max_error, opt.min_inlier_ratio, opt.min_num_trials, opt.max_num_trials, opt.confidence)
        tt2 = datetime.datetime.now()
        R = qvec2rotmat(result['qvec'])
        t = result['tvec']

    err_R = rotation_angle(instance['R'] @ R.T)
    err_c = np.linalg.norm(instance['R'].T @ instance['t'] - R.T @ t)


    PX = (R @ instance['p3d'].T + t[:,None]).T
    proj = np.array(cam.project(PX))
    res = proj - instance['p2d']
    rms_err = np.sqrt(np.mean(np.sum(res**2, axis=1)))
    return [err_R, err_c, rms_err], (tt2-tt1).total_seconds()



def main(dataset_path='data/absolute', force_opt = {}, dataset_filter=[], method_filter = []):
    datasets = [
        ('calib/OV_single_plane_2012-A0', 5.0),
        ('calib/UZH_Snapdragon_outdoor_45', 5.0),
        ('calib/Kalibr_TUMVI', 5.0),
        ('calib/UZH_DAVIS_outdoor_forward', 5.0),
        ('calib/OV_single_plane_3136-H0', 5.0),
        ('calib/OV_corner_ov01', 5.0),
        ('calib/Kalibr_BM2820', 5.0),
        ('calib/OCamCalib_GOPR', 5.0),
        ('calib/UZH_Snapdragon_indoor_forward', 5.0),
        ('calib/OV_corner_ov00', 5.0),
        ('calib/OV_corner_ov07', 5.0),
        ('calib/UZH_Snapdragon_outdoor_forward', 5.0),
        ('calib/OV_cube_ov00', 5.0),
        ('calib/OV_cube_ov01', 5.0),
        ('calib/OV_single_plane_5501-C4', 5.0),
        ('calib/Kalibr_GOPRO', 5.0),
        ('calib/Kalibr_ENTANIYA', 5.0),
        ('calib/OCamCalib_Fisheye2', 5.0),
        ('calib/OV_corner_ov06', 5.0),
        ('calib/Kalibr_BM4018S118', 5.0),
        ('calib/OCamCalib_Omni', 5.0),
        ('calib/OV_cube_ov03', 5.0),
        ('calib/OCamCalib_KaidanOmni', 5.0),
        ('calib/UZH_Snapdragon_indoor_45', 5.0),
        ('calib/OV_corner_ov05', 5.0),
        ('calib/OCamCalib_Ladybug', 5.0),
        ('calib/OV_single_plane_130108MP', 5.0),
        ('calib/UZH_DAVIS_indoor_45', 5.0),
        ('calib/OCamCalib_Fisheye1', 5.0),
        ('calib/OV_cube_ov02', 5.0),
        ('calib/OV_corner_ov04', 5.0),
        ('calib/UZH_DAVIS_indoor_forward', 5.0),
        ('calib/Kalibr_BF2M2020S23', 5.0),
        ('calib/UZH_DAVIS_outdoor_45', 5.0),
        ('calib/Kalibr_EUROC', 5.0),
        ('calib/OCamCalib_MiniOmni', 5.0),
        ('calib/Kalibr_BF5M13720', 5.0),
        ('calib/OCamCalib_VMRImage', 5.0),
        ('calib/OCamCalib_Fisheye190deg', 5.0),
    ]

    if len(dataset_filter) > 0:
        datasets = [(n,t) for (n,t) in datasets if substr_in_list(n,dataset_filter)]

    evaluators = {
        'PnPfr (poselib)': lambda i: eval_pnp_estimator(i, estimator='poselib_pnp', estimate_focal=True, estimate_extra=True, gt_pp=False),
        'PnPfr (poselib) GT PP': lambda i: eval_pnp_estimator(i, estimator='poselib_pnp', estimate_focal=True, estimate_extra=True, gt_pp=True),
        #'PnPf (poselib)': lambda i: eval_pnp_estimator(i, estimator='poselib_pnp', estimate_focal=True, estimate_extra=False),
        'PnP (poselib)': lambda i: eval_pnp_estimator(i, estimator='poselib_pnp'),
        #'PnP (COLMAP)': lambda i: eval_pnp_estimator(i, estimator='pycolmap'),
        #'PnPL (poselib)': lambda i: eval_pnp_estimator(i, estimator='poselib_pnpl')
    }

    if len(method_filter) > 0:
        evaluators = {k:v for (k,v) in evaluators.items() if substr_in_list(k,method_filter)}

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

        # RANSAC options
        opt = {
            'max_error': threshold*10,
            'ransac': {
                'max_iterations': 1000,
                'min_iterations': 100,
                'success_prob': 0.9999
            }
        }

        # Add in global overrides
        for k, v in force_opt.items():
            opt[k] = v

        for k, v in tqdm(f.items(), desc=dataset):
            instance = {
                'p2d': v['p2d'][:],
                'p3d': v['p3d'][:],
                'cam': h5_to_camera_dict(v['camera']),
                'R': v['R'][:],
                't': v['t'][:],
                'threshold': threshold,
                'opt': opt  
            }

            # Run each of the evaluators 
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

    print('\nAverage metrics:')
    avg_metrics = posebench.compute_average_metrics(metrics)
    posebench.print_metrics_per_method_table(avg_metrics)

