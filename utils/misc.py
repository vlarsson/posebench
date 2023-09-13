import numpy as np
import h5py
import pycolmap

def substr_in_list(s, lst):
    return np.any([s.find(t) >= 0 for t in lst])

def poselib_opt_to_pycolmap_opt(opt):
    pyc_opt = pycolmap.RANSACOptions()

    if 'max_reproj_error' in opt:
        pyc_opt.max_error = opt['max_reproj_error']
    elif 'max_epipolar_error' in opt:
        pyc_opt.max_error = opt['max_epipolar_error']

    if 'max_iterations' in opt:
        pyc_opt.max_num_trials = opt['max_iterations']
    if 'min_iterations' in opt:
        pyc_opt.min_num_trials = opt['max_iterations']

    if 'success_prob' in opt:
        pyc_opt.confidence = opt['success_prob']

    return pyc_opt

def h5_to_camera_dict(data):
    camera_dict = {}
    camera_dict['model'] = data['model'].asstr()[0]
    camera_dict['width'] = int(data['width'][0])
    camera_dict['height'] = int(data['height'][0])
    camera_dict['params'] =data['params'][:]
    return camera_dict


def calib_matrix_to_camera_dict(K):
    camera_dict = {}
    camera_dict['model'] = 'PINHOLE'
    camera_dict['width'] = int(np.ceil(K[0,2] * 2))
    camera_dict['height'] = int(np.ceil(K[1,2] * 2))
    camera_dict['params'] = [K[0,0], K[1,1], K[0,2], K[1,2]]
    return camera_dict

def camera_dict_to_calib_matrix(cam):
    if cam['model'] == 'PINHOLE':
        p = cam['params']
        return np.array([[p[0], 0.0, p[2]], [0.0, p[1], p[3]], [0.0, 0.0, 1.0]])
    else:
        raise Exception('nyi model in camera_dict_to_calib_matrix')
        

# From Paul
def compute_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


# From Mihai
def compute_auc_mihai(method, errors, thresholds=[0.001, 0.01, 0.1], verbose=True, outfile=None):
    n_images = len(errors)
    errors = np.sort(errors)
    if verbose:
        if outfile is None:
            print('[%s]' % method)
        else:
            print('[%s]' % method, file=outfile)
    gt_acc = 0.001
    errors[errors <= gt_acc] = 0.000
    results = []
    for threshold in thresholds:
        previous_error = 0
        previous_score = 0
        mAA = 0

        # Initialization.
        previous_error = gt_acc
        previous_score = np.sum(errors <= gt_acc)
        mAA = gt_acc * previous_score
        for error in errors:
            # Skip initialization.
            if error <= gt_acc:
                continue
            if error > threshold:
                break
            score = previous_score + 1
            mAA += trapezoid_area(previous_score, score, error - previous_error)
            previous_error = error
            previous_score = score
        mAA += trapezoid_area(previous_score, previous_score, threshold - previous_error)
        mAA /= (threshold * n_images)
        if verbose:
            if outfile is None:
                print('AUC @ %2.3f - %.2f percents' % (threshold, mAA * 100))
            else:
                print('AUC @ %2.3f - %.2f percents' % (threshold, mAA * 100), file=outfile)
        results.append(mAA * 100)
    return results
