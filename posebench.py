import numpy as np
import datetime
import absolute_pose
import relative_pose
import homography
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_iterations', required=False, type=int)
    parser.add_argument('--max_iterations', required=False, type=int)
    parser.add_argument('--success_prob', required=False, type=float)
    args = parser.parse_args()
    args = vars(args)
    args = { k:v for (k,v) in args.items() if v is not None}
    return args


def format_metric(name, value):
    if 'AUC' in name:
        return f'{name}={100.0 * value:-5.2f}'
    elif '_rot' in name:
        return f'{name}={value:-3.1E}'
    elif '_pos' in name:
        return f'{name}={value:-3.1E}'
    elif '_rt' in name:
        if value < 1e-6:
            return f'{name}={1e9 * value:-5.1f}ns'
        elif value < 1e-3:
            return f'{name}={1e6 * value:-5.1f}us'
        elif value < 1.0:
            return f'{name}={1e3 * value:-5.1f}ms'
        else:
            return f'{name}={value:.2}s'
    else:
        return f'{name}={value}'

def print_metrics_per_method(metrics):
    for name, res in metrics.items():
        s = f'{name:13s}: '
        for metric_name, value in res.items():
            s = s + format_metric(metric_name,value) + ', '
        s = s[0:-2]
        print(s)

def print_metrics_per_dataset(metrics):
    for dataset in metrics.keys():
        print(f'Dataset: {dataset}')
        print_metrics_per_method(metrics[dataset])

def compute_average_metrics(metrics):
    avg_metrics = {}
    for dataset, dataset_metrics in metrics.items():
        for method, res in dataset_metrics.items():
            if method not in avg_metrics:
                avg_metrics[method] = {}

            for m_name, m_val in res.items():
                if m_name not in avg_metrics[method]:
                    avg_metrics[method][m_name] = []
                avg_metrics[method][m_name].append(m_val)

    for method in avg_metrics.keys():
        for m_name, m_vals in avg_metrics[method].items():
            avg_metrics[method][m_name] = np.mean(m_vals)

    return avg_metrics

if __name__ == '__main__':
    force_opt = parse_args()
    problems = {
        'absolute pose': absolute_pose.main,
        'relative pose': relative_pose.main,
        'homography': homography.main,
    }
    start_time = datetime.datetime.now()
    compiled_metrics = []
    dataset_names = []
    for name, problem in problems.items():
        print(f'Running problem {name}')
        metrics, _ = problem(force_opt = force_opt)

        avg_metrics = compute_average_metrics(metrics)
        compiled_metrics.append(avg_metrics)
        dataset_names += metrics.keys()

    end_time = datetime.datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print(f'Finished running evaluation in {total_time:.1f} seconds ({len(dataset_names)} datasets)')
    print('Datasets: ' + (','.join(dataset_names)) + '\n')

    # Output all the average metrics
    for avg_metrics in compiled_metrics:
        print_metrics_per_method(avg_metrics)
        