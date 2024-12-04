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
    parser.add_argument('--method',  required=False, type=str)
    parser.add_argument('--dataset',  required=False, type=str)
    
    args = parser.parse_args()
    args = vars(args)
    args = { k:v for (k,v) in args.items() if v is not None}

    method_filter = []
    if 'method' in args:
        method_filter = args['method'].split(',')
        del args['method']
    dataset_filter = []
    if 'dataset' in args:
        dataset_filter = args['dataset'].split(',')
        del args['dataset']
    return args, method_filter, dataset_filter


def format_metric(name, value):
    name = name.upper()
    if 'AUC' in name:
        return f'{100.0 * value:>6.2f}'
    elif 'ROT' in name:
        return f'{value:-3.1E}'
    elif 'INL' in name:
        return f'{value:-3.5f}'
    elif 'RMS' in name:
        return f'{value:-3.2f}px'
    elif 'PX' in name:
        return f'{value:-3.2f}px'
    elif 'PROJ' in name:
        return f'{value:-3.2f}px'
    elif 'MSAC' in name:
        return f'{value:-3.4f}px'
    elif 'POS' in name:
        return f'{value:-3.1f}'
    elif 'TIME' in name or 'RT' in name:
        if value < 1e-6:
            return f'{1e9 * value:-5.1f}ns'
        elif value < 1e-3:
            return f'{1e6 * value:-5.1f}us'
        elif value < 1.0:
            return f'{1e3 * value:-5.1f}ms'
        else:
            return f'{value:.2}s'
    else:
        return f'{value}'

def print_metrics_per_method(metrics):
    for name, res in metrics.items():
        s = f'{name:18s}: '
        for metric_name, value in res.items():
            s = s + f'{metric_name}={format_metric(metric_name,value)}' + ', '
        s = s[0:-2]
        print(s)

def print_metrics_per_method_table(metrics, sort_by_metric=None, reverse_sort=False):
    method_names = list(metrics.keys())
    if len(method_names) == 0:
        return
    metric_names = list(metrics[method_names[0]].keys())

    if sort_by_metric is not None:
        vals = [metrics[m][sort_by_metric] for m in method_names]
        if reverse_sort:
            ind = np.argsort(-np.array(vals))
        else:
            ind = np.argsort(np.array(vals))
        method_names = np.array(method_names)[ind]
            

    field_lengths = {x:len(x)+2 for x in metric_names}
    name_length = np.max([len(x) for x in metrics.keys()])

    # print header
    print(f'{"":{name_length}s}',end=' ')
    for metric_name in metric_names:
        print(f'{metric_name:>{field_lengths[metric_name]}s}',end=' ')
    print('')

    for name in method_names:
        res = metrics[name]
        print(f'{name:{name_length}s}',end=' ')
        for metric_name, value in res.items():
            print(f'{format_metric(metric_name,value):>{field_lengths[metric_name]}s}',end=' ')
        print('')


def print_metrics_per_dataset(metrics, as_table=True, sort_by_metric=None, reverse_sort=False):
    for dataset in metrics.keys():
        print(f'Dataset: {dataset}')
        if as_table:
            print_metrics_per_method_table(metrics[dataset],sort_by_metric=sort_by_metric, reverse_sort=reverse_sort)        
        else:
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
    force_opt,  method_filter, dataset_filter = parse_args()
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
        metrics, _ = problem(force_opt = force_opt, method_filter=method_filter, dataset_filter=dataset_filter)

        avg_metrics = compute_average_metrics(metrics)
        compiled_metrics.append(avg_metrics)
        dataset_names += metrics.keys()

    end_time = datetime.datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print(f'Finished running evaluation in {total_time:.1f} seconds ({len(dataset_names)} datasets)')
    print('Datasets: ' + (','.join(dataset_names)) + '\n')

    # Output all the average metrics
    for avg_metrics in compiled_metrics:
        print_metrics_per_method_table(avg_metrics)
        print('')
        