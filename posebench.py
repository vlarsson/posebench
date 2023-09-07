import numpy as np
import absolute_pose
import relative_pose

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

def print_metrics(metrics):
    for dataset in metrics.keys():
        print(f'Dataset: {dataset}')
        for name, res in metrics[dataset].items():
            s = f'{name:10s}: '
            for metric_name, value in res.items():
                s = s + format_metric(metric_name,value) + ', '
            s = s[0:-2]
            print(s)


if __name__ == '__main__':
    run_evals()