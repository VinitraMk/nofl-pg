import yaml
import numpy as np

def sample_hyperparameter(params, param_ranges):
    sample_params = {}
    for i,param in enumerate(params):
        sample_params[param] = np.random.uniform(param_ranges[i][0], param_ranges[i][1])
    return sample_params

def save_params(params, filepath):
    lines = []
    for k in params:
        ln = f'{k}: {params[k]}\n'
        lines.append(ln)
    with open(filepath, 'w+') as fp:
        fp.writelines(lines)

def read_yaml(ypath):
    yml_params = {}
    with open(ypath, "r") as stream:
        try:
            yml_params = yaml.safe_load(stream)
        except yaml.YAMLError as err:
            print(err)
    return yml_params

def dump_yaml(ypath, datadict):
    with open(ypath, 'w') as outfile:
        yaml.dump(datadict, outfile, default_flow_style=False)
