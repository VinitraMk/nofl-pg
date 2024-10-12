import numpy as np

def sample_hyperparameter(params, param_ranges):
    sample_params = {}
    for i,param in enumerate(params):
        sample_params[param] = np.random.uniform(param_ranges[i][0], param_ranges[i][1])
    return sample_params