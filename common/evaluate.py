import numpy as np
import torch.nn as nn
from scipy.linalg import sqrtm

'''
def NLL(observed_df, generated_df, outcome_vars):
    NLLS = []
    for var in outcome_vars:
        squared_diff = (observed_df[var] - generated_df[var]) ** 2
        var = np.var(generated_df[var])
    
        nll = 0.5 * squared_diff / var + 0.5 * np.log(2 * np.pi * var)
        NLLS.append(np.mean(nll))
    return np.array(NLLS)

def reconstruction_error(observed_df, generated_df, outcome_vars):
    recons = []
    for var in outcome_vars:
        diff = (generated_df[var] - observed_df) ** 2
        recons.append(np.mean(diff))
    return np.array(recons)
'''

def fid_score(observed_df, generated_df, outcome_vars):
    fids = []
    for var in outcome_vars:
        mu_real = np.mean(generated_df[var])
        sigma_real = np.var(observed_df[var])

        mu_gen = np.mean(generated_df[var])
        sigma_gen = np.var(observed_df[var])

        mean_diff_squared = (mu_real - mu_gen) ** 2

        sqrt_product = np.sqrt(sigma_real * sigma_gen)
        trace_component = sigma_real + sigma_gen - 2 * sqrt_product

        fid = mean_diff_squared + trace_component

        fids.append(fid)
    return np.array(fids)

def inception_score(generated_df, outcome_vars):
    iss = []
    for var in outcome_vars:
        p_y_given_x = np.mean(generated_df[var], axis=0)
        p_y = np.mean(p_y_given_x)

        # Calculate the score
        kl_divergence = np.mean(np.log(p_y_given_x / p_y))
        inception_score = np.exp(kl_divergence)
        iss.append(inception_score)
    return np.array(iss)
