# all imports
import argparse
import numpy as np
import pandas as pd
import models.autoencoder as vae
import models.nfl as nfl
import models.credence as credence
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from common.utils import sample_hyperparameter, save_params
from common.evaluate import *
from datetime import datetime
import os

# define function for showing probability distributions
def view_distribution(gt_df, generated_df, colnames, chart_path, figsize = (15,30)):

    # print(X[:,0].shape, generated_df[1]["X1"].shape)
    # print('true value range', X[:, 0].min(), X[:, 0].max())
    # print('gen value range', generated_df[0]["X1"].min(), generated_df[0]["X1"].max())

    #plt.show()
    nrows = len(colnames)
    size = figsize
    fig,ax = plt.subplots(nrows=nrows,ncols=2,figsize=size)
    

    for i,yvar in enumerate(colnames):
        sns.kdeplot(gt_df[yvar], ax = ax[i,0], fill = True)
        ax[i,0].set_title(f'Observed {yvar}')
        sns.kdeplot(generated_df[yvar], ax = ax[i,1], fill = True)
        ax[i,1].set_title(f'Generated {yvar}')
    
    plt.savefig(chart_path)

    #plt.show()

# conduct experiment

def use_modified_credence(gt_df, x_vars, y_vars, out_vars, treat_vars, categorical_vars, num_vars, sample_params, output_dir):
    nfl_obj = nfl.NFL(
        data = gt_df,
        outcome_var = out_vars,
        treatment_var = treat_vars,
        categorical_var = categorical_vars,
        numerical_var=x_vars+num_vars
    )
    #sample_params = sample_hyperparameter(['kld_rigidity'], [(0.0,1.0)])
    #max_epochs = 5 
    #print('\n\nHyperparameters')
    #print('kld rigidity:', sample_params['kld_rigidity'])
    #print('max epochs', max_epochs, '\n\n')
    gen_models = nfl_obj.fit(latent_dim = 4, hidden_dim = [8,16,8], kld_rigidity = sample_params['kld_rigidity'], max_epochs = sample_params['max_epochs'])

    # generated samples
    generated_df, generated_df_prime = nfl_obj.sample()
    generated_df_prime['Y'] = (generated_df_prime['A'] * generated_df_prime['Y1']) + ((1 - generated_df_prime['A']) * generated_df_prime['Y0'])
    generated_df_prime['Y_cf'] = (generated_df_prime['A'] * generated_df_prime['Yprime1']) + ((1 - generated_df_prime['A']) * generated_df_prime['Yprime0'])
    experiment_name = "exp-{:%Y%m%d%H%M%S}".format(datetime.now())
    output_dir = os.path.join(output_dir, 'modified-credence')
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
    exp_dir = os.path.join(output_dir, experiment_name)
    if not(os.path.exists(exp_dir)):
        os.mkdir(exp_dir)

    filename = f'{experiment_name}-Y'
    chart_path = os.path.join(exp_dir, filename)
    params_path = os.path.join(exp_dir, f'{experiment_name}-params.txt')
    csv_path = os.path.join(exp_dir, f'{experiment_name}-gendata.csv')
    view_distribution(gt_df, generated_df_prime, y_vars, chart_path, (10,10))
    save_params(sample_params, params_path)
    generated_df_prime.to_csv(csv_path, index = False)
    #print(generated_df_prime['Y'].shape, gt_df['Y'].shape, generated_df_prime['Y_cf'].shape, gt_df['Y_cf'].shape)
    #fids = fid_score(gt_df, generated_df_prime, y_vars)
    #iss = inception_score(generated_df_prime, y_vars)
    #print('FID score: ', fids)
    #print('Inception score: ', iss)


def use_credence(gt_df, x_vars, y_vars, out_vars, treat_vars, categorical_vars, num_vars, sample_params, output_dir):
    cred_obj = credence.Credence(
        data = gt_df,
        outcome_var = out_vars,
        treatment_var = treat_vars,
        categorical_var = categorical_vars,
        numerical_var=x_vars+num_vars
    )
    #sample_params = sample_hyperparameter(['kld_rigidity'], [(0.0,1.0)])
    #max_epochs = 5
    #print('\n\nHyperparameters')
    #print('kld rigidity:', sample_params['kld_rigidity'])
    #print('max epochs', max_epochs, '\n\n')
    gen_models = cred_obj.fit(latent_dim = 4, hidden_dim = [8,16,8], kld_rigidity = sample_params['kld_rigidity'], max_epochs = sample_params['max_epochs'])

    # generated samples
    generated_df, generated_df_prime = cred_obj.sample()
    generated_df_prime['Y'] = (generated_df_prime['A'] * generated_df_prime['Y1']) + ((1 - generated_df_prime['A']) * generated_df_prime['Y0'])
    generated_df_prime['Y_cf'] = (generated_df_prime['A'] * generated_df_prime['Yprime1']) + ((1 - generated_df_prime['A']) * generated_df_prime['Yprime0'])

    experiment_name = "exp-{:%Y%m%d%H%M%S}".format(datetime.now())
    output_dir = os.path.join(output_dir, 'credence')
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
    exp_dir = os.path.join(output_dir, experiment_name)
    if not(os.path.exists(exp_dir)):
        os.mkdir(exp_dir)

    filename = f'{experiment_name}-X'
    params_path = os.path.join(exp_dir, f'{experiment_name}-params.txt')
    chart_path = os.path.join(exp_dir, filename)
    view_distribution(gt_df, generated_df_prime, x_vars, chart_path, (25,50))
    filename = f'{experiment_name}-Y'
    chart_path = os.path.join(exp_dir, filename)
    csv_path = os.path.join(exp_dir, f'{experiment_name}-gendata.csv')
    view_distribution(gt_df, generated_df_prime, y_vars, chart_path, (10,10))
    save_params(sample_params, params_path)
    generated_df_prime.to_csv(csv_path, index = False)
    #print(generated_df_prime['Y'].shape, gt_df['Y'].shape)
    #fids = fid_score(gt_df, generated_df_prime, y_vars)
    #iss = inception_score(generated_df_prime, y_vars)
    #print('FID score: ', fids)
    #print('Inception score: ', iss)

def run_experiment(sample_params, dataset_type, framework_type, output_dir):
    if dataset_type == 'toy':
    # generating toy dataset
        X = np.random.normal(0, 1, (2000, 5))
        Y0 = np.random.normal(np.sum(X,axis=1),1)
        T = np.random.binomial(1,0.5,size=(X.shape[0],))
        Y1 = Y0**2 + np.random.normal(np.mean(X,axis=1),5)
        Y = T*Y1 + (1 - T)*Y0
        xnames = ['X%d'%(i) for i in range(X.shape[1])]
        ynames = ['Y', 'Y_cf']

        gt_df = pd.DataFrame(X, columns=['X%d'%(i) for i in range(X.shape[1])])
        gt_df['Y'] = T*Y1 + (1 - T)*Y0
        gt_df['T'] = T

        if framework_type == 'modified_credence':
            print("\nRunning modified credence")
            use_modified_credence(gt_df, xnames, ynames, ['Y'], ['T'], ['T'], ['Y'], sample_params, output_dir)
        elif framework_type == 'credence':
            print("\nRunning credence")
            use_credence(gt_df, xnames, ynames, ['Y'], ['T'], ['T'], ['Y'], sample_params, output_dir)
        else:
            SystemExit('Invalid framework type provided')
    elif dataset_type == 'acic19_linear':
        gt_df = pd.read_csv('./data/datasets/acic19_low_dim_1_linear.csv')

        x_vars = ['V%d'%(i) for i in range(1,11)]
        y_vars = ['Y', 'Y_cf']

        if framework_type == 'modified_credence':
            print("\nRunning modified credence")
            use_modified_credence(gt_df, x_vars, y_vars, ['Y'], ['A'], ['A'], ['Y'], sample_params, output_dir)
        elif framework_type == 'credence':
            print("\nRunning Credence")
            use_credence(gt_df, x_vars, y_vars, ['Y'], ['A'], ['A'], ['Y'], sample_params, output_dir)
        else:
            SystemExit('Invalid framework type provided')
    elif dataset_type == 'acic19_polynomial':
        gt_df = pd.read_csv('./data/datasets/acic19_low_dim_1_polynomial.csv')

        x_vars = ['V%d'%(i) for i in range(1,11)]
        y_vars = ['Y', 'Y_cf']

        if framework_type == 'modified_credence':
            print("\n\n\nRunning modified credence")
            use_modified_credence(gt_df, x_vars, y_vars, ['Y'], ['A'], ['A'], ['Y'], sample_params, output_dir)
        elif framework_type == 'credence':
            print("\n\n\nRunning credence")
            use_credence(gt_df, x_vars, y_vars, ['Y'], ['A'], ['A'], ['Y'], sample_params, output_dir)
        else:
            SystemExit('Invalid framework type provided')
    else:
        SystemExit('Invalid dataset value provided!')

def run_job(job_output_dir, dataset_type, max_epochs = 250, no_of_experiments = 10):
    for j in range(no_of_experiments):
        print(f"\n\nRunning experiment #{j+1}\n\n")
        sample_params = sample_hyperparameter(['kld_rigidity'], [(0.0,0.3)])
        sample_params['max_epochs'] = max_epochs
        sample_params['dataset'] = dataset_type

        print('Hyperparameters:')
        print(sample_params)
        print('Dataset to be used: ', dataset_type)

        run_experiment(sample_params, dataset_type, 'modified_credence', job_output_dir)

        run_experiment(sample_params, dataset_type, 'credence', job_output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('job_name', type = str)
    parser.add_argument('--dataset_type', type = str, default = 'acic19_linear')
    parser.add_argument('--max_epochs', type = int, default = 3)
    args = parser.parse_args()
    job_name = f'job-{args.job_name}'
    job_dir = os.path.join(os.getcwd(), f'outputs/{job_name}')
    #job_output_dir = os.path.join(os.getcwd(), f'./outputs/{job_name}')
    if not(os.path.exists(job_dir)):
        os.mkdir(job_dir)
        run_job(job_dir, args.dataset_type, args.max_epochs, 5)
    else:
        print('Job with the same name already exists! Pick another name!')

