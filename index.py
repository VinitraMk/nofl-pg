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
from common.utils import sample_hyperparameter, save_params, read_yaml
from common.evaluate import *
from datetime import datetime
import os

# define function for showing probability distributions
def view_distribution(gt_df, generated_df, colnames, chart_path, figsize = (15,30), merge_plots = False):

    # print(X[:,0].shape, generated_df[1]["X1"].shape)
    # print('true value range', X[:, 0].min(), X[:, 0].max())
    # print('gen value range', generated_df[0]["X1"].min(), generated_df[0]["X1"].max())

    #plt.show()
    nrows = len(colnames)
    size = figsize
    
    if merge_plots:
        fig,ax = plt.subplots(nrows=nrows,ncols=1,figsize=size)
        for i,yvar in enumerate(colnames):
            sns.kdeplot(gt_df[yvar], ax = ax[i], fill = True)
            #ax[i,0].set_title(f'Observed {yvar}')
            sns.kdeplot(generated_df[yvar], ax = ax[i], fill = True)
            ax[i].set_title(f'Observed vs Generated {yvar}')
    else:
        fig,ax = plt.subplots(nrows=nrows,ncols=2,figsize=size)
        for i,yvar in enumerate(colnames):
            sns.kdeplot(gt_df[yvar], ax = ax[i,0], fill = True)
            ax[i,0].set_title(f'Observed {yvar}')
            sns.kdeplot(generated_df[yvar], ax = ax[i,1], fill = True)
            ax[i,1].set_title(f'Generated {yvar}')
    
    plt.savefig(chart_path)

    #plt.show()
def view_treatment_plot(gt_df, generated_df, treat_var, chart_path, figsize = (15, 30)):
    plt.figure(figsize = figsize)
    Y0 = gt_df.loc[gt_df[treat_var] == 0]
    Y1 = gt_df.loc[gt_df[treat_var] == 1]
    Y0gen = generated_df.loc[generated_df[treat_var] == 0]
    Y1gen = generated_df.loc[generated_df[treat_var] == 1]

    d = pd.DataFrame({
        'Ground truth': {
            'Y0': Y0.shape[0],
            'Y1': Y1.shape[0],
        },
        'Generated data': {
            'Y0': Y0gen.shape[0],
            'Y1': Y1gen.shape[0]
        }
    })
    #print(list(d.keys()))
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=figsize)
    d.plot(kind = 'bar', ax = ax, color = ['skyblue', 'orange'])
    
    plt.savefig(chart_path)

# conduct experiment

def use_modified_credence(gt_df, x_vars, y_vars, out_vars, treat_vars, categorical_vars, num_vars, sample_params, output_dir, exp_params):
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
    gen_models = nfl_obj.fit(latent_dim = exp_params['latent_dim'],
        hidden_dim = exp_params['hidden_dims'],
        kld_rigidity = sample_params['kld_rigidity'], max_epochs = sample_params['max_epochs'], lr = exp_params['lr'], use_uniform_autoencoder = exp_params['use_uniform_autoencoder'])
    treat_var = treat_vars[0]
    # generated samples
    generated_df, generated_df_prime = nfl_obj.sample()
    generated_df_prime['Y'] = (generated_df_prime[treat_var] * generated_df_prime['Y1']) + ((1 - generated_df_prime[treat_var]) * generated_df_prime['Y0'])
    generated_df_prime['Y_cf'] = (generated_df_prime[treat_var] * generated_df_prime['Yprime1']) + ((1 - generated_df_prime[treat_var]) * generated_df_prime['Yprime0'])
    experiment_name = "exp-{:%Y%m%d%H%M%S}".format(datetime.now())
    output_dir = os.path.join(output_dir, 'modified-credence')
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
    exp_dir = os.path.join(output_dir, experiment_name)
    if not(os.path.exists(exp_dir)):
        os.mkdir(exp_dir)

    filename = f'{experiment_name}-Y'
    chart_path = os.path.join(exp_dir, filename)
    view_distribution(gt_df, generated_df_prime, y_vars, chart_path, (10,10), exp_params['merge_plots'])
    filename = f'{experiment_name}-{treat_var}'
    chart_path = os.path.join(exp_dir, filename)
    view_treatment_plot(gt_df, generated_df, treat_var, chart_path, (15, 30))
    
    params_path = os.path.join(exp_dir, f'{experiment_name}-params.txt')
    save_params(sample_params, params_path)
    params_path = os.path.join(exp_dir, f'{experiment_name}-exp-params.txt')
    save_params(exp_params, params_path)
    csv_path = os.path.join(exp_dir, f'{experiment_name}-gendata.csv')
    generated_df_prime.to_csv(csv_path, index = False)
    #print(generated_df_prime['Y'].shape, gt_df['Y'].shape, generated_df_prime['Y_cf'].shape, gt_df['Y_cf'].shape)
    #fids = fid_score(gt_df, generated_df_prime, y_vars)
    #iss = inception_score(generated_df_prime, y_vars)
    #print('FID score: ', fids)
    #print('Inception score: ', iss)


def use_credence(gt_df, x_vars, y_vars, out_vars, treat_vars, categorical_vars, num_vars, sample_params, output_dir, exp_params):
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
    gen_models = cred_obj.fit(latent_dim = exp_params['latent_dim'],
        hidden_dim = exp_params['hidden_dims'],
        kld_rigidity = sample_params['kld_rigidity'], max_epochs = sample_params['max_epochs'], lr = exp_params['lr'], use_uniform_autoencoder = exp_params['use_uniform_autoencoder'])
    treat_var = treat_vars[0]
    # generated samples
    generated_df, generated_df_prime = cred_obj.sample()
    generated_df_prime['Y'] = (generated_df_prime[treat_var] * generated_df_prime['Y1']) + ((1 - generated_df_prime[treat_var]) * generated_df_prime['Y0'])
    generated_df_prime['Y_cf'] = (generated_df_prime[treat_var] * generated_df_prime['Yprime1']) + ((1 - generated_df_prime[treat_var]) * generated_df_prime['Yprime0'])

    experiment_name = "exp-{:%Y%m%d%H%M%S}".format(datetime.now())
    output_dir = os.path.join(output_dir, 'credence')
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
    exp_dir = os.path.join(output_dir, experiment_name)
    if not(os.path.exists(exp_dir)):
        os.mkdir(exp_dir)

    filename = f'{experiment_name}-X'
    chart_path = os.path.join(exp_dir, filename)
    view_distribution(gt_df, generated_df_prime, x_vars, chart_path, (25,50), exp_params['merge_plots'])
    filename = f'{experiment_name}-Y'
    chart_path = os.path.join(exp_dir, filename)
    view_distribution(gt_df, generated_df_prime, y_vars, chart_path, (10,10), exp_params['merge_plots'])
    filename = f'{experiment_name}-{treat_var}'
    chart_path = os.path.join(exp_dir, filename)
    view_treatment_plot(gt_df, generated_df, treat_var, chart_path, (15, 30))
    params_path = os.path.join(exp_dir, f'{experiment_name}-params.txt')
    save_params(sample_params, params_path)
    params_path = os.path.join(exp_dir, f'{experiment_name}-exp-params.txt')
    save_params(exp_params, params_path)
    csv_path = os.path.join(exp_dir, f'{experiment_name}-gendata.csv')
    generated_df_prime.to_csv(csv_path, index = False)
    #print(generated_df_prime['Y'].shape, gt_df['Y'].shape)
    #fids = fid_score(gt_df, generated_df_prime, y_vars)
    #iss = inception_score(generated_df_prime, y_vars)
    #print('FID score: ', fids)
    #print('Inception score: ', iss)

def run_experiment(sample_params, dataset_type, framework_type, output_dir, exp_params):
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
        gt_df['Y_cf'] = (1 - T) * Y1 + T*Y0
        gt_df['T'] = T

        if framework_type == 'modified_credence':
            #print("\nRunning modified credence")
            use_modified_credence(gt_df, xnames, ynames, ['Y'], ['T'], ['T'], ['Y'], sample_params, output_dir, exp_params)
        elif framework_type == 'credence':
            #print("\nRunning credence")
            use_credence(gt_df, xnames, ynames, ['Y'], ['T'], ['T'], ['Y'], sample_params, output_dir, exp_params)
        else:
            SystemExit('Invalid framework type provided')
    elif dataset_type == 'acic19_linear':
        gt_df = pd.read_csv('./data/datasets/acic19_low_dim_1_linear.csv')

        x_vars = ['V%d'%(i) for i in range(1,11)]
        y_vars = ['Y', 'Y_cf']

        if framework_type == 'modified_credence':
            #print("\nRunning modified credence")
            use_modified_credence(gt_df, x_vars, y_vars, ['Y'], ['A'], ['A'], ['Y'], sample_params, output_dir, exp_params)
        elif framework_type == 'credence':
            #print("\nRunning Credence")
            use_credence(gt_df, x_vars, y_vars, ['Y'], ['A'], ['A'], ['Y'], sample_params, output_dir, exp_params)
        else:
            SystemExit('Invalid framework type provided')
    elif dataset_type == 'acic19_polynomial':
        gt_df = pd.read_csv('./data/datasets/acic19_low_dim_1_polynomial.csv')

        x_vars = ['V%d'%(i) for i in range(1,11)]
        y_vars = ['Y', 'Y_cf']

        if framework_type == 'modified_credence':
            #print("\n\n\nRunning modified credence")
            use_modified_credence(gt_df, x_vars, y_vars, ['Y'], ['A'], ['A'], ['Y'], sample_params, output_dir, exp_params['frameworks']['modified_credence'])
        elif framework_type == 'credence':
            #print("\n\n\nRunning credence")
            use_credence(gt_df, x_vars, y_vars, ['Y'], ['A'], ['A'], ['Y'], sample_params, output_dir, exp_params['frameworks']['credence'])
        else:
            SystemExit('Invalid framework type provided')
    else:
        SystemExit('Invalid dataset value provided!')

def run_job(job_output_dir, dataset_type, exp_params):
    frameworks = list(exp_params['frameworks'].keys())
    for j in range(args.no_of_exps):
        print(f"\n\nRunning experiment #{j+1}")
        for f in frameworks:
            sample_params = sample_hyperparameter(['kld_rigidity'], tuple([exp_params['frameworks'][f]['kld_rigidity_range']]))
            sample_params['max_epochs'] = exp_params['frameworks'][f]['max_epochs']
            sample_params['dataset'] = dataset_type
            print(f"\nRunning {f}")
            print('Hyperparameters:')
            print(sample_params)
            print('Dataset to be used: ', dataset_type)
            print('Experiment params:')
            print(exp_params, "\n")

            run_experiment(sample_params, dataset_type, f, job_output_dir, exp_params['frameworks'][f])

        #run_experiment(sample_params, dataset_type, 'credence', job_output_dir, exp_params['frameworks']['credence'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type = str, default = "")
    #parser.add_argument('--dataset_type', type = str, default = 'acic19_linear')
    parser.add_argument('--no_of_exps', type = int, default = 5)
    args = parser.parse_args()
    if args.job_name == "":
        nj = len(os.listdir('./outputs'))
        job_name = f"job-{nj + 1}"
    job_dir = os.path.join(os.getcwd(), f'outputs/{job_name}')
    #job_output_dir = os.path.join(os.getcwd(), f'./outputs/{job_name}')

    exp_params = read_yaml('./exp_config.yaml')
    #print(exp_params, list(exp_params['frameworks'].keys()))

    if not(os.path.exists(job_dir)):
        os.mkdir(job_dir)
        run_job(job_dir, exp_params['dataset_type'], exp_params)
    else:
        print('Job with the same name already exists! Pick another name!')

