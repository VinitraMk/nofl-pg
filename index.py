import argparse
import numpy as np
import pandas as pd
import models.autoencoder as vae
import models.nfl as nfl 
import seaborn as sns
import matplotlib.pyplot as plt

def view_distribution(observed, generated_df):

    X, Y0, Y1, _ = observed
    #print(generated_df)

    # print(X[:,0].shape, generated_df[1]["X1"].shape)
    # print('true value range', X[:, 0].min(), X[:, 0].max())
    # print('gen value range', generated_df[0]["X1"].min(), generated_df[0]["X1"].max())
    #sns.scatterplot(y="Y0",x="T",data=generated_df[0])
    fig,ax = plt.subplots(nrows=5,ncols=2, figsize=(30,30))
    sns.kdeplot(X[:,0], ax=ax[0,0], fill=True)
    ax[0,0].set_title('Observed X_0')
    sns.kdeplot(generated_df[1]["X0"], ax=ax[0,1], fill=True)
    ax[0,1].set_title('Generated X_0')

    sns.kdeplot(X[:, 1], ax = ax[1, 0], fill = True)
    ax[1,0].set_title('Observed X_1')
    sns.kdeplot(generated_df[1]["X1"], ax = ax[1, 1], fill = True)
    ax[1,1].set_title('Generated X_1')

    sns.kdeplot(X[:, 2], ax = ax[2, 0], fill = True)
    ax[2,0].set_title('Observed X_2')
    sns.kdeplot(generated_df[1]["X2"], ax = ax[2, 1], fill = True)
    ax[2,1].set_title('Generated X_2')

    sns.kdeplot(X[:, 3], ax = ax[3, 0], fill = True)
    ax[3,0].set_title('Observed X_3')
    sns.kdeplot(generated_df[1]["X3"], ax = ax[3, 1], fill = True)
    ax[3,1].set_title('Generated X_3')

    sns.kdeplot(X[:, 4], ax = ax[4, 0], fill = True)
    ax[4,0].set_title('Observed X_4')
    sns.kdeplot(generated_df[1]["X4"], ax = ax[4, 1], fill = True)
    ax[4,1].set_title('Generated X_4')

    plt.show()

    fig,ax = plt.subplots(nrows=4,ncols=2,figsize=(15,30))

    sns.kdeplot(Y0, ax = ax[0,0], fill = True)
    ax[0,0].set_title('Observed Y0')
    sns.kdeplot(generated_df[1]["Y0"], ax = ax[0,1], fill = True)
    ax[0,1].set_title('Generated Y0')

    sns.kdeplot(Y1, ax = ax[1,0], fill = True)
    ax[1,0].set_title('Observed Y1')
    sns.kdeplot(generated_df[1]["Y1"], ax = ax[1,1], fill = True)
    ax[1,1].set_title('Generated Y1')

    sns.kdeplot(Y0, ax = ax[2,0], fill = True)
    ax[2,0].set_title('Observed Y0')
    sns.kdeplot(generated_df[1]["Yprime0"], ax = ax[2,1], fill = True)
    ax[2,1].set_title('Generated Y0 Prime')

    sns.kdeplot(Y1, ax = ax[3,0], fill = True)
    ax[3,0].set_title('Observed Y1')
    sns.kdeplot(generated_df[1]["Yprime1"], ax = ax[3,1], fill = True)
    ax[3,1].set_title('Generated Y1 Prime')

    plt.show()


def main(args):

    if args.dataset_type == 'toy':
        # generating toy dataset
        X = np.random.normal(0, 1, (2000, 5))
        Y0 = np.random.normal(np.sum(X,axis=1),1)
        T = np.random.binomial(1,0.5,size=(X.shape[0],))
        Y1 = Y0**2 + np.random.normal(np.mean(X,axis=1),5)
        df = pd.DataFrame(X, columns=['X%d'%(i) for i in range(X.shape[1])])
        df['Y'] = T*Y1 + (1 - T)*Y0
        df['T'] = T
    
        cred_obj = nfl.NFL(
            data = df,
            outcome_var = ['Y'],
            treatment_var = ['T'],
            categorical_var = ['T'],
            numerical_var=['X%d'%(i) for i in range(X.shape[1])]+['Y']
        )
        gen_models = cred_obj.fit(kld_rigidity = 1.0, max_epochs = 5)

        # generate samples
        generated_df = cred_obj.sample()
        observed = (X, Y0, Y1, T)
        #print(generated_df)
        view_distribution(observed, generated_df)
    elif args.dataset_type == 'acic19_linear':
        gt_df = pd.read_csv('./data/datasets/acic19_low_dim_1_linear.csv')

        x_vars = ['V%d'%(i) for i in range(1,11)]
        y_vars = ['Y', 'Y_cf']

        cred_obj = nfl.NFL(
            data = gt_df,
            outcome_var = ['Y'],
            treatment_var = ['A'],
            categorical_var = ['A'],
            numerical_var=x_vars+['Y']
        )
        gen_models = cred_obj.fit(kld_rigidity = 1.0, max_epochs = 5)

        # generated samples
        generated_df, generated_df_prime = cred_obj.sample()
        generated_df_prime['Y'] = (generated_df_prime['A'] * generated_df_prime['Y1']) + ((1 - generated_df_prime['A']) * generated_df_prime['Y0'])
        generated_df_prime['Y_cf'] = (generated_df_prime['A'] * generated_df_prime['Yprime1']) + ((1 - generated_df_prime['A']) * generated_df_prime['Yprime0'])
        #print(gt_df)
        #print(generated_df_prime)
        #print(len(x_vars), len(y_vars))
        view_distribution(gt_df, generated_df_prime, x_vars, y_vars, (25,55), (10,10))
    else:
        SystemExit('Invalid dataset value provided!')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type = str, default = 'toy')
    args = parser.parse_args()

    main(args)
