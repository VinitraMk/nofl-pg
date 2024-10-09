import argparse
import numpy as np
import pandas as pd
import models.autoencoder as vae
import models.credence as cred

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
    
        cred_obj = cred.Credence(
            data = df,
            outcome_var = ['Y'],
            treatment_var = ['T'],
            categorical_var = ['T'],
            numerical_var=['X%d'%(i) for i in range(X.shape[1])]+['Y']
        )
        gen_models = cred_obj.fit(kld_rigidity = 1.0, max_epochs = 5)

        # generate samples
        df_gen = cred_obj.sample()
        print(df_gen)

    else:
        SystemExit('Invalid dataset value provided!')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type = str, default = 'toy')
    args = parser.parse_args()

    main(args)
