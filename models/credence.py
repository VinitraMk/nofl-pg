import models.autoencoder as autoencoder
import pytorch_lightning.callbacks.progress as pb
import torch
import pytorch_lightning as pl
import tqdm
import numpy as np
import pandas as pd


class Credence:

    def __init__(
        self,
        data,
        outcome_var,
        treatment_var,
        categorical_var,
        numerical_var,
        var_bounds = {}
    ):
        self.data_raw = data
        self.Ynames = outcome_var
        self.Tnames = treatment_var
        self.categorical_var = categorical_var
        self.numerical_var = numerical_var

        self.var_bounds = var_bounds

        # preprocess data
        self.data_processed = self.preprocess(
            self.data_raw,
            self.Ynames,
            self.Tnames,
            self.categorical_var,
            self.numerical_var,
        )

        self.Xnames = [
            x for x in self.data_processed.columns if x not in self.Ynames + self.Tnames
        ]

    def fit(self,
        latent_dim = 4,
        hidden_dim = [8],
        batch_size = 64,
        kld_rigidity = 0.2,
        max_epochs = 100,
        treatment_effect_fn=lambda x: 0,
        selection_bias_fn=lambda x, t: 0,
        lr = 1e-3
    ):
        
        # generator for treatment
        #self.model_treat = self.data_processed[self.Tnames].mean()
        # generate T without from gt T
        self.model_treat = autoencoder.conVAE(
             df=self.data_processed,
             Xnames=[],
             Ynames=self.Tnames,
             cat_cols=self.categorical_var,
             var_bounds=self.var_bounds,
             latent_dim=latent_dim,
             hidden_dim=hidden_dim,
             kld_rigidity=kld_rigidity,
             lr = lr
        )

        bar = pb.ProgressBar()
        self.trainer_treat = pl.Trainer(
            max_epochs = max_epochs,
            callbacks = [bar],
        )
        self.trainer_treat.fit(
            self.model_treat, self.model_treat.train_loader, self.model_treat.val_loader
        )

        # generator for X | T
        self.model_cov = autoencoder.conVAE(
            df = self.data_processed,
            Xnames=self.Tnames,
            Ynames = self.Xnames,
            cat_cols = self.categorical_var,
            var_bounds = self.var_bounds,
            latent_dim = latent_dim,
            hidden_dim = hidden_dim,
            kld_rigidity = kld_rigidity,
            lr = lr
        )
        bar = pb.ProgressBar()
        self.trainer_cov = pl.Trainer(
            max_epochs = max_epochs,
            callbacks = [bar]
        )
        self.trainer_cov.fit(
            self.model_cov, self.model_cov.train_loader, self.model_cov.val_loader
        )

        # generator for Y | X,T
        self.model_out = autoencoder.conVAE(
            df = self.data_processed,
            Xnames = self.Xnames + self.Tnames,
            Ynames = self.Ynames,
            cat_cols = self.categorical_var,
            var_bounds = self.var_bounds,
            latent_dim = latent_dim,
            hidden_dim = hidden_dim,
            potential_outcome = True,
            treatment_cols = self.Tnames,
            treatment_effect_fn = treatment_effect_fn,
            selection_bias_fn = selection_bias_fn,
            kld_rigidity = kld_rigidity,
            lr = lr
        ).float()
        bar = pb.ProgressBar()
        self.trainer_out = pl.Trainer(
            max_epochs = max_epochs,
            callbacks = [bar]
        )
        self.trainer_out.fit(
            self.model_out, self.model_out.train_loader, self.model_out.val_loader
        )

        return [self.model_treat, self.model_cov, self.model_out]
    
    def sample(self, data = None):
        num_samples = self.data_processed.shape[0]
        print("no of samples: ", num_samples)
        
        if data is None:
            pi_treat = (
                torch.zeros((num_samples, self.model_treat.latent_dim)),
                torch.zeros((num_samples, self.model_treat.latent_dim)),
            )

            pi_cov = (
                torch.zeros((num_samples, self.model_cov.latent_dim)),
                torch.ones((num_samples, self.model_cov.latent_dim)),
            )
            
            pi_out = (
                torch.zeros((num_samples, self.model_out.latent_dim)),
                torch.ones((num_samples, self.model_out.latent_dim)),
            )
            
        else:
            num_samples = data.shape[0]
            T = torch.tensor(data[self.Tnames].values.astype(float)).float()
            Y = torch.tensor(data[self.Ynames].values.astype(float)).float()
            X = torch.tensor(data[self.Xnames].values.astype(float)).float()
            pi_treat = self.model_treat.forward(T)
            pi_cov = self.model_cov.forward(X)
            pi_out = self.model_out.forward(Y)

        #print('gt X shape', X.shape)
        Tgen = self.model_treat.sample(pi = pi_treat, x = torch.empty(size=(num_samples, 0)))
        #print('pi cov',pi_cov)
        Xgen = self.model_cov.sample(pi = pi_cov, x = Tgen)
        #print('pi out', pi_out[0].size(), pi_out[1].size())
        #print('gen shapes', Xgen.size(), T.size())
        Ygen = self.model_out.sample(pi = pi_out, x = torch.cat((Xgen, Tgen),1))
        Ygen_prime = self.model_out.sample(pi = pi_out, x = torch.cat((Xgen, 1 - Tgen), 1))

        df = pd.DataFrame(Xgen.detach().numpy(), columns = self.Xnames)
        df_T = pd.DataFrame(Tgen.detach().numpy(), columns=self.Tnames)
        df_Y = pd.DataFrame(Ygen.detach().numpy(),columns=['Y%d'%i for i in range(Ygen.detach().numpy().shape[1])])
        df_Y_prime = pd.DataFrame(Ygen_prime.detach().numpy(),columns=['Yprime%d'%i for i in range(Ygen.detach().numpy().shape[1])])
        df = df.join(df_T).join(df_Y)
        df_prime = df.join(df_Y_prime)
        return df, df_prime

    def preprocess(
        self, df, outcome_var, treatment_var, categorical_var, numerical_var
    ):
        # this function preprocesses the categorical variables from objects to numerics 

        # codifying categorical variables
        df_cat = (df[categorical_var]).astype("category")
        for col in categorical_var:
            df_cat[col] = df_cat[col].cat.codes

        # codifying numeric variables
        df_num = df[numerical_var]

        # joining columns
        df_ = df_cat.join(df_num)

        return df_
