"""
lasso rapide avec split et avec lD
"""
import os
import pdb
import time
import statsmodels.api as sm
import numpy as np
import pandas as pd
from gsroptim.lasso import lasso_path
from scipy.sparse import load_npz
from scipy.stats import chi2
from custom_enet import CustomENet

os.chdir("DML/")

def lasso_fit(matrix_x,resultat1_part,lambdas):
    """
    Fit a LASSO model for a given dataset and return the selected variants for all lambda values.
    """
    index_variant = []
    betas = lasso_path(matrix_x, resultat1_part, lambdas)[1]
    # betas = pd.DataFrame(betas)
    # betas.loc[]
    for lambda_i in enumerate(lambdas):
        coef = betas[:lambda_i] # 3 get the coefficients
        l1grid_which = np.where(coef!=0)[0] # 4 get coefficient different from 0
        index_variant.append(resultat1_part.iloc[l1grid_which].index.tolist())
    return index_variant

def dml(dataset,sd_outcome,scenar,dts_index,method="linear"):
    """
    A function to perform double machine learning strategy.
    """
    # choose the method
    dict_method = {"linear":sm.WLS,"method_2":0,"method_3":1}
    method = dict_method[method]
    liste_index = []
    exposure = [x for x in dataset.columns if x.startswith("X")]
    outcome = [y for y in dataset.columns if y.startswith("Y")]
    half = int(len(exposure)/2)
    exps = dataset.loc[:,exposure[0:half]] # take half of the exposures
    otcs = dataset.loc[:,outcome]

    # exp_to_analyse = np.random.randint(0,len(exposure)-25)
    # np.random.seed(34)
    exp10 = ["X_45"]
    for exposure_column in exp10:
        liste_index = []
        start_time = time.time()
        X = exps.drop(columns = exposure_column)
        x = exps.loc[:,exposure_column]
        model2 = method(x, X).fit()
        resultat2 = model2.resid
        ld_matrix = load_npz("sparse_matrix_chunk_4.npz").toarray()
        starting_time = time.time()
        matrix_res2 = np.dot(ld_matrix,np.diag(resultat2))#  c'est un peu long
        print("temps pour produit matriciel",time.time()-starting_time)
        # get the residual of the model for all exposure (except one) and one outcome
        for outcome_column in otcs.columns:
            y = otcs.loc[:,outcome_column]
            if len(y) > 0:
                model1 = method(y, X, weights=sd_outcome[outcome_column]**-2).fit()
                resultat1 = model1.resid
            else:
                resultat1 = None
                print("None",exposure_column,outcome_column)
            debut = 0
            for resultat_part in np.array_split(pd.concat([resultat1,resultat2],axis=1), 10):
                # use MR-LASSO to find the potential IVs
                resultat1_part = resultat_part.iloc[:,0]
                resultat2_part = resultat_part.iloc[:,1]
                fin = debut + len(resultat2_part)
                matrix_x = np.column_stack((resultat2_part,matrix_res2[debut:fin,debut:fin]))# matrix_x = np.column_stack((resultat2_part,np.diag(resultat2_part)))
                debut = fin
                penalty_factor = np.ones(matrix_x.shape[1])
                penalty_factor[0] = 0
                l1grid = [18] # np.arange(20,40,1) # ,18,22,26,30
                ysd = np.ones(len(resultat1_part)*100000)  # to change with real value
                lambdas = np.arange(20,25,1)
                index_variant = lasso_fit(matrix_x,resultat1_part,lambdas)
                # Stock the results for each lambda
                for i,liste_i in enumerate(index_variant):
                    if isinstance(liste_i,list):
                        if len(liste_index) <= i:
                            liste_index.append([])
                        liste_index[i].extend(liste_i)

        dtf_param = pd.DataFrame(index=[scenar],columns=["rse","length","beta","se","index","r2"])
        dtf_param = dtf_param.map(lambda x: [[]]) # all results are stored in a list

        for indexes in liste_index:
            if len(indexes) == 0:
                for column in dtf_param.columns:
                    dtf_param.loc[scenar, column][0].append(np.nan)
            else:
                model = sm.WLS(resultat1[indexes],
                               resultat2[indexes],
                               weights=ysd[indexes]**-2).fit() # 5 on fit un modèle linéaire
                dtf_param.loc[scenar,"rse"][0].append(np.sqrt(model.scale)) # 6 get the rse
                dtf_param.loc[scenar,"length"][0].append(len(indexes)) # 7 get number of coefficients
                dtf_param.loc[scenar,"beta"][0].append(model.params.iloc[0]) # 8 get the coefficients
                dtf_param.loc[scenar,"se"][0].append((model.bse/np.sqrt(model.scale)).iloc[0]) # 9 get se
                dtf_param.loc[scenar,"index"][0].append(indexes) # 10 get the variants
                dtf_param.loc[scenar,"r2"][0].append(model.rsquared) # 11 get the r2
        dtf_param.to_pickle(f"resultat_DML_130524/resultat_dml_100e_43142v_200split_S{scenar}_D{dts_index}_E{exposure_column}.pkl")
        print(time.time()-start_time)
    return dtf_param

results_df = pd.read_pickle("simu_43142v_030524.pkl") # simu_120338v_240424.pkl

for scenar in range(0,4):
    dtf_param_all = None
    # Split the DataFrame into chunks of "num_datasets"
    CHUNK_SIZE = 10
    chunk = results_df.iloc[CHUNK_SIZE*scenar:CHUNK_SIZE*(scenar+1),:]
    dtf_param_all = None
    for dataset_index in range(CHUNK_SIZE*scenar,CHUNK_SIZE*(scenar+1)):
        # Extract exposures and outcomes
        dataset = chunk.loc[dataset_index,"Base Dataset"]
        sd_all = chunk.loc[dataset_index,"Standard Deviation"]
        dtf_param = dml(dataset,sd_all,method="linear",scenar=scenar,dts_index=dataset_index)
        if isinstance(dtf_param_all, pd.DataFrame):
            dtf_param_all += dtf_param
        else:
            dtf_param_all = dtf_param
    dtf_param_all.to_pickle(f"resultat_DML_130524/resultat_dml_100e_43142v_200split_S{scenar}.pkl")
