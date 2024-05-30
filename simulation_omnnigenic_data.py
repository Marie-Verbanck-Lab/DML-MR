"""
on va modifier la pléiotropie pour qu'elle touche l'intégalité des variants étant donné qu'on se trouve dans le modèle omnigénique ...
maj : on modèle 5% de variants "core", 20% de variants périphériques et 75% de variants neutres
"""

import os
import numpy as np
import itertools
import pandas as pd
from scipy.sparse import load_npz
from itertools import combinations
import pdb

class GeneticSimulation:

    def __init__(self, dataset_size=43142, num_exposures=100, num_outcomes=1, seed=72):
        self.dataset_size = dataset_size
        self.num_exposures = num_exposures
        self.num_outcomes = num_outcomes
        self.num_causal_variants = int(dataset_size * 0.05)
        self.num_variables = num_exposures + num_outcomes
        self.rng = np.random.default_rng(seed=seed)
        self.LD_sparse = None
        self.h2 = 1
        self.pleiotropy_prop = 0.5
        self.prop_directionnal = 1
        # Generate correlated_effect matrix once during initialization
        # self.correlated_effect = np.array([[0.2],
        #                                    [0.2],
        #                                    [0.2],
        #                                    [0.2],
        #                                    [0.2],
        #                                    [0.2]])
        # self.correlated_effect = self.generate_correlated_effect_matrix(
        #    num_rows=self.num_exposures, num_columns=self.num_outcomes,
        #    null_prob=0.5, low=0.1, high=0.3, seed=self.rng.integers(0, 1000)
        # )
        self.correlated_effect = self.generate_correlated_effect_matrix(
           num_rows=self.num_exposures, num_columns=self.num_outcomes,
           null_prob=0.5, low=0.1, high=0.3, seed=self.rng.integers(0, 1000)
        )
        #Save the correlated_effect matrix as a NumPy file
        np.savetxt('cem_43142v_030524.csv', self.correlated_effect, delimiter='\t')

    def load_LD_sparse(self, file_path):
        self.LD_sparse = load_npz(file_path) # csr_matrix(np.array(load_npz(file_path).todense())[:5000,:5000])
        
    def set_causal_variants(self, pleiotropy_prop, liste_variants, prop): # modif
        """
        Set the causal variants for exposures in the simulated dataset.
        Parameters:
        - pleiotropy_prop (float): Proportion of pleiotropic variants.
        Returns:
        - np.ndarray: Array containing lists of causal variants for each exposure.
        """
        
        pleio = []
        num_causal_variants = int(self.dataset_size * prop)

        if pleiotropy_prop == 0:
            all_causal_variants = []
            # Choose significant variants for all exposures
            # all_variants = np.arange(liste_variants) # pas besoin peut etre # np.arange(self.dataset_size)
            causal_list = self.rng.choice(liste_variants, size=num_causal_variants, replace=False)
            # Divide the list into exposures
            all_causal_variants = np.array_split(causal_list, self.num_exposures)

        if pleiotropy_prop > 0:
            all_causal_variants = [[] for _ in range(self.num_exposures)]
            # Generate all pairs of exposures
            exposure_pairs = list(itertools.combinations(range(self.num_exposures), 2))
            # Calculate the number of pleiotropic variants per pair
            causal_variants_per_exposure = int(num_causal_variants / self.num_exposures)
            pleiotropic_variant_count_per_pair = int((causal_variants_per_exposure * pleiotropy_prop) / (self.num_exposures - 1)) # pb : fait 0

            # Assign pleiotropic variants to each pair
            for pair in exposure_pairs:
                # solve problem if 0 pleiotropic_variant
                if pleiotropic_variant_count_per_pair == 0:
                    num_pleio = (pleiotropy_prop/(2-pleiotropy_prop)) * num_causal_variants
                    proba = num_pleio / len(exposure_pairs)
                    # tirer avec proba d'avoir 0 ou 1
                    pleiotropic_variant_count_per_pair_2 = np.random.choice([0,1],p=[1-proba,proba])
                    pleiotropic_variants_for_pair = self.rng.choice(liste_variants, size=pleiotropic_variant_count_per_pair_2, replace=False) # self.dataset_size
                    pleio.append(pleiotropic_variants_for_pair)
                else:
                    print("erreur potentielle")
                    pleiotropic_variants_for_pair = self.rng.choice(self.dataset_size, size=pleiotropic_variant_count_per_pair, replace=False)
                    pleio.append(pleiotropic_variants_for_pair)
                # Add these pleiotropic variants to the exposures in the pair
                for exposure in pair:
                    all_causal_variants[exposure] = np.unique(np.concatenate((all_causal_variants[exposure], pleiotropic_variants_for_pair)))
                    
            # Assign unique variants to each exposure
            for exposure_index in range(self.num_exposures):
                # all_causal_variants[exposure_index]
                unique_variant_count = int(causal_variants_per_exposure - len(all_causal_variants[exposure_index])/2) # euh/2 ?
                unique_variants = self.rng.choice(liste_variants, size=unique_variant_count, replace=False) # liste_variant - np.concatenate(all_causal_variants) ?
                all_causal_variants[exposure_index] = np.unique(np.concatenate((all_causal_variants[exposure_index], unique_variants)))            

        return all_causal_variants,pleio

    def generate_correlated_effect_matrix(self, num_rows=100, num_columns=3, null_prob=0.5, low=0.1, high=0.3, seed=None):
        # NOTE: THIS FUNCTION WILL NOT BE USED IN THIS TUTORIAL, correlated_effect will be predifined in the testing code 
        """
        Generate a correlated_effect matrix with random values and introduce variability.

        Parameters:
        - num_rows (int): Number of rows in the matrix.
        - num_columns (int): Number of columns in the matrix.
        - null_prob (float): Probability of setting an effect to zero.
        - low (float): Lower bound for random values.
        - high (float): Upper bound for random values.
        - seed (int): Seed for random number generation.

        Returns:
        - np.ndarray: Correlated_effect matrix with random values and variability.
        """
        num_rows = self.num_exposures if num_rows is None else num_rows
        num_columns = self.num_outcomes if num_columns is None else num_columns

        rng = np.random.default_rng(seed=seed)
        correlated_effect = rng.uniform(low=low, high=high, size=(num_rows, num_columns))

        # Introduce variability by randomly setting some effects to zero
        null_mask = rng.random(size=(num_rows, num_columns)) < null_prob
        correlated_effect[null_mask] = 0.0

        return correlated_effect

    
    def simulate_base_dataset(self, h2=0.5, pleiotropy_prop=0.5,prop_directionnal=0.5,seed = None):
        """ 
        - simule les effets causaux/non causaux en tirant dans une loi normal
        - met en place la pléiotropie directionelle dans notre jeu de données (un variant , 2 exposure, meme direction)
        """
        liste_variants = np.arange(self.dataset_size)
        core_variants,core_pleio = self.set_causal_variants(pleiotropy_prop,liste_variants,0.05)
        liste_variants = list(set(liste_variants) - set(np.concatenate(core_variants)))
        peri_variants,peri_pleio = self.set_causal_variants(pleiotropy_prop,liste_variants,0.20)
        pleio = [np.concatenate((ci, pi)) for ci, pi in zip(core_pleio, peri_pleio )]
        all_causal_variants = {"core" : core_variants ,"peri" : peri_variants}
        # Calculate the overall phenotypic variance for core variants using heritability (h2)
        sigma_causal = (1 + (self.dataset_size * h2) / self.num_causal_variants)

        # noise variants
        # matrix = np.random.uniform(low=0.1, high=1, size=(self.num_variables,self.num_variables))
        base_dataset = self.rng.normal(loc=0, scale=np.random.uniform(0.1, 1, (43142,101)), size=(self.dataset_size, self.num_variables))
        # base_dataset  = self.rng.multivariate_normal(np.zeros(self.num_variables), matrix, self.dataset_size)
        # base_dataset = self.rng.normal(loc=0.0,scale=0.1,size=(self.dataset_size, self.num_variables))
        base_dataset = pd.DataFrame(base_dataset)

        # core variants
        for exposure_index, significant_variants in enumerate(core_variants):
            significant_variants = np.array(np.unique(significant_variants), dtype = int)
            sigmas_exposure = self.rng.normal(loc=0.0, scale=sigma_causal, size=len(significant_variants))
            base_dataset.iloc[significant_variants, self.num_outcomes + exposure_index] = sigmas_exposure

        # peripheral variants
        for exposure_index, significant_variants in enumerate(peri_variants):
            significant_variants = np.array(np.unique(significant_variants), dtype = int)
            sigmas_exposure = self.rng.normal(loc=0.0, scale=1, size=len(significant_variants))
            base_dataset.iloc[significant_variants, self.num_outcomes + exposure_index] = sigmas_exposure

        # directionnal pleiotropy
        paires = list(combinations(range(self.num_exposures), 2))
        dir_pleio = []

        if prop_directionnal>0 and pleiotropy_prop>0:
            i = 0
            all_pleio = np.concatenate(pleio)
            # we keep only x percent of the list to be directionnal
            dir_pleio = np.random.choice(all_pleio,int(prop_directionnal*len(all_pleio)), replace=False)
            # the rest is non directionnal (or "balanced" but we didn't really implement the balance yet)
            bal_pleio = list(set(all_pleio) - set(dir_pleio))
            # for each pleiotropic pair
            for i,pleio_exposure in enumerate(pleio):
                if len(pleio_exposure) > 0:
                    # we keep directionnal pleiotropic variant
                    significant_variants = list(set(pleio_exposure) - set(bal_pleio))
                    # we gave them a big causal effect : not anymore
                    # sigmas_exposure = self.rng.normal(0, sigma_causal, (len(significant_variants),2))
                    # we replace them in the base dataset with absolute value (same positive directionnal effect)
                    colonnes = [self.num_outcomes+paires[i][0],self.num_outcomes+paires[i][1]]
                    beta_pleio = base_dataset.iloc[significant_variants,colonnes]
                    beta_pleio = abs(beta_pleio)
                    # half of the time we change the sign of the effect
                    # number of negative effect
                    size_neg = int(len(beta_pleio)/2)
                    if size_neg == 0:
                        size_neg = np.random.choice([0,1])
                    index_negative = np.random.choice(a=range(len(beta_pleio)),size=size_neg,replace=False)
                    if len(index_negative) > 0:
                        beta_pleio.iloc[index_negative,:] *= -1
                    base_dataset.iloc[significant_variants,colonnes] = beta_pleio

        # now , we add pleiotropy for the rest of the variants
        # all_variant - np.concatenate(all_causal_variants)
        # (prop/2-prop) * ...
        # number of cobinaison : self.num_exposures*(self.num_exposures-1)/2
        # tier num_pleio_variant / cbn
        # rendre pleiotropique

        # Create column names for the DataFrame
        col_names = [f"Y_{i + 1}" for i in range(self.num_outcomes)] + [f"X_{i + 1}" for i in range(self.num_exposures)]
        # Create the final DataFrame
        base_dataset.columns = col_names
        # base_dataset_df = pd.DataFrame(base_dataset, columns=col_names)

        return base_dataset,all_causal_variants,pleio

    def propagate_LD_sparse(self, ld_sparse_matrix, effect_sizes):
        """
        Propagate LD effects through the simulated dataset.

        - ld_sparse_matrix (scipy.sparse.csr_matrix): Sparse matrix representing LD information.
        - effect_sizes (np.ndarray): Array of effect sizes.

        Returns:
        - np.ndarray: Array of propagated effect sizes.
        """
        propagated_effect_sizes = effect_sizes.copy()
        for i in range(ld_sparse_matrix.shape[0]):
            if np.any(ld_sparse_matrix.getrow(i).data):
                ld_row = ld_sparse_matrix.getrow(i).toarray().flatten()
                ld_adjustment = np.dot(ld_row, effect_sizes)
                propagated_effect_sizes.iloc[i,:] += ld_adjustment

        return propagated_effect_sizes

    def simulate_outcomes(self, base_dataset_LD, correlated_effect):
        """
        Simulate outcomes based on the correlated_effect matrix.

        Parameters:
        - base_dataset_LD (pd.DataFrame): Simulated dataset with genetic components and LD propagation.
        - correlated_effect (np.ndarray): Array representing the correlation between outcomes and exposures.

        Returns:
        - pd.DataFrame: DataFrame containing simulated outcomes.
        """
        X = base_dataset_LD.iloc[:,self.num_outcomes:]
        X = pd.DataFrame(X)
        matrix = np.random.uniform(low=0.1, high=0.5, size=(correlated_effect.shape[1], correlated_effect.shape[1]))
        noise=self.rng.multivariate_normal(np.zeros(correlated_effect.shape[1]), matrix, X.shape[0])
        outcomes = np.dot(X, correlated_effect)+noise
        base_dataset_LD.iloc[:,:self.num_outcomes] = outcomes

        return base_dataset_LD
        
        
    def simulate_dataset(self, correlated_effect, pleiotropy_prop=0.1, h2=0.1,prop_directionnal=0.5, seed=None):
        """
        Simulate a complete final dataset including base genetic components, LD propagation, and outcomes.

        Parameters:
        - correlated_effect (np.ndarray): Array representing the correlation between outcomes and exposures.
        - pleiotropy_prop (float): Proportion of pleiotropic variants.
        - h2 (float): Heritability.
        - seed (int): Seed for random number generation.

        Returns:
        - pd.DataFrame: Simulated dataset with outcomes and exposures.
        - np.ndarray: Array containing lists of causal variants for each exposure.
        """
        if self.LD_sparse is None:
            raise ValueError("LD matrix not loaded. Use 'load_LD_sparse' method to load the LD matrix.")

        # Simulate the base dataset with genetic components and LD propagation
        base_dataset, all_causal_variants,pleio = self.simulate_base_dataset(h2=h2,
                                                                             pleiotropy_prop=pleiotropy_prop,
                                                                             seed=seed,
                                                                             prop_directionnal=prop_directionnal)
        # Simulate outcomes based on the correlated dataset
        base_dataset = self.simulate_outcomes(base_dataset, correlated_effect)

        # Update outcomes in the base dataset
        # base_dataset.iloc[np.concatenate(all_causal_variants).astype(int), :self.num_outcomes] = outcomes_df.values

        # Apply LD propagation to the entire dataset
        LD_sparse = self.LD_sparse.tocsr()
        base_dataset_LD = self.propagate_LD_sparse(LD_sparse, base_dataset)

        # Center exposures and outcomes by subtracting the mean of all variables
        mean_all_variables = base_dataset_LD.mean()
        base_dataset_LD -= mean_all_variables # une sorte de nomalisation
        
        # Rename columns
        col_names = [f"Y_{i + 1}" for i in range(self.num_outcomes)] + [f"X_{i + 1}" for i in range(self.num_exposures)]
        base_dataset_LD.columns = col_names

        # Return the base dataset and indices of significant variants
        return base_dataset_LD, all_causal_variants,pleio

    def compute_std_deviation(self, simulated_dataset):
        """
        Compute the standard deviation of the final simulated dataset.

        Parameters:
        - simulated_dataset (pd.DataFrame): The simulated dataset.

        Returns:
        - std_deviation (float): The standard deviation of the dataset.
        """

        # Compute the standard deviation of the entire dataset
        std_deviation = simulated_dataset.std()

        return std_deviation

    
    def verify_number_of_IVs(self, simulated_dataset):
        """
        Verify the number of independent variables (IVs) based on the specified criterion.

        Parameters:
        - simulated_dataset (pd.DataFrame): The simulated dataset.

        Returns:
        - num_IVs (int): The number of IVs that meet the specified criterion.
        - num_IVs_pleiotropic (int): The number of IVs with normalized betas > 5.45131 for at least two exposures.
        - IVs_indices (np.ndarray): The indices of identified IVs (row indices).
        """

        # Extract the beta values for the exposures only
        exposure_betas = simulated_dataset.iloc[:, self.num_outcomes:self.num_outcomes + self.num_exposures].values

        # Calculate the variance of the beta values
        variance_betas = np.var(exposure_betas, axis=0)
        # Calculate the normalized beta values using the variance
        normalized_betas = np.abs(exposure_betas / np.sqrt(variance_betas))

        # Identify IVs based on the threshold
        dict_IVs = {}
        for exposure in range(self.num_exposures): # self.num_exposures
            IVs_indices = np.where(np.abs(normalized_betas[:,exposure]) > 5.45131)[0]
            dict_IVs[f"X_{exposure+1}"] = IVs_indices

        # Count the number of identified IVs
        num_IVs = sum(len(valeurs) for valeurs in dict_IVs.values())
        # Identify pleiotropic IVs (normalized betas > 5.45131 for at least two exposures)
        pleiotropic_IVs_indices = np.where(np.sum(normalized_betas > 5.45131, axis=1) >= 2)[0]
        num_pleiotropic_IVs = len(pleiotropic_IVs_indices)

        return num_IVs, num_pleiotropic_IVs, dict_IVs, pleiotropic_IVs_indices

    def run_scenario(self, pleiotropy_prop, h2, num_datasets, prop_directionnal, parent_folder="PleioLin"):
        """
        Run a scenario of the genetic simulation for multiple datasets.

        Parameters:
        - pleiotropy_prop (float): Proportion of pleiotropic variants.
        - h2 (float): Heritability.
        - num_datasets (int): Number of datasets to simulate.
        - parent_folder (str): Parent folder to save results.

        Returns:
        - list: List of dictionaries containing information for each simulated dataset.
        """
        model_results = []
        
        # Create parent folder if it doesn't exist
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)

        for dataset_num in range(1, num_datasets + 1):
            # Change the seed for each iteration
            seed_for_iteration = self.rng.integers(0, 2**31 - 1)
            
            # Run the simulation for a single dataset
            base_dataset, all_causal_variants,pleio = self.simulate_dataset(
                correlated_effect=self.correlated_effect,
                pleiotropy_prop=pleiotropy_prop,
                h2=h2,
                seed=seed_for_iteration,
                prop_directionnal=prop_directionnal
            )

            # Save the base dataset to a CSV file
            scenario_folder = f'Scenario{int(pleiotropy_prop * 100)}_h2{h2}'
            scenario_path = os.path.join(parent_folder, scenario_folder)
            if not os.path.exists(scenario_path):
                os.makedirs(scenario_path)
            base_dataset.to_csv(os.path.join(scenario_path, f'dataset_{dataset_num}.csv'), index=False)

            # Verify the number of IVs and get relevant information
            num_IVs, num_pleiotropic_IVs, IVs_indices, pleiotropic_IVs_indices = self.verify_number_of_IVs(base_dataset)

            # Compute the standard deviation of the final simulated dataset
            std_deviation = self.compute_std_deviation(base_dataset)

            # Save causal variants to CSV files
            # causal_variants_df = pd.DataFrame({'causal_variants': all_causal_variants})
            # causal_variants_df.to_csv(os.path.join(scenario_path, f'causal_variants_{dataset_num}.csv'), index=False)

            # Save IVs_indices to CSV for each dataset
            IVs_indices_df = pd.DataFrame({'IVs_indices': IVs_indices})
            IVs_indices_df.to_csv(os.path.join(scenario_path, f'IVs_indices_{dataset_num}.csv'), index=False)

            # Save pleiotropic_IVs_indices to CSV for each dataset
            pleiotropic_IVs_indices_df = pd.DataFrame({'pleiotropic_IVs_indices': pleiotropic_IVs_indices})
            pleiotropic_IVs_indices_df.to_csv(os.path.join(scenario_path, f'pleiotropic_IVs_indices_{dataset_num}.csv'), index=False)

            # Save std_deviation to CSV for each dataset
            pd.DataFrame({'std_deviation': [std_deviation]}).to_csv(os.path.join(scenario_path, f'std_deviation_{dataset_num}.csv'), index=False)

            # Save params.txt
            with open(os.path.join(scenario_path, 'params.txt'), 'w') as params_file:
                params_file.write(f"Pleiotropic Variants (%): {pleiotropy_prop * 100}\n")
                params_file.write(f"Heritability: {h2}\n")
                params_file.write(f"Seed for Iteration: {seed_for_iteration}\n")

            # Store the relevant information in a dictionary
            result = {
                "Pleiotropic Variants (%)": pleiotropy_prop * 100,
                "Heritability": h2,
                "Base Dataset": base_dataset,
                # "Causal Variants": np.concatenate(all_causal_variants),
                "Divided CV": all_causal_variants,
                "IV Indices": IVs_indices,
                "Num IVs":num_IVs,
                "pleio":pleio,
                "num pleiotropic IVs": num_pleiotropic_IVs,
                "pleiotropic IVs indices": pleiotropic_IVs_indices,
                "Standard Deviation": std_deviation,
                "Seed for Iteration": seed_for_iteration
            }
            model_results.append(result)

        return model_results

genetic_simulator = GeneticSimulation()
genetic_simulator.load_LD_sparse("/home/mariomf/nouvelle_simulation/sparse_matrix_chunk_4.npz") # sparse_matrix/sparse_matrix_chunk_0.npz
new_num_outcomes = 1
new_num_exposures = 100

# Update the number of outcomes and exposures
genetic_simulator.num_outcomes = new_num_outcomes
genetic_simulator.num_exposures = new_num_exposures
genetic_simulator.num_variables = new_num_exposures + new_num_outcomes
# correlated_effect = np.array([
#                 [0.15, 0.1, 0.2],
#                 [0.2, 0.15, 0.1],
#                 [0.15, 0.0, 0.0],
#                 [0.0, 0.25, 0.0],
#                 [0.0, 0.0, 0.25]
#             ])

scenarios = [
    {"pleiotropy_prop": 0, "prop_directionnal":0, "h2": 0.1, "num_datasets": 10},
    {"pleiotropy_prop": 0, "prop_directionnal":0, "h2": 0.5, "num_datasets": 10},
    {"pleiotropy_prop": 0.5, "prop_directionnal":1, "h2": 0.1, "num_datasets": 10},
    {"pleiotropy_prop": 0.5, "prop_directionnal":1, "h2": 0.5, "num_datasets": 10}
]

all_results = []
for scenario in scenarios:
    model_results = genetic_simulator.run_scenario(
        pleiotropy_prop=scenario["pleiotropy_prop"],
        h2=scenario["h2"],
        prop_directionnal=scenario["prop_directionnal"],
        num_datasets=scenario["num_datasets"],
        parent_folder="PleioLin"
    )
    all_results.extend(model_results)

# Convert the list of dictionaries to a DataFrame
results_df = pd.DataFrame(all_results)

results_df.to_pickle("simu_43142v_220524_omni.pkl")