import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class GraphDataProcessor:

    def __init__(self, element_properties=None, target_elements=None):

        self.target_elements = target_elements or self._get_default_elements()
        self.element_properties = element_properties or self._get_default_properties()
        self.domain_features = None
        self.phase_features = None

    def _get_default_elements(self):

        return ['Ni', 'Al', 'Co', 'Cr', 'Mo', 'Re', 'Ru', 'Ti', 'Ta', 'W', 'Hf',
                'Nb', 'Si', 'C', 'Y', 'Ce', 'B']

    def _get_default_properties(self):


        s_electrons = {'Ni': 2, 'Al': 2, 'Co': 2, 'Cr': 1, 'Mo': 1, 'Re': 2, 'Ru': 1,
                       'Ti': 2, 'Ta': 2, 'W': 2, 'Hf': 2, 'Nb': 1, 'Si': 2, 'C': 2,
                       'Y': 2, 'Ce': 2, 'B': 2}

        d_electrons = {'Ni': 8, 'Al': 0, 'Co': 7, 'Cr': 5, 'Mo': 5, 'Re': 5, 'Ru': 7,
                       'Ti': 2, 'Ta': 3, 'W': 4, 'Hf': 2, 'Nb': 4, 'Si': 0, 'C': 0,
                       'Y': 1, 'Ce': 1, 'B': 0}

        f_electrons = {'Ni': 0, 'Al': 0, 'Co': 0, 'Cr': 0, 'Mo': 0, 'Re': 14, 'Ru': 0,
                       'Ti': 0, 'Ta': 14, 'W': 14, 'Hf': 14, 'Nb': 0, 'Si': 0, 'C': 0,
                       'Y': 0, 'Ce': 1, 'B': 0}


        atomic_radii = {'Ni': 0.149, 'Al': 0.118, 'Co': 0.152, 'Cr': 0.166, 'Mo': 0.19,
                        'Re': 0.188, 'Ru': 0.178, 'Ti': 0.176, 'Ta': 0.2, 'W': 0.193,
                        'Hf': 0.208, 'Nb': 0.198, 'Si': 0.111, 'C': 0.067, 'Y': 0.212,
                        'Ce': 0.185, 'B': 0.087}


        gamma_energies = {'Ni': -287, 'Al': -284, 'Co': -284.169, 'Cr': -287, 'Mo': -267.585,
                          'Re': -278.817, 'Ru': -304.489, 'Ti': -256.9, 'Ta': -267.729,
                          'W': -282.13, 'Hf': -251.956, 'Nb': 0, 'Si': 0, 'C': 0, 'Y': 0,
                          'Ce': 1, 'B': 0}

        gamma_prime_energies = {'Ni': -303, 'Al': -258, 'Co': -325, 'Cr': -366, 'Mo': -493,
                                'Re': -467.5, 'Ru': -318.7, 'Ti': -468, 'W': 0, 'Ta': -425,
                                'Hf': 0, 'Nb': 0, 'Si': 0, 'C': 0, 'Y': 0, 'Ce': 1, 'B': 0}

        return {
            's_electrons': s_electrons,
            'd_electrons': d_electrons,
            'f_electrons': f_electrons,
            'atomic_radii': atomic_radii,
            'gamma_energies': gamma_energies,
            'gamma_prime_energies': gamma_prime_energies
        }

    def _normalize_features(self, features):

        features = np.array(list(features.values()))
        max_abs = np.max(np.abs(features))
        return features / max_abs if max_abs > 0 else features

    def _convert_to_tensor(self, data):

        if isinstance(data, pd.Series):
            data = data.values
        return torch.from_numpy(data).type(torch.float)

    def setup_domain_features(self, feature_names):

        self.domain_features = feature_names

    def setup_phase_features(self, gamma_phase_cols, gamma_prime_cols):

        self.phase_features = {
            'gamma': gamma_phase_cols,
            'gamma_prime': gamma_prime_cols
        }

    def normalize_data(self, dataframe):

        df_normalized = dataframe.copy()
        element_scaler = MinMaxScaler()
        df_normalized[self.target_elements] = element_scaler.fit_transform(df_normalized[self.target_elements])

        if self.domain_features:
            domain_scaler = MinMaxScaler()
            df_normalized[self.domain_features] = domain_scaler.fit_transform(df_normalized[self.domain_features])

        return df_normalized

    def create_embeddings(self, dataframe):

        n_samples = dataframe.shape[0]
        n_elements = len(self.target_elements)
        n_features = 11  # Total feature dimensions per node
        embeddings = np.zeros((n_samples, n_elements + 1, n_features))
        embeddings[:, :-1, 0] = dataframe[self.target_elements].values

        if self.phase_features:
            gamma_data = dataframe[self.phase_features['gamma']].values
            gamma_prime_data = dataframe[self.phase_features['gamma_prime']].values
            gamma_padded = np.concatenate([gamma_data, np.zeros((n_samples, n_elements - gamma_data.shape[1]))], axis=1)
            gamma_prime_padded = np.concatenate(
                [gamma_prime_data, np.zeros((n_samples, n_elements - gamma_prime_data.shape[1]))], axis=1)

            embeddings[:, :-1, 1] = gamma_padded
            embeddings[:, :-1, 2] = gamma_prime_padded

        property_keys = list(self.element_properties.keys())
        for idx, prop_key in enumerate(property_keys):
            normalized_props = self._normalize_features(self.element_properties[prop_key])
            # Broadcast to all samples
            embeddings[:, :-1, idx + 3] = np.tile(normalized_props, (n_samples, 1))

        if self.domain_features and len(self.domain_features) >= 2:
            condition_data = dataframe[self.domain_features[:2]].values
            # Replicate for all elements
            embeddings[:, :, [9, 10]] = np.repeat(condition_data[:, np.newaxis, :], n_elements + 1, axis=1)

        embeddings[:, -1, :-2] = 1

        return self._convert_to_tensor(embeddings)

    def apply_masking(self, embeddings):

        mask = torch.ones_like(embeddings)

        zero_composition = (embeddings[:, :, 0] == 0)
        mask[zero_composition] = 0

        return mask * embeddings

    def generate_element_ids(self, dataframe):

        def map_to_ids(composition_row):

            element_ids = [i if composition_row[i] != 0 else len(self.target_elements)
                           for i in range(len(self.target_elements))]

            element_ids.append(len(self.target_elements))
            return element_ids

        compositions = dataframe[self.target_elements].values
        id_mappings = [map_to_ids(row) for row in compositions]
        return torch.tensor(id_mappings, dtype=torch.long)

    def process_dataset(self, dataframe, target_column=None):

        normalized_df = self.normalize_data(dataframe)

        embeddings = self.create_embeddings(normalized_df)
        masked_embeddings = self.apply_masking(embeddings)

        element_ids = self.generate_element_ids(normalized_df)

        domain_tensor = None
        if self.domain_features:
            domain_tensor = self._convert_to_tensor(normalized_df[self.domain_features].values)

        target_tensor = None
        if target_column and target_column in dataframe.columns:

            target_values = np.log(dataframe[target_column].values)
            target_tensor = self._convert_to_tensor(target_values)

        return {
            'embeddings': masked_embeddings,
            'element_ids': element_ids,
            'domain_features': domain_tensor,
            'targets': target_tensor
        }