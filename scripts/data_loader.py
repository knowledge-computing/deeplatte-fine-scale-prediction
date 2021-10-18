import os
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataObj:
    def __init__(self, label_mat, dynamic_x, static_x,
                 dynamic_feature_names, static_feature_names,
                 mapping_mat, **kwargs):
        """

        Args:
            label_mat (ndarray): (696,1,92,76)
            dynamic_x (ndarray): (696,14,92,76)
            static_x (ndarray): (1,82,92,76)
            dynamic_feature_names (list): 14, list of dynamic features, such as 'temperature', 'dew_point', 'humidity' .. etc.
            static_feature_names (list): 82, list of static features, such as 'landuse_a_allotments', 'landuse_a_military' .. etc.
            mapping_mat (ndarray): (Pixel width, height) = (92,76)


        """

        self.label_mat = label_mat
        self.dynamic_x = self.norm(dynamic_x)
        self.static_x = self.norm(static_x)

        self.dynamic_feature_names = dynamic_feature_names
        self.static_feature_names = static_feature_names
        self.features_names = dynamic_feature_names + static_feature_names
        self.mapping_mat = mapping_mat
        self.num_features = len(self.dynamic_feature_names) + len(self.static_feature_names)
        self.num_dynamic_features = len(self.dynamic_feature_names)
        self.num_static_features = len(self.static_feature_names)
        self.num_times, _, self.num_rows, self.num_cols = self.dynamic_x.shape

        # set up training
        self.use_test = kwargs.get('use_test', False)
        self.train_size = kwargs.get('train_size', 0.8)
        self.test_size = kwargs.get('test_size', 0.2) * self.use_test
        self.train_loc, self.val_loc, self.test_loc = self.split_train_val_test_locations()
        self.train_y = self.set_label_mat(self.train_loc)
        self.val_y = self.set_label_mat(self.val_loc)
        self.test_y = self.set_label_mat(self.test_loc)

    def set_label_mat(self, locations):
        """ return a new label mat containing only given locations """

        mat = np.full(self.label_mat.shape, np.nan)
        for loc in locations:
            r, c = np.where(self.mapping_mat == loc)
            mat[..., r[0], c[0]] = self.label_mat[..., r[0], c[0]]
        return mat

    def split_train_val_test_locations(self):
        """ split labeled locations into train, val, (test) set
        return: train_loc, val_loc, test_loc -> list, list, list """

        #  find locations that have more than 0.01 x num_times labels
        candidate_mat = np.sum(~np.isnan(self.label_mat), axis=(0, 1)) == self.num_times

        sub_region_locations = [
            self.mapping_mat[np.where(candidate_mat[0:self.num_rows // 2, 0:self.num_cols // 2])].tolist(),
            self.mapping_mat[np.where(candidate_mat[0:self.num_rows // 2, self.num_cols // 2:])].tolist(),
            self.mapping_mat[np.where(candidate_mat[self.num_rows // 2:, 0:self.num_cols // 2])].tolist(),
            self.mapping_mat[np.where(candidate_mat[self.num_rows // 2:, self.num_cols // 2:])].tolist()
        ]

        train_loc, val_loc, test_loc = [], [], []
        for loc in sub_region_locations:
            if self.use_test:
                train, test = train_test_split(loc, test_size=self.test_size, random_state=1234)
                train, val = train_test_split(train, train_size=self.train_size, random_state=1234)
                test_loc += test_loc
            else:
                train, val = train_test_split(loc, train_size=self.train_size, random_state=1234)
            train_loc += train
            val_loc += val
        return sorted(train_loc), sorted(val_loc), sorted(test_loc)

    @staticmethod
    def norm(mat):
        num_times, num_features, num_rows, num_cols = mat.shape
        mat_2d = np.moveaxis(mat, 1, -1)  # [num_times x num_rows x num_cols x num_features]
        mat_2d = mat_2d.reshape(-1, num_features)  # [num_samples x num_features]
        norm_mat = StandardScaler().fit_transform(mat_2d)
        norm_mat = norm_mat.reshape(num_times, num_rows, num_cols, num_features)
        norm_mat = np.moveaxis(norm_mat, -1, 1)  # [num_times x num_features x num_rows x num_cols]
        return norm_mat


def load_data_from_file(data_path):
    """ load data from a file """

    if not os.path.isfile(data_path):
        raise FileNotFoundError

    data = np.load(data_path)
    dynamic_feature_names, static_feature_names = list(data['dynamic_features']), list(data['static_features'])
    data_obj = DataObj(label_mat=data['label_mat'],
                       dynamic_x=data['dynamic_mat'],
                       static_x=data['static_mat'],
                       dynamic_feature_names=dynamic_feature_names,
                       static_feature_names=static_feature_names,
                       mapping_mat=data['mapping_mat'])
    return data_obj
