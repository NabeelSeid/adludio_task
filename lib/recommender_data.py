import look_up_table as look_up_table
import os
import sys
import pandas as pd
import json
import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
data_dir = this_dir + '../data/'
sys.path.append(this_dir)


class RECCOMEND_DATA():
    """
    This is a class for operation on the recomanded data.

    Attributes:
        data (dict): The dataset for games.
        game_features (dict): Collection of game features.
    """

    def __init__(self):
        """
        The constructor for RECCOMEND_DATA class.

        Attributes:
            data (dict): The dataset for games.
            game_features (dict): Collection of game features.
        """
        # load the data for the recommendation
        self.data = self.update_data()
        self.game_features = self.load_game_features()

    def update_data(self):
        """
        Reads dataset.csv and converts it to a dictinary format.

        Returns:
            dict: A dictinary that contains all games.
        """
        # connec to S3 get latest data
        df = pd.read_csv(data_dir + 'dataset.csv')
        dict_ = {}
        for ky in df.keys():
            dict_[ky] = df[ky].values
        return dict_

    def load_game_features(self):
        """
        Read and return game features

        Return:
            dict: A dictinary that contains game features
        """
        return json.load(open(data_dir + 'features.json', 'r'))

    def add_design_data(self):
        """
        adds additional design data to the data strucutre
        """
        pass

    def list_input_keys_values(self, input_columns=look_up_table.COLUMN_INPUTS):
        """ 
        returns a list of input column names, and potential selectable values, gathered from the data file.
        """
        res = {}
        for ky in input_columns:
            if ky in self.data:
                res[ky] = np.sort(np.unique(self.data[ky])).tolist()
        return res

    def extract_data_slice(self, keys_vals_dict):
        """
        Make a slice of self.data using if the keys and values found in keys_vals_dict are present.

        Parameters:
            keys_vals_dict (dict): A dictinary, key-value pairs used for matching values in self.data

        Returns:
            list: A boolean list with true value for matching rows
        """

        # extract keys for game dataset self.data
        data_keys = list(self.data.keys())
        # create all ones array with length same as self.data
        ind = np.ones(len(self.data[data_keys[0]]), dtype=bool)
        if keys_vals_dict is not None:
            for ky in keys_vals_dict:
                # ind_key = np.zeros(len(ind), dtype=bool)
                if ky in self.data:
                    # iterativly checks if a row exist in self.data that match
                    # a key-value pair in keys_vals_dict
                    ind_key = [i in keys_vals_dict[ky] for i in self.data[ky]]
                # multiplie ind with ind_key to replace unmatched rows with 0's
                ind *= np.array(ind_key)
        return ind

    def sum_events(self, ind_rows, matching_keys, event_list):
        """sum events in event_list group by matching_keys, for data with indicies given by ind_rows
        """
        if isinstance(matching_keys, list) is False:
            matching_keys = [matching_keys]
        res = {}

        # identifies the unique things to group by
        mtch_keys_arr = np.array(
            self.data[matching_keys[0]][ind_rows], dtype=str)
        if len(matching_keys) > 1:
            for ky in matching_keys[1:]:
                # add the uniq
                mtch_keys_arr += np.array(self.data[ky][ind_rows], dtype=str)

        # for each unique machting key entry
        for u_key in np.unique(mtch_keys_arr):
            ind_key = mtch_keys_arr == u_key
            print(ind_key)
            res[u_key] = {}
            # sum all events in event_list
            for event in event_list:
                res[u_key][event] = np.sum(self.data[event][ind_rows][ind_key])
        return res

    def calculates_rates(self, arr1, arr2):
        """calculates ratios, with careful error handling
        """
        #print('arr1, arr2', type(arr1), arr2, file=sys.stdout)

        #div = np.nan
        #ind = np.isfinite(arr1) * np.isfinite(arr2) * (arr2>0)
        # div[ind] = arr1[ind] / arr2[ind]
        div = arr1 / arr2
        return div

    def calculate_score(self, vals):
        """
        determines a score by mutliplying values together
        """
        score = np.prod(vals)

        return score

    def extract_game_features(self, game_id=None, feature_list=None):
        """Return a list of features and values for a set of game_ids.
        """

        if feature_list is None:
            if game_id in self.game_features:
                feature_list = list(self.game_features[game_id].keys())

        if isinstance(feature_list, list) is False:
            feature_list = [feature_list]

        game_feats = {}
        if game_id in self.game_features:
            for feature in feature_list:
                if feature in self.game_features[game_id]:
                    game_feats[feature] = self.game_features[game_id][feature]

        return game_feats

    def extract_game_version(self, game_id=None):
        """
        Return a version for a game_id
        """
        return self.data['version'][np.where(self.data['first_key'] == game_id)]
