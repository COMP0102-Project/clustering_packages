import pandas as pd


class ChexpertDatasetLoader:
    def __init__(self, targets):
        self.chexnet_targets = targets

    def get_dataset(self, CSVPATH, number_of_rows):
        '''
        Returns a dataframe containing the chexpert csv uploaded from PATH.
        Final column is appended with 'positive' indicating all pathologies present
        delimited by a semicolon.
        :param CSVPATH: path to chexpert csv
        :param number_of_rows: number of rows in the csv to return
        :return:
        '''
        full_train_df = pd.read_csv(CSVPATH)
        full_train_df['positive'] = full_train_df.apply(self._positive, axis=1).fillna('')
        full_train_df['positive'] = full_train_df['positive'].apply(lambda x: x.split(";"))
        return full_train_df.head(number_of_rows)

    def _positive(self, row):

        feature_list = []
        for feature in self.chexnet_targets:
            if row[feature] == 1:
                feature_list.append(feature)
        return ';'.join(feature_list)



