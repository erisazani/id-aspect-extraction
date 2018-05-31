# @author: Erryan Sazany

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfKeywordExtractor:
    def __init__(self):
        self.vec = TfidfVectorizer()

    def fit_transform(self, df):
        assert isinstance(df, pd.DataFrame)

        tweets = df['text'].tolist()
        weight_matrix = self.vec.fit_transform(tweets).todense()
        feature_names = self.vec.get_feature_names()

        weight_list = weight_matrix.tolist()

        feature_weight = pd.DataFrame(weight_list, columns=feature_names)

        return feature_weight

    def extract(self, text):
        print text
