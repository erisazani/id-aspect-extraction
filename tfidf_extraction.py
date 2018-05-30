from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfKeywordExtractor:
    def __init__(self):
        vec = TfidfVectorizer()

    def fit_transform(self, df):
        df_tfidf = self.vec.fit_transform(df)
        return df_tfidf

    def extract(self, text):
        print text
