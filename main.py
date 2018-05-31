# @author: Erryan Sazany

import pandas as pd

from aspect_category import Aspects
from dataset import Dataset
from evaluation import Evaluation
from rake_extraction import RakeKeywordExtractor
from tfidf_extraction import TfidfKeywordExtractor
from word_embedding import WordEmbedding


def run_experiment_with_rake():
    print "\nBegin experiment using RAKE algorithm..."
    # RAKE: predict keyword dengan RAKE, ambil words dengan RAKE skor tertinggi
    rake = RakeKeywordExtractor()
    tweets_rake['keyword'] = tweets_rake.apply(
        lambda t: rake.extract_keyword(rake.extract_candidates(t['text'], incl_scores=True)), axis=1
    )

    # RAKE: infer aspect dengan aspect mapping, dengan similarity terbesar
    tweets_rake['selected_keyword'] = tweets_rake.apply(
        lambda t: asp.find_nearest_inferred_aspect(t['keyword'], emb)[1], axis=1
    )
    tweets_rake['inferred_aspect'] = tweets_rake.apply(
        lambda t: asp.find_nearest_inferred_aspect(t['keyword'], emb)[0], axis=1
    )
    tweets_rake['gold_aspect'] = tweets_rake.apply(lambda t: asp.INVERTED_ASPECTS[t['inferred_aspect']], axis=1)

    tweets_rake.to_csv('dump/result_rake.csv', encoding='utf-8', index=False)

    # RAKE: Evaluasi dengan accuracy
    eva_rake = Evaluation(tweets_rake)
    conf_matrix = eva_rake.build_confusion_matrix(tweets_rake)
    print "Confusion matrix:"
    print conf_matrix
    print "Accuracy using RAKE algorithm: {}".format(eva_rake.accuracy())
    print "Average Precision using RAKE algorithm: {}".format(eva_rake.average_precision())
    print "Average Recall using RAKE algorithm: {}".format(eva_rake.average_recall())


def run_experiment_with_tfidf(tweets_tfidf):
    print "\nBegin experiment using TF-IDF weighting algorithm..."
    # TF-IDF: cari keyword dengan TF-IDF, ambil yang single word aja dengan bobot tertinggi
    tfidf = TfidfKeywordExtractor()
    tfidf_weight = tfidf.fit_transform(tweets_tfidf)
    tfidf_weight['keyword'] = tfidf_weight.idxmax(axis=1)

    # MUST BE after extracting keyword
    # OTHERWISE, the keyword will be "tweet_no" for all tweets
    tfidf_weight = tfidf_weight.reset_index().rename(columns={'index': 'tweet_no'})
    tfidf_weight['tweet_no'] = tfidf_weight['tweet_no'] + 1
    tfidf_weight = tfidf_weight[['tweet_no', 'keyword']]
    tfidf_weight.to_csv('tfidf_keyword.csv', encoding='utf-8', index=False)

    tweets_tfidf = tweets_tfidf.reset_index().rename(columns={'index': 'tweet_no'})
    tweets_tfidf['tweet_no'] = tweets_tfidf['tweet_no'] + 1
    tweets_tfidf.to_csv('tweets_tfidf.csv', encoding='utf-8', index=False)

    tweets_tfidf = pd.merge(tweets_tfidf, tfidf_weight,
                            how='left', on='tweet_no')
    tweets_tfidf.to_csv('tweets_tfidf_after_merge.csv', encoding='utf-8', index=False)

    # TF-IDF: infer aspect dengan aspect mapping, dengan similarity terbesar
    tweets_tfidf['selected_keyword'] = tweets_tfidf.apply(
        lambda t: asp.find_nearest_inferred_aspect(t['keyword'], emb)[1], axis=1
    )
    tweets_tfidf['inferred_aspect'] = tweets_tfidf.apply(
        lambda t: asp.find_nearest_inferred_aspect(t['keyword'], emb)[0], axis=1
    )
    tweets_tfidf['gold_aspect'] = tweets_tfidf.apply(lambda t: asp.INVERTED_ASPECTS[t['inferred_aspect']], axis=1)

    tweets_tfidf.to_csv('dump/result_tfidf.csv', encoding='utf-8', index=False)

    # RAKE: Evaluasi dengan accuracy
    eva_tfidf = Evaluation(tweets_tfidf)
    conf_matrix = eva_tfidf.build_confusion_matrix(tweets_tfidf)
    print "Confusion matrix:"
    print conf_matrix
    print "Accuracy using TF-IDF weighting algorithm: {}".format(eva_tfidf.accuracy())
    print "Average Precision using TF-IDF weighting algorithm: {}".format(eva_tfidf.average_precision())
    print "Average Recall using TF-IDF weighting algorithm: {}".format(eva_tfidf.average_recall())





# baca dataset
# preprocess data: buang punctuation
data = Dataset()
data.load_dataset()
data.cleanse_dataset()
data.build_dictionaries()

tweets = data.get_dataset()
tweets_rake = tweets.copy()
tweets_tfidf = tweets.copy()

emb = WordEmbedding()
emb_vec = emb.load_embedding(emb_type='fasttext-id')

asp = Aspects()

run_experiment_with_rake()
run_experiment_with_tfidf(tweets_tfidf)
