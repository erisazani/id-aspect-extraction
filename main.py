from aspect_category import Aspects
from dataset import Dataset
from evaluation import Evaluation
from rake_extraction import RakeKeywordExtractor
from tfidf_extraction import TfidfKeywordExtractor
from word_embedding import WordEmbedding

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

tweets_rake.to_csv('dump/result_rake.csv', encoding='utf-8')

# RAKE: Evaluasi dengan accuracy
eva_rake = Evaluation(tweets_rake)
conf_matrix = eva_rake.build_confusion_matrix(tweets_rake)
print "Confusion matrix:"
print conf_matrix
print "Accuracy using RAKE algorithm: {}".format(eva_rake.accuracy())
print "Average Precision using RAKE algorithm: {}".format(eva_rake.average_precision())
print "Average Recall using RAKE algorithm: {}".format(eva_rake.average_recall())


# TF-IDF: predict keyword dengan TF-IDF, ambil yang single word aja dengan bobot tertinggi


# TF-IDF: infer aspect dengan aspect mapping, dengan similarity terbesar


# RAKE: Evaluasi dengan accuracy
