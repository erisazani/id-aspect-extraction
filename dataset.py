import pandas as pd


class Dataset:
    def __init__(self):
        self.train_set = None
        self.vocab_index = {}
        self.index_vocab = {}
        self.vocab_length = -1

    def load_dataset(self):
        self.train_set = pd.read_csv('data/dataset_annotated.csv', encoding='ISO-8859-1')
        # print(train_set.head(5))

    def cleanse_dataset(self):
        # buang punctuation
        self.train_set['text'] = self.train_set['text'].str.replace(r'[\`\-\=]', '')
        self.train_set['text'] = self.train_set['text'].str.replace(r'[\~\!\@\#\$\%\^\&\*\(\_\+]', '')
        self.train_set['text'] = self.train_set['text'].str.replace(r'[\[\]\{\}\\\|\;\'\:\"\,\.\/\<\>\?]', '')

    def build_dictionaries(self):
        # Constructing dictionary: Vocab --> Index
        self.vocab_index = {}
        all_tweets = self.train_set['text'].tolist()
        index_counter = 1
        for t in all_tweets:
            t = t.lower()
            cur_words = t.split(' ')
            for w in cur_words:
                if w not in self.vocab_index:
                    self.vocab_index[w] = index_counter
                    index_counter += 1

        # Constructing dictionary: Index --> Vocab
        self.index_vocab = {}
        for w, i in self.vocab_index.items():
            self.index_vocab[i] = w

        self.vocab_length = len(self.vocab_index.keys())

    def get_dataset(self):
        return self.train_set

    def vocab_to_index(self):
        return self.vocab_index

    def index_to_vocab(self):
        return self.index_vocab



