from scipy import spatial


class WordEmbedding():
    def __init__(self):
        self.EMBEDDING_FILE_PATH = {
            'fasttext-id': 'embeddings/kyubyong-fasttext.vec',
            'word2vec-id': 'embeddings/kyubyong-word2vec-cleansed.tsv'
        }

        self.embedding_vector = {}

    def load_embedding(self, emb_type='fasttext-id'):
        assert emb_type in ['fasttext-id', 'word2vec-id']

        with open(self.EMBEDDING_FILE_PATH[emb_type], 'rU') as f:
            for line in f:
                fields = line.split(' ')
                cur_word = fields[0]
                cur_vector = list(map(float, fields[1:-1]))

                self.embedding_vector[cur_word] = cur_vector

        return self.embedding_vector

    def similarity(self, word1, word2):
        # Using cosine similarity
        assert len(self.embedding_vector) > 0

        if word1 in self.embedding_vector and word2 in self.embedding_vector:
            return 1 - spatial.distance.cosine(
                self.embedding_vector[word1], self.embedding_vector[word2]
            )
        else:
            return 0
