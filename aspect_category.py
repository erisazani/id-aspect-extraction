class Aspects:
    def __init__(self):
        self.GOLD_ASPECTS = ['harga', 'daya tarik', 'fasilitas/akomodasi', 'makanan/minuman', 'fail_to_decide_aspect']

        # Gold Aspects --> Inferred Aspects
        self.INFERRED_ASPECTS = {
            'harga': ['harga'],
            'daya tarik': ['wahana', 'atraksi', 'keramaian', 'acara'],
            'fasilitas/akomodasi': ['kebersihan', 'servis', 'pelayanan', 'cuaca', 'jarak', 'transportasi'],
            'makanan/minuman': ['makanan', 'minuman'],
            'fail_to_decide_aspect': ['']
        }

        # Inferred Aspects --> Gold Aspects
        self.INVERTED_ASPECTS = {}
        for gold_asp in self.INFERRED_ASPECTS:
            for inf_asp in self.INFERRED_ASPECTS[gold_asp]:
                self.INVERTED_ASPECTS[inf_asp] = gold_asp

    def find_nearest_inferred_aspect(self, keyword, emb_obj):
        cur_nearest_aspect = ''
        cur_highest_similarity = -1
        cur_selected_keyword = ''

        for k in keyword:
            for gold_asp in sorted(self.INFERRED_ASPECTS.keys()):
                for inf_asp in self.INFERRED_ASPECTS[gold_asp]:
                    cur_sim = emb_obj.similarity(k, inf_asp)

                    if cur_sim > cur_highest_similarity:
                        cur_highest_similarity = cur_sim
                        cur_nearest_aspect = inf_asp
                        cur_selected_keyword = k

        return cur_nearest_aspect, cur_selected_keyword
