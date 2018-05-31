# @author: Erryan Sazany

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
        nearest_aspect = ''
        highest_similarity = -1
        selected_keyword = ''

        if isinstance(keyword, list):
            for k in keyword:
                for gold_asp in sorted(self.INFERRED_ASPECTS.keys()):
                    for inf_asp in self.INFERRED_ASPECTS[gold_asp]:
                        cur_sim = emb_obj.similarity(k, inf_asp)

                        if cur_sim > highest_similarity:
                            highest_similarity = cur_sim
                            nearest_aspect = inf_asp
                            selected_keyword = k
        else:
            for gold_asp in sorted(self.INFERRED_ASPECTS.keys()):
                for inf_asp in self.INFERRED_ASPECTS[gold_asp]:
                    cur_sim = emb_obj.similarity(keyword, inf_asp)

                    if cur_sim > highest_similarity:
                        highest_similarity = cur_sim
                        nearest_aspect = inf_asp
                        selected_keyword = keyword

        return nearest_aspect, selected_keyword
