# @author: Erryan Sazany

import pandas as pd
from aspect_category import Aspects


class Evaluation:
    def __init__(self, dataframe):
        self.asp = Aspects()
        self.confusion_matrix = None

    def build_confusion_matrix(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        assert 'label' in dataframe.columns
        assert 'gold_aspect' in dataframe.columns

        self.confusion_matrix = pd.DataFrame(
            [[0] * len(self.asp.GOLD_ASPECTS) for _ in range(len(self.asp.GOLD_ASPECTS))],
            index=self.asp.GOLD_ASPECTS,  # actual label,
            columns=self.asp.GOLD_ASPECTS  # predicted label
        )

        def _fill_matrix(actual_label, predicted_label):
            self.confusion_matrix.loc[actual_label, predicted_label] += 1

        dataframe.apply(lambda row: _fill_matrix(row['label'], row['gold_aspect']), axis=1)

        return self.confusion_matrix

    def accuracy(self):
        correct_prediction = 0
        for gasp in self.asp.GOLD_ASPECTS:
            correct_prediction += self.confusion_matrix.loc[gasp, gasp]

        return correct_prediction / (self.confusion_matrix.sum().sum() * 1.0)

    def average_precision(self):
        total_precision = 0
        aspects = [a for a in self.asp.GOLD_ASPECTS if a != 'fail_to_decide_aspect']
        total_label = len(aspects)

        for gasp in aspects:
            predicted_as_gasp = self.confusion_matrix.loc[:, gasp].sum()
            correct_prediction_at_gasp = self.confusion_matrix.loc[gasp, gasp]
            precision_at_gasp = correct_prediction_at_gasp / (predicted_as_gasp * 1.0)
            total_precision += precision_at_gasp

        return total_precision / (total_label * 1.0)

    def average_recall(self):
        total_recall = 0
        aspects = [a for a in self.asp.GOLD_ASPECTS if a != 'fail_to_decide_aspect']
        total_label = len(aspects)

        for gasp in aspects:
            actual_label_is_gasp = self.confusion_matrix.loc[gasp, :].sum()
            correct_prediction_at_gasp = self.confusion_matrix.loc[gasp, gasp]
            recall_at_gasp = correct_prediction_at_gasp / (actual_label_is_gasp * 1.0)
            total_recall += recall_at_gasp

        return total_recall / (total_label * 1.0)
