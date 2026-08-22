import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score


def evaluate_binary_interaction(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(y_true, y_score)
    real_score = np.asmatrix(y_true)
    predict_score = np.asmatrix(y_score)
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    thresholds = sorted_predict_score[np.int32(len(sorted_predict_score) * np.arange(1, 1000) / 1000)]
    thresholds = np.asmatrix(thresholds)
    predict_score_matrix = np.tile(predict_score, (thresholds.shape[1], 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    tp = predict_score_matrix.dot(real_score.T)
    fp = predict_score_matrix.sum(axis=1) - tp
    fn = real_score.sum() - tp
    tn = len(real_score.T) - tp - fp - fn
    recall_list = tp / (tp + fn)
    f1_score_list = 2 * tp / (len(real_score.T) + tp - tn)
    accuracy_list = (tp + tn) / len(real_score.T)
    max_index = np.argmax(f1_score_list)
    recall_at_f1 = recall_list[max_index][0, 0]
    f1_at_f1 = f1_score_list[max_index][0, 0]
    precision_at_f1 = f1_at_f1 * recall_at_f1 / (2 * recall_at_f1 - f1_at_f1 + 1e-12) if (2 * recall_at_f1 - f1_at_f1) > 1e-8 else 0.0
    return auc, aupr, f1_score_list[max_index][0, 0], accuracy_list[max_index][0, 0], recall_list[max_index][0, 0], precision_at_f1
