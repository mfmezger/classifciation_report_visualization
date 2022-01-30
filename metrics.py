import sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score,  average_precision_score, jaccard_score
from sklearn.metrics import brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score
#top_k_accuracy_score,

def calculate_multiclass_classification_metrics(y_pred=None, y_test=None):
    """https://scikit-learn.org/stable/modules/model_evaluation.html"""

    # calculate metrics.
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    bal_acc = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    # top_k_acc = top_k_accuracy_score(y_true=y_test, y_pred=y_pred, k=5)
    # avg_prec = average_precision_score(y_true=y_test, y_score=y_pred)
    # brier_score = brier_score_loss(y_true=y_test, y_prob=y_pred)
    
    f1_macro = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
    f1_micro = f1_score(y_true=y_test, y_pred=y_pred, average='micro')
    f1_weighted = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')

    # log = log_loss(y_true=y_test, y_pred=y_pred)
    prec_macro = precision_score(y_true=y_test, y_pred=y_pred, average='macro')
    prec_micro = precision_score(y_true=y_test, y_pred=y_pred, average='micro')
    prec_weighted = precision_score(y_true=y_test, y_pred=y_pred, average='weighted')

    recall_macro = recall_score(y_true=y_test, y_pred=y_pred, average='macro')
    recall_micro = recall_score(y_true=y_test, y_pred=y_pred, average='micro')
    recall_weighted = recall_score(y_true=y_test, y_pred=y_pred, average='weighted')

    roc_auc_macro = roc_auc_score(y_true=y_test, y_score=y_pred, average='macro')
    roc_auc_micro = roc_auc_score(y_true=y_test, y_score=y_pred, average='micro')
    roc_auc_weighted = roc_auc_score(y_true=y_test, y_score=y_pred, average='weighted')
    
    jaccard_macro = jaccard_score(y_true=y_test, y_pred=y_pred, average='macro')
    jaccard_micro = jaccard_score(y_true=y_test, y_pred=y_pred, average='micro')
    jaccard_weighted = jaccard_score(y_true=y_test, y_pred=y_pred, average='weighted')



    # create dict out of scores.
    scores = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "precision_macro": prec_macro,
        "precision_micro": prec_micro,
        "precision_weighted": prec_weighted,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "recall_weighted": recall_weighted,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_micro": roc_auc_micro,
        "roc_auc_weighted": roc_auc_weighted,
        "jaccard_macro": jaccard_macro,
        "jaccard_micro": jaccard_micro,
        "jaccard_weighted": jaccard_weighted,

    }

    return scores


def calculate_binary_classification_metrics(y_pred=None, y_test=None):
    """https://scikit-learn.org/stable/modules/model_evaluation.html"""

    # calculate metrics.
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    bal_acc = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    # top_k_acc = top_k_accuracy_score(y_true=y_test, y_pred=y_pred, k=5)
    avg_prec = average_precision_score(y_true=y_test, y_score=y_pred)
    brier_score = brier_score_loss(y_true=y_test, y_prob=y_pred)
    
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='binary')

    log = log_loss(y_true=y_test, y_pred=y_pred)
    prec = precision_score(y_true=y_test, y_pred=y_pred, average='binary')

    recall = recall_score(y_true=y_test, y_pred=y_pred, average='binary')

    roc_auc = roc_auc_score(y_true=y_test, y_score=y_pred, average='binary')
    
    jaccard = jaccard_score(y_true=y_test, y_pred=y_pred, average='binary')



    # create dict out of scores.
    scores = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "log_loss": log,
        "precision": prec,
        "recall": recall,
        "roc_auc": roc_auc,
        "jaccard": jaccard,
        

    }

    return scores


def binary_classification(y_pred=None, y_test=None):
    # if y_pred or y_test is not passed, throw an error.
    if y_pred is None or y_test is None:
        raise ValueError("y_pred or y_test is None.")
        
    # test if y_pred is binary, if not throw error
    if len(y_pred.unique()) != 2:
        raise ValueError("y_pred is not binary.")

    # test if y_test is as long as y_pred, if not throw error.
    if len(y_test) != len(y_pred):
        raise ValueError("y_test is not the same length as y_pred.")

    # calculate metrics.
    scores =calculate_classification_metrics(y_pred=y_pred, y_test=y_test)

    print(scores)

    # create confusion matrix.


    # save into pdf.




def multiclass_classification(y_pred=None, y_test=None):
    # if y_pred or y_test is None, throw an error.
    if y_pred is None or y_test is None:
        raise ValueError("y_pred or y_test is None.")

    # test if y_test is as long as y_pred, if not throw error.
    if len(y_test) != len(y_pred):
        raise ValueError("y_test is not the same length as y_pred.")

    # calculate metrics.
    scores = calculate_classification_metrics(y_pred=y_pred, y_test=y_test)
    
    print(scores)
    # create confusion matrix.


    # save into pdf.


def regression():
        # if y_pred or y_test is None, throw an error.
    if y_pred is None or y_test is None:
        raise ValueError("y_pred or y_test is None.")

    # test if y_test is as long as y_pred, if not throw error.
    if len(y_test) != len(y_pred):
        raise ValueError("y_test is not the same length as y_pred.")
    
    pass



def main():


    pass

if __name__ == "__main__":
    main()