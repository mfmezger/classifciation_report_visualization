import sklearn
import pandas as pd
import numpy as np


def create_dataset_plots(dataset=None, class_id=None, x=None, y=None):
    pass

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


    pass

def multiclass_classification(y_pred=None, y_test=None):
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