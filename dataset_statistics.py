import sklearn
import pandas as pd
import seaborn as sns
from seaborn import histplot
import numpy as np


def calculate_dataset_statistics(dataset=None, class_id=None, x=None, y=None):
    """Input can be either dataframe or x and y."""
    if isinstance(dataset, pd.DataFrame):
        if class_id == None:
            # assume that the last column is the class
            class_id = dataset.columns[-1]


        dataset_stats = {
            "num_rows": dataset.shape[0],
            "num_cols": dataset.shape[1],
            "num_nulls": dataset.isnull().sum().sum(),
            "num_unique_values": dataset.nunique().sum(),
            "num_unique_values_per_column": dataset.nunique(),
            "num_duplicates": dataset.duplicated().sum(),
            "num_duplicates_per_column": dataset.duplicated(),
            "num_nan": np.isnan(dataset).sum(),
            "num_classes": len(dataset[class_id].unique()),
            
            }
    # check if x and y are passed
    elif x is not None and y is not None:
        print("Calculating dataset statistics from x and y")

        # convert numpy array to pandas dataframe.
        dataset = pd.DataFrame(x)
        dataset_stats = {
            "num_rows": dataset.shape[0],
            "num_cols": dataset.shape[1],
            "num_nan": np.isnan(dataset).sum().sum(),
            "num_classes": len(np.unique(y)),
        }

    return dataset_stats
    

def create_dataset_plots(dataset=None, class_id=None, x=None, y=None):
    if isinstance(dataset, pd.DataFrame):
        if class_id == None:
            # assume that the last column is the class
            class_id = dataset.columns[-1]

    # check if x and y are passed
    elif x is not None and y is not None:
        print("Calculating dataset statistics from x and y")

        # convert numpy array to pandas dataframe.
        dataset = pd.DataFrame(x)
        # add y to the dataframe
        dataset["class"] = y
        class_id = "class"
    
    # create a histogramm plot for the classes.
    histplot(data=dataset, x="class", y="count")
    

    # create plots for the different variables and their distributions
    for col in dataset.columns:
        if col != class_id:
            sns.distplot(dataset[col])
    
    # create a pdf containing the plots.
    


    pass