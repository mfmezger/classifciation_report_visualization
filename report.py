import sklearn
import pandas as pd
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


def main():


    pass

if __name__ == "__main__":
    main()