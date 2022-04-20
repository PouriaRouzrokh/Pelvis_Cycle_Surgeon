#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

# Standard built-in modules
import warnings

# Third-party modules
import pandas as pd
import sklearn.model_selection as skm

#-------------------------------------------------------------------------------
# Configurations
#-------------------------------------------------------------------------------

warnings.filterwarnings("ignore") 

#-------------------------------------------------------------------------------
# General functions
#-------------------------------------------------------------------------------

#-----------------------------------
# - F: is_notebook_running

def is_notebook_running():
    """Checks if the code is running on a notebook or a Python file.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

#-------------------------------------------------------------------------------
# Data helper functions
#-------------------------------------------------------------------------------
    
#-----------------------------------
# - F: split_data

def split_data(df: pd.DataFrame, 
               n_splits: int, 
               y_column: str=None, 
               group_column:str=None, 
               fold_column: str="Fold", 
               shuffle=False, 
               random_state=None):
    """Split a dataset into different folds, given a datadrame for the data.

    Args:
        df (pd.DataFrame): the data dataframe.
        n_splits (int): desired number of folds
        y_column (str, optional): the column to be used for stratification. 
            Defaults to None.
        group_column (str, optional): the column to be used for grouping. 
            Defaults to None.
        fold_column (str, optional): name of the fold column. 
            Defaults to "Fold".
        shuffle (bool, optional): whether to build folds using shuffling. 
            Defaults to False.
        random_state (_type_, optional): the random state to use. 
            Defaults to None.

    Returns:
        df (pd.DataFrame): the updated dataframe with the new fold column.
    """
    # Setting the random state
    if random_state is not None:
        shuffle = True
    elif shuffle and random_state is None:
        random_state = 42
    
    # Splitting the data
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    if y_column is None and group_column is None:
        splitter = skm.sklearnKFold(n_splits=n_splits, shuffle=shuffle, 
                                    random_state=random_state)
        print("Using simple KFold split...")
    elif y_column is not None and group_column is None:
        splitter = skm.StratifiedKFold(n_splits=n_splits, shuffle=shuffle, 
                                       random_state=random_state)
        print("Using StratifiedKFold split...")
    elif y_column is None and group_column is not None:
        splitter = skm.GroupKFold(n_splits=n_splits, shuffle=shuffle, 
                                  random_state=random_state)
        print("Using GroupKFold split...")
    elif y_column is not None and group_column is not None:
        splitter = skm.StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, 
                                            random_state=random_state)
        print("Using StratifiedGroupKFold split...")
    
    # Adding the fold column to the dataframe.
    df[fold_column] = 0
    for fold_idx, (_, val_index) in enumerate(
        splitter.split(df, y=df[y_column].tolist() \
            if y_column is not None else None, 
            groups=df[group_column].tolist() \
                if group_column is not None else None)):
        df.loc[val_index,fold_column]=fold_idx
    return df

