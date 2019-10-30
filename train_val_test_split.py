"""
This function is used to easily split a pandas DataFrame into six datasets:
* X_train:  An X Training DataFrame
* X_val:    An X Validation DataFrame
* X_test:   An X Testing DataFrame
* y_train:  A Y Training DataFrame
* y_val:    A Y Validation DataFrame
* y_test:   A Y Testing DataFrame

Make sure you are inputting a pandas DataFrame objects.

train_val_test_split(input_df, targ, feat, v_size = 0.25, t_size = 0.25, r_state = 138)

  input_df :  The pandas DataFrame object to be split up

  targ :      Name of column to be predicted, used for the y_train, y_val and
              y_test DataFrames

  feat:       List of columns to be included as features, used in the X_train,
              X_val and X_test DataFrames

  v_size:     A float between 0.0 and 1.0, represents the proportion of X_Train
              to include in X_Val

  t_size:     A float between 0.0 and 1.0, represents the proportion of input_df
              to include in X_Test

  r_state:    The seed used by the random number generator used in the
              train/test split, and the train/val split

Returns: (X_train, y_train, X_val, y_val, X_test, y_test)
"""

# Used for the pandas DataFrames:
import pandas as pd

# Used for the train/test splits:
from sklearn.model_selection import train_test_split

def train_val_test_split(input_df, targ, feat, v_size = 0.25, t_size = 0.25, r_state = 138):

  # Split up input_df into df_train, df_val and df_test
  df = input_df.copy()
  df_train, df_test = train_test_split(df, test_size = t_size, random_state = r_state)
  df_train, df_val = train_test_split(df, test_size = v_size, random_state = r_state)

  # Converts the split DataFrames into features DataFrames and target DataFrames
  X_train = df_train[feat]
  X_val = df_val[feat]
  X_test = df_test[feat]
  y_train = df_train[targ]
  y_val = df_val[targ]
  y_test = df_test[targ]

  # Returns the finished DataFrames:
  return (X_train, y_train, X_val, y_val, X_test, y_test)
