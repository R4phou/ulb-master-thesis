import pandas as pd
from plot import *

def read_data_2(path):
    data = pd.read_csv(path)
    return data


def normalize_list(lst):
    """
        Normalize a list
    """
    return [(x) / (max(lst)) for x in lst]


def normalize_matrix_columns(matrix):
    """
        Normalize the columns of a matrix
    """
    for col in range(matrix.shape[1]):
        matrix[:, col] = normalize_list(matrix[:, col])
    return matrix


def plot_heatmap(data, title):
    """
        Plot a heatmap of the data
        data : matrix to plot
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(data, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_title(title)
    plt.show()


def normalize_columns(data1, data2):
    """
    Normalize each columns of the dataframes data1 and data2 independently except the date column 
    such that the values are in [-1, 1]
    """
    data1_norm = data1.copy()
    data2_norm = data2.copy()
    for column in data1.columns[1:]:
        # Use the mean and standard deviation of the column of both dataframes to normalize
        mean = (data1[column].mean() + data2[column].mean()) / 2
        std = (data1[column].std() + data2[column].std()) / 2

        data1_norm[column] = (data1[column] - mean) / std
        data2_norm[column] = (data2[column] - mean) / std
    return data1_norm, data2_norm


def transform_cols_in_sequence(data, col_id, range_of_years):
    cols = [col_id + year for year in range_of_years]

    # Transform col_id to remove "_"
    col_id = col_id.replace("_", "")

    # Add a new column named col_id (will be the sequence)
    data[col_id] = None

    for i, row in data.iterrows(): # For each row
        sequence = []
        for col in cols: # For each column
            sequence.append(row[col])
        sequence = np.array(sequence) # Transform the sequence into a numpy array
        data.at[i, col_id] = sequence # Replace the value of the new column with the sequence

    # Drop the columns that were transformed
    data = data.drop(columns=cols)

    return data

def read_data():
    PATH = "../data/HDI/HDR23-24_Composite_indices_complete_time_series.csv"
    data = pd.read_csv(PATH, encoding='latin1')
    data = data.dropna()

    years = [str(i) for i in range(1990, 2023)]

    fixed_col_names_to_keep = ['iso3']
    var_col_names_to_keep = ["co2_prod_", "pop_total_", "hdi_", "le_", "gdi_", "eys_"]

    col_names_to_keep = fixed_col_names_to_keep + [var + year for var in var_col_names_to_keep for year in years]

    data = data[col_names_to_keep]

     # Transform the variable columns into sequences
    for col in var_col_names_to_keep: 
        data = transform_cols_in_sequence(data, col, years)

    return data


def scale_column(data, col):
    """
    Scale a column between 0 and 1 knowing that each row is a sequence np.array
    Scaling using the min and max of the column and not of the sequence each time!
    """
    min_val = data[col].apply(lambda x: x.min()).min()
    max_val = data[col].apply(lambda x: x.max()).max()
    data[col] = data[col].apply(lambda x: (x - min_val) / (max_val - min_val))
    return data

def scale_data(data):
    """
    Scale all the columns that are sequences (all except the first one, here it is 'iso3')
    """
    for col in data.columns[1:]:
        data = scale_column(data, col)
    return data

