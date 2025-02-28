print("utils.py Loading")
if __name__ == "__main__":
    from plot_functions import *
else:
    from utils.plot_functions import *


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


def score_function(column, maximize=True):
    """ 
    Function that gets the datasets and modify the data to evaluate the scores of the different criterias
    - column: each value is a time series, if we want to minimize it, we have to invert it
    - mininimize: boolean, if we want to minimize the value
    """
    def invert_values(column):
        return [-x for x in column]

    if maximize:
        return column
    else:
        # Invert the values: the higher the value, the lower the score
        column = invert_values(column)
        # Get the minimum value of the column
        min_value = min([min(x) for x in column])
        # Add the minimum value to the column in order to have only positive values
        column = [x + abs(min_value) for x in column]
        return column

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
    """ 
    # le = life expectancy -> maximum is better
    # gdi = gender development index -> maximum is better
    # hdi = human development index -> maximum is better
    # eys = expected years of schooling -> maximum is better
    # poptotal = total population -> maximum is better
    # co2prod = production of co2 -> minimum is better
    # Pass through the score functions
    """
    print("Reading HDI dataset")
    PATH = "../data/HDI/HDR23-24_Composite_indices_complete_time_series.csv"
    data = pd.read_csv(PATH, encoding='latin1')

    years = [str(i) for i in range(1990, 2023)]

    fixed_col_names_to_keep = ['iso3']
    var_col_names_to_keep = ["co2_prod_", "hdi_", "le_", "gdi_", "eys_", "mys_"]

    col_names_to_keep = fixed_col_names_to_keep + [var + year for var in var_col_names_to_keep for year in years]

    data = data[col_names_to_keep]
    data = data.dropna()

     # Transform the variable columns into sequences
    for col in var_col_names_to_keep: 
        data = transform_cols_in_sequence(data, col, years)

    data["co2prod"] = score_function(data["co2prod"], maximize=False)

    # Remove the last 3 rows
    data = data.iloc[:-3]

    get_min_max_criteria(data, init=True)

    data.index = data["iso3"]
    data = data.drop(columns=["iso3"])

    return data

def read_stock_data(path):
    """ 
        Read the stock data from the csv file and return a dataframe with the following columns:
        name (index), open, high, low, close, volume
        each cell is the numpy array of the values of the stock over time
        It also returns the dates of the stocks (for plotting purposes)
    """

    def check_dates(dates):
        """ 
            Check that dates are the same for all stocks (necessary condition for applying the promethee method)
        """
        for i in range(1, len(dates)):
            if not np.array_equal(dates[i], dates[0]):
                return False
        return True

    def filter_by_dates(data):
        """ 
            The idea here is to only keep the stocks that have the same dates each time
        """
        filtered_data = data.copy()
        for i in range(len(data)):
            if not np.array_equal(data["date"].iloc[i], data["date"].iloc[0]):
                filtered_data = filtered_data.drop(data.index[i])
        return filtered_data

    # Read the data
    df = pd.read_csv(path)

    # Get the dates
    dates = df.groupby("Name").apply(lambda x: x["date"].values)
    
    # Create a new df with name as index and open, high, low, close, volume, dates as columns
    # Each cell is a np.array of the calues of the corresponding column for the stock
    data = df.groupby("Name").apply(lambda x: x[["open", "high", "low", "close", "volume"]].values)

    data = pd.DataFrame(data, index=data.index, columns=["data"])
    data["open"] = data["data"].apply(lambda x: x[:, 0])
    data["high"] = data["data"].apply(lambda x: x[:, 1])
    data["low"] = data["data"].apply(lambda x: x[:, 2])
    data["close"] = data["data"].apply(lambda x: x[:, 3])
    data["volume"] = data["data"].apply(lambda x: x[:, 4])

    # Add the dates
    dates = df.groupby("Name").apply(lambda x: x["date"].values)
    data["date"] = dates

    # Check that dates are the same for all stocks
    if not check_dates(data["date"].values):
        print("Dates are not the same for all stocks, a filtering will be applied")
        data = filter_by_dates(data)

        if not check_dates(data["date"].values):
           raise Exception("Dates are still not the same for all stocks")
        else:
            print("Data filtered successfully!")
            dates = data["date"].values[0]
        

    # Drop the useless column
    data = data.drop(columns=["data", "date"])
    return data, dates

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

def get_min_max_criteria(data, init=False):
    min_max = {}

    startcol = 0
    if init:
        startcol = 1

    for column in data.columns[startcol:]:
        min_max[column] = {"min": float('inf'), "max": float('-inf')}
        # Each column is a column of time series data
        column_data = data[column]

        for row in column_data:
            tempmin = row.min()
            tempmax = row.max()
            if tempmin < min_max[column]["min"]:
                min_max[column]["min"] = tempmin
            if tempmax > min_max[column]["max"]:
                min_max[column]["max"] = tempmax

        # Print the min and max values for each criteria
    for key, value in min_max.items():
        print(f"{key}: min={round(value.get('min'),4)}, max={round(value.get('max'),4)}")

# if main file
if __name__ == "__main__":
    print("utils.py loaded")