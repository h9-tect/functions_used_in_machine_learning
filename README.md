# function_used_in_machine_learning

### I created the most used function in machine learning projects i tried to arrange them from siple to complex 'please note this a structure of function maybe you have to modify it or importing necssary libraries to work better for you, please feel free to take any function and dont forget to start and fork if you found it usefuel'

here is our functions
- Data preparation functions: functions for data preprocessing such as renaming columns, removing columns, replacing values, sorting values, grouping values, removing duplicates, applying a function to a column, type casting, extracting part of a string, removing whitespace, concatenating columns, filtering rows and counting values.
- Date functions: functions for formatting dates, extracting parts of dates.
- Data loading functions: functions for reading data from various sources, such as CSV, Excel, JSON, and databases.
- Data cleaning functions: functions for handling missing values, removing duplicates, and dealing with outliers.
- Data transformation functions: functions for normalizing, scaling, and encoding categorical variables.
- Data splitting functions: functions for dividing data into training, validation and testing sets
- Data augmentation functions: functions for generating new data samples through techniques such as rotation, scaling, and flipping.
- Feature extraction functions: functions for extracting features such as image, text, audio, and video.
- Data reshaping functions: functions for reshaping data into the format required by a particular model, such as converting data into tensors.
- Data balancing functions: functions for balancing the dataset when there is a class imbalance.
- feature engineering ,Dimensionality reduction, Clustering,Anomaly detection,and Imputing Missing Values


### Data preparation functions
```

# rename columns 
def rename_columns(data, columns_mapping):
    data = data.rename(columns=columns_mapping)
    return data
```
```
# remove columns
def remove_columns(data, columns_to_remove):
    data = data.drop(columns_to_remove, axis=1)
    return data
```
```
# replace values
def replace_values(data, columns_mapping):
    data = data.replace(columns_mapping)
    return data
```
```
def sort_values(data, sort_by, ascending=True):
    data = data.sort_values(by=sort_by, ascending=ascending)
    return data
```
```
def group_values(data, group_by, agg_func):
    data = data.groupby(group_by).agg(agg_func)
    return data
```
```
def remove_duplicates(data):
    data = data.drop_duplicates()
    return data
```
```
def apply_func_to_col(data, col_name, func):
    data[col_name] = data[col_name].apply(func)
    return data
```
```
def type_cast_col(data, col_name, cast_type):
    data[col_name] = data[col_name].astype(cast_type)
    return data
```
```
def concatenate_cols(data, cols_to_concat, new_col_name):
    data[new_col_name] = data[cols_to_concat].apply(lambda x: ' '.join(x), axis=1)
    return data
```
```
def extract_string_part(data, col_name, start, end):
    data[col_name] = data[col_name].str[start:end]
    return data
```
```
def count_values(data, col_name):
    value_counts = data[col_name].value_counts()
    return value_counts
```
```
def filter_rows(data, filter_by):
    data = data[data[filter_by[0]] == filter_by[1]]
    return data
```
## Data loading functions
```
def load_csv(filepath):
    data = pd.read_csv(filepath)
    return data
```
```
def load_excel(filepath):
    data = pd.read_excel(filepath)
    return data
```
```
def load_json(filepath):
    data = pd.read_json(filepath)
    return data
```
```
def load_database(connection_string):
    conn = sqlite3.connect(connection_string)
    data = pd.read_sql_query("SELECT * FROM tablename", conn)
    return data
```
## Data cleaning functions
```
def deal_with_outliers(data):
    data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]
    return data
```
```
def remove_duplicates(data):
    data = data.drop_duplicates()
    return data
```
```
def handle_missing_values(data):
    data = data.fillna(data.mean())
    return data
```
## Data transformation functions
```
def encode_categorical_variables(data):
    data = pd.get_dummies(data, columns=['column1', 'column2'])
    return data
```
```
def scale_data(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data
```
```
def normalize_data(data):
    data = (data - data.min()) / (data.max() - data.min())
    return data
```
## Data splitting functions
```
def divide_data(data):
    X_train, X_test, y_train, y_test = train_test_split(data.drop('label', axis=1), data['label'], test_size=0.2)
    return X_train, X_test, y_train, y_test
```
## Data augmentation functions
```
def generate_new_data(data):
    data_gen = ImageDataGenerator(rotation_range=90, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)
    data_gen.fit(data)
    return data_gen
```
## Feature extraction functions
```
def extract_image_features(data):
    features = [np.mean(data), np.std(data)]
    return features
```
```
def extract_text_features(data):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(data).toarray()
    return features
```
```
def extract_audio_features(data):
    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=40)
    return mfccs
```
```
def extract_video_features(data):
    hog_features = hog(data, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), transform_sqrt=True)
    return hog_features
```
## Data reshaping functions
```
def reshape_data(data):
    data = data.reshape(-1, 28, 28, 1)
    return data
``` 

## Data balancing functions
```
def balance_dataset(data):
    data = RandomUnderSampler().fit_resample(data, data['label'])
    return data
```
## feature engineering 
This function takes in a dataframe and creates five new features. The first feature is the log of the 'column1' column. The second feature is the interaction between 'column2' and 'column3'. The third feature is the square of 'column4', the fourth feature is 'column5' with categorical variable and the fifth feature is the round of 'column1'.

```
import numpy as np

def engineer_features(data):
    # create new feature 1 - log of column1
    data['log_column1'] = np.log(data['column1'])
    # create new feature 2 - interaction between column2 and column3
    data['column2_x_column3'] = data['column2'] * data['column3']
    # create new feature 3 - square of column4
    data['square_column4'] = data['column4'] ** 2
    # create new feature 4 - column5 with categorical variable
    data['column5_category'] = pd.cut(data['column5'], bins=[0, 10, 20, 30], labels=['low', 'medium', 'high'])
    # create new feature 5 - round columns
    data['round_column1'] = data['column1'].round()
    return data

engineered_data = engineer_features(data)
```
## Dimensionality reduction
```
from sklearn.decomposition import PCA

def reduce_dimensions(data, n_components):
    pca = PCA(n_components=n_components)
    data = pca.fit_transform(data)
    return data

reduced_data = reduce_dimensions(data, n_components=5)
```
## Clustering
```
from sklearn.cluster import KMeans

def cluster_data(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    data['cluster'] = kmeans.predict(data)
    return data

clustered_data = cluster_data(data, n_clusters=3)
```
## Anomaly detection
```
from sklearn.ensemble import IsolationForest

def detect_anomalies(data):
    isolation_forest = IsolationForest(contamination=0.1)
    isolation_forest.fit(data)
    data['anomaly'] = isolation_forest.predict(data)
    data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})
    return data
```
anomalies_data = detect_anomalies(data)
## Imputing Missing Values
```
from sklearn.impute import SimpleImputer

def impute_missing_values(data):
    imputer = SimpleImputer(strategy='mean')
    data = imputer.fit_transform(data)
    return data

imputed_data = impute_missing_values(data)
```


