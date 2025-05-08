#Importing necessary packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/lajja/Desktop/se489-mlops/data/music-dataset/features_3_sec.csv")
df2 = pd.read_csv("C:/Users/lajja/Desktop/se489-mlops/data/music-dataset/features_30_sec.csv")
df.head()

df2.head()

# Data type of the data
df.dtypes

# Let's simulate loading the data and calculating descriptive statistics
# Display basic statistics of the dataset
df.describe()

# Shape of the data
df.shape

print("Columns containing missing values", list(df.columns[df.isnull().any()]))

from sklearn.preprocessing import LabelEncoder

# Assuming df.iloc[:, :-1] contains the feature columns and df.iloc[:, -1] contains the genre labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]  # Genre labels

# For the 30 sec dataset 
X1 = df2.iloc[:, :-1]
y1 = df2.iloc[:, -1] 

# Initialize the label encoder
label_encoder = LabelEncoder()

# Fit the encoder on the full dataset's labels to capture all possible classes
y_encoded = label_encoder.fit_transform(y)
y1_encoded = label_encoder.fit_transform(y1)

#features
print(df.iloc[:,:-1])

# Drop the column 'filename' as it is no longer required for training
df.drop(labels="filename", axis=1, inplace=True) 

df2.drop(labels="filename", axis=1, inplace=True)