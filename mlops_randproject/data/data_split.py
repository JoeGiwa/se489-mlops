#Importing necessary packages
import pandas as pd

#data/music-dataset/features_3_sec.csv
df = pd.read_csv("/Users/mitanshi/Desktop/se489-mlops/data/music-dataset/features_3_sec.csv")
df2 = pd.read_csv("/Users/mitanshi/Desktop/se489-mlops/data/music-dataset/features_30_sec.csv")

from sklearn.preprocessing import LabelEncoder

# Encode y for features_3_sec
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Encode y1 for features_30_sec
y1_encoded = label_encoder.fit_transform(y1)

from sklearn.model_selection import train_test_split

# For the features_3_sec dataset
X = df.iloc[:,:-1]
y = df['label']  

#For the features_30_sec dataset
X1 = df2.iloc[:,:-1]
y1 = df2['label']  

# Splitting the data into training and testing sets(feautres_3_sec)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
# Splitting the data into training and testing sets(feautres_30_sec)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1_encoded, test_size=0.2, random_state=42)

