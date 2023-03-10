# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle

# Load raw data into dataset
df = pd.read_csv('../data/raw_data/shelter_outcomes.csv', index_col = 0)

# Drop redundant columns
df = df.drop(['date_of_birth'], axis = 1)
df = df.drop(['date_of_intake'], axis = 1)
df = df.drop(['date_of_outcome'], axis = 1)

# Drop ID columns
df = df.drop(['animal_id'], axis = 1)
df = df.drop(['name'], axis = 1)

# Label encode target variable and save the labels
pd.factorize(df.outcome_type)
df['out_type_ec'], y_labels = pd.factorize(df.outcome_type)
df = df.drop(['outcome_type'], axis = 1)

# Convert binary categorical columns to boolean
is_dog = [True if i == 'Dog' else False for i in df.animal_type]
df['is_dog'] = is_dog
df = df.drop(['animal_type'], axis = 1)

is_male = [True if i == 'Male' else False for i in df.sex]
df['is_male'] = is_male
df = df.drop(['sex'], axis = 1)

# Label encode multicategorical columns
num_cols = df._get_numeric_data().columns
cat_cols = list(df.drop(num_cols, axis = 1).columns)

for col in cat_cols:
    pd.factorize(df[col])
    df[col] = pd.factorize(df[col])[0]
    
# Use correlation analysis to select features
corr = np.abs(df.corr(method='pearson')['out_type_ec']).sort_values(ascending=False)

bad_corr_feat = corr[corr<0.05]
bad_corr_feat = bad_corr_feat.index.values

df_ca = df.drop(columns=bad_corr_feat, inplace=False)

# Define X and Y
y = df_ca['out_type_ec']
X = df_ca.drop(columns = ['out_type_ec']).copy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Train a gradient boosting classifier model using RandomizedSearchCV to select the best hyperparameters
gbc = GradientBoostingClassifier()

parameters = {"learning_rate": [0.01, 0.1, 0.2, 0.3, 0.4],
              "subsample"    : np.arange(0.5, 1.0, 0.1),
              "n_estimators" : [100, 250, 500, 750],
              "max_depth"    : [3, 6, 10, 15]
                 }

clf = RandomizedSearchCV(estimator = gbc,
                         cv = 4,
                         n_iter = 100,
                         param_distributions = parameters,
                         scoring ='neg_mean_squared_error' 
                         )

clf.fit(X_train, y_train)

filename = 'shelter_outcomes_classifier.sav'
pickle.dump(clf, open(filename, 'wb'))