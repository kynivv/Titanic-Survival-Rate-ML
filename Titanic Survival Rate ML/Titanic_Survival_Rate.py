
import pandas as pd
import numpy as np

# ML algorithm
from sklearn.ensemble import RandomForestClassifier

# Function import
from sklearn.model_selection import train_test_split

#
pd.options.mode.chained_assignment = None

# For model
import joblib

# Data import
data = pd.read_csv('titanic_train.csv')
data.head()

# Median age
median_age = data['age'].median()

# Filling empty slots
data['age'].fillna(median_age, inplace=True)
data['age'].head()

# Intputs
data_inputs = data[['pclass', 'age', 'sex']]
data_inputs.head()

# Outputs
expected_output = data[['survived']]
expected_output.head()

# Data Preprocessing
data_inputs['pclass'].replace('3rd', 3, inplace=True)
data_inputs['pclass'].replace('2nd', 2, inplace=True)
data_inputs['pclass'].replace('1st', 1, inplace=True)
data_inputs.head()

data_inputs['sex'] = np.where(data_inputs['sex'] == 'female', 0, 1)
data_inputs.head()

# Dividing data
inputs_train, inputs_test, expected_output_train, expected_output_test = train_test_split(data_inputs, expected_output, test_size = 0.33, random_state = 42)

# Teaching algorithm
rf = RandomForestClassifier(n_estimators=100)
rf.fit(inputs_train,expected_output_train)

# Defining accuracy
accuracy = rf.score(inputs_test, expected_output_test)
print(accuracy)

joblib.dump(rf, "Titanic_survival_rate_model", compress=9)

