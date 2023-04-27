import pandas as pd
import numpy as np
import random
import graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from random import randint

df = pd.read_csv('weatherAUS.csv')

# Data cleaning
df.columns = ["Date", "Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustDir", "WindGustSpeed", " WindDir9am", "WindDir3pm",
              "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RainToday", "RainTomorrow"]
df.drop(['Date', 'Location'], axis=1, inplace=True)
df['RainToday'] = df['RainToday'].replace('No', 0)
df['RainToday'] = df['RainToday'].replace('Yes', 1)
df['RainTomorrow'] = df['RainTomorrow'].replace('No', 0)
df['RainTomorrow'] = df['RainTomorrow'].replace('Yes', 1)
df = df.dropna()
df_x = df[["MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
           "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm", "RainToday"]]
df_y = df[["RainTomorrow"]]
df_corr = df[["MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
              "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm", "RainToday", "RainTomorrow"]]


# Split data
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y,
                                                    test_size=0.2,
                                                    random_state=42)

clf = tree.DecisionTreeClassifier(random_state=0, max_depth=8)

clf = clf.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = clf.predict(X_test)

# Accuracy
accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Correlation Matrix
plt.matshow(df_corr.corr())
cb = plt.colorbar()
plt.title('Correlation Matrix', fontsize=16)
plt.show()

exit = "no"
while exit == "no":

    if input('Press anything to write your own data or write ran for randomized data: ') == "ran":
        minTemp = round(random.uniform(-6.5, 31.5), 2)
        maxTemp = round(random.uniform(4.1, 48.5), 2)
        rainfall = round(random.uniform(0, 206.5), 2)
        windGustSpeed = round(random.uniform(9, 124.5), 2)
        windSpeed9am = round(random.uniform(1.5, 67.5), 2)
        windSpeed3pm = round(random.uniform(1.5, 77.5), 2)
        humidity9am = round(random.uniform(0, 100), 2)
        humidity3pm = round(random.uniform(0, 100), 2)
        pressure9am = round(random.uniform(980.5, 1040.5), 2)
        pressure3pm = round(random.uniform(977.5, 1038.9), 2)
        temp9am = round(random.uniform(-0.7, 39.5), 2)
        temp3pm = round(random.uniform(3.5, 46.5), 2)
        rainToday = randint(0, 1)
        ran = [[minTemp, maxTemp, rainfall, windGustSpeed, windSpeed9am, windSpeed3pm,
                humidity9am, humidity3pm, pressure9am, pressure3pm, temp9am, temp3pm, rainToday]]
        numpy_array = np.array(ran)

    else:
        minTemp = input('write your minTemp: ')
        maxTemp = input('write your maxTemp: ')
        rainfall = input('write your rainfall: ')
        windGustSpeed = input('write your windGustSpeed: ')
        windSpeed9am = input('write your windSpeed9am: ')
        windSpeed3pm = input('write your windSpeed3pm: ')
        humidity9am = input('write your humidity9am: ')
        humidity3pm = input('write your humidity3pm: ')
        pressure9am = input('write your pressure9am: ')
        pressure3pm = input('write your pressure3pm: ')
        temp9am = input('write your temp9am: ')
        temp3pm = input('write your temp3pm: ')
        rainToday = input('write your rainToday (1 = yes, 0 = no): ')
        ran = [[minTemp, maxTemp, rainfall, windGustSpeed, windSpeed9am, windSpeed3pm,
                humidity9am, humidity3pm, pressure9am, pressure3pm, temp9am, temp3pm, rainToday]]
        numpy_array = np.array(ran)

    print("minTemp, maxTemp, rainfall, windGustSpeed, windSpeed9am, windSpeed3pm, humidity9am, humidity3pm, pressure9am, pressure3pm, temp9am, temp3pm, rainToday")
    print(ran)
    new_df = pd.DataFrame(numpy_array, columns=["MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am",
                          "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm", "RainToday"])

    pred = clf.predict(new_df)
    if (pred[0] == 1):
        print('It is going to rain tomorrow')
        print("｀、ヽ｀ヽ｀、ヽ(ノ＞＜)ノ ｀、ヽ｀☂️ヽ｀、ヽ")
    else:
        print('No Rain Tomorrow')
        print("╲⎝⧹ ( ͡° ͜ʖ ͡°) ⎠╱☀️")
    exit = input('exit? (write no to continue): ')

# Visualization, requires Graphviz, install it with pip install graphviz
# Prints the extracted tree to a pdf file
dot = tree.export_graphviz(clf, out_file=None,
                           feature_names=["MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
                                          "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm", "RainToday"],
                           class_names="RainTomorrow",
                           filled=True, rounded=True,
                           special_characters=True,
                           max_depth=3)
graph = graphviz.Source(dot)
graph.view()
