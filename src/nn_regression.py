import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense


def specify_compile_mod(n_cols):
   model = Sequential()
   model.add(Dense(100, activation='relu', input_shape=(n_cols, )))
   model.add(Dense(100, activation='relu'))
   model.add(Dense(1))
   model.compile(optimizer='adam', loss='mean_squared_error')
   return model


def plot_results(prediction, title_name, fig_name):
    fig_path = os.path.join(os.getcwd(), 'figures')
    plt.scatter(prediction, y_test)
    plt.xlabel('predicted')
    plt.ylabel('true values')
    plt.title(title_name)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.show()


if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), 'data/concrete_data.xls')
    data = pd.read_excel(data_path)

    new_cols = []
    for col in data.columns:
        new_cols.append(col.split('(')[0].strip().lower().replace(' ', '_'))

    data.columns = new_cols

    X = data.drop(['concrete_compressive_strength'], axis=1).as_matrix()
    y = data.concrete_compressive_strength

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    n_cols = X_train.shape[1]

    model = specify_compile_mod(n_cols=n_cols)
    model.fit(X_train, y_train, epochs=1000)
    pred1000 = model.predict(X_test)

    plot_results(pred1000, 'Predicted vs True Values w/ Training @ 1000 Epochs', 'pred_vs_true_1000epochs.png')

    
