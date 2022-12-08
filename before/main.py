import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_DIR = '../data'
FILE_NAME = 'Iris.csv'
DATA_PATH = Path(f'{DATA_DIR}/{FILE_NAME}')

TEST_SIZE = 0.3
RANDOM_STATE = 42

N_INIT = 10
N_CLUSTERS = 3
MAX_ITER = 300
TOL = 0.0001
ALGORITHM ='lloyd' #['lloyd', 'elkan', 'auto', 'full']

def main():
    df = pd.read_csv(str(DATA_PATH))

    #Label Encoding - for encoding categorical features into numerical ones
    encoder = LabelEncoder()
    df['Species'] = encoder.fit_transform(df['Species'])

    #DROPPING ID
    df= df.drop(['Id'], axis = 1)

    #converting dataframe to np array
    data = df.values
    X = data[:, 0:5]
    y = data[:, -1]

    # split into train/test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    #KMeans
    kmeans = KMeans(n_init=N_INIT,
                    n_clusters=N_CLUSTERS,
                    max_iter=MAX_ITER,
                    tol=TOL,
                    algorithm=ALGORITHM
                    )

    kmeans.fit(X_train, y_train)

    # training predictions
    train_labels = kmeans.predict(X_train)

    #testing predictions
    test_labels = kmeans.predict(X_test)

    #KMeans model accuracy
    #training accuracy
    print(accuracy_score(y_train, train_labels)*100)
    #testing accuracy
    print(accuracy_score(test_labels, y_test)*100)

    #classification report for training set
    print(classification_report(y_train, train_labels))

if __name__ == '__main__':
    main()