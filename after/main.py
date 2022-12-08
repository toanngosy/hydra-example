import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path

import logging
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# from config import MainConfig
# cs = ConfigStore.instance()
# cs.store(name='main_config', node=MainConfig)

# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(version_base='1.2', config_path='conf', config_name='config')
def main(cfg: DictConfig):

    log.info(f"Run with configuration: {cfg}")

    log.info(f"Load dataset: {cfg.files.file_name}")
    # load dataset
    file_path = f'{cfg.paths.data_dir}/{cfg.files.file_name}'
    df = pd.read_csv(file_path)

    log.info("Encoding categorical features")
    #Label Encoding - for encoding categorical features into numerical ones
    encoder = LabelEncoder()
    df['Species'] = encoder.fit_transform(df['Species'])

    #DROPPING ID
    df= df.drop(['Id'], axis = 1)

    #converting dataframe to np array
    data = df.values
    X = data[:, 0:5]
    y = data[:, -1]

    log.info(f"Split train/test set: {1-cfg.runparams.test_size}/{cfg.runparams.test_size}")
    # split into train/test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.runparams.test_size,
        random_state=cfg.runparams.random_state
    )

    #KMeans
    kmeans = KMeans(n_init=cfg.knnparams.n_init,
                    n_clusters=cfg.knnparams.n_clusters,
                    max_iter=cfg.knnparams.max_iter,
                    tol=cfg.knnparams.tol,
                    algorithm=cfg.knnparams.algorithm
                    )

    kmeans.fit(X_train, y_train)

    # training predictions
    train_labels = kmeans.predict(X_train)

    #testing predictions
    test_labels = kmeans.predict(X_test)

    #KMeans model accuracy
    #training accuracy
    log.info(f"Accuracy on train set: {accuracy_score(y_train, train_labels)*100}")
    log.info(f"Accuracy on test set: {accuracy_score(test_labels, y_test)*100}")
    log.info(f"classification report on train set: \n{classification_report(y_train, train_labels)}")


if __name__ == '__main__':
    main()