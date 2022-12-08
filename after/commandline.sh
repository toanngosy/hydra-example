# select an item from a config group
python main.py knnparams=knnparams_2

# run and show config for the run (https://hydra.cc/docs/tutorials/basic/running_your_app/debugging/)
python main.py --cfg all

# run with default, overwrite knnparams.n_init
python main.py knnparams.n_init=15

# multirun on knnparams
python main.py --multirun knnparams=knnparams,knnparams_2

# multirun on knnparams.n_init
python main.py --multirun knnparams.n_init=15,20

# tab completion https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion/


