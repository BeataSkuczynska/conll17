# PolEval 2018 Shared Task 1(A)


### Installing

Make sure you use Python 3.6

Take data from poleval.pl/tasks
Install requirements 

```
pip install -r requirements.py
```

## Run training and evaluation

To run full cycle run
```
$ python train.py path_to_train_conll_file path_to_test_conll_file
```
Optional arguments:
```
-p            set this flag to only predict dependency tree for given CONLL-U file
--max_len     change maximal number of tokens in sentence
```
Also you can change RNN parameters in the `config.py` file.
