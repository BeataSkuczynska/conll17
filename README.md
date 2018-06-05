# CONLL Shared Task 2017


### Installing

Make sure you use Python 3.6

Take language treebanks from this site https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2515 .
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
--max_len     change maximal number of tokens in sentence
```
Also you can change RNN parameters in the `config.py` file.
