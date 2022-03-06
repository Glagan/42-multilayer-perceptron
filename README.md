# Multilayer Perceptron

## Requirements

``Python 3`` and other requirements in ``requirements.txt``.

## Usage

You first need to **train** a model to predict with it.  
If no datasets as the first argument is given, the default one will be used.

```python
python3 train.py
```

You can then predict for another dataset with the same format.  
if no datasets as the first argument is given, it will try to use ``datasets/correction.csv``.

```python
python3 predict.py
```

You can see the effect of changing the learning rate and the neural network size with the ``learning_rate.py`` and ``network_size.py`` scripts, which will train a model with different hyperparameters and compare their accuracy and loss over each epoch.

```python
python3 learning_rate.py
python3 network_size.py
```

## Resources

-   No 42-AI for this subject !
-   Andrew Ng course
    -   Theory: https://www.coursera.org/learn/machine-learning/home/week/4
    -   Application: https://www.coursera.org/learn/machine-learning/home/week/5
-   https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
-   https://cs231n.github.io/
