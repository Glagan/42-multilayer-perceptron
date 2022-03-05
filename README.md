# Multilayer Perceptron

## TODO

-   [~] Train
    -   [x] Read dataset
    -   [x] Normalize
    -   [x] Split dataset, one for training and one for testing
    -   [~] Main loop
        -   [x] Initialize neural network of given size
            -   [x] Weights
            -   [x] Bias
        -   [ ] Add batch processing
        -   [ ] Add regularization
        -   [x] Calculate forward pass
            -   [x] Calculate each layers neurons values (Gradient descent)
                -   [x] ReLU plus bias for the next layer
            -   [x] Calculate Output layer result with softmax
        -   [x] Save error value on each iterations for graph
        -   [x] Backpropagation
            -   [x] softmax derivative for the last layer
            -   [x] Reverse order from forward pass
    -   [x] Save result (network + weights)
-   [x] Predict
    -   [x] Open saved results
    -   [x] Simple prediction formula with given weights
        -   [x] [binary cross-entropy error function](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression)

## Resources

-   No 42-AI for this subject !
-   Andrew Ng course
    -   Theory: https://www.coursera.org/learn/machine-learning/home/week/4
    -   Application: https://www.coursera.org/learn/machine-learning/home/week/5
-   https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
-   https://cs231n.github.io/
