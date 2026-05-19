# Logistic Regression Implementation

This project contains a simple NumPy implementation of logistic regression in [logistic_regression.py](/home/bruno-zabot/Studies/MachineLearning/assignment/logistic_regression.py).

The file defines four functions:

1. `sigmoid`
2. `prediction`
3. `binary_crossentropy`
4. `gradient_descent`

## Overall Flow

The implementation follows the standard binary logistic regression pipeline:

1. Start with weights and bias initialized to zero.
2. Compute the linear combination `Z = Xw + b`.
3. Pass `Z` through the sigmoid function to convert it into probabilities.
4. Compare predictions with the real labels.
5. Compute gradients for the weights and bias.
6. Update the parameters repeatedly with gradient descent.

## Function Explanations

### `sigmoid(y)`

```python
def sigmoid(y):
    return 1 / (1 + np.exp(-1 * y))
```

Purpose:
Converts any real-valued input into a value between `0` and `1`.

Why it is needed:
Logistic regression predicts probabilities. The sigmoid function maps the linear output of the model into a probability-like value.

Formula:

```text
sigmoid(y) = 1 / (1 + e^(-y))
```

Input:
- `y`: a number, vector, or NumPy array

Output:
- A value or array with all elements between `0` and `1`

Interpretation:
- Values close to `1` mean the model is confident in class `1`
- Values close to `0` mean the model is confident in class `0`

Example:

```python
sigmoid(0)      # 0.5
sigmoid(2)      # about 0.88
sigmoid(-2)     # about 0.12
```

### `prediction(att, a, b)`

```python
def prediction(att, a, b):
    return sigmoid(np.dot(att, a) + b)
```

Purpose:
Computes the predicted probability for one sample or a set of samples.

How it works:
- `np.dot(att, a)` computes the weighted sum of the input features
- `b` adds the bias term
- `sigmoid(...)` transforms the result into a probability

Input:
- `att`: input attributes or feature vector/matrix
- `a`: weight vector
- `b`: bias term

Output:
- Predicted probability or probabilities

Interpretation:
This function represents the logistic regression model itself:

```text
P(y=1|x) = sigmoid(x · w + b)
```

Notes:
- In your current implementation, this function is separate from training and can be used after learning `weights` and `bias`
- It is useful for making predictions on new data

### `binary_crossentropy(A, y)`

```python
def binary_crossentropy(A, y):
    A = np.clip(A, 1e-15, 1 - 1e-15)
    loss = -np.mean(y * np.log(A) + (1 - y) * np.log(1 - A))
    return loss
```

Purpose:
Calculates the binary cross-entropy loss, which measures how far the predicted probabilities are from the true labels.

Why it is needed:
During logistic regression, binary cross-entropy is the standard loss function for binary classification problems.

How it works:
- `A` contains predicted probabilities
- `y` contains true labels (`0` or `1`)
- `np.clip(...)` avoids taking `log(0)`, which would cause numerical errors
- The formula averages the error across all examples

Formula:

```text
Loss = -mean(y log(A) + (1 - y) log(1 - A))
```

Input:
- `A`: predicted probabilities
- `y`: true binary labels

Output:
- A single scalar loss value

Interpretation:
- Lower loss means better predictions
- Loss is `0` only for perfect predictions

Notes:
- This function is defined correctly, but it is not currently used inside `gradient_descent`
- It could be added there to monitor training progress every few iterations

### `gradient_descent(X, y)`

```python
def gradient_descent(X, y):
    weights = np.zeros(X.shape[1])
    bias = 0

    learning_rate = 0.01
    iterations = 1000

    for i in range(iterations):
        Z = np.dot(X, weights) + bias
        A = sigmoid(Z)
        error = A - y

        dw = 1 / len(y) * np.dot(X.T, error)
        db = 1 / len(y) * np.sum(error)

        weights = weights - (learning_rate * dw)
        bias = bias - (learning_rate * db)

    return weights, bias
```

Purpose:
Trains the logistic regression model by learning the best weights and bias from the data.

Input:
- `X`: feature matrix with shape `(number_of_samples, number_of_features)`
- `y`: target vector with binary labels `0` and `1`

Output:
- `weights`: learned weight vector
- `bias`: learned bias scalar

Step-by-step explanation:

1. `weights = np.zeros(X.shape[1])`
   Initializes one weight for each feature.

2. `bias = 0`
   Initializes the bias term.

3. `learning_rate = 0.01`
   Controls the size of each update step.

4. `iterations = 1000`
   Sets how many times the algorithm updates the parameters.

5. `Z = np.dot(X, weights) + bias`
   Computes the linear output for every sample.

6. `A = sigmoid(Z)`
   Converts linear outputs into predicted probabilities.

7. `error = A - y`
   Measures the difference between predicted probabilities and true labels.

8. `dw = 1 / len(y) * np.dot(X.T, error)`
   Computes the gradient of the loss with respect to the weights.

9. `db = 1 / len(y) * np.sum(error)`
   Computes the gradient of the loss with respect to the bias.

10. `weights = weights - (learning_rate * dw)`
    Updates the weights in the direction that reduces the loss.

11. `bias = bias - (learning_rate * db)`
    Updates the bias in the same way.

12. `return weights, bias`
    Returns the trained parameters.

Why this works:
Gradient descent iteratively adjusts the model parameters so that predicted probabilities become closer to the true labels.

## Relationship Between the Functions

- `sigmoid` is the activation function used by logistic regression
- `prediction` uses `sigmoid` to make model predictions
- `binary_crossentropy` evaluates how good those predictions are
- `gradient_descent` trains the model by updating the weights and bias

In the current code:
- `gradient_descent` uses `sigmoid`
- `prediction` and `binary_crossentropy` are helper functions that can be used separately

## Example Usage

An example exists in [test_example.py](/home/bruno-zabot/Studies/MachineLearning/assignment/test_example.py):

```python
weight, bias = gradient_descent(X, y)
print(weight, bias)
```

After training, predictions for new data can be made with:

```python
probabilities = prediction(X, weight, bias)
```

If needed, probabilities can be converted into class labels:

```python
labels = (probabilities >= 0.5).astype(int)
```

## Current Limitations

This implementation is a good educational version, but it has some limitations:

- It does not track or print the loss during training
- It does not include a separate `fit` and `predict` interface
- It assumes the target values are already binary (`0` or `1`)
- It uses fixed hyperparameters (`learning_rate` and `iterations`) inside the function

## Summary

Your implementation covers the essential parts of binary logistic regression:

- `sigmoid` converts scores into probabilities
- `prediction` computes model outputs
- `binary_crossentropy` measures prediction error
- `gradient_descent` learns the model parameters

Together, these functions form a complete basic logistic regression workflow using NumPy.
