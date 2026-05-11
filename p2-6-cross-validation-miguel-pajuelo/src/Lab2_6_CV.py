from sklearn.base import clone
import numpy as np

def cross_validation(model, X, y, nFolds):
    """
    Perform cross-validation on a given machine learning model to evaluate its performance.

    This function manually implements n-fold cross-validation if a specific number of folds is provided.
    If nFolds is set to -1, Leave One Out (LOO) cross-validation is performed instead, which uses each
    data point as a single test set while the rest of the data serves as the training set.

    Parameters:
    - model: scikit-learn-like estimator
        The machine learning model to be evaluated. This model must implement the .fit() and .score() methods
        similar to scikit-learn models.
    - X: array-like of shape (n_samples, n_features)
        The input features to be used for training and testing the model.
    - y: array-like of shape (n_samples,)
        The target values (class labels in classification, real numbers in regression) for the input samples.
    - nFolds: int
        The number of folds to use for cross-validation. If set to -1, LOO cross-validation is performed.

    Returns:
    - mean_score: float
        The mean score across all cross-validation folds.
    - std_score: float
        The standard deviation of the scores across all cross-validation folds, indicating the variability
        of the score across folds.
    """
    if nFolds == -1:
        # Implement Leave One Out CV
        nFolds = X.shape[0]

    # Number of samples
    n_samples = X.shape[0]

    # Validate number of folds
    if nFolds <= 1 or nFolds > n_samples:
        raise ValueError("nFolds must be between 2 and the number of samples, or -1 for LOO.")

    # Create array of sample indices
    indices = np.arange(n_samples)

    # TODO: Calculate fold_size based on the number of folds
    # We distribute samples as evenly as possible among folds
    fold_sizes = np.full(nFolds, n_samples // nFolds, dtype=int)
    fold_sizes[:n_samples % nFolds] += 1

    # TODO: Initialize a list to store the accuracy values of the model for each fold
    accuracy_scores = []

    current = 0
    for i in range(nFolds):
        fold_size = fold_sizes[i]

        # TODO: Generate indices of samples for the validation set for the fold
        start = current
        end = current + fold_size
        valid_indices = indices[start:end]

        # TODO: Generate indices of samples for the training set for the fold
        train_indices = np.concatenate((indices[:start], indices[end:]))

        # TODO: Split the dataset into training and validation
        X_train_fold, X_valid_fold = X[train_indices], X[valid_indices]
        y_train_fold, y_valid_fold = y[train_indices], y[valid_indices]

        # TODO: Train the model with the training set
        # Clone the model so each fold uses a fresh estimator
        model_copy = clone(model)
        model_copy.fit(X_train_fold, y_train_fold)

        # TODO: Calculate the accuracy of the model with the validation set and store it in accuracy_scores
        score = model_copy.score(X_valid_fold, y_valid_fold)
        accuracy_scores.append(score)

        current = end

    # TODO: Return the mean and standard deviation of the accuracy_scores
    mean_score = np.mean(accuracy_scores)
    std_score = np.std(accuracy_scores)
    return mean_score, std_score