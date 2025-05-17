import numpy as np
import argparse
import matplotlib.pyplot as plt

def euclidean_dist(x1: np.ndarray, x2: np.ndarray):
    """
    Calculate the Euclidean distance between two data points

    Parameters:
        x1 (np.array): Feature values of the first data point
        x2 (np.array): Feature values of the second data point

    Returns:
        float: Euclidean distance between the two data points.
    """
    # TODO: Implement the Euclidean distance calculation
    pass

def manhattan_dist(x1: np.ndarray, x2: np.ndarray):
    """
    Calculate the Manhattan distance between two data points

    Parameters:
        x1 (np.array): Feature values of the first data point
        x2 (np.array): Feature values of the second data point

    Returns:
        float: Manhattan distance between the two data points.
    """
    # TODO: Implement the Manhattan distance calculation
    pass

def predict(k: int, dist_metric: int, train_data: np.ndarray, X: np.ndarray):
    """
    Compute the predictions for the data points in X using a kNN
    classified defined by the first three parameters
    
    Parameters:
        k (int): Number of nearest neighbors to use when computing
                the predictions
        dist_metric (int): Distance metric to use when computing 
                            the predictions
        train_data (np.ndarray): Training dataset
        X (np.ndarray): Feature values of the data points to be 
                        predicted 

    Returns:
        np.ndarray: Vector of predicted labels
    """
    # TODO: Implement kNN prediction
    pass
            
            
                

def compute_error(preds: np.ndarray, labels: np.ndarray):
    """
    Compute the error rate for a given set of predictions 
    and labels
    
    Parameters:
        preds (np.ndarray): Your models predictions
        labels (np.ndarray): The correct labels

    Returns:
        float: Error rate of the predictions 
    """
    # TODO: Implement the error rate computation
    pass
    

def val_model(ks: range, dist_metric: int, train_data: np.ndarray, val_data: np.ndarray):
    """
    For each value in ks, compute the training and validation
    error rates
    
    Parameters:
        ks (range): Set of values 
        dist_metric (int): Distance metric to use when computing 
                            the predictions
        train_data (np.ndarray): Training dataset
        val_data (np.ndarray): Validation dataset

    Returns:
        tuple(train_preds: np.ndarray, 
              val_preds: np.ndarray,
              train_errs: np.ndarray,
              val_errs: np.ndarray): tuple of predictions and error rate arrays 
                                    where each row corresponds to one of the k
                                    values in ks. For the preds arrays, the length
                                    of each row will be the number of data points 
                                    in the relevant dataset and for the errs arrays,
                                     each row will consistent of a single error rate. 
    """
    # TODO: Implement validation
    pass
        
        
	
def crossval_model(ks: range, num_folds: int, dist_metric: int, train_data: np.ndarray):
    """
    For each value in ks, compute the cross-validation error rate
    
    Parameters:
        ks (range): Set of values 
        dist_metric (int): Distance metric to use when computing 
        				   the predictions
        num_folds(int): number of folds to split the training 
        				dataset into
        train_data (np.ndarray): Training dataset

    Returns:
        tuple(crossval_preds: np.ndarray,
              crossval_errs: np.ndarray): tuple of predictions and error rate arrays
                                          where each row corresponds to one of the k
                                          values in ks. For the preds array, the 
                                          length of each row will be the number of 
                                          training data points and should contain the
                                          prediction for the corresponding data point
                                          when held out as a validation data point. 
                                          For the errs array, each row will contain the
                                          corresponding num_folds-fold cross-validation
                                          error rate. 
    """
    # TODO: Implement cross-validation
    pass
        
            
            

if __name__ == '__main__':
        # This takes care of command line argument parsing for you!
        # To access a specific argument, simply access args.<argument name>.
        # For example, to get the learning rate, you can use `args.learning_rate`.
        parser = argparse.ArgumentParser()
        parser.add_argument("train_input", type=str, help='path to formatted training data')
        parser.add_argument("val_input", type=str, help='path to formatted validation data')
        parser.add_argument("test_input", type=str, help='path to formatted test data')
        parser.add_argument("val_type", type=int, choices=[0,1], help='validation type; 0 = validation, 1 = cross-validation')
        parser.add_argument("dist_metric", type=int, choices=[0,1], help='distance metric; 0 = Euclidean, 1 = Manhattan')
        parser.add_argument("min_k", type=int, help='smallest value of k to consider (inclusive)')
        parser.add_argument("max_k", type=int, help='largest value of k to consider (inclusive)')
        parser.add_argument("train_out", type=str, help='file to write train predictions to')
        parser.add_argument("val_out", type=str, help='file to write train predictions to')
        parser.add_argument("test_out", type=str, help='file to write test predictions to')
        parser.add_argument("metrics_out", type=str, help='file to write metrics to')
        args = parser.parse_args()

        
