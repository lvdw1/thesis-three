# nndriver.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

class NNDriver:
    """
    A class to encapsulate an MLPRegressor neural network.
    
    By default, the network:
      - Uses all CSV columns except the last three as input.
      - Uses the last three CSV columns as the output.
    
    You can also override these behaviors by specifying custom column selections.
    """
    
    def __init__(self, 
                 hidden_layer_sizes=(64, 56, 48, 40, 32, 24, 16, 8), 
                 alpha_value=0.01, 
                 learning_rate='adaptive', 
                 learning_rate_init=0.001, 
                 max_iter=1000, 
                 tol=1e-6, 
                 random_state=42, 
                 verbose=True, 
                 early_stopping=False):
        """
        Initialize the neural network model.
        """
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='sgd',
            alpha=alpha_value,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            shuffle=True,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            early_stopping=early_stopping
        )
    
    @staticmethod
    def load_data(csv_path, input_columns=None, output_columns=None):
        """
        Load data from a CSV file.
        
        Parameters:
            csv_path (str): Path to the CSV file.
            input_columns (list or None): Column names or indices to be used as inputs.
                                          If None, all columns except the last three are used.
            output_columns (list or None): Column names or indices to be used as outputs.
                                           If None, the last three columns are used.
        
        Returns:
            X (ndarray): Input features.
            y (ndarray): Output targets.
        """
        data = pd.read_csv(csv_path)
        
        if input_columns is None:
            X = data.iloc[:, :-3].values  # Use all columns except the last three as input
        else:
            X = data[input_columns].values
        
        if output_columns is None:
            y = data.iloc[:, -3:].values  # Use the last three columns as output
        else:
            y = data[output_columns].values
        
        return X, y
    
    def train(self, X_train, y_train):
        """
        Train the neural network on the provided training data.
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
            X (ndarray): Input features.
            
        Returns:
            ndarray: Predictions.
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model using Mean Squared Error (MSE).
        
        Parameters:
            X_test (ndarray): Test input features.
            y_test (ndarray): True test outputs.
            
        Returns:
            float: The MSE value.
        """
        y_pred = self.model.predict(X_test)
        mse_value = mean_squared_error(y_test, y_pred)
        return mse_value
    
    def get_loss(self):
        """
        Get the final loss value from the MLPRegressor.
        """
        return self.model.loss_

if __name__ == "__main__":
    # Example usage when running this module as a script.
    csv_file = "session3/run1.csv"  # Update this path as needed
    
    # Load data
    X, y = NNDriver.load_data(csv_file)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize the neural network driver
    nn_driver = NNDriver()
    
    # Train the model
    nn_driver.train(X_train, y_train)
    
    # Evaluate the model
    mse_value = nn_driver.evaluate(X_test, y_test)
    
    print("Test MSE:", mse_value)
    print("Final Loss:", nn_driver.get_loss())
