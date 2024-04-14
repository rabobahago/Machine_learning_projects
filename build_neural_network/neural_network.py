import numpy as np
class NeuralNetwork:
    """
    A simple neural network with one hidden layer.

    Parameters:
    -----------
    input_size: int
        The number of input features
    hidden_size: int
        The number of neurons in the hidden layer
    output_size: int
        The number of neurons in the output layer
    loss_func: str
        The loss function to use. Options are 'mse' for mean squared error, 'log_loss' for logistic loss, and 'categorical_crossentropy' for categorical crossentropy.
    """
    def __init__(self, input_size, hidden_size, output_size, loss_func = 'mse'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.loss_func = loss_func
        # initialize weights and biases
        self.weights1 = np.random.rand(self.input_size, self.hidden_size)
        self.bias1 = np.ones((1, self.hidden_size))
        self.weights2 = np.random.rand(self.hidden_size, self.output_size)
        self.bias2 = np.ones((1, self.output_size))
        # track the loss
        self.train_loss = []
        self.test_loss = []


    def forward(self, X):
        """
        Perform forward propagation.

        Parameters:
        -----------
        X: numpy array
            The input data

        Returns:
        --------
        numpy array
            The predicted output
        """
        # Perform forward propagation
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.z1, self.weights2)
        self.a2 = self.sigmoid(self.z2)
        if self.loss_func == 'categorical_crossentropy':
            self.a2 = self.softmax(self.z2)
        else:
            self.a2 = self.sigmoid(self.z2)
        return self.a2