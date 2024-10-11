import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        return input_error


class ActivationLayer:
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(input_data)
        return self.output

    def backward(self, output_error, learning_rate):
        return output_error * self.activation_derivative(self.input)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


# Example usage:
# Assuming you have a CSV file named "chat_data.csv" with columns "input" and "output"
# where "input" contains the user's input and "output" contains the chatbot's corresponding response.
try:
    data = pd.read_csv("updated_chat_data.csv")  # Replace with your actual file name
except FileNotFoundError:
    print("Error: 'chat_data.csv' not found. Please upload your data file.")
    exit()

# Preprocess the text data
# This is a basic example. You can use more advanced techniques like TF-IDF or word embeddings for better performance.

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["input"]).toarray()
y = vectorizer.transform(data["output"]).toarray()

# Create the network
layer1 = Layer(X.shape[1], 20)
activation1 = ActivationLayer(sigmoid, sigmoid_derivative)
layer2 = Layer(20, y.shape[1])  # Output layer with the same size as the vocabulary.
activation2 = ActivationLayer(sigmoid, sigmoid_derivative)

# Training loop
epochs = 10000
learning_rate = 10

for epoch in range(epochs):
    # Forward pass
    output1 = layer1.forward(X)
    activated_output1 = activation1.forward(output1)
    output2 = layer2.forward(activated_output1)
    y_pred = activation2.forward(output2)

    # Calculate error
    error = mse(y, y_pred)

    # Backward pass
    output_error = mse_derivative(y, y_pred)
    output_error = activation2.backward(output_error, learning_rate)
    output_error = layer2.backward(output_error, learning_rate)
    output_error = activation1.backward(output_error, learning_rate)
    layer1.backward(output_error, learning_rate)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Error: {error}")

print("Training finished.")

# Chatbot interaction loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Preprocess user input
    user_input_vector = vectorizer.transform([user_input]).toarray()

    # Make predictions
    output1 = layer1.forward(user_input_vector)
    activated_output1 = activation1.forward(output1)
    output2 = layer2.forward(activated_output1)
    y_pred = activation2.forward(output2)

    # Calculate cosine similarity between the predicted output and the training outputs
    similarities = cosine_similarity(y_pred, y)

    # Find the index of the most similar output
    most_similar_index = np.argmax(similarities)

    # Get the corresponding response from the training data
    predicted_response = data["output"][most_similar_index]
    print("Chatbot:", predicted_response)
