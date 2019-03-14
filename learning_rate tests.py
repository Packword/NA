import numpy as np
learning_rates = [0.001,0.01,0.1,1,10,100,1000]
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
def sigmoid_output_to_derivative(output):
    return output*(1-output)
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0],[1],[1],[0]])
for learning_rate in learning_rates:
    print ("\nТренируемся при learning_rate:" + str(learning_rate))
    np.random.seed(1)
    input_0 = 2*np.random.random((3,4)) - 1   # случайная инициализация весов со средним 0
    input_1 = 2*np.random.random((4,1)) - 1
    for j in np.arange(60000):
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0,input_0))
        layer_2 = sigmoid(np.dot(layer_1,input_1))
        layer_2_error = layer_2 - y
        if (j% 10000) == 0 and j!=0:
            print ("Ошибка после "+str(j)+" повторений:" + str(np.mean(np.abs(layer_2_error))))
        layer_2_delta = layer_2_error*sigmoid_output_to_derivative(layer_2)
        layer_1_error = layer_2_delta.dot(input_1.T)
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        input_1 -= learning_rate * (layer_1.T.dot(layer_2_delta))
        input_0 -=learning_rate * (layer_0.T.dot(layer_1_delta))