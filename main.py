import numpy as np


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)

        # инцициализация весов
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        """
        Сжимает числовой результат в дипазоне от -1 до 0
        """
        # Подобный формат комментариев, может быть использован для автоматизации документации
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Выводит производную для весов
        """
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            # Тренировка через нейронную сеть
            output = self.think(training_inputs)

            # Подсчет количества ошибок
            error = training_outputs - output

            # Менее правдоподобные веса подстраиваются
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        """
        Пропуск введенных данных через нееросеть для получения выходных данных
        """

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # тут нудно ввести значиния инпута и значения аутпута (вход и выход)
    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])

    # изменены значения аутпута, чтоб результат был инверсивным от оригинала

    training_outputs = np.array([[1, 0, 0, 1]]).T

    neural_network.train(training_inputs, training_outputs, 10000)

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))

    print("New situation: input data = ", A, B, C)
    print("Output data: ")
    print(neural_network.think(np.array([A, B, C])))