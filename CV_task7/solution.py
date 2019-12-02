from interface import *


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values,
                n - batch size, ... - arbitrary input shape
        :return: np.array((n, ...)), output values,
                n - batch size, ... - arbitrary output shape (same as input)
        """
        ans = inputs.copy()
        ans[inputs < 0] = 0
        return ans

    def backward(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs,
                n - batch size, ... - arbitrary output shape
        :return: np.array((n, ...)), dLoss/dInputs,
                n - batch size, ... - arbitrary input shape (same as output)
        """
        inputs = self.forward_inputs
        ans = grad_outputs.copy()
        ans[inputs < 0] = 0
        return ans


# ============================== 2.1.2 Softmax ===============================
class Softmax(Layer):
    def forward(self, inputs):
        """
        :param inputs: np.array((n, d)), input values,
                n - batch size, d - number of units
        :return: np.array((n, d)), output values,
                n - batch size, d - number of units
        """
        ans = np.exp(inputs)
        ans /= np.sum(ans, axis=1, keepdims=True)
        return ans

    def backward(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs,
                n - batch size, d - number of units
            :return: np.array((n, d)), dLoss/dInputs,
                n - batch size, d - number of units
        """
        n, d = grad_outputs.shape
        outputs = self.forward_outputs
        matrix = np.array([- outputs[i].reshape(d, 1) @ outputs[i].reshape(1, d) + np.diag(outputs[i]) for i in range(n)]).reshape(n, d, d)
        return np.array([grad_outputs[i, :].reshape(1, d) @ matrix[i] for i in range(n)]).reshape(n, d)


# =============================== 2.1.3 Dense ================================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_shape = (units,)
        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        print(self.input_shape, self.output_shape)
        input_units, = self.input_shape
        output_units, = self.output_shape

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

    def forward(self, inputs):
        """
        :param inputs: np.array((n, d)), input values,
                n - batch size, d - number of input units
        :return: np.array((n, c)), output values,
                n - batch size, c - number of output units
        """
        batch_size, input_units = inputs.shape
        output_units, = self.output_shape
        return inputs @ self.weights + self.biases

    def backward(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs,
                n - batch size, c - number of output units
        :return: np.array((n, d)), dLoss/dInputs,
                n - batch size, d - number of input units
        """
        # your code here \/
        batch_size, output_units = grad_outputs.shape
        input_units, = self.input_shape
        inputs = self.forward_inputs

        # Don't forget to update current gradients:
        # dLoss/dWeights
        self.weights_grad = np.mean(np.array([inputs[i, :].reshape(input_units, 1) @ grad_outputs[i, :].reshape(1, output_units)
                                    for i in range(batch_size)]).reshape(batch_size, input_units, output_units), axis=0)
        # dLoss/dBiases
        self.biases_grad = np.mean(grad_outputs, axis=0)

        return np.array([self.weights @ grad_outputs[i, :].reshape(output_units, 1)
                        for i in range(batch_size)]).reshape(batch_size, input_units)


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def __call__(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values
        :return: np.array((n,)), loss scalars for batch
        """
        batch_size, output_units = y_gt.shape
        return -np.sum(y_gt * np.log(y_pred), axis=1)

    def gradient(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values
        :return: np.array((n, d)), gradient loss to y_pred
        """
        return - y_gt / y_pred


# ================================ 2.3.1 SGD =================================
class SGD(Optimizer):
    def __init__(self, lr):
        self._lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter
        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam
            :return: np.array, new parameter values
            """
            # your code here \/
            assert parameter_shape == parameter.shape
            assert parameter_shape == parameter_grad.shape
            return parameter - self._lr * parameter_grad
            # your code here /\

        return updater


# ============================ 2.3.2 SGDMomentum =============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self._lr = lr
        self._momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter
        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam
            :return: np.array, new parameter values
            """
            # your code here \/
            assert parameter_shape == parameter.shape
            assert parameter_shape == parameter_grad.shape
            assert parameter_shape == updater.inertia.shape

            updater.inertia = self._momentum * updater.inertia + self._lr * parameter_grad
            return parameter - updater.inertia

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ======================= 2.4 Train and test on MNIST ========================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    model = Model(CategoricalCrossentropy(), SGDMomentum(0.001))
    print("shapes", x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    sh = x_train.shape
    model.add(Dense(100, input_shape=(sh[1],)))
    model.add(ReLU(100))
    model.add(Dense(10)) #change for number prediction
    model.add(Softmax())

    model.fit(x_train, y_train, 2, 4, x_valid=x_valid, y_valid=y_valid)
    return model

# ============================================================================
