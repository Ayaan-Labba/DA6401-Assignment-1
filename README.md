# DA6401-Assignment-1
Link to Github repo: https://github.com/Ayaan-Labba/DA6401-Assignment-1
Link to Wandb report: https://wandb.ai/ch21b021-indian-institute-of-technology-madras/DA6401-Assignment-1/reports/DA6401-Assignment-1--VmlldzoxMTcwNjk1Nw


## Project Structure
```
.
├── dataset.py       # Loads and preprocesses data
├── model.py         # Neural network model with optimizers & training
├── train.py         # Script to train the model
├── config.json      # Config file for hyperparameters
├── requirements.txt # Dependencies
└── README.md        # Documentation
```


## Installing Required Packages

Create a virtual environment with:
`python -m venv .venv`
`source .venv/bin/activate`

Install the required packages with:
`pip install -r requirement.txt`


## Functions
- `load_data()` in **`dataset.py`**: Downloads either the fashion_mnist dataset or mnist dataset
- `visualize_samples` in **`visualize.py`**: Creates an image of the first elements in each class from the chosen dataset


## Classes
`NeuralNetwork` in **`neural_network.py`**:
- Parameters: Implements the feedforward neural network with backpropagation and the given optimizers
    - `input_size` (int) – Number of input neurons
    - `hidden_layers` (list) – List of hidden layer sizes
    - `output_size` (int) – Number of output neurons
    - `activation` (str) – Activation function (ReLU, sigmoid, tanh)
    - `weight_init` (str) – Weight initialization (random, Xavier)
    - `optimizer` (str) – Optimizer (sgd, momentum, nesterov, adam, nadam)
    - `lr` (float) – Learning rate
    - `momentum` (float) – Momentum factor (for momentum, nesterov)
    - `beta` (float) - Momentum factor for rmsprop
    - `beta1` (float) – First moment decay rate (for adam, nadam)
    - `beta2` (float) – Second moment decay rate (for adam, nadam)
    - `epsilon` (float) – Small constant to avoid division by zero
- Functions:
    - `initialize_weights(self)` - Initializes weights using random or Xavier initialization
    - `activation_function(self, x, derivative=False)` - Applies activation function and its derivative for backpropagation
    - `forward(self, X)` - Performs forward propagation through the network
    - `loss(self, Y_pred, Y_true, function="cross_entropy")` - Computes categorical loss (cross-entropy loss or squared-error loss)
    - `gradients(self, H, A, Y_true)` - Computes gradients using backpropagation
    - `gradient_descent(self, grads)` - Updates weights using the selected optimizer
    - `train(self, X_train, Y_train, epochs, batch_size, log_interval=10)` - Trains the model with batch processing and logs loss/accuracy
    - `predict(self, X)` - Predicts class probabilities using softmax


## Training and Testing
To train the model and obtain the model performance, run:
`python train.py`

To test the performance on the test set:
`python test.py`

To use Wandb, run on the terminal:
`pip install wandb`
`wandb login`
and update train.py with:

```
import wandb

wandb.init(project="neural-network")

for epoch in range(epochs):
    # Training code...
    wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy})

```
