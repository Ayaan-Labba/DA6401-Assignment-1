# DA6401-Assignment-1
Link to Github repo: https://github.com/Ayaan-Labba/DA6401-Assignment-1
Link to Wandb report: https://wandb.ai/ch21b021-indian-institute-of-technology-madras/DA6401-Assignment-1/reports/DA6401-Assignment-1--VmlldzoxMTcwNjk1Nw

## Project Structure
.
├── dataset.py       # Loads and preprocesses data

├── model.py         # Neural network model with optimizers & training

├── train.py         # Script to train the model

├── config.json      # Config file for hyperparameters

├── requirements.txt # Dependencies

└── README.md        # Documentation

## Installing Required Packages

Create a virtual environment with:
`python -m venv .venv`
`source .venv/bin/activate`

Install the required packages with:
`pip install -r requirement.txt`

## Functions
`load_data()` in **`dataset.py`**: Downloads either the fashion_mnist dataset or mnist dataset
`visualize_samples` in **`visualize.py`**: Creates an image of the first elements in each class from the chosen dataset

## Classes


