## Usage

### Configuring the Project
1. Set the data paths and output file paths in `config.py`.
2. Adjust the model hyperparameters in `config.py` as needed.

### Training and Testing
1. Run `python main.py` to start the training and testing process.
2. The trained model checkpoints and evaluation results will be saved to the specified output paths.

## Project Structure

### `main.py`
This is the entry point of the project. It handles the overall execution flow, including data loading, model training, and evaluation.

### `trainer.py`
This module contains the code for model training and testing. It includes functions for model initialization, training loop, and evaluation.

### `config.py`
This module stores all the configurable parameters for the project, such as data paths, output paths, and model hyperparameters.

### `datasets.py`
This module defines the data processing and loading logic for the project's dataset(s).

### `models.py`
This module contains the definitions of all the machine learning models used in the project.