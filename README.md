# Speech Recognition with PyTorch
CNN implementation in Python with PyTorch, on audio (``.wav``) files (94+ on test).

1. [General](#General)
    - [Background](#background)
    - [Model Structure](#model-structure)
2. [Dependencies](#dependencies) 

## General

### Background
Implementation of a neural network on the audio files. using ``gcommand_dataset.py`` that converts the ``.wav`` files into a 2D matrix (of ``161 x 101``).

The audio files in this dataset are ``~`` ``1sec`` long, and there are `` 30`` optional commands that can be heard in the files.

### Model Structure
In short, the model has ``5`` convolutional layers, with ``Batch Normalize``, ``ReLU`` and ``Max Pooling`` after each one. Then ``2`` more ``Fully Connected`` layers. The output of the neural network is ``30``.

In more detail:
- First layer: Convolutional layer, kernel size = 5, stride = 2, padding = 2, batch norm = 16 => [1,16]
- Second layer: Convolutional layer, kernel size = 3, stride = 1, padding = 1, batch norm = 32 => [16,32]
- Third layer: Convolutional layer, kernel size = 3, stride = 1, padding = 1, batch norm = 64 => [32,64]
- Fourth layer: Convolutional layer, kernel size = 3, stride = 1, padding = 1, batch norm = 128 => [64,128]
- Fifth lyaer: Convolutional layer, kernel size = 3, stride = 1, padding = 1, batch norm = 256 => [128,256]
- Sixth layer: Fully Connected layer, batch norm = 128 => [512, 128]
- Seventh layer: Fully Connected layer => [128,30]

Throughout the model building, I monitored the loss and accuracy values so that I could get value of the accuracy of the model, and how
And where can I improve it. This can be seen under the section called: 𝑀𝑜𝑑𝑒𝑙𝑠 𝑣𝑎𝑙𝑖𝑑𝑎𝑡𝑖𝑜𝑛 𝑓𝑢𝑛𝑐𝑛𝑐𝑛𝑐𝑛 in the attached code.


#### About The Output Files
The program code exports a total of 2 files:
* A ``test_y`` file that contains the predictions for the test.
* The ``BestModelcpu.png`` or ``BestModelcuda.png`` file (based on the device on which the code runs), which contains a graph of the accuracy percentage and loss values of the training and the validation depending on the epochs.
* 


Note that for using the dataset given in this repo, you need to download the dataset (about ``1GB``). You can also use ``google colab`` for running this program.
## Dependencies
* [Python 3.6+](https://www.python.org/downloads/)
* [NumPy](https://numpy.org/install/)
* [Matplotlib](https://matplotlib.org/stable/users/installing.html)
* [PyTorch](https://pytorch.org/get-started/locally/)
