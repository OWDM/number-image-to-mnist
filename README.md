# number-image-to-mnist
 An innovative tool that converts images containing numerical digits into MNIST-like formatted data for machine learning applications.
An innovative tool that converts images containing numerical digits into MNIST-like formatted data for machine learning applications. This project bridges the gap between real-world digit images and the standardized format of the MNIST dataset, facilitating easier integration and application of machine learning models trained on MNIST.

# The Goal
my project aims to transform any image featuring a number into a format similar to that of the MNIST dataset. This involves detecting the numbers in the image, then scaling and processing them to match the dataset used for model training.

# Project Structure
MNIST-Data: Contains sample images from the MNIST dataset for reference and testing.
RawDataForTesting: Includes raw images used to test the algorithm's effectiveness in transforming real-world images into MNIST-compatible format.
src: Contains the core project files including the main script and Jupyter notebooks.
models: Stores trained models, including mnist_model2.h5 which is the output from RunTheModel.py.
notebooks: Jupyter notebooks for detailed exploration and explanation of the project's methodology.
