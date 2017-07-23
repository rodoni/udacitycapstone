# README

This ṕython script is a part of capstone proposal to submitted for Udacity, 
in this script was resolved a classification problem using Convolutional Neural
Network applied for a set of satellite images from Amazon forest, this was a Kaggle
competition named Planet: Understanding the Amazon from Space.


Requirements:

_Python_ _2.7_

_Numpy_

_Pandas_

_TensorFlow_ _1.1.0_

_Keras_ _2.0_

_Scikit-learn_ _0.18_


Commands:

To training the entire dataset an predict and create the csv file

python amazon.py <path_to_dataset>

Ex:.  python amazon.py /home/user/DataSet/Amazon/

To only predicut and create the csv file, but is necessary weights files

python amazon.py <path_to_dataset> --only-predict