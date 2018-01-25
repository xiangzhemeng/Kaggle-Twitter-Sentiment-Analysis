# Twitter Sentiment Analysis (Text classification)

**`Team`:** Hello World

**`Team Members`:** Sung Lin Chan, Xiangzhe Meng, Süha Kagan Köse

This repository is the final project of CS-433 Machine Learning Fall 2017 at EPFL. The private competition was hosted on Kaggle [EPFL ML Text Classification](https://www.kaggle.com/c/epfml17-text)
we had a complete dataset of 2500000 tweets. One half of tweets are positive labels and the other half are negative labels Our task was to build a classifier to predict the test dataset of 10000 tweets. This README.md illustrates the 
the implementation of the classifier, and present the procedure to reproduce our works. The details of our implementation were written in the report. Ultimately, we ranked **9th of 63 teams** on the leaderboard.

## Project Specification

See [Project Specification](https://github.com/epfml/ML_course/tree/master/projects/project2/project_text_classification) at EPFL Machine Learning Course CS-433 github page.

## Hardware Environment
In this project, we use two instances on GCP (Google Cloud Platform) to accelerate  the neural network training by GPU the text preprocessing by multiprocessing technique.

For neural network training:
- GPU Platform:
    - CPU: 6 vCPUs Intel Broadwell
    - RAM: 22.5 GB
    - GPU: 1 x NVIDIA Tesla P100
    - OS: Ubuntu 16.04 LTS

For text preprocessing:
- Pure CPU Platform:
    - CPU: 24 vCPUs Intel Broadwell
    - RAM: 30GB
    - OS: Ubuntu 16.04 LTS


## Dependencies

All the scripts in this project ran in Python 3.5.2, the generic version on GCP instance. For nueral network framework, we used Keras, a high-level neural networks API, and use Tensorflow as backend.
 
The NVIDIA GPU CUDA version is 8.0 and the cuDNN version is v6.0. Although, there are newer version of CUDA and cuDNN at this time, we use the stable versions that are recommended by the official website of Tensorflow. For more information and installation guide about how to set up GPU environment for Tensorflow, please see [here](https://www.tensorflow.org/install/install_linux)



* [Scikit-Learn] (0.19.1）- Install scikit-learn library with pip

    ```sh
    $ sudo pip3 install scikit-learn
    ```

* [Gensim] (3.2.0) - Install Gensim library 

    ```sh
    $ sudo pip3 install gensim
    ```

* [FastText] (0.8.3) - Install FastText implementation

    ```sh
    $ sudo pip3 install fasttext
    ```

* [NLTK] (3.2.5) - Install NLTK and download all packages

    ```sh
    // Install
    $ sudo pip3 install nltk
    
    // Download packages
    $ python3
    $ >>> import nltk
    $ >>> nltk.download()
    ```
    
* [Tensorflow] (1.4.0) - Install tensorflow. Depends on your platfrom, choose either without GPU version or with GPU version

    ```sh
    // Without GPU version
    $ sudo pip3 install tensorflow
    
    // With GPU version
    $ sudo pip3 install tensorflow-gpu
    ```

* [Keras] (1.4.0) - Install Keras
    
    ```sh
    $ sudo pip3 install keras
    ```

* [XGBoost] (0.6a2) - Install XGboost
    
    ```sh
    $ sudo pip3 install xgboost
    ```


## Folder / Files

* `segmenter.py`: <br/>
    helper function for preprocessing step

* `data_loading.py`: <br/>
    helper function for loading the original dataset and output pandas dataframe object as pickles.

* `data_preprocessing.py`: <br/>
    Module of preprocessing. Take output of `data_loading.py` and output preprocessed tweets

* `cnn_training.py`: <br/>
    Module of three cnn models The the output of `data_preprocessing.py` and generate result as input of `xgboost_training.py`
    
* `xgboost_training.py`: <br/>
    Module of xgboost model. Take the output of `cnn_training.py` and generate the prediction result.

* `run.py`: <br/>
    Script for running the modules, `data_loading.py`, `data_preprocessing.py`, `cnn_training.py` and `xgboost_training.py`.
    
* `data`: <br/>
    This folder contains the necessary metadata and intermediate files while running our scripts.
    
    - `tweets`: Contain the original train and test dataset downloaded from Kaggle.
    - `dictionary`: Contain the text files for text preprocessing
    - `pickles`: Contain the intermediate files of preprocessed text as the input of CNN model
    - `xgboost`: Contain the intermediate output files of CNN model and there are the input of XGboost model.
    - `output` : Contain output file of kaggle format from `run.py` 

    Note: The files inside `tweets` and `dictionary` are essential for running the scripts from scratch.
    Download [tweets and dictionary](https://www.dropbox.com/s/pmwewfpcfqdrmz5/Archive.zip)
    Then, unzip the downloaded file and move the extracted `tweets` and `dictionary` folder in `data/` directory.
    
    If you want to skip the preprocessing step and CNN training step, download [preprocessed data and pretrained model](https://www.dropbox.com/s/pmwewfpcfqdrmz5/Archive.zip).
    Then, unzip the downloaded file and move all the extracted folders  in `data/` directory.

* `othermodels`: <br/>

    The files in this folder are the models we explored, before coming out the best model.

    - `keras_nn_model.py`: This is the classifier using NN model and the word representation method is GloVE. Each was represented by the average of the sum of each word and fit into NN model.

    - `fastText_model.py`: This is the classifier using FastText. The word representation is FastText english pre-trained model.

    - `svm_model.py`: This is the classifier using support vector machine. The word representation is TF-IDF by using Scikit-Learn built-in method.


## Reproduce Our Best Score on Kaggle

Here are our steps from original dataset to kaggle submission file in order. We had modulized each step into .py file, they can be executed individually. For your convenience, we provide `run.py` which could run the modules with simple command.

1. Transform dataset to pandas dataframe - `data_loading.py`
2. Preprocessing dataset - `data_preprocessing.py`
3. CNN model training  - `cnn_training.py`
4. XGboost model training and generate submission file - `xgboost_training.py`


**First**, make sure all the essential data is put into "data/" directory

**Second**, there are three options to generate Kaggle submission file. We recommand the first options, which takes less than 10 minutes to reproduct the result with pretrianed models.

   -if you want to skip preprocessing step and CNN model training step, execute run.py with -m argument "xgboost"

        $ python3 run.py -m xgboost
   
   Note: Make sure that there are `test_model1.txt`, `test_model2.txt`, `test_model3.txt`, `train_model1.txt`, `train_model2.txt` and `train_model3.txt` in "data/xgboost in order to launch run.py successfully.
   
   -if you want to skip preprocessing step and start from CNN model training setp, execute run.py with -m argument "cnn"
   
        $ python3 run.py -m cnn
   
   Note: Make sure that there are `train_clean.pkl` and `test_clean.pkl`  in "data/pickles in order to launch run.py            successfully.
    
   -if you want to run all the steps from scratch, execute run.py with -m argument "all"

        $ python3 run.py -m all
    
  Note: our preprocessing step require larges amount of CPU resource. It is a multiprocessing step, and will occupy all the     cores of CPU. It took one hour to finish this step on 24  vCPUs instance on GCP and extra one and half hour more to finish CNN model training step with NVIDIA P100.
  
**Finally**, you can find `prediction.csv` in "data/output" directory

## Contributors
- Sung Lin Chan
- Xiangzhe Meng
- Süha Kagan Köse
___

License: [MIT](https://opensource.org/licenses/MIT)
