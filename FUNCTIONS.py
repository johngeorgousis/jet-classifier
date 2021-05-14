import numpy as np
#import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from IPython.display import display

import pprint
pp = pprint.PrettyPrinter(indent=4)

import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  # Decision Trees
from sklearn.naive_bayes import MultinomialNB    # Naive Bayes
from sklearn.naive_bayes import GaussianNB       # Gaussian Naive Bayes
from sklearn.svm import SVC                      # SVM
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
print('TensorFlow Version: ', tf.__version__)
                


'''
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW TENSORFLOW 
'''
def create_dataset(file, pixels=40, R=1.5):
    '''
    Takes dat file of events
    Labels events (background = 0, signal = 1)
    Preprocessed events and turns into images
    Returns 2d array where rows: events and columns: (image, label) 
    '''

    image = np.zeros((pixels, pixels))                           # Define initial image
    data = {}
    a = 0
    
    with open(file) as infile:
        for line in infile:

            # Preprocessing
            event = line.strip().split()
            event = pd.Series(event)                         # Turn into Series
            event = preprocess(event)                        # Preprocess
            max1 = find_max1(event)                          # Extract maxima
            event = center(event, max1)                      # Center 
            max2 = find_max2(event)                          # Extract maxima
            event = rotate(event, max2)                      # Rotate 
            max3 = find_max3(event)                          # Extract maxima
            event = flip(event, max3)                        # Flip 
            event = create_image(event, pixels=pixels, R=R)  # Create image
            image = event                                    # Rename
            #image /= np.sum(image)
            #image /= np.amax(image)                          # Normalise final image between 0 and 1
            #image = np.log(image)                            # Log image
            
            event=max1=max2=max3=None
            
            a += 1
            data[a] = image
            
    data = list(data.values())
    data = np.array(data)
    data = np.reshape(data, (a*pixels, pixels))
    
        
    return data
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def normalise(image, label):
    
    '''
    To be used in preprocess_ML_tf
    '''
    
    image = tf.cast(image, tf.int64)                 # Set dtype to int64
    label = tf.cast(label, tf.int64)
    #image /= np.amax(image.numpy())                    # Normalise Image
    #image = tf.image.resize(image, (pixels, pixels, 1))   # Resize image to 40x40
    return image, label

def preprocess_ML_tf(data_s, data_b, batch_size):
    
    '''Prepares dataset for TensorFlow Machine Learning algorithm'''
    
    '''
    Input1: Signal Dataset created using create_dataset
    Input2: Background Dataset created using create_dataset
    Input3: Batch size
    
    Process:
    - Create labels for signals (1) and backgrounds (0)
    - Combine signal and background datasets
    - Combine signal and background labels
    - Define useful (local) variables
    - Reshape main dataset (for CNN)
    - Train-Val-Test Split examples and labels
    - Turn into tf datasets (train, val, split)
    - Create Batches (train, val, split)
    - Define useful (global) variables (returned later)
    - Plot Events to Visualise & make sure everything's right (e.g. normalised vs non-normalised)
    
    Output1: train_batches
    Output2: val_batches
    Output3: test_batches
    Output4: num_of_batches_train
    Output5: num_of_batches_val
    Output6: num_of_batches_test
    '''
    
    
    
    '''Preprocess'''
    # Create s&b labels
    slabels = np.ones(data_s.shape[0]//40)
    blabels = np.zeros(data_b.shape[0]//40)

    # Concatenate s&b and s&b labels
    data = np.concatenate((data_s, data_b), axis=0)
    labels = np.concatenate((slabels, blabels), axis=0)

    # Define & Print useful variables (local)
    num_of_examples = data.shape[0] // 40     # divide by 40 because 1st dim is 40 * num_of_examples
    num_of_labels = labels.shape[0]
    print('Total Events:', num_of_examples)
    print('Total Labels:', num_of_labels)

    # Reshape examples (for CNN)
    examples = data.reshape(num_of_examples, 40, 40, 1)
    print('Shape: ', examples.shape)
    print(' ')
    
    
    '''Train-Val-Test Split'''
    train_examples, test_examples, train_labels, test_labels = train_test_split(examples, labels, test_size=0.15, random_state=42)
    train_examples, val_examples, train_labels, val_labels = train_test_split(train_examples, train_labels, test_size=0.18, random_state=42)

    print('Train: ', train_examples.shape, train_labels.shape)
    print('Val: ', val_examples.shape, val_labels.shape)
    print('Test: ', test_examples.shape, test_labels.shape)
    print(' ')
    
    
    train_data = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    val_data = tf.data.Dataset.from_tensor_slices((val_examples, val_labels))
    test_data = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    print(train_data)
    print(val_data)
    print(test_data)
    
    
    
    '''BATCHES'''
    batch_size = batch_size

    train_batches = train_data.cache().shuffle(num_of_examples).map(normalise).batch(batch_size, drop_remainder=True).prefetch(1)
    val_batches = val_data.cache().shuffle(num_of_examples).map(normalise).batch(batch_size, drop_remainder=True).prefetch(1)  # or prefetch(buffer_size=tf.data.experimental.AUTOTUNE) 
    test_batches = test_data.cache().shuffle(num_of_examples).map(normalise).batch(batch_size, drop_remainder=True).prefetch(1)
    
    # Define useful variables (will be returend)
    num_of_batches_train = len(train_labels) // batch_size
    num_of_batches_val = len(val_labels) // batch_size
    num_of_batches_test = len(test_labels) // batch_size

    print(train_batches)
    print('\ntrain, val, test: ', num_of_batches_train, num_of_batches_val, num_of_batches_test)
    
    
    '''VISUALISE'''
    plt.figure(figsize=(15,10))

    for images, labels in train_batches.take(1):
        for i in range(3):
            ax = plt.subplot(3, 3, i + 1)
            sns.heatmap(images[i].numpy().reshape(40, 40))
            #plt.imshow(images[i].numpy().astype("uint8"))
            plt.title('Image {}'.format(i+1))
            plt.axis("off")
    
    
    return train_batches, val_batches, test_batches, num_of_batches_train, num_of_batches_val, num_of_batches_test
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def visualise_preds(model, test_batches):
    
    '''
    
    Visualise Predictions for TensorFlow
    
    Input1: Model
    Input 2: Predictions
    Output: NULL
    
    '''
    
    '''1st'''
    
    class_names = ['Background', 'Signal']

    for event, label in test_batches.take(1):
        ps = model.predict(event)
        images = event.numpy().squeeze()
        labels = label.numpy()


    plt.figure(figsize=(15,10))

    for n in range(6):
        plt.subplot(3,3,n+1)
        sns.heatmap(images[n])
        #plt.imshow(images[n], cmap = plt.cm.binary)
        color = 'green' if np.argmax(ps[n]) == labels[n] else 'red'
        plt.title(class_names[np.argmax(ps[n])], color=color)
        plt.axis('off')
        
        
        
    '''2nd'''
    
    for event, label in test_batches.take(1):
        ps = model.predict(event)
        first_image = event.numpy().squeeze()[0]


        fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
        #sns.heatmap(first_image)
        ax1.imshow(first_image)
        ax1.axis('off')
        ax2.barh(np.arange(2), ps[0])
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(2))
        ax2.set_yticklabels(np.arange(2))
        ax2.set_title('Class Probability')
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def learning_curve(train_batches, val_batches, test_batches, num_of_batches_train, num_of_batches_val):
    
    '''
    
    Plots a learning curve to determine whether more data would improve the model (i.e. detect underfitting)
    
    '''
    
    
    input_shape=(40, 40, 1)
    kernel_size = 2
    padding='valid'
    activation = 'tanh'

    prop = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    loss_list = []
    accuracy_list = []

    for i in prop:
        model = tf.keras.Sequential([
                      tf.keras.Input(shape=input_shape),
                      tf.keras.layers.Conv2D(16, kernel_size=kernel_size, padding=padding, activation=activation),
                      tf.keras.layers.MaxPooling2D(),
                      tf.keras.layers.Conv2D(32, kernel_size=kernel_size, padding=padding, activation=activation),
                      tf.keras.layers.MaxPooling2D(),
                      tf.keras.layers.Conv2D(64, kernel_size=kernel_size, padding=padding, activation=activation),
                      tf.keras.layers.MaxPooling2D(),
                      tf.keras.layers.Flatten(),
                      tf.keras.layers.Dense(128, activation=activation),
                      tf.keras.layers.Dense(2, activation = 'softmax')
            ])


        # Compile Model
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])


        print('\n\n', i, '\n')

        # Fit model to training data
        EPOCHS = 4

        history = model.fit(train_batches.take(int(i*num_of_batches_train)), 
                  epochs=EPOCHS,
                  validation_data=val_batches.take(int(i*num_of_batches_val)), 
                  verbose=0
                  )

        loss, accuracy = model.evaluate(test_batches, verbose=0)
        loss_list.append(loss)
        accuracy_list.append(accuracy)

        loss, accuracy = model.evaluate(test_batches, verbose=0)
        print('Accuracy on the Test Set: {:.1%}'.format(accuracy))
        
    plt.plot(np.array(prop)*100, accuracy_list, linestyle='--', marker='o')
    plt.xlabel('% of Dataset Used')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def model_complexity_graph(history):
    
    '''
    
    Plots model complexity graph to determine how many epochs you need (i.e. when the model starts overfitting the data)
    
    '''
    
    
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']

    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    epochs_range=range(len(training_accuracy))

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, training_accuracy, label='Training Accuracy')
    plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, training_loss, label='Training Loss')
    plt.plot(epochs_range, validation_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cmx_tf(models, test_batches, num_of_batches_test):
    
    '''
    Plots Confusion Matrix for TensorFlow list of models
    
    '''
    
    for model in models:
    
        '''Extract Preds & Labels'''

        preds_all = []
        labels_all = []
        preds_batch = []
        labels_batch = []


        # For all batches
        for batch, labels in test_batches.take(num_of_batches_test):

            # 64 preds and labels added to list
            pp = model.predict(batch)
            preds_batch = np.array([np.argmax(pp[i]) for i in range(len(pp))])
            labels_batch = labels.numpy()

            preds_all.append(preds_batch)
            labels_all.append(labels_batch)


        # Convert list of lists to ndarray and flatten to get 1D ndarray of all preds and 1D ndarray of all labels
        preds = np.array(preds_all).flatten()
        labels = np.array(labels_all).flatten()      


        '''Build CMX'''

        cmx_non_normal = tf.math.confusion_matrix(labels, preds).numpy() # Create Confusion Matrix
        cmx0 = cmx_non_normal[0] / cmx_non_normal[0].sum()
        cmx1 = cmx_non_normal[1] / cmx_non_normal[1].sum()
        cmx = np.stack((cmx0, cmx1), axis=0)


        # Plot confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(cmx, cmap=['skyblue', 'deepskyblue', 'dodgerblue', 'blue',  'darkblue'])

        # xylabels and title
        plt.title('Confusion Matrix')
        plt.xlabel('PREDICTIONS')
        plt.ylabel('LABELS')

        # Label ticks
        ax.set_xticklabels(['Background', 'Signal'])
        ax.set_yticklabels(['Background', 'Signal'])
        # Align ticks
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
                 rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=90, ha="center",
                 rotation_mode="anchor")

        # Text Annotations for Blocks in CMX
        for i in range(2):
            for j in range(2):

                value = int(np.round(100*cmx[i, j], 0))

                text = ax.text(j+0.5, 
                               i+0.5, 
                               value,
                               ha="center", 
                               va="center", 
                               color="orangered", 
                               fontsize = 20)

        plt.show()



        # # Print P(signal|signal) and P(signal|background)
        # pss = cmx[1,1] / (cmx[1,1]+cmx[1,0])
        # pbs = 1 - pss
        # psb = cmx[0,1] / (cmx[0,1]+cmx[0,0])
        # pbb = 1 - psb
        # precision = cmx[1,1] / (cmx[1,1]+cmx[0,1])
        # recall = cmx[1,1] / (cmx[1,1]+cmx[1,0])
        # print('\n')
        # print('P(signal|signal) = {:.0f}%'.format(100*pss))
        # print('P(signal|background) = {:.0f}%'.format(100*psb)) 
        # print('P(background|background) = {:.0f}%'.format(100*pbb))
        # print('P(background|signal) = {:.0f}%'.format(100*pbs))
        # print('Precision = {:.0f}'.format(precision*100))
        # print('Recall = {:.0f}'.format(recall*100))
        # print('\n')
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def ROC3_tf(model0, model1, model2, test_batches, num_of_batches_test):
    
    ''' 
    Plot ROC curve for exactly 3 TensorFlow models
    
    '''
    
    '''Extract Preds & Labels'''

    preds_batch = []
    labels_batch = []
    preds_all = []
    labels_all = []


    # For all batches
    for batch, labels in test_batches.take(num_of_batches_test):

        # 64 preds and labels added to list
        pp = model0.predict(batch)
        preds_batch = np.array([np.argmax(pp[i]) for i in range(len(pp))])
        labels_batch = labels.numpy()

        preds_all.append(preds_batch)
        labels_all.append(labels_batch)


    # Convert list of lists to ndarray and flatten to get 1D ndarray of all preds and 1D ndarray of all labels
    preds0 = np.array(preds_all).flatten()
    labels0 = np.array(labels_all).flatten()


    ##########################################################################################################################################################################


    preds_batch = []
    labels_batch = []
    preds_all = []
    labels_all = []


    # For all batches
    for batch, labels in test_batches.take(num_of_batches_test):

    # 64 preds and labels added to list
        pp = model1.predict(batch)
        preds_batch = np.array([np.argmax(pp[i]) for i in range(len(pp))])
        labels_batch = labels.numpy()

        preds_all.append(preds_batch)
        labels_all.append(labels_batch)


    # Convert list of lists to ndarray and flatten to get 1D ndarray of all preds and 1D ndarray of all labels
    preds1 = np.array(preds_all).flatten()
    labels1 = np.array(labels_all).flatten()


    ##########################################################################################################################################################################


    preds_batch = []
    labels_batch = []
    preds_all = []
    labels_all = []


    # For all batches
    for batch, labels in test_batches.take(num_of_batches_test):

    # 64 preds and labels added to list
        pp = model2.predict(batch)
        preds_batch = np.array([np.argmax(pp[i]) for i in range(len(pp))])
        labels_batch = labels.numpy()

        preds_all.append(preds_batch)
        labels_all.append(labels_batch)


    # Convert list of lists to ndarray and flatten to get 1D ndarray of all preds and 1D ndarray of all labels
    preds2 = np.array(preds_all).flatten()
    labels2 = np.array(labels_all).flatten()     


    '''Build ROC'''
        
    ##########################################################################################################################################################################


    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc


    fpr0, tpr0, thresholds = roc_curve(labels0, preds0)
    auc0 = auc(fpr0, tpr0)

    fpr1, tpr1, thresholds1 = roc_curve(labels1, preds1)
    auc1 = auc(fpr1, tpr1)

    fpr2, tpr2, thresholds2 = roc_curve(labels2, preds2)
    auc2 = auc(fpr2, tpr2)

    ##########################################################################################################################################################################

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr0, tpr0, label='0 (area = {:.3f})'.format(auc0))
    plt.plot(fpr1, tpr1, label='1 (area = {:.3f})'.format(auc1))
    plt.plot(fpr2, tpr2, label='2 (area = {:.3f})'.format(auc2))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
#     # Zoom in view of the upper left corner.
#     plt.figure(2)
#     plt.xlim(0, 0.2)
#     plt.ylim(0.8, 1)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.plot(fpr0, tpr0, label='0 (area = {:.3f})'.format(auc0))
#     plt.plot(fpr1, tpr1, label='1 (area = {:.3f})'.format(auc1))
#     plt.plot(fpr2, tpr2, label='1 (area = {:.3f})'.format(auc2))
#     plt.xlabel('False positive rate')
#     plt.ylabel('True positive rate')
#     plt.title('ROC curve (zoomed in at top left)')
#     plt.legend(loc='best')
#     plt.show()

    ##########################################################################################################################################################################
'''
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN SKLEARN 
'''
    
def preprocess_ML_sklearn(data_s, data_b):
    
    '''Prepares dataset for sklearn Machine Learning algorithm'''
    
    '''
    Input1: Signal Dataset created using create_dataset
    Input2: Background Dataset created using create_dataset
    
    Process:
    - Create labels for signals (1) and backgrounds (0)
    - Combine signal and background datasets
    - Combine signal and background labels
    - Define useful (local) variables
    - Reshape main dataset (for sklearn)
    - Train-Val-Test Split examples and labels
    - Plot Events to Visualise & make sure everything's right (e.g. normalised vs non-normalised)
    
    Output1: train_examples
    Output2: train_examples
    Output3: val_examples
    Output4: val_labels
    Output5: test_examples
    Output6: test_labels
    '''
    
    # Create s&b labels
    slabels = np.ones(data_s.shape[0]//40)
    blabels = np.zeros(data_b.shape[0]//40)

    # Concatenate examples and labels
    data = np.concatenate((data_s, data_b), axis=0)
    labels = np.concatenate((slabels, blabels), axis=0)

    # Define useful quantities
    num_of_examples = data.shape[0] // 40     # divide by 40 because 1st dim is 40 * num_of_examples
    num_of_labels = labels.shape[0]
    print('Total Events:', num_of_examples)
    print('Total Labels:', num_of_labels)

    # Reshape examples (for sklearn)
    examples = data.reshape(num_of_examples, 1600)
    print('\nShape: ', examples.shape)

    train_examples, test_examples, train_labels, test_labels = train_test_split(examples, labels, test_size=0.15, random_state=42)
    train_examples, val_examples, train_labels, val_labels = train_test_split(train_examples, train_labels, test_size=0.18, random_state=42)

    print('\nTrain: ', train_examples.shape, train_labels.shape)
    print('Val: ', val_examples.shape, val_labels.shape)
    print('Test: ', test_examples.shape, test_labels.shape)
    print(' ')

    eg = train_examples[5].reshape(40, 40)

    sns.heatmap(eg)
    plt.title("Train Image Example")
    plt.show()
    
    return train_examples, train_labels, val_examples, val_labels, test_examples, test_labels
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def compare_f1(models, test_examples, test_labels):
    
    '''Plots f1-score of Models using Test Data and prints top 5 models'''
    
    '''
    Input1: list of models
    Input2: test examples
    Input3: test labels
    
    Output1: NULL
    '''
    
    # Local Variables
    model_names = []
    scores = []

    # Get plot data
    for model in models:

        labels = test_labels
        preds = model.predict(test_examples)
        scores.append(f1_score(labels, preds))
        model_names.append(model.__class__.__name__)

    # Make Plots
    fig, ax = plt.subplots(figsize=(27, 6))
    plt.bar(model_names, scores, color="darkcyan")
    plt.show()

    # Print top 5 algorithms
    max_i = np.flip(np.argsort(scores))
    print('F1score')
    for i in max_i:
        print('{:.4f} {}'.format(scores[i], model_names[i])) 
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def compare_accuracy(models, test_examples, test_labels):
    
    '''Plots f1-score of Models using Test Data and prints top 5 models'''
    
    '''
    Input1: list of models
    Input2: test examples
    Input3: test labels
    
    Output1: NULL
    '''
    
    # Local Variables
    model_names = []
    scores = []

    # Get plot data
    for model in models:

        labels = test_labels
        preds = model.predict(test_examples)
        scores.append(accuracy_score(labels, preds))
        model_names.append(model.__class__.__name__)

    # Make Plots
    fig, ax = plt.subplots(figsize=(27, 6))
    plt.bar(model_names, scores, color="darkcyan")
    plt.show()

    # Print top 5 algorithms
    max_i = np.flip(np.argsort(scores))
    print('Accuracy')
    for i in max_i:
        print('{:.4f} {}'.format(scores[i], model_names[i]))  
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cmx_sklearn(models, test_examples, test_labels, dim=4):
    
    '''
    Plots Confusion Matrix for sklearn list of models
    
    '''
    cmxs = []
    
    for model in models:
            
            preds = model.predict(test_examples)
            labels = test_labels

            cmx_non_normal = tf.math.confusion_matrix(labels, preds).numpy() # Create Confusion Matrix
            cmx0 = cmx_non_normal[0] / cmx_non_normal[0].sum()
            cmx1 = cmx_non_normal[1] / cmx_non_normal[1].sum()
            cmx = np.stack((cmx0, cmx1), axis=0)
            cmxs.append(cmx)

    plt.figure(figsize=(25,20))
    for n in range(len(cmxs)):
        # Plot confusion matrix
        ax = plt.subplot(dim, dim, n+1)
        sns.heatmap(cmxs[n], cmap=['skyblue', 'deepskyblue', 'dodgerblue', 'blue',  'darkblue'])

        # xylabels and title
        plt.title(remove_text_inside_brackets(str(models[n])))
        plt.xlabel('PREDICTIONS')
        plt.ylabel('LABELS')

        # Label ticks
        ax.set_xticklabels(['Background', 'Signal'])
        ax.set_yticklabels(['Background', 'Signal'])
        # Align ticks
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
                 rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=90, ha="center",
                 rotation_mode="anchor")

        # Text Annotations for Blocks in CMX
        for i in range(2):
            for j in range(2):

                value = int(np.round(100*cmxs[n][i, j], 0))

                text = ax.text(j+0.5, 
                               i+0.5, 
                               value,
                               ha="center", 
                               va="center", 
                               color="orangered", 
                               fontsize = 20)
        plt.axis("off")
    plt.show()                   
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def learning_curve_sklearn(models, train_examples, train_labels, val_examples, val_labels):
    
    total_train = train_labels.shape[0]
    total_val = val_labels.shape[0]
    print('Total No. of Training Examples:', total_train)
    
    props = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    
    accuracies = []
    
    for model in models:
        print('\n=================================================================================================================================================================================\n')
        accuracy = []
        times = []
        
        for prop in props:
            prop_examples = int(prop*total_train)
            #print('Current No. of Training Examples:', prop_examples)
            start = time.time()
            model.fit(train_examples[0:prop_examples], train_labels[0:prop_examples])
            end = time.time()
            times.append((end-start)/60)
            #print('Training time for {} examples: {:.3f} minutes'.format(total_train, (end-start)/60))
            start = time.time()
            prop_examples_val = int(prop*total_val)
            val_preds = model.predict(val_examples[0:prop_examples_val])
            accuracy.append(accuracy_score(val_labels[0:prop_examples_val], val_preds[0:prop_examples_val]))
            end = time.time()
            #print('Prediction time for {} examples: {:.3f} minutes\n'.format(total_val, (end-start)/60))
            
            
            
        
        plt.plot(np.array(props)*100, accuracy, linestyle='--', marker='o')
        plt.xlabel('% of Dataset Used')
        plt.ylabel('Accuracy')
        plt.title('{} - Learning Curve'.format(model.__class__.__name__))
        plt.show()
        
        plt.plot(np.array(props)*100, times, linestyle='--', marker='o')
        plt.xlabel('% of Dataset Used')
        plt.ylabel('Training Time (minutes)')
        plt.title('{} - Training Time'.format(model.__class__.__name__))
        plt.show()
        
        print("Proportions:", props)
        print("Accuracy: {}".format(accuracy))
        print("Time: {}".format(times))
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    
def create_dataset_sklearn(file, pixels=40, R=1.5):
    '''
    Takes dat file of events
    Labels events (background = 0, signal = 1)
    Preprocessed events and turns into images
    Returns 2d array where rows: events and columns: (image, label) 
    '''

    data = ((0, 0))
    image = np.zeros((pixels, pixels))                           # Define initial image
    
        
    if file=='data_background.dat':
        label = 0
    elif file=='data_signal.dat':
        label = 1
    else: 
        print("ERROR: File name unclear")
        return
    
    with open(file) as infile:
        for line in infile:

            # Preprocessing
            event = line.strip().split()
            event = pd.Series(event)                         # Turn into Series
            event = preprocess(event)                        # Preprocess
            max1 = find_max1(event)                          # Extract maxima
            event = center(event, max1)                      # Center 
            max2 = find_max2(event)                          # Extract maxima
            event = rotate(event, max2)                      # Rotate 
            max3 = find_max3(event)                          # Extract maxima
            event = flip(event, max3)                        # Flip 
            event = create_image(event, pixels=pixels, R=R)  # Create image
            #event = event.flatten()                          # Flatten image from 2D to 1D for NN
            image = event                                   # Rename
            #image /= np.amax(image)                          # Normalise final image between 0 and 1
            
            event=max1=max2=max3=None                            # Delete from memory

            event = np.array((image, label))
            data = np.vstack((data, event))
    
    data = np.delete(data, 0, axis=0)
    return data
'''
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING
PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING


'''

   
                    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def average_image_un(pixels=60, R=1.5, event_no=12178, display=False, file='data/dataset_s_100k.dat'):
    '''
    Reads events directly from a file and creates an average image of the events. 
    
    pixels: int. Image Resolution
    R: float. Fat jet radius
    event_no: int/list. Number of events for which images are created. If int, then single image (faster). If list, then multiple images (slower)
    display: boolean. Indicates whether images should be displayed automatically (return null) or returned as an ndarray. 
    '''

    image = np.zeros((pixels, pixels))                           # Define initial image
    a = 0                                                        # Define Counter
    
    
    #Return single image
    if type(event_no) == int:
        
        with open(file) as infile:
            for line in infile:

                event = line.strip().split()
                event = pd.Series(event)                         # Turn into Series
                event = preprocess(event)                        # Preprocess 
                event = create_image(event, pixels=pixels, R=R)  # Create image
                image += event                                   # Add event image to average image
                #image /= np.amax(image)                          # Normalise final image between 0 and 1
                event=max1=max2=max3=None                            # Delete from memory

                a += 1
                if a == event_no:                                 # Break if max sample size for average image is exceeded 
                    return image



def average_image(pixels=60, R=1.5, event_no=12178, display=False, file='data/dataset_s_100k.dat'):
    '''
    Reads events directly from a file and creates an average image of the events. 
    
    pixels: int. Image Resolution
    R: float. Fat jet radius
    event_no: int/list. Number of events for which images are created. If int, then single image (faster). If list, then multiple images (slower)
    display: boolean. Indicates whether images should be displayed automatically (return null) or returned as an ndarray. 
    '''

    image = np.zeros((pixels, pixels))                           # Define initial image
    a = 0                                                        # Define Counter
    
    
    #Return single image
    if type(event_no) == int:
        
        with open(file) as infile:
            for line in infile:

                event = line.strip().split()
                event = pd.Series(event)                         # Turn into Series
                event = preprocess(event)                        # Preprocess
                max1 = find_max1(event)           # Extract maxima
                event = center(event, max1)                    # Center 
                max2 = find_max2(event)
                event = rotate(event, max2)                   # Rotate 
                max3 = find_max3(event)
                event = flip(event, max3)                     # Flip 
                event = create_image(event, pixels=pixels, R=R)  # Create image
                image += event                                   # Add event image to average image
                #image /= np.amax(image)                          # Normalise final image between 0 and 1
                event=max1=max2=max3=None                            # Delete from memory

                a += 1
                if a == event_no:                                 # Break if max sample size for average image is exceeded 
                    return image

                    
                
    
    # Display Images
    elif display == True and type(event_no) == list:
                
        with open(file) as infile:
            for line in infile:

                event = line.strip().split()
                event = pd.Series(event)                         # Turn into Series
                event = preprocess(event)                        # Preprocess
                max1 = find_max1(event)           # Extract maxima
                event = center(event, max1)                    # Center 
                max2 = find_max2(event)
                event = rotate(event, max2)                   # Rotate 
                max3 = find_max3(event)
                event = flip(event, max3)                     # Flip 
                event = create_image(event, pixels=pixels, R=R)  # Create image
                image += event                                   # Add event image to average image
                #image /= np.amax(image)                          # Normalise final image between 0 and 1
                event=max1=max2=max3=None                            # Delete from memory
                
                a += 1
                if a in event_no:
                    sns.heatmap(image, robust=True)
                    plt.show()
#                     sns.heatmap(image)
#                     plt.show()
                    if a >= max(event_no):                        # Break if max sample size for average image is exceeded 
                        break
                    
    
        
    # Return multiple images
    ##### Not working properly
    elif type(event_no) == list:
        images = []                                         # List containing the output images
        
        with open(file) as infile:
            for line in infile:

                event = line.strip().split()
                event = pd.Series(event)                         # Turn into Series
                event = preprocess(event)                        # Preprocess
                max1 = find_max1(event)           # Extract maxima
                event = center(event, max1)                    # Center 
                max2 = find_max2(event)
                event = rotate(event, max2)                   # Rotate 
                max3 = find_max3(event)
                event = flip(event, max3)                     # Flip 
                event = create_image(event, pixels=pixels, R=R)  # Create image
                image += event                                   # Add event image to average image
                #image /= np.amax(image)                          # Normalise final image between 0 and 1
                event=max1=max2=max3=None                            # Delete from memory

                a += 1
                if a in event_no:                                 # Store images
                    images.append(image)
                    if a >= max(event_no):                        # Break if max sample size for average image is exceeded
                        return images

                    
                    
                    
                    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





def preprocess(event):
    '''
    Input: Series (event) to be processed
    Output: Preprocessed Series
    
    -Drops constituents element
    -Replaces NaN values with 0
    -Converts all values to floats
    '''

    
    # Drop constituents 
    event = event.drop(event.index[0])
    
    # Replace NaN with 0
    event = event.fillna(0)

    # Convert values to floats
    event = event.astype(float)
    
    return event

                    
                    
                    
                    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





def find_max1(event):

    '''
    Takes an event and outputs a tuple containing 3 Series, each for the highest pT and its φ, η.
    
    Input: Series (event). 

    Output[0]: [Series of 1st max pT, φ, η]
    '''


    # Separate η, φ, pT
    hdata = event[::3]
    fdata = event[1::3]
    pdata = event[2::3]


    # 1. Extract index of 1st maximum pT
    maxid1 = pdata.idxmax()
    maxlist1 = []

    # 2. Extract max η, φ, pT for event
    if pdata.max() != 0:                                                                     # Brief explanation of if statement below)
        maxlist1.append([event.iloc[maxid1-1], event.iloc[maxid1-2], event.iloc[maxid1-3]])   # From event, add to list the max pT and its η, φ
    else:
        maxlist1.append([0., event.iloc[maxid1-2], event.iloc[maxid1-3]])                    # If max pT is 0, then add it as 0 and not the first value

    # 3. Create series of max pT, η, φ
    max1 = pd.Series(data=maxlist1[0], index=['pT', 'φ', 'η'])

    return max1
    
    
    
    
def find_max2(event):
    
    '''
    Takes an event and outputs a tuple containing 3 Series, each for the highest pT and its φ, η.
    
    Input: Series (event). 
    Output: [Series of 2nd max pT, φ, η]
    '''
    
    # Separate η, φ, pT
    hdata = event[::3]
    fdata = event[1::3]
    pdata1 = event[2::3]


    # 0. 1st pT = 0 to find 2nd Max pT
    pdata = pdata1.copy(deep=True)
    pdata.loc[pdata.idxmax()] = 0

    # 1. Extract index of 2nd max pT
    maxid2 = pdata.idxmax()
    maxlist2 = []
    
    # Extract numerical index of φ of 2nd max pT
    f_id_2 = maxid2 - 1      
    h_id_2 = maxid2 - 2
    
    

    # 2. Extract max η, φ, pT for event
    if pdata.max() != 0:                                                                     # Brief explanation of if statement below)
        maxlist2.append([event.iloc[maxid2-1], event.iloc[maxid2-2], event.iloc[maxid2-3]])   # From event, add to list the max pT and its η, φ
    else:
        maxlist2.append([0., event.iloc[maxid2-2], event.iloc[maxid2-3]])                    # If max pT is 0, then add it as 0 and not the first value

    # 3. Create series of max pT, η, φ
    max2 = pd.Series(data=maxlist2[0], index=['pT', 'φ', 'η'])
    
    return max2
    
    
    
def find_max3(event):
    
    '''
    Takes an event and outputs a Series containing the 3rd highest pT, and its φ, η
    
    Input: Series (event). 
    Output: [Series of 3rd max pT, φ, η]
    '''

    # Separate η, φ, pT
    hdata = event[::3]
    fdata = event[1::3]
    pdata1 = event[2::3]


    # 0. 1st, 2nd pT = 0 to find 3rd Max pT
    pdata = pdata1.copy(deep=True)
    pdata.loc[pdata.idxmax()] = 0
    pdata.loc[pdata.idxmax()] = 0


    # 1. Extract index of 3rd max pT
    maxid3 = pdata.idxmax()
    maxlist3 = []
    


    # 2. Extract max η, φ, pT for event
    if pdata.max() != 0:                                                                     # Brief explanation of if statement below)
        maxlist3.append([event.iloc[maxid3-1], event.iloc[maxid3-2], event.iloc[maxid3-3]])   # From event, add to list the max pT and its η, φ
    else:
        maxlist3.append([0., event.iloc[maxid3-2], event.iloc[maxid3-3]])                    # If max pT is 0, then add it as 0 and not the first value

    # 3. Create series of max pT, η, φ
    max3 = pd.Series(data=maxlist3[0], index=['pT', 'φ', 'η'])

    return  max3

# **Why the if statement?** (note to self) <br />
# Because if maximum pT is 0 in the pdata vector, it picks the ID of the first pT by default as the max (because they're all 0). <br />
# Then, it goes to the non-zero'd event vector and adds its non-zero pT as the max, when the value of that max should clearly have been 0.

# So the if statement fixes this: <br />
# - If max pT != 0, then add it as normal.
# - If max pT = 0, then add '0' as its value instead. (with the coordinates of the first pT, which is incorrect, but this doesn't matter since pT = 0 are not taken into account in the image) <br />     
                    
                    
                    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





def center(event, max1, output='event', R=1.5, pixels=60):
    
    '''
    Centers image around (φ', η') = (0, 0). Both transformations are linear (so far). 
    
    event1: Series (event)
    max123: Tuple of 3 series of max pT, η, φ. Returned by extract_max123() function
    output: 'event' to return a Series of the transformed event1. 'image' to return a transformed dataframe representing an image 
    '''
    
    # Define η, φ indices to be used later
    h_indices = event[::3].index
    f_indices = event[1::3].index

    
    
    # For all η, φ in the event
    for h_index, f_index in zip(h_indices, f_indices):             

        # Define Useful Quantities
        num_index = event.name         # REDUNTANT? REMOVE IT. index of event, so that we can find its corresponding φ in the max123[0] dataframe of max pT's and φ, η's
        maxh = max1.loc['η']                # η of max1 pT value
        maxf = max1.loc['φ']                # φ of max1 pT value
        f = event.iloc[1::3][f_index]            # φ original value
        
        # η Transformation
        event.iloc[::3][h_index] -= maxh         # Subtract max η from current η
        
        # φ Transformation (Note: the if statements take periodicity into account, making sure that range does not exceed 2π)
        if (f - maxf) < -np.pi:
            event.iloc[1::3][f_index] = f + 2*np.pi - maxf

        elif (f - maxf) > np.pi:
            event.iloc[1::3][f_index] = f - 2*np.pi - maxf

        else: 
            event.iloc[1::3][f_index] -= maxf     # Subtract max φ from current φ


    if output == 'event':
        return event
    
    
    elif output == 'image':
        # Initiate bin lists
        bin_h = []
        bin_f = []
        bin_p = []

        # Define max number of constituents 
        max_const = event.shape[0] // 3
        # For all constituents
        for i in range(max_const):
            # Add constituent's η, φ, p to bins
            bin_h.append(list(event.iloc[::3])[i])
            bin_f.append(list(event.iloc[1::3])[i])
            bin_p.append(list(event.iloc[2::3])[i])

        # Turn lists into Series
        bin_h = pd.Series(bin_h)
        bin_f = pd.Series(bin_f)
        bin_p = pd.Series(bin_p)

        # Define no. of bins
        bin_count = np.linspace(-R, R, pixels + 1)

        # Create bins from -R to R and convert to DataFrame
        bins = np.histogram2d(bin_h, bin_f, bins=bin_count, weights=bin_p)[0] # x and y are switch because when the bins were turned into a Series the shape[0] and shape[1] were switched
        image = bins
        
        return image

                    
                    
                    
                    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





def rotate(event, max2):
    '''
    Input: Series (event), max2 series obtained from find_max2()
    Output: Rotated Series (event)
    
    -Calculates the angle of rotation so that φ of 2nd highest pT ends up at (φ', η') = (0, h) for some h > 0 
    -Transforms all η and φ of event using the formulas below (from mathematics) 
    (η' = ηcosθ + φsinθ)
    (φ' = φcosθ - ηsinθ)
    => θ = arctan(φ/η), with if statements taking care of η = 0 cases and making sure η' is positive and not negative
    '''

    # Calculate Angle
    hmax=max2.loc['η']
    fmax=max2.loc['φ']
    
    angle = 0
    
    if (hmax == 0) and (fmax > 0):
        angle = np.pi/2
    elif (hmax == 0) and (fmax < 0):
        angle = -np.pi/2
    elif hmax > 0:
        angle = np.arctan(fmax/hmax)
    elif hmax < 0:
        angle = np.arctan(fmax/hmax) + np.pi
        

    # Rotate Image
    h_indices = event[::3].index
    f_indices = event[1::3].index
    

    for h_index, f_index in zip(h_indices, f_indices): 
        
        h = event.iloc[0::3][h_index]
        f = event.iloc[1::3][f_index]
        
        event.iloc[1::3][f_index] = f*np.cos(angle) - h*np.sin(angle)
        event.iloc[::3][h_index] = f*np.sin(angle) + h*np.cos(angle) 
    
        
    return event

                    
                    
                    
                    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------






def flip(event, max3):
    '''
    Input: Series (event), max3 series obtained from find_max3()
    Output: Flipped Series (event)
    
    -Checks if φ is on left-hand side
    -If yes, it multiplies all φ with -1 to flip the image
    '''
    
    # Check if 2nd highest pT is on left-hand side
    if max3.loc['φ'] < 0:
        
        # Define φ indices for transformation
        f_indices = event[1::3].index
        
        # For all φ 
        for f_index in f_indices: 
            # Multiply φ by -1
            event.iloc[1::3][f_index] *= -1
    
    return event

                    
                    
                    
                    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





def create_image(event, R=1.5, pixels=60):
    
    '''    
    Creates an image of single event.
    
    Input: Series (event)
    Output: ndarray (image)
    '''
    
    # Turn into DataFrame
    event = pd.DataFrame(event).T
    
    # Initiate bin lists
    bin_h = []
    bin_f = []
    bin_p = []

        
    # Add constituent's coordinates to bin lists
    const = event.shape[1] // 3     # For all constituents
    for i in range(const):
        bin_h.append(list(event.iloc[0][::3])[i])
        bin_f.append(list(event.iloc[0][1::3])[i])
        bin_p.append(list(event.iloc[0][2::3])[i])


    # Turn lists into Series
    bin_h = pd.Series(bin_h)
    bin_f = pd.Series(bin_f)
    bin_p = pd.Series(bin_p)

   # Define number & range of bins
    bin_count = np.linspace(-R, R, pixels + 1)

    # Create image (array)
    bins = np.histogram2d(bin_h, bin_f, bins=bin_count, weights=bin_p)[0] # x and y are switch because when the bins were turned into a Series the shape[0] and shape[1] were switched

    # Convert to DataFrame
    image = bins
    
    return image

                    
                    
                    
                    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





def remove_text_inside_brackets(text, brackets="()[]"):
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b: # found bracket
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0  # keep it
                else:  # found bracket to remove
                    break
        else: # character is not a [balanced] bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)

                    
                    
                    
                    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------