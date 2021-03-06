import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from IPython.display import display
import time

# Possibly Redundant
from scipy import ndimage, misc
from skimage.feature import peak_local_max
from skimage import data, img_as_float




# ### 🔴 (Sidetrack) Normalise/Log pT (Still doesn't work without robust)
# Note: the log transformation can simply be done on the final ndarray and it's much more efficient (and doesnt require a function)

# Also tried normalising the whole image using image /= np.amax(image) and results were even worse. **Though this might be a good form to feed it to the algorithm in**

def normalise_p(event1):
    # Define p indices to be used later
    p_indices = event1[2::3].index

    # Create copy of event
    event = event1.copy(deep=True)

    # Calculate sum (only for normalisation)
    total_p = 0
    for p_index in p_indices:
        total_p += event.iloc[2::3][p_index]
    
    # For all p in the event
    for p_index in p_indices:             

        # Define Useful Quantities
        num_index = event.name                   
        
        # p Normalisation
        event.iloc[2::3][p_index] /= total_p                                         # Normalise
        #event.iloc[2::3][p_index] = math.log(event.iloc[2::3][p_index], 10)         # Log Transform

        
#     if output == 'event':
#         return event
    return event







OLD ML Dataset create_dataset

# # # Load Previously saved data
# # sdata = pd.read_pickle('sdata')
# # bdata = pd.read_pickle('bdata')

# # Import, Preprocess, Create Dataset
# sdata = create_dataset('data_signal.dat')
# bdata = create_dataset('data_background.dat')

# # Save Datasets
# pd.DataFrame(sdata).to_pickle('sdata')
# pd.DataFrame(sdata).to_pickle('bdata')

# stest = pd.read_pickle('sdata')
# btest = pd.read_pickle('bdata')

# # Concat and Suffle
# data = np.concatenate((sdata, bdata), axis=0)
# data = sklearn.utils.shuffle(data)

# num_of_examples = data.shape[0]
# print('Total Events:', num_of_examples)

# # Separate Examples and Labels
# data= pd.DataFrame(data, columns=['examples', 'labels'])
# examples= data['examples']         
# labels = data['labels']    

# from sklearn.model_selection import train_test_split

# train_examples, test_examples, train_labels, test_labels = train_test_split(examples, labels, test_size=0.15, random_state=42)

# train_examples, val_examples, train_labels, val_labels = train_test_split(train_examples, train_labels, test_size=0.18, random_state=42)

# print('Train: ', train_examples.shape, train_labels.shape)
# print('Val: ', val_examples.shape, val_labels.shape)
# print('Test: ', test_examples.shape, test_labels.shape)

# train_data = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
# val_data = tf.data.Dataset.from_tensor_slices((val_examples, val_labels))
# test_data = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

# print(train_data)
# print(val_data)
# print(test_data)





















Lame NN

# model = tf.keras.Sequential([
#            tf.keras.layers.Flatten(input_shape=(batch_size, pixels, pixels)),
#            tf.keras.layers.Dense(256, activation = 'relu'),
#            tf.keras.layers.Dropout(0.2),
#            tf.keras.layers.Dense(128, activation = 'relu'),
#            tf.keras.layers.Dropout(0.2),
#            tf.keras.layers.Dense(64, activation = 'relu'),
#            tf.keras.layers.Dropout(0.2),
#            tf.keras.layers.Dense(2, activation = 'softmax')
# ])













# ⭐(Ignore) Step 0: Read the data (tar.gz file) & Explore it
**Read**

As a first step, we unzipped the tar.gz file into a .dat file using 7-zip. 
Then, we convert the .dat file into a string and then into a DataFrame.

.strip() --> remove spaces on the sides

.split() --> separate values by spaces (otherwise we'd get a single conlumn)

**Explore**

**Physics**

Jonas: "The file was produced from a simulation of pp->tt~H where the top decays hadronically
and the anti-top decays leptonically. <br /> I selected events with exactly 1 fat jet with R=1.5."


**Notes**
- The rows represent events (of 1 fat jet each, R = 1.5) 
- The first column represents the number of constituents of the jet  
- The following columns represent the coordinates of the constituents, η, φ, pT, cycling in that order. <br />(e.g. columns 1, 2, 3 are η, φ, pT for the 1st constituent, columns 4, 5, 6 are η, φ, pT for the 2nd constituent etc.)


- -infinity < η < infinity 
- -π < φ < π
- pT[GeV] > 0



# Convert .dat file into string (list comprehension)
datContent = [i.strip().split() for i in open("tth_semihad.dat").readlines()]

# Convert list into DataFrame
mydata = pd.DataFrame(datContent)





# # Display the data
# mydata = mydata.rename(columns={0: 'Const'})
# display(mydata.head())

# # Print statements
# events = mydata.shape[0]
# print('There are {} rows (events).'.format(events))
# print('The maximum number of constituents in an event is {}.'.format((mydata.shape[1] - 1) // 3))

## Display data types
#print('\nData Types: \n', mydata.dtypes)

## Descriptive statistics on data
#mydata.describe()













































def preprocessdf(df1):
    '''
    -Extracts no. of constituents
    -Drops constituents column
    -Replaces NaN values with 0
    -Converts all values to floats
    
    Input: DataFrame to be transformed
    Output: Transformed DataFrame, constituents Series 
    '''
    
    # Create df copy
    df = df1.copy(deep=True)
    
    # Extract constituents column
    df = df['Const']
    
    # Drop constituents from df
    df = df.drop('Const', axis=1)
    
    # Replace NaN with 0
    df = df.fillna(0)

    # Convert values to floats
    df = df.astype(float)
    
    return df, const

# Create Preprocessed DF
#mydata_prep = preprocessdf(mydata)[0]



##### CREATE CENTERED DF 
# ### φ, η transform all events
# centered = []

# start = time.time()

# # Create matrix of transformed events
# for i in range(events):
    
#     centered.append(center(mydata_prep.iloc[i], max123, output='event'))
    
# end = time.time()

# # Turn matrix into DataFrame
# mydata_centered = pd.DataFrame(centered)

# print('Time taken to centre image: {0:.2f}s'.format(end-start))


##### Create MAXIMA123 DF
# maxpt1 = []
# maxpt2 = []
# maxpt3 = []

# start = time.time()

# # For all events, add maxima to & coordinates to list
# for i in range(events):
#     maxpt1.append(extract_max123(mydata_prep.iloc[i])[0])
#     maxpt2.append(extract_max123(mydata_prep.iloc[i])[1])
#     maxpt3.append(extract_max123(mydata_prep.iloc[i])[2])
    
# # Turn lists into DataFrames
# max1 = pd.DataFrame(maxpt1)
# max2 = pd.DataFrame(maxpt2)
# max3 = pd.DataFrame(maxpt3)

# # Create list of DataFrames containing all df for the 3 maxima
# max123 = [max1, max2, max3]

# end = time.time()
# print('Time taken: {0:.2f}s'.format(end-start))








def plot_events(df1, R=1.5, pixels=60, title='(φ\', η\') ∈ [-R, R]',  ylabel='η\'', xlabel='φ\'', df=False):
    
    '''
    Displays an image of single event, or multiple events (input can be either Series or DataFrame). If DataFrame, then average image is created.  
    
    Input: dataframe (multiple events)
    Output: null, just plots image
    
    df: if df=True, then display the image as a DataFrame as well
    '''
    
    # Create copy of df so that it's not accidentally modified
    df = df1.copy(deep=True)
    
    # If input is Series (single event) then turn into DataFrame. This makes it so that single events are processed correctly
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df).T

    # Initiate bin lists
    bin_h = []
    bin_f = []
    bin_p = []

    # Define max number of constituents 
    max_const = df.shape[1] // 3

    # For all rows
    for i in range(df.shape[0]):             

        # For all constituents
        for j in range(max_const):
            # Add constituent's coordinates to bin lists
            bin_h.append(list(df.iloc[i][::3])[j])
            bin_f.append(list(df.iloc[i][1::3])[j])
            bin_p.append(list(df.iloc[i][2::3])[j])

    # Turn lists into Series
    bin_h = pd.Series(bin_h)
    bin_f = pd.Series(bin_f)
    bin_p = pd.Series(bin_p)

   # Define no. of bins
    bin_count = np.linspace(-R, R, pixels + 1)

    # Create bins from -R to R (using bins vector)
    bins = np.histogram2d(bin_h, bin_f, bins=bin_count, weights=bin_p)[0] # x and y are switch because when the bins were turned into a Series the shape[0] and shape[1] were switched

    # Convert to DataFrame
    bins = pd.DataFrame(bins)
    
    if df:
        display(bins)

    # Plot Heat Map
    sns.heatmap(bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def f_transform(event1, max123, output='event', R=1.5, pixels=60):
    
    '''
    event1: the event (row) to be transformed
    max123: list of 3 dataframes of max pT, η, φ. Obtained using the extract_max123() function
    output: 'event' to return a Series of the transformed event1. 'image' to return a transformed dataframe representing an image 
    '''
    # For testing only. Redundant
    mydata_prep = preprocessdf(mydata)[0]
    
    # Define φ indices to be used later
    f_indices = mydata_prep.iloc[0][1::3].index

    # Create copy of event
    event = event1.copy(deep=True)

    # Transformation (Note: the if statements take periodicity into account, making sure that range does not exceed 2π)

    for f_index in f_indices:                    # For all φ in the row

        f = event.iloc[1::3][f_index]            # φ original value
        num_index = event.name                   # index of event, so that we can find its corresponding φ in the max123[0] dataframe of max pT's and φ, η's
        k = max123[0].iloc[num_index]['φ']       # φ of max1 pT value


        if (f - k) < -np.pi:
            event.iloc[1::3][f_index] = f + 2*np.pi - k

        elif (f - k) > np.pi:
            event.iloc[1::3][f_index] = f - 2*np.pi - k

        else: 
            event.iloc[1::3][f_index] -= k                  # Subtract φ corresponding to max pT 
            
            

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
        bins = pd.DataFrame(bins)
        image = bins
        
        return image
    
################# Create dataset after only f transformation
# # Define
# ftrans = []

# start = time.time()

# # Create matrix of transformed events
# for i in range(events):
#     ftrans.append(f_transform(mydata_prep.iloc[i], max123))

# end = time.time()

    
# # Turn matrix into DataFrame
# mydata_fprime = pd.DataFrame(ftrans)

# print('Time taken: {0:.2f}s'.format(end-start))    

################# Compare images before and after only f transformation
# # Single Event 
# e = 15
# plot_events(mydata_prep.iloc[e], title='before φ transformation', xlabel='φ', ylabel='η')

# image=f_transform(mydata_prep.iloc[e], max123, output='image')

# sns.heatmap(image)
# plt.title('3rd Event after φ Transformation')
# plt.xlabel('φ\'')
# plt.ylabel('η')
# plt.show()

# # Average Image
# plot_events(mydata_prep, R=3, title='Before φ transformation with R = 3', xlabel='φ pixels', ylabel='η pixels')
# plot_events(mydata_fprime, R=3, title='After φ transformation (φ\') with R = 3', xlabel='φ\' pixels', ylabel='η pixels')





#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def h_transform(event1, max123, output='event', R=1.5, pixels=60):
    
    '''
    event1: the event (row) to be transformed
    max123: list of 3 dataframes of max pT, η, φ. Obtained using the extract_max123() function
    output: 'event' to return a Series of the transformed event1. 'image' to return a transformed dataframe representing an image 
    '''
    # For testing only. Redundant
    mydata_prep = preprocessdf(mydata)[0]
    
    # Define η, φ indices to be used later
    h_indices = mydata_prep.iloc[0][::3].index

    # Create copy of event
    event = event1.copy(deep=True)

    # For all η, φ in the event
    for h_index in h_indices:             

        # Define Useful Quantities
        num_index = event.name                   # index of event, so that we can find its corresponding φ in the max123[0] dataframe of max pT's and φ, η's
        maxh = max123[0].iloc[num_index]['η']    # η of max1 pT value
        
        # η Transformation
        event.iloc[::3][h_index] -= maxh         # Subtract max η from current η
    
    return event
     
# ### η transform all events
# hh = []

# start = time.time()

# # Create matrix of transformed events
# for i in range(events):
#     hh.append(h_transform(mydata_prep.iloc[i], max123))

# end = time.time()

# # Turn matrix into DataFrame
# mydata_hprime = pd.DataFrame(hh)

# print('Time taken to η trans: {0:.2f}s'.format(end-start))