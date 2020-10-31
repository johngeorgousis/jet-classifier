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





##### CODE TO DISPLAY IMAGE FOR MULTIPLE EVENTS (READ FROM FILE)

# pixels = 100                                        # Image Resolution, int.
# frequency = [1, 100, 500, 3000, 8000, 12000]        # Image Display Frequency (display at no. of samples in list)
# image = np.zeros((pixels, pixels))                  # Define initial image
# a = 0                                               # Define Counter



# with open("tth_semihad.dat") as infile:
#     for line in infile:
        
#         event=line.strip().split()
#         event = pd.Series(event)                    # Turn into Series
#         event = preprocess(event)                   # Preprocess
#         max123 = extract_max123(event)              # Extract maxima
#         event = center(event, max123)               # Center 
#         #event = rotate(event, max123)               # Rotate 
#         #event = flip(event, max123)                 # Flip 
#         event = create_image(event, pixels=pixels)  # Create image
#         image += event                              # Add event image to average image
#         event = max123 = None                       # Delete from memory
        
#         a += 1
#         if a in frequency:
#             sns.heatmap(image, robust=True)
#             plt.show()



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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


def preprocess(event1):
    '''
    -Extracts no. of constituents
    -Drops constituents column
    -Replaces NaN values with 0
    -Converts all values to floats
    
    Input: Series (event) to be processed
    Output: Processed Series, constituents Series 
    '''
    
    # Create series copy
    event = event1.copy(deep=True)
    
    # Extract constituents 
    #const = event['Const']
    # Drop constituents from row
    event = event.drop(event.index[0])
    
    # Replace NaN with 0
    event = event.fillna(0)

    # Convert values to floats
    event = event.astype(float)
    
    return event, const

# # Create Preprocessed DF
# mydata_prepr = [preprocess(mydata.iloc[i])[0] for i in range(mydata.shape[0])]

# mydata_prepr = pd.DataFrame(mydata_prepr)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
    
def extract_max123(event1):

    '''
    Input: event (row). 
    e.g. mydata_prep.iloc[0]

    Output[0]: [Series of 1st max p, φ, η]
    Output[1]: [Series of 2nd max p, φ, η]
    Output[2]: [Series of 3rd max p, φ, η]
    '''


    # Create event copy
    event = event1.copy(deep=True)

    # Separate η, φ, pT
    hdata = event[::3]
    fdata = event[1::3]
    pdata1 = event[2::3]



    # 1. Extract index of maximum pT
    maxid1 = pdata1.idxmax()
    maxlist1 = []

    # 2. Extract max η, φ, pT for event
    if pdata1.max() != 0:                                                                     # Brief explanation of if statement below)
        maxlist1.append([event.iloc[maxid1-1], event.iloc[maxid1-2], event.iloc[maxid1-3]])   # From event, add to list the max pT and its η, φ
    else:
        maxlist1.append([0., event.iloc[maxid1-2], event.iloc[maxid1-3]])                    # If max pT is 0, then add it as 0 and not the first value

    # 3. Create & Display dataframe of max pT, η, φ
    row_max1 = pd.Series(data=maxlist1[0], index=['pT', 'φ', 'η'])




    # 0. Set Max pT to 0 to find next Max pT
    pdata2 = pdata1.copy(deep=True)
    pdata2.loc[maxid1] = 0

    # 1. Extract index of maximum pT
    maxid2 = pdata2.idxmax()
    maxlist2 = []

    # 2. Extract max η, φ, pT for event
    if pdata2.max() != 0:                                                                     # Brief explanation of if statement below)
        maxlist2.append([event.iloc[maxid2-1], event.iloc[maxid2-2], event.iloc[maxid2-3]])   # From event, add to list the max pT and its η, φ
    else:
        maxlist2.append([0., event.iloc[maxid2-2], event.iloc[maxid2-3]])                    # If max pT is 0, then add it as 0 and not the first value

    # 3. Create & Display dataframe of max pT, η, φ
    row_max2 = pd.Series(data=maxlist2[0], index=['pT', 'φ', 'η'])




    # 0. Set Max pT to 0 to find next Max pT
    pdata3 = pdata2.copy(deep=True)
    pdata3.loc[maxid2] = 0

    # 1. Extract index of maximum pT
    maxid3 = pdata3.idxmax()
    maxlist3 = []

    # 2. Extract max η, φ, pT for event
    if pdata3.max() != 0:                                                                     # Brief explanation of if statement below)
        maxlist3.append([event.iloc[maxid3-1], event.iloc[maxid3-2], event.iloc[maxid3-3]])   # From event, add to list the max pT and its η, φ
    else:
        maxlist3.append([0., event.iloc[maxid3-2], event.iloc[maxid3-3]])                    # If max pT is 0, then add it as 0 and not the first value

    # 3. Create & Display dataframe of max pT, η, φ
    row_max3 = pd.Series(data=maxlist3[0], index=['pT', 'φ', 'η'])



    return row_max1, row_max2, row_max3


##### Create DF
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


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def center(event1, max123, output='event', R=1.5, pixels=60):
    
    '''
    Centers image around (φ', η') = (0, 0). Both transformations are linear (so far). 
    
    event1: the event (row) to be transformed
    max123: list of 3 dataframes of max pT, η, φ. Obtained using the extract_max123() function
    output: 'event' to return a Series of the transformed event1. 'image' to return a transformed dataframe representing an image 
    '''
    # For testing only. Redundant
    mydata_prep = preprocessdf(mydata)[0]
    
    # Define η, φ indices to be used later
    h_indices = mydata_prep.iloc[0][::3].index
    f_indices = mydata_prep.iloc[0][1::3].index

    # Create copy of event
    event = event1.copy(deep=True)

    # For all η, φ in the event
    for h_index, f_index in zip(h_indices, f_indices):             

        # Define Useful Quantities
        num_index = event.name                   # index of event, so that we can find its corresponding φ in the max123[0] dataframe of max pT's and φ, η's
        maxh = max123[0].iloc[num_index]['η']    # η of max1 pT value
        maxf = max123[0].iloc[num_index]['φ']    # φ of max1 pT value
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
        bins = pd.DataFrame(bins)
        image = bins
        
        return image
    
    
    
   ##### CREATE DF 
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

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


