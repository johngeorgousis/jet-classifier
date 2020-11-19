import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from IPython.display import display
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

# # Create Preprocessed DF
# mydata_prepr = [preprocess(mydata.iloc[i]) for i in range(mydata.shape[0])]

# mydata_prepr = pd.DataFrame(mydata_prepr)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------










#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Method 1: Numerical rotation. Rotate until 2nd highest φ (f_id_2) absolute value is less than 0.1 


from scipy import ndimage

def rotate(evenτ, max123):
    
    # Define η, φ indices to be used later
    h_indices = event[::3].index
    f_indices = event[1::3].index
    
    hmax=max123[1]['η']
    fmax=max123[1]['φ']
    
    if (hmax == 0) and (fmax > 0):
        angle = 90
    
    elif (hmax == 0) and (fmax < 0):
        angle = -90
        
    elif hmax > 0:
        angle = np.arctan(fmax/hmax) / np.pi * 180
        
    elif hmax < 0:
        angle = np.arctan(fmax/hmax) / np.pi * 180 + 180
    
    # For all η, φ in the event
    for h_index, f_index in zip(h_indices, f_indices): 
        num_index = event.name
        
        # φ, η transform
        
        h = event.iloc[0::3][h_index]
        f = event.iloc[1::3][f_index]
        if f != 0 and h != 0:
            #event.iloc[::3][h_index] = (((h**2) * np.sin(angle)) / f) + (f**2 * np.cos(angle) / h)
            event.iloc[::3][h_index] = f*np.sin(angle) + h*np.cos(angle)
            event.iloc[1::3][f_index] -= max123[1]['φ']
        
    return event










#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test(event, pixels=60, R=1.5, display=False):
    
    '''
    Given an event from my_data return an image (this is for quick testing of transformation steps)
    '''

    image = np.zeros((pixels, pixels))
    
    event = pd.Series(event)                         # Turn into Series
    event = preprocess(event)                        # Preprocess
    max123, f_id_2, flip_img = extract_max123(event)           # Extract maxima
    event = center(event, max123)                    # Center 
    #event = rotate(event, f_id_2)                   # Rotate 
    event = flip(event, flip_img)                     # Flip 
    event = create_image(event, pixels=pixels, R=R)  # Create image
    image += event                                   # Add event image to average image
    #image /= np.amax(image)                          # Normalise final image between 0 and 1
    event = max123 = None                            # Delete from memory
   
    
    if display:
        sns.heatmap(image)
        return None
    
    return image    

def test_no_flip(event, pixels=60, R=1.5, display=False):
    
    '''
    Given an event from my_data return an image (this is for quick testing of transformation steps)
    '''

    image = np.zeros((pixels, pixels))
    
    event = pd.Series(event)                         # Turn into Series
    event = preprocess(event)                        # Preprocess
    max123, f_id_2, flip_img = extract_max123(event)           # Extract maxima
    event = center(event, max123)                    # Center 
    #event = rotate(event, f_id_2)                   # Rotate 
    event = create_image(event, pixels=pixels, R=R)  # Create image
    image += event                                   # Add event image to average image
    #image /= np.amax(image)                          # Normalise final image between 0 and 1
    event = max123 = None                            # Delete from memory
   
    
    if display:
        sns.heatmap(image)
        return None
    
    return image    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------











#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def average_image(pixels=60, R=1.5, event_no=12178, display=False):
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
        
        with open("tth_semihad.dat") as infile:
            for line in infile:

                event=line.strip().split()
                event = pd.Series(event)                         # Turn into Series
                event = preprocess(event)                        # Preprocess
                max123, f_id_2, flip_img = extract_max123(event)           # Extract maxima
                event = center(event, max123)                    # Center 
                #event = rotate(event, f_id_2)                   # Rotate 
                event = flip(event, flip_img)                     # Flip 
                event = create_image(event, pixels=pixels, R=R)  # Create image
                image += event                                   # Add event image to average image
                #image /= np.amax(image)                          # Normalise final image between 0 and 1
                event = max123 = None                            # Delete from memory

                a += 1
                if a == event_no:                                 # Break if max sample size for average image is exceeded 
                    return image

                    
                
    
    # Display Images
    elif display == True and type(event_no) == list:
                
        with open("tth_semihad.dat") as infile:
            for line in infile:

                event=line.strip().split()
                event = pd.Series(event)                         # Turn into Series
                event = preprocess(event)                        # Preprocess
                max123, f_id_2, flip_img = extract_max123(event)           # Extract maxima
                event = center(event, max123)                    # Center 
                event = rotate(event, max123)                   # Rotate 
                #event = flip(event, flip_img)                     # Flip 
                event = create_image(event, pixels=pixels, R=R)  # Create image
                #event = rotate(event, max123)
                image += event                                   # Add event image to average image
                #image /= np.amax(image)                          # Normalise final image between 0 and 1
                event = max123 = None                            # Delete from memory
                
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
        
        with open("tth_semihad.dat") as infile:
            for line in infile:

                event=line.strip().split()
                event = pd.Series(event)                         # Turn into Series
                event = preprocess(event)                        # Preprocess
                max123, f_id_2, flip_img = extract_max123(event)           # Extract maxima
                event = center(event, max123)                    # Center 
                #event = rotate(event, f_id_2)                   # Rotate 
                event = flip(event, flip_img)                     # Flip 
                event = create_image(event, pixels=pixels, R=R)  # Create image
                image += event                                   # Add event image to average image
                #image /= np.amax(image)                          # Normalise final image between 0 and 1
                event = max123 = None                            # Delete from memory

                a += 1
                if a in event_no:                                 # Store images
                    images.append(image)
                    if a >= max(event_no):                        # Break if max sample size for average image is exceeded
                        return images
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------










#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def create_image(event, R=1.5, pixels=60):
    
    '''    
    Creates an image of single event.
    
    Input: Series (event)
    Output: ndarray (image)
    '''
    
    # Turn into pd.Series
    #event = pd.Series(event)
    event = pd.DataFrame(event).T
    
    # If input is Series (single event) then turn into DataFrame. This makes it so that single events are processed correctly
#     if isinstance(event, pd.Series):
#         event = pd.DataFrame(event).T

    # Initiate bin lists
    bin_h = []
    bin_f = []
    bin_p = []

    # Define max number of constituents 
    max_const = event.shape[1] // 3

    # For all rows
    #for i in range(event.shape[0]):             

    # For all constituents (I tested it using only meaningful constituents from first column and the code was slower)
    for i in range(max_const):
        # Add constituent's coordinates to bin lists
        bin_h.append(list(event.iloc[0][::3])[i])
        bin_f.append(list(event.iloc[0][1::3])[i])
        bin_p.append(list(event.iloc[0][2::3])[i])

# Tried not doing it for pT=0 constituents. Was less efficient
#     i = 0
#     while i < max_const and list(event.iloc[0][2::3])[i] != 0.:
#         bin_h.append(list(event.iloc[0][::3])[i])
#         bin_f.append(list(event.iloc[0][1::3])[i])
#         bin_p.append(list(event.iloc[0][2::3])[i])
#         i += 1

    

    # Turn lists into Series
    bin_h = pd.Series(bin_h)
    bin_f = pd.Series(bin_f)
    bin_p = pd.Series(bin_p)

   # Define no. of bins
    bin_count = np.linspace(-R, R, pixels + 1)

    # Create bins from -R to R (using bins vector)
    bins = np.histogram2d(bin_h, bin_f, bins=bin_count, weights=bin_p)[0] # x and y are switch because when the bins were turned into a Series the shape[0] and shape[1] were switched

    # Convert to DataFrame
    image = bins
    
    return image
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------










#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def extract_max123(event):

    '''
    Takes an event and outputs a tuple containing 3 Series, each for the highest pT and its φ, η.
    
    Input: Series (event). 

    Output[0]: [Series of 1st max pT, φ, η]
    Output[1]: [Series of 2nd max pT, φ, η]
    Output[2]: [Series of 3rd max pT, φ, η]
    Output[3]: [numerical index of φ of 2nd highest pT]
    Output[4]: [boolean of whether to flip (True) or not flip (False) image]
    '''


    # Separate η, φ, pT
    hdata = event[::3]
    fdata = event[1::3]
    pdata1 = event[2::3]



    # 1. Extract index of 1st maximum pT
    maxid1 = pdata1.idxmax()
    maxlist1 = []

    # 2. Extract max η, φ, pT for event
    if pdata1.max() != 0:                                                                     # Brief explanation of if statement below)
        maxlist1.append([event.iloc[maxid1-1], event.iloc[maxid1-2], event.iloc[maxid1-3]])   # From event, add to list the max pT and its η, φ
    else:
        maxlist1.append([0., event.iloc[maxid1-2], event.iloc[maxid1-3]])                    # If max pT is 0, then add it as 0 and not the first value

    # 3. Create series of max pT, η, φ
    row_max1 = pd.Series(data=maxlist1[0], index=['pT', 'φ', 'η'])




    # 0. Set Max pT to 0 to find 2nd Max pT
    pdata2 = pdata1.copy(deep=True)
    pdata2.loc[maxid1] = 0

    # 1. Extract index of 2nd max pT
    maxid2 = pdata2.idxmax()
    maxlist2 = []
    
    # Extract numerical index of φ of 2nd max pT
    f_id_2 = maxid2 - 1      
    h_id_2 = maxid2 - 2
    
    

    # 2. Extract max η, φ, pT for event
    if pdata2.max() != 0:                                                                     # Brief explanation of if statement below)
        maxlist2.append([event.iloc[maxid2-1], event.iloc[maxid2-2], event.iloc[maxid2-3]])   # From event, add to list the max pT and its η, φ
    else:
        maxlist2.append([0., event.iloc[maxid2-2], event.iloc[maxid2-3]])                    # If max pT is 0, then add it as 0 and not the first value

    # 3. Create series of max pT, η, φ
    row_max2 = pd.Series(data=maxlist2[0], index=['pT', 'φ', 'η'])
    



    # 0. Set Max pT to 0 to find 3rd Max pT
    pdata3 = pdata2.copy(deep=True)
    pdata3.loc[maxid2] = 0

    # 1. Extract index of 3rd maximum pT
    maxid3 = pdata3.idxmax()
    maxlist3 = []
    
    # Determine whether image will be flipped
    flip_img = None
    if event.iloc[maxid3-2] < 0:
        flip_img = True

    # 2. Extract max η, φ, pT for event
    if pdata3.max() != 0:                                                                     # Brief explanation of if statement below)
        maxlist3.append([event.iloc[maxid3-1], event.iloc[maxid3-2], event.iloc[maxid3-3]])   # From event, add to list the max pT and its η, φ
    else:
        maxlist3.append([0., event.iloc[maxid3-2], event.iloc[maxid3-3]])                    # If max pT is 0, then add it as 0 and not the first value

    # 3. Create series of max pT, η, φ
    row_max3 = pd.Series(data=maxlist3[0], index=['pT', 'φ', 'η'])

    
    max123 = (row_max1, row_max2, row_max3)
    
    return  max123, f_id_2, flip_img
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def extract_max123(event):

    '''
    Takes an event and outputs a tuple containing 3 Series, each for the highest pT and its φ, η.
    
    Input: Series (event). 

    Output[0]: [Series of 1st max pT, φ, η]
    Output[1]: [Series of 2nd max pT, φ, η]
    Output[2]: [Series of 3rd max pT, φ, η]
    Output[3]: [numerical index of φ of 2nd highest pT]
    Output[4]: [boolean of whether to flip (True) or not flip (False) image]
    '''


    # Separate η, φ, pT
    hdata = event[::3]
    fdata = event[1::3]
    pdata1 = event[2::3]



    # 1. Extract index of 1st maximum pT
    maxid1 = pdata1.idxmax()
    maxlist1 = []

    # 2. Extract max η, φ, pT for event
    if pdata1.max() != 0:                                                                     # Brief explanation of if statement below)
        maxlist1.append([event.iloc[maxid1-1], event.iloc[maxid1-2], event.iloc[maxid1-3]])   # From event, add to list the max pT and its η, φ
    else:
        maxlist1.append([0., event.iloc[maxid1-2], event.iloc[maxid1-3]])                    # If max pT is 0, then add it as 0 and not the first value

    # 3. Create series of max pT, η, φ
    row_max1 = pd.Series(data=maxlist1[0], index=['pT', 'φ', 'η'])

    
    
    
    
    
    # Separate η, φ, pT
    hdata = event[::3]
    fdata = event[1::3]
    pdata1 = event[2::3]


    # 0. Set Max pT to 0 to find 2nd Max pT
    pdata2 = pdata1.copy(deep=True)
    pdata2.loc[pdata1.idxmax()] = 0

    # 1. Extract index of 2nd max pT
    maxid2 = pdata2.idxmax()
    maxlist2 = []
    
    # Extract numerical index of φ of 2nd max pT
    f_id_2 = maxid2 - 1      
    h_id_2 = maxid2 - 2
    
    

    # 2. Extract max η, φ, pT for event
    if pdata2.max() != 0:                                                                     # Brief explanation of if statement below)
        maxlist2.append([event.iloc[maxid2-1], event.iloc[maxid2-2], event.iloc[maxid2-3]])   # From event, add to list the max pT and its η, φ
    else:
        maxlist2.append([0., event.iloc[maxid2-2], event.iloc[maxid2-3]])                    # If max pT is 0, then add it as 0 and not the first value

    # 3. Create series of max pT, η, φ
    row_max2 = pd.Series(data=maxlist2[0], index=['pT', 'φ', 'η'])
    



    # 0. Set Max pT to 0 to find 3rd Max pT
    pdata3 = pdata2.copy(deep=True)
    pdata3.loc[maxid2] = 0

    # 1. Extract index of 3rd maximum pT
    maxid3 = pdata3.idxmax()
    maxlist3 = []
    
    # Determine whether image will be flipped
    flip_img = None
    if event.iloc[maxid3-2] < 0:
        flip_img = True

    # 2. Extract max η, φ, pT for event
    if pdata3.max() != 0:                                                                     # Brief explanation of if statement below)
        maxlist3.append([event.iloc[maxid3-1], event.iloc[maxid3-2], event.iloc[maxid3-3]])   # From event, add to list the max pT and its η, φ
    else:
        maxlist3.append([0., event.iloc[maxid3-2], event.iloc[maxid3-3]])                    # If max pT is 0, then add it as 0 and not the first value

    # 3. Create series of max pT, η, φ
    row_max3 = pd.Series(data=maxlist3[0], index=['pT', 'φ', 'η'])

    
    max123 = (row_max1, row_max2, row_max3)
    
    return  max123, f_id_2, flip_img








#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def center(event, max123, output='event', R=1.5, pixels=60):
    
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
        maxh = max123[0].loc['η']                # η of max1 pT value
        maxf = max123[0].loc['φ']                # φ of max1 pT value
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

def flip(event, flip_img):
    
    # Check if 2nd highest pT is on left-hand side
    if flip_img:
        
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
