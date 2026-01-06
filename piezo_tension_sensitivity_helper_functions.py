################################## REQUIRED PACKAGES ########################################################
import cv2
import re
import math
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import widgets
from IPython.display import display



################################## DIC MICROSCOPY IMAGE DATA ANALYSIS ########################################################
def circle_fit(x, r, h, k):
    '''This function calculates the y-coordinate of a circle given the x-coordinate, radius, and center coordinates.

    Inputs:
        x: The x-coordinate(s) at which to calculate the y-coordinate(s) of the circle.
        r: The radius of the circle.
        h: The x-coordinate of the center of the circle.
        k: The y-coordinate of the center of the circle.

    Outputs:
        The y-coordinate(s) of the circle at the given x-coordinate(s).'''
    return k + np.sqrt(r**2 - (x - h)**2)

def get_cropped_image(file_path_name):
    '''This function opens an image, allows the user to select a region of interest (ROI) using a bounding box, 
    crops the image based on the ROI, and returns the cropped image, cropped coordinates, and the original image.

    Inputs:
        file_path_name: The path and filename of the image to be processed.

    Outputs:
        roi_cropped: The cropped image based on the selected ROI.
        cropped_coordinates: The x, y, width, and height of the selected ROI.
        img_raw: The original image.'''
    
    #open image, get coordinates for the x, y (top left corner), w (width to the right), and h (hight down)
    # Naming a window
    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)

    # Using resizeWindow()
    cv2.resizeWindow("Resized_Window", 900, 900)
    img_raw = cv2.imread(file_path_name)[:, :, 0]
    x, y, w, h = cv2.selectROI('Resized_Window', img_raw)
    cropped_coordinates = [x, y, w, h]
    cv2.destroyAllWindows()

    #crop image based on x, y, w, h coordinates and show cropped section
    roi_cropped = img_raw[int(y):int(y+h),int(x):int(x+w)]
    sns.heatmap(roi_cropped, cmap = 'Greys_r', cbar = False)
    plt.show()

    return roi_cropped, cropped_coordinates, img_raw

def find_membrane(roi_cropped):
    '''This function takes a cropped image as input, processes the image to find the coordinates of the darkest pixel 
    in each column (representing the membrane), and returns a dataframe containing the x and y coordinates of the membrane.

    Inputs:
        roi_cropped: The cropped image.

    Outputs:
        membrane_df: A dataframe containing the x and y coordinates of the membrane.'''

    #making an array of repeating large number. This way later argmin won't pick up zeros like it would with np.zero. 1000000 is arbitrary
    #really need to find a clwaner way to do this
    avgpix_array = np.full((roi_cropped.shape[0], roi_cropped.shape[1]), 1000000)

    #go through each colum and find the average value for each chunk of five pixels
    #Add that value to the blank zero array in the center position of each finve pixel chunk
    for i in range(roi_cropped.shape[1]):
        pix_start = 0
        while pix_start + 3 <= roi_cropped.shape[0]:
            pix_to_avg = np.average(roi_cropped[pix_start:pix_start+3, i])
            pix_start += 3
            avgpix_array[pix_start-2, i] = pix_to_avg

    #find darkest pixel (center pixel of 5 pixel chunk) for each column and generate x coordinates to graph with, convert to dataframe
    y_coordinates = np.argmin(avgpix_array, axis = 0)
    x_coordinates = np.linspace(0, roi_cropped.shape[1]-1, roi_cropped.shape[1], dtype=int)
    mem_coordinates = {'x_coordinates':x_coordinates, 'y_coordinates':y_coordinates}
    membrane_df = pd.DataFrame(mem_coordinates)

    #plot and check to make sure overall the membrane is selected
    plt.scatter(membrane_df['x_coordinates'], membrane_df['y_coordinates'], s = 4, color ='red')
    plt.plot(membrane_df['x_coordinates'], membrane_df['y_coordinates'], color = 'gold')
    plt.imshow(roi_cropped)
    plt.show()

    return membrane_df

def fix_mem_loc(df):
    '''This function takes a dataframe containing x and y coordinates of a membrane, allows the user to specify start and end points
    for the membrane, and optionally removes problem points. It returns a filtered dataframe with the selected membrane coordinates.

    Inputs:
      df: The dataframe containing x and y coordinates of the membrane.

    Outputs:
      df_fixed_mem_loc: The filtered dataframe with selected membrane coordinates.

    '''

    # Create a scatter plot of the raw membrane location
    raw_mem_loc_fig = go.Figure(go.Scatter(x=df['x_coordinates'], y=df['y_coordinates']))
    raw_mem_loc_fig.show()

    # Prompt the user to input the start and end points of the membrane
    front_cut = input('Membrane start: ')
    end_cut = input('Membrane end: ')

    # Prompt the user to input problem points (x values) to remove, separated by spaces
    problem_points = input('Problem points (x val) / n: ').split()

    # Filter the dataframe based on the specified start and end points
    df_fixed_mem_loc = df.loc[(df['x_coordinates'] > int(front_cut)) & (df['x_coordinates'] < int(end_cut))]

    # Remove problem points from the dataframe, if specified by the user
    if problem_points[0] != 'n':
        for i in range(len(problem_points)):
            df_fixed_mem_loc = df_fixed_mem_loc[df_fixed_mem_loc['x_coordinates'] != int(problem_points[i])]

        # Create a scatter plot of the fixed membrane location
        fixed_mem_loc_fig = go.Figure(go.Scatter(x=df_fixed_mem_loc['x_coordinates'], y=df_fixed_mem_loc['y_coordinates']))
        fixed_mem_loc_fig.show()
    else:
        # Create a scatter plot of the fixed membrane location
        fixed_mem_loc_fig = go.Figure(go.Scatter(x=df_fixed_mem_loc['x_coordinates'], y=df_fixed_mem_loc['y_coordinates']))
        fixed_mem_loc_fig.show()

    return df_fixed_mem_loc

def _1gaussian(x, amp1,cen1,sigma1):
    '''This function calculates the value of a Gaussian distribution at a given point.

    Inputs:
    x: The point at which to evaluate the Gaussian distribution.
    amp1: The amplitude of the Gaussian curve.
    cen1: The center (mean) of the Gaussian curve.
    sigma1: The standard deviation of the Gaussian curve.

    Outputs:
    The calculated value of the Gaussian distribution at the given point `x`.'''

    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))

def gaus_find_membrane(mem_loc_fixed, roi_cropped):
    '''This function performs Gaussian fitting on the membrane coordinates and the cropped image. It calculates the Gaussian 
    position and sigma for each x-coordinate of the membrane.

    Inputs:
      mem_loc_fixed: The dataframe containing fixed membrane coordinates.
      roi_cropped: The cropped image.

    Outputs:
      df_gaus_data: A dataframe containing the Gaussian positions and sigmas for each x-coordinate of the membrane.
      fig: The plotted figure showing the membrane coordinates and the cropped image.'''

    mem_gaus_position_list = []
    sigma_list = []
    
    # Iterate over each x-coordinate in the fixed membrane coordinates
    for i in mem_loc_fixed['x_coordinates']:
        midpoint = mem_loc_fixed.loc[i][1]
        
        # Generate a list of 1D point locations for fitting
        point_location_1d_list_prefix = np.linspace(midpoint - 10, midpoint + 10, 21)
        shift_value = min(point_location_1d_list_prefix)
        point_location_1d_list_adjusted = point_location_1d_list_prefix - shift_value

        # Extract the corresponding pixel values from the cropped image
        pixel_val_list_prefix = roi_cropped[int(min(point_location_1d_list_prefix)):int(max(point_location_1d_list_prefix)) + 1, mem_loc_fixed.loc[i][0]] * -1
        pixel_val_list_adjusted = pixel_val_list_prefix + (min(pixel_val_list_prefix) * -1)

        # Perform curve fitting using the Gaussian function
        popt, pcov = curve_fit(_1gaussian, point_location_1d_list_adjusted, pixel_val_list_adjusted)

        # Calculate the Gaussian position by adding the shift value
        mem_gaus_point = popt[1] + shift_value
        mem_gaus_position_list.append(mem_gaus_point)
        sigma_list.append(popt[2])

    # Create a dataframe containing the Gaussian positions and sigmas
    df_gaus_data = pd.DataFrame({'gaus_x': mem_loc_fixed['x_coordinates'],
                                'gaus_y': mem_gaus_position_list,
                                'sigma': sigma_list})

    # Plot the membrane coordinates and the cropped image
    fig = plt.figure(figsize=(3, 2), dpi=250)
    plt.scatter(mem_loc_fixed['x_coordinates'], mem_gaus_position_list, s=10, label='data', color='red', edgecolor='black', linewidth=0.5)
    plt.imshow(roi_cropped)

    return df_gaus_data, fig

def remove_low_gaus_sigma(df_gaus_data, roi_cropped):
    '''This function removes data points from df_gaus_data based on a threshold for the sigma value. It creates a new dataframe 
    with the points that have a sigma greater than 1.

    Inputs:
      df_gaus_data: The dataframe containing Gaussian positions and sigmas.
      roi_cropped: The cropped image.

    Outputs:
      df_gaus_data_cutsigma: The new dataframe with points that have a sigma greater than 1.'''

    #make a new dataframe with these points removed
    df_gaus_data_cutsigma=df_gaus_data.loc[df_gaus_data['sigma']>1]
    #getting the cut points to graph
    df_gaus_data_highsigma=df_gaus_data.loc[df_gaus_data['sigma']<1]
    #make figure to show cut points
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_gaus_data_cutsigma['gaus_x'], df_gaus_data_cutsigma['gaus_y'],  s=100, label='data',
               color = 'red', edgecolor='black', linewidth=1)
    ax.scatter(df_gaus_data_highsigma['gaus_x'], df_gaus_data_highsigma['gaus_y'], s=100, label='data',
                color = 'gold', edgecolor='black', linewidth=1)
    ax.imshow(roi_cropped)
    
    return df_gaus_data_cutsigma

def fix_mem_loc_gaus(df_gaus_data):
    '''This function creates a scatter plot of the Gaussian positions in df_gaus_data and allows for the removal of specific 
    points based on user input.

    Inputs:
      df_gaus_data: The dataframe containing Gaussian positions.

    Outputs:
      df_gaus_data: The modified dataframe after removing specific points.'''

    # Create a scatter plot of the raw Gaussian positions
    raw_mem_loc_fig = go.Figure(go.Scatter(x=df_gaus_data['gaus_x'], y=df_gaus_data['gaus_y']))
    raw_mem_loc_fig.show()

    problem_points = input('Problem points (x val) / n: ').split()

    # Remove specific points based on user input
    if problem_points[0] != 'n':
        for i in range(len(problem_points)):
            df_gaus_data = df_gaus_data[df_gaus_data['gaus_x'] != int(problem_points[i])]

    # Create a scatter plot of the fixed Gaussian positions
    fixed_mem_loc_fig = go.Figure(go.Scatter(x=df_gaus_data['gaus_x'], y=df_gaus_data['gaus_y']))
    fixed_mem_loc_fig.show()

    return df_gaus_data

def fit_circle_gaus_weighted(circle_fit, df_gaus_data, roi_cropped):
    '''This function fits a circle to the Gaussian positions in df_gaus_data using a weighted standard deviation approach.

    Inputs:
      circle_fit: The circle fitting function.
      df_gaus_data: The dataframe containing Gaussian positions and sigmas.
      roi_cropped: The cropped image.

    Outputs:
      popt: The optimized parameters of the circle fit.
      fit_stdvs: The standard deviations of the fitted parameters.
    '''

    # Guess initial values for the circle fit parameters
    r_guess = 100
    h_guess = df_gaus_data['gaus_x'].median()
    k_guess = 0
    guess_para = [r_guess, h_guess, k_guess]

    # Perform curve fitting with weighted standard deviation
    popt, pcov = curve_fit(circle_fit, df_gaus_data['gaus_x'], df_gaus_data['gaus_y'], guess_para, maxfev=5000, sigma=df_gaus_data['sigma'])

    # Extract variances and convert to standard deviations for each fit parameter
    pcov_diag = np.diag(pcov).tolist()
    radius_std = pcov_diag[0]
    h_std = pcov_diag[1]
    k_stdv = pcov_diag[2]
    fit_stdvs = [radius_std, h_std, k_stdv]

    # Plot the circular fit along with the original heatmap, scatter of points used, and line graph of connected dots
    fitline = np.linspace(0, roi_cropped.shape[1] - 1, num=1000)
    plt.plot(fitline, circle_fit(fitline, *popt), color='gold', linewidth=4, linestyle='dashed', dashes=(1, 1))
    plt.imshow(roi_cropped)
    plt.plot(df_gaus_data['gaus_x'], df_gaus_data['gaus_y'], color='red')
    plt.show()

    return popt, fit_stdvs

def combined_image_traces(popt, cropped_coordinates, mem_loc_df, img_raw, pressure_step):
    '''This function generates a combined plot of the fitted circle on the original image and calculates the radius and tension of the membrane.

    Inputs:
      popt: The optimized parameters of the circle fit.
      cropped_coordinates: The coordinates of the cropped region (x, y, width, height).
      mem_loc_df: The dataframe containing membrane location data.
      img_raw: The original image.
      pressure_step: The pressure step value.

    Outputs:
      fig: The figure object of the combined plot.
      tension: The calculated tension of the membrane.
      true_r: The calculated radius of the membrane.'''
    
    x, y, h, w = cropped_coordinates
    r, xpoint, ypoint = popt

    #plotting fit onto the original image
    fitline = np.linspace(x-r, x+w+r, num = 1000)

    fig = plt.figure(figsize=(6, 4), dpi=300)

    plt.plot(fitline, circle_fit(fitline, r, xpoint+x, ypoint+y), color = 'red', linewidth = 2, linestyle='dashed', dashes=(0.5, 1))
    plt.imshow(img_raw, cmap='gray')
    plt.show()


    true_r = r * 0.03225 * 10**-6
    #mmHg to F/m^2 conversion factor = 133.322
    tension = ((true_r * (pressure_step * 133.322)))/2*10**3
    print('Radius', round(true_r*10**6, 3),('um'))
    print('Tension', round(tension, 3), 'mN/m')

    return fig, round(tension,3), round(true_r*10**6, 3)

def update_tension_pressure(pressure_tension, tension, pressure_step, circle_fit_data, fit_stdvs):
    '''This function updates the pressure_tension dictionary with the calculated tension, circle fit data, and fit standard deviations for a specific pressure step.

    Inputs:
      pressure_tension: The dictionary storing tension and fit data for each pressure step.
      tension: The calculated tension for the current pressure step.
      pressure_step: The current pressure step.
      circle_fit_data: The optimized parameters and fit data from the circle fit.
      fit_stdvs: The standard deviations of the fit parameters.

    Outputs:
      pressure_tension: The updated pressure_tension dictionary.'''

    # Update the pressure_tension dictionary with the calculated values
    pressure_tension[pressure_step] = [tension, round(circle_fit_data[0] * 0.03225, 3), round(np.sqrt(fit_stdvs[0]) * 0.03225, 3), round(np.sqrt(fit_stdvs[1]) * 0.03225, 3), round(np.sqrt(fit_stdvs[2]) * 0.03225, 3)]

    # Print the updated pressure_tension dictionary
    print(pressure_tension)

    return pressure_tension

def generate_final_tension_df(updated_pressure_tension_dict):
    '''This function generates a final DataFrame containing tension, radius, pressure, radius standard deviation,
    h standard deviation, k standard deviation, and percent standard error values.

    Inputs:
      updated_pressure_tension_dict: The updated pressure_tension dictionary.

    Outputs:
      df_tension_final: The final DataFrame with tension and fit data.'''

    # Initialize empty lists to store the data
    rad_list = []
    tension_list = []
    pressure_list = list(updated_pressure_tension_dict)
    radius_stdv_list = []
    h_stdv_list = []
    k_stdv_list = []

    # Iterate over the updated_pressure_tension_dict and extract the data
    for i in updated_pressure_tension_dict:
        current_list = updated_pressure_tension_dict[i]
        tension_list.append(current_list[0])
        rad_list.append(current_list[1])
        radius_stdv_list.append(current_list[2])
        h_stdv_list.append(current_list[3])
        k_stdv_list.append(current_list[4])

    # Calculate the percent standard error
    percent_std_error_list = [(x/y)*100 for x,y in zip(radius_stdv_list, rad_list)]

    # Create the final DataFrame
    df = pd.DataFrame({'pressure': pressure_list,
                       'rad': rad_list,
                       'tension': tension_list,
                       'radius_stdv': radius_stdv_list,
                       'h_stdv': h_stdv_list,
                       'k_stdv': k_stdv_list,
                       'percent_std_error': percent_std_error_list})

    # Sort the DataFrame by pressure
    df_tension_final = df.sort_values(by=['pressure'])

    return df_tension_final



################################## STRETCH EPHYS DATA ACQUISITION ########################################################

def load_dataframe(file):
  '''Loads and processes the raw ascii file and converts it into a useable dataframe. It removes the headers, empty lines scattered throughout, renames
  the headers, and adds a "sweep" clumn to determine what data belongs to what sweep and therefore what negative pressure.
  
  Inputs:
    file: ascii file generated by the electrophysiology recording software patchmaster.

  Outputs:
    df: organized and processed dataframe containing original file data along with a sweep number column'''

  with open(file, 'r') as fhand:
    #removes spaces and separates string at \n
    raw_file = fhand.read().strip().split('\n')

  line_index = []
  count = 0
  #finding the lines that are not headers/have text in them/are blank and indexing them
  for line in raw_file:
    if re.search(r'[a-z]+', line) == None:
      line_index.append(count)
    count += 1

  #picking out data lines and adding them to this new list of lists
  processed_file = [raw_file[i].strip().replace(" ", "").split(",") for i in line_index]

  #determining the number of sweeps
  #original file has title (1 line) and each sweep has a header (2 lines)
  nsweeps = int((len(raw_file) - len(processed_file)-1)/2)

  #determining column names based on the length of  processed_file[0]
  if len(processed_file[0]) == 5:
      colnames = ['index','ti','i','tp','p']
  else:
      colnames = ['index','ti','i','tp','p','tx','x', 'ty', 'y']

  df = pd.DataFrame(columns = colnames, data = processed_file)
  df = df.apply(pd.to_numeric)
  df = df.dropna(axis=0)

  #adding in sweeps
  datapoint_per_sweep = len(df) / nsweeps
  df['sweep'] = np.repeat(np.arange(nsweeps), datapoint_per_sweep)

  #converting values to more user friendly units
  df['p'] = df['p'] / 0.02
  df['ti'] *= 1000
  df['i'] *= 1e12
  df['tp'] *= 1000

  df = df.drop(labels=['tx', 'x', 'ty', 'y'], axis = 1, inplace=False)
  return(df)

def fit_boltz(x, A,  x0, k):

    """Equation used to fit sigmoidal data.

    Inputs:
        - x = raw data to be fit
        - A=ampliture of the fit (max plateau value)
        - x0=p50 (midpoint of the ride of the fit)
        - k=slope of the fit
    Output:
        -y = values  based on all other parameters"""
    
    y = A / (1 + np.exp(-k*(x-x0)))
    return y

def max_currents(df):
    """Takes the dataframe generated from the original ascii file and 1) determines the max currents for each sweep and 2)generates a summary dataframe that
    can be used for the rest of the ephys analysis. For the max current, rather than taking just the instantaneous/absolute max, this function finds the absolute
    max current then averages it with the currents from five points above and five points below. This is meant to limit the chance an artifiact atrifically drives 
    up the max current.
    
    Inputs:
        - df: oirignal dataframe containing all ephys data.
    Output:
        -df_summary: dataframe with the summary data for that patch. Headings are ['sweep', 'pressure', 'max_current', 'p50_max_current']"""
    
    grouped = df.groupby('sweep')
    sweep_list = []
    pressure_list = []
    max_current_list=[]
    abs_max_current_list = []
    subplot_titles = []

    grouped = df.groupby('sweep')
    for name, group in grouped:
        sweep = group['sweep'].max()
        sweep_string = str(sweep)
        subplot_titles.append(sweep_string)
    fig = make_subplots(rows = 1, cols= int(df['sweep'].max()+1), subplot_titles=subplot_titles)


    for name,group in grouped:
        #isolating area of the sweep where pressure step is being applied
        iso_data = group.loc[(group['ti']>4900) & (group['ti']<5250)]
        max_current_index = iso_data['i'].argmin()

        #averaging 5 above and below max current to smooth data and limit noise
        max_current_val = iso_data['i'].iloc[max_current_index]
        max_current_avg = iso_data['i'].iloc[max_current_index-2:max_current_index+3].mean()
        abs_max_current = iso_data['i'].min()

        if abs(max_current_val) > abs(max_current_avg*1.2):
            print('max current larger than average max current:'+' '+str(name))
        max_current_time = iso_data['ti'].iloc[max_current_index]
        
        #appending each list to make them the same length with proper values
        sweep_list.append(name)
        pressure_list.append(name*5)
        max_current_list.append(max_current_avg*-1)
        abs_max_current_list.append(abs_max_current*-1)

        fig.add_trace(go.Scatter(mode='lines', name=name, x = iso_data['ti'], y=iso_data['i'],  marker=dict(color='black')), row= 1, col= name + 1)          
        fig.add_trace(go.Scatter(x = [4900, 5250], y = [max_current_avg, max_current_avg], marker=dict(color='orange', size = 0)),row= 1, col= name + 1)

        fig.update_layout(
            autosize=True,
            width=5000,
            height=300,
            margin=dict(
                l=10,
                r=10,
                b=10,
                t=40,
                pad=4),
                showlegend=False)


    #create a summary dataframe that will be used for the rest of the notebook
    df_summary=pd.DataFrame({'sweep':sweep_list,
                            'pressure':pressure_list,
                            'max_current':max_current_list,
                            'abs_max_current':abs_max_current_list})
    return df_summary, fig

def create_checkbox_list(df_ephys):
    """Used to manually select what ephys traces need to be fixed. This function will generate a series of checkboxes that have corresponding sweep numbers associated
    with each. For traces that need to be fixed, click the corresponding box. This will generate a list that will be used later and indicates whch seeps you want to fix.
    
    Inputs:
        - df_ephys: oirignal dataframe containing all ephys data.
    Output:
        -selected_items: a list of nubers corresponding to the sweeps that will need to be edited"""
    

    df_sweep_list = [str(x) for x in df_ephys['sweep'].unique().tolist()]
    
    # Create an empty list to store selected items
    selected_items = []

    # Create a function to handle checkbox clicks
    def on_checkbox_click(b):
        if b['new']:
            selected_items.append(b['owner'].description)
        else:
            selected_items.remove(b['owner'].description)

    # Create a checkbox for each item
    checkboxes = [widgets.Checkbox(value=False, description=item, layout={'width': 'max-content'}) for item in df_sweep_list]

    # Attach the click handler to each checkbox
    for cb in checkboxes:
        cb.observe(on_checkbox_click, names='value')

    # Create a horizontal box to display the checkboxes
    hbox = widgets.HBox(checkboxes)

    display(hbox)
    return selected_items

def problem_traces_fig(df, bad_max_current_sweeps): 
    """Used to generate a figure that contains plots of all the sweeps that need to be eidted. Can be used as a reference for the later function fix_problem_trace.
    
    Inputs:
        - df: oirignal dataframe containing all ephys data.
        - bad_max_current_sweeps: list of numbers that corresponds to the sweeps that need edited. Generated from create_checkbox_list function.
    Output:
        -fig: a figure containing all of the sweeps that will be edited later"""

    #generate a figure, one row but as many columns as there are sweeps that need edited
    fig = make_subplots(rows = 1, cols = len(bad_max_current_sweeps))
    for i in range(len(bad_max_current_sweeps)):
        #plot each problem sweep from 4900<t<5300
        sweep_val = bad_max_current_sweeps[i]
        df_to_fix = df.loc[(df['sweep'] == int(sweep_val)) & (df['ti']>4900) & (df['ti']<5300)]
        fig.add_trace(go.Scatter(mode='lines', x = df_to_fix['ti'], y = df_to_fix['i'],  marker = dict(color='black')), row = 1, col = i + 1, )
    fig.update_layout(showlegend=False)
    fig.show()
    return fig

def find_max_current_problem(df, sweep_num, start_t, end_t):
    """Used in the fix_problem_trace function. The function will find the max current between the start and end times given and replot the data. 
    A red dot will indicate the new max current value.
    
    Inputs:
        - df: oirignal dataframe containing all ephys data.
        - sweep_num: a number indicating which sweep in the original dataframe is being edited.
        - start_t: start time (in ms) that the function will use when searching for the max current (lower time limit)
        - end_t: end time (in ms) that the function will use when searching for the max current (upped time limit)

    Output:
        - max_current_avg: the value for newly determined max current. This is the average max current
        - abs_max_current: the value for newly determined max current. This is the absolute max current"""
    
    #isolating area of the sweep where pressure step is being applied
    iso_data = df.loc[(df['ti']>start_t) & (df['ti']<end_t) & (df['sweep'] == sweep_num)]
    max_current_index = iso_data['i'].argmin()

    #averaging 5 above and below max current to smooth data and limit noise
    max_current_val = iso_data['i'].iloc[max_current_index]
    max_current_avg = iso_data['i'].iloc[max_current_index-2:max_current_index+3].mean()
    abs_max_current = iso_data['i'].min()
    max_current_time = iso_data['ti'].iloc[max_current_index]

    #generate a graph of the original sweep overlayed with a red dot to indicate the new max current (average max current)
    df_iso = df.loc[(df['sweep'] == sweep_num) & (df['ti']>4800) & (df['ti']<5400)]
    fig = go.Figure(go.Scatter(mode='lines', x = df_iso['ti'], y=df_iso['i'],  marker=dict(color='black')))
    fig.add_trace(go.Scatter(x = [max_current_time], y = [max_current_avg], marker=dict(color='maroon', size = 20)))
    fig.update_layout(showlegend=False)

    fig.update_layout(
    autosize=True,
    width=500,
    height=300,
    margin=dict(
        l=10,
        r=10,
        b=10,
        t=10,
        pad=4),
        showlegend=False)
    fig.show()
    
    return max_current_avg, abs_max_current

def fix_problem_trace(df, avg_max_current_list, abs_max_current_list, bad_max_current_sweeps):

    """Used to determine the max current for traces that had an erroneous max current from the max_currents function. Ths function will find both the 
    new average max and new absolute max. This function also uses the find_max_current_problem as a way 1) find the max current and 2) visualize where 
    the new max current is to kake sure the function worked.


    Inputs:
        - df: oirignal dataframe containing all ephys data.
        - avg_max_current_list: original list of average max current data.
        - abs_max_current_list: original list of average absolute current data.
        - bad_max_current_sweeps: list of numbers that corresponds to the sweeps that need edited. Generated from create_checkbox_list function.

    Output:
        - avg_max_current_list: updated verion of the ofiginal avg_max_current_list with corrected values
        - abs_max_current_list: updated verion of the ofiginal abs_max_current_list with corrected values"""  
        
    corrected_avg_max_currents = {}
    corrected_abs_max_currents = {}
    #Create text boxes to input the start and end times the max current falls between
    for i in range(len(bad_max_current_sweeps)):
        sweep_num = int(bad_max_current_sweeps[i])
        print('Ranges for sweep number', sweep_num)
        x_low_val = float(input('x lower limit value: '))
        x_high_val = float(input('x upper limit value: '))
        print('')
        print('')
        #use find_max_current_problem to find new max current (avg and abs) and then plot them to check visually
        max_current_avg, abs_max_current = find_max_current_problem(df, sweep_num, x_low_val, x_high_val)

        corrected_avg_max_currents[sweep_num] = max_current_avg
        corrected_abs_max_currents[sweep_num] = abs_max_current

    #correct avg and abs max lists by replacing sweep(key) list index with the new current (value)
    for key, value in corrected_avg_max_currents.items():
        avg_max_current_list[key] = value*-1

    for key, value in corrected_abs_max_currents.items():
        abs_max_current_list[key] = value*-1

    return avg_max_current_list, abs_max_current_list 

def fix_rundown(df_summary):

    '''Correct for rundown. This function takes the summary data, determines the max current in the p50 curve, and then removes
    any sweep with a max current >75% of the max p50 current.
  
  Inputs:
    df_summary: summary data containing the max currents for each pressure step

  Outputs:
    df_summary: the updated summary dataframe with sweeps with >75% max p50 current removed'''

    #determining max current and the current cutoff
    max_index=df_summary['max_current'].argmax()
    average_max=df_summary['max_current'].iloc[max_index-1: max_index+2].mean()
    cutoff_value=average_max*0.75
    remove_list=[]
    for i in range(len(df_summary['max_current'])):
        #skipping over pressure steps before the max p50 current is achieved
        if i<max_index:
            continue

    #going through each sweep after the max p50 current and eliminating sweeps with a max current <75% the max p50 current
        else:
            if df_summary['max_current'].iloc[i]<cutoff_value:
                print(str(df_summary['pressure'].iloc[i])+'mmHg : '+str(df_summary['max_current'].iloc[i])+' pA')
                # df_summary.drop(labels=i, inplace=True)
                remove_list.append(i)
    df_summary=df_summary[~df_summary['sweep'].isin(remove_list)]
    return df_summary

def generate_p50_nonnorm(df_summary):
    '''Takes the summary df generated originally from the max_currents function and performs a P50 fit on the non-normalized data. The fit is constrained to +/-30pA of the max current.
     This data, along with the max current determined by the plateau of the fit, are then added to the summary dataframe (df_summary).
     
     Inputs:
        -df_summary: summary dataframe originally generated from the max_currents function.

    Outputs:
        -df_summary: updated version of the inpur df_summary that now contains a p50 and p50_max_current column'''
    #doing all of this first for the average max current values 
    p50_avg_max_current_list=[]

    #setting arbitrary upper and lower bounds to the fit is not constrained but its close to where the true plateau should be
    amin = df_summary['max_current'].max()-30
    amax = df_summary['max_current'].max()+30
    popt, pcov = curve_fit(fit_boltz, df_summary['pressure'], df_summary['max_current'],
                           bounds=([amin, -np.inf, -np.inf], [amax, np.inf, np.inf]))

    #adding p50 and max from p50 values to respective lists with the same length as the number of pressure steps for that patch
    for i in range(len(df_summary)):
        p50_avg_max_current_list.append(popt[0])

    #updating the summary dataframe to contain the p50 and max current determined by the plateau of the p50 fit
    df_summary['p50_avg_max_current'] = p50_avg_max_current_list


    #plotting the data as a visual check that the fit is reasonable
    xdata_gen = np.linspace(0, df_summary['pressure'].max(), 500)
    ydata_gen = fit_boltz(xdata_gen, *popt)
    print('Plateau value from Boltzman Fit: ', popt[0], 'pA')
    plt.scatter(df_summary['pressure'], df_summary['max_current'], color='lightslategrey',s=70, edgecolors='black')
    plt.plot(xdata_gen, ydata_gen, linewidth=5, color='tomato')
    plt.show()

    #repeating this process for the mabolute max current values
    p50_abs_max_current_list=[]

    #setting arbitrary upper and lower bounds to the fit is not constrained but its close to where the true plateau should be
    amin = df_summary['abs_max_current'].max()-30
    amax = df_summary['abs_max_current'].max()+30
    popt, pcov = curve_fit(fit_boltz, df_summary['pressure'], df_summary['abs_max_current'],
                           bounds=([amin, -np.inf, -np.inf], [amax, np.inf, np.inf]))
    
    #adding p50 and max from p50 values to respective lists with the same length as the number of pressure steps for that patch
    for i in range(len(df_summary)):
        p50_abs_max_current_list.append(popt[0])

    #updating the summary dataframe to contain the p50 and max current determined by the plateau of the p50 fit
    df_summary['p50_abs_max_current'] = p50_abs_max_current_list

    return df_summary

def generate_norm_current(df_summary):
    '''Takes the summary df generated from the generate_p50 function and calculates/adds a normalized current column. The currents are normalized
    to the max current value determined by the plateau of the p50 fit (A in the fit_boltz function).
     
     Inputs:
        -df_summary: summary dataframe  generated from the generate_p50 function.

    Outputs:
        -df_summary: updated version of the inpur df_summary that now contains a normalized_current column'''
    #doing all of this first for the average max current values 
    #normalizes each max current to the p50 fit determined max current
    avg_norm_current_list = [x/df_summary['p50_avg_max_current'].iloc[0] for x in df_summary['max_current']]

    #plotting as a sanity check to make sure the current plot retains the same shape just the y axis is now normalized
    plt.scatter(df_summary['pressure'], avg_norm_current_list,  color='lightslategrey',s=70, edgecolors='black')
    plt.ylabel('Normalized Current') #add y axis title
    plt.xlabel('Pressure (mmHg)') #add x axis title
    plt.show()

    #adding the normalized currents to the original summary dataframe
    df_summary['avg_norm_current'] = avg_norm_current_list
    
    #repeating this process for the abolute max current values
    #normalizes each max current to the p50 fit determined max current
    abs_norm_current_list = [x/df_summary['p50_abs_max_current'].iloc[0] for x in df_summary['abs_max_current']]

    #adding the normalized currents to the original summary dataframe
    df_summary['abs_norm_current'] = abs_norm_current_list

    return df_summary

def generate_p50_norm(df_summary):
    '''Takes the summary df generated originally from the max_currents function and performs a P50 fit on the normalized data. The fit is constrained to +/-30pA of the max current.
     This data, along with the max current determined by the plateau of the fit, are then added to the summary dataframe (df_summary).
     
     Inputs:
        -df_summary: summary dataframe originally generated from the max_currents function.

    Outputs:
        -df_summary: updated version of the inpur df_summary that now contains a p50 and p50_max_current column'''
    
    #doing all of this first for the average max current values 
    p50_list_avgm=[]
    k_avgm_slope_list=[]
    #setting arbitrary upper and lower bounds to the fit is not constrained but its close to where the true plateau should be
    amin = df_summary['avg_norm_current'].max()-0.2
    amax = df_summary['avg_norm_current'].max()+0.2
    popt, pcov = curve_fit(fit_boltz, df_summary['pressure'], df_summary['avg_norm_current'])
    print(popt)

    #adding p50 and max from p50 values to respective lists with the same length as the number of pressure steps for that patch
    for i in range(len(df_summary)):
        p50_list_avgm.append(popt[1])
        k_avgm_slope_list.append(popt[2])

    #updating the summary dataframe to contain the p50 and max current determined by the plateau of the p50 fit
    df_summary['p50_avgm'] = p50_list_avgm
    df_summary['k_avgm_slope'] = k_avgm_slope_list

    #plotting the data as a visual check that the fit is reasonable
    xdata_gen = np.linspace(0, df_summary['pressure'].max(), 500)
    ydata_gen = fit_boltz(xdata_gen, *popt)
    print('P50 value from Boltzman Fit: ', popt[1], 'mmHg')
    print('k value from Boltzman Fit: ', popt[2])
    
    #plotting P50
    plt.scatter(df_summary['pressure'], df_summary['avg_norm_current'], color='orange',s=70, edgecolors='black', zorder=3)
    plt.plot(xdata_gen, ydata_gen, linewidth=5, color='blue', zorder=2)

    #plotting lines to show P50 value
    plt.plot([0, popt[1]], [0.5, 0.5], linewidth=5, color='red', linestyle='dotted', zorder=1)
    plt.plot([popt[1], popt[1]], [0, 0.5], linewidth=5, color='red', linestyle='dotted', zorder=1)
    plt.scatter(popt[1], 0.5, color='red', s=100, edgecolors='black', zorder=4)
    plt.ylabel('Normalized Current') #add y axis title
    plt.xlabel('Pressure (mmHg)') #add x axis title
    plt.show()

    #repeating this process for the mabolute max current values
    p50_list_absm=[]
    k_absm_slope_list=[]

    #setting arbitrary upper and lower bounds to the fit is not constrained but its close to where the true plateau should be
    amin = df_summary['abs_norm_current'].max()-0.2
    amax = df_summary['abs_norm_current'].max()+0.2
    popt, pcov = curve_fit(fit_boltz, df_summary['pressure'], df_summary['abs_norm_current'],
                           bounds=([amin, -np.inf, -np.inf], [amax, np.inf, np.inf]))
    
    #adding p50 and max from p50 values to respective lists with the same length as the number of pressure steps for that patch
    for i in range(len(df_summary)):
        p50_list_absm.append(popt[1])
        k_absm_slope_list.append(popt[2])

    #updating the summary dataframe to contain the p50 and max current determined by the plateau of the p50 fit
    df_summary['p50_absm'] = p50_list_absm
    df_summary['k_absm_slope'] = k_absm_slope_list

    return df_summary

def steady_state_currents(df_ephys, df_summary):
    """Used to determine the steady state currents (both raw and nromalized) for each pressure step of each sweep for one patch. This fiunction will
    average the currents for all values recorded in the last 100ms of each pressure step to get the raw ssc. For normalized ssc, this value is just divided
    by the avg max current for the sweep.


    Inputs:
        - df_ephys: oirignal dataframe containing all ephys data.
        - df_summary: dataframe containing summary data such as max current, P50, k

    Output:
        - df_summary: updated verion of the unput df_summary that now has columns for raw and normalized ssc data for each sweep"""  

    ssc_list = []
    ssc_norm_list = []
    grouped = df_summary.groupby('sweep')
    
    #determine the raw ssc for each sweep
    for name, group in grouped:
        iso_ephys=df_ephys.loc[df_ephys['sweep']==name]
        time_start = iso_ephys.loc[iso_ephys['ti'] == 5200].index.values[0]
        time_end = iso_ephys.loc[iso_ephys['ti'] == 5300].index.values[0]
        ssc_time_iso = iso_ephys.truncate(before = time_start, after = time_end)
        ssc_avg = ssc_time_iso['i'].mean()*-1
        ssc_list.append(ssc_avg)
        
    #divide each raw ssc by its corresponding sweep avg max current to get normalized ssc
    for i in range(len(df_summary)):
        ssc_norm_list.append(ssc_list[i] / df_summary['max_current'].iloc[i])
        
    #update the summary df to have raw and nromalized ssc data added
    df_summary['ssc_raw']=ssc_list
    df_summary['ssc_norm']=ssc_norm_list
    
    return df_summary




    fig,ax = plt.subplots(figsize=(10, 5))

    test_data=df_ephys_cut.loc[df_ephys_cut['sweep']==df_ephys_cut['sweep'].max()]
    
    #getting time of max current
    start_index_current=test_data['i'].argmin() #getting the max current value index
    start_time_current=test_data['ti'].iloc[start_index_current] #converting that index into time
    start_cut_df=test_data.loc[(test_data['ti']>950) & (test_data['ti']<1550)] 

    #plotting the current and max current time
    ax.plot(start_cut_df['ti'], start_cut_df['i'], color='darkblue', linewidth=3)
    ax.scatter(start_time_current, test_data['i'].iloc[start_index_current], s=100, color='gold', zorder=2, edgecolor='black')
    ax.set_xlim(1000, 1040)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Current (pA)')

    #getting time of max poke
    start_index_poke=test_data['d'].argmax() #getting the max poke value index
    start_time_poke=test_data['td'].iloc[start_index_poke] #converting that index into time

    ax = ax.twinx()
    
    #plotting indeptation and max indentation time
    ax.plot(start_cut_df['td'], start_cut_df['d'], color='maroon', linewidth=3)
    ax.scatter(start_time_poke, test_data['d'].iloc[start_index_poke], s=100, color='gold', zorder=2, edgecolor='black')
    ax.set_xlim(1000, 1040)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Indentation (um)')
    
    #plotting vertical lines to show delta t
    ax.axvline(x=start_time_current, color='blue', linewidth=3)
    ax.axvline(x=start_time_poke, color='red', linewidth=3)
    
    plt.subplots_adjust(wspace=0.4)  # Increase the horizontal spacing
    plt.tight_layout()

    delay_time=start_time_current-start_time_poke
    
    df_poke['current_delay']=[delay_time]*len(df_poke)
    return df_poke