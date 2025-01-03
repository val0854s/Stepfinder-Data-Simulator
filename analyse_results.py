import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib.colors as mcolors
import os
from scipy.stats import norm

'''input_data = "C:/Users/lsega/Documents/Master/Ries Group/data/simulated data/noisy_data_typII_02/data_typII_02.txt"
ASF_Steps = "C:/Users/lsega/Documents/Master/Ries Group/results/ASF_results/simulated data/ASF_data_typII/ASF_noisy_data_typII_02_FitX.csv"
BNP_Steps = "C:/Users/lsega/Documents/Master/Ries Group/results/BNP_results/simulated data/BNP_data_typII/50000_BNP_noisy_data_typII_02.csv"
BNP_TypI = "C:/Users/lsega/Documents/Master/Ries Group/results/BNP_results/simulated data/BNP_data_typI"
BNP_TypII = "C:/Users/lsega/Documents/Master/Ries Group/results/BNP_results/simulated data/BNP_data_typII"'''

BNP_exp_mPAINT = "C:/Users/lsega/Documents/Master/Ries Group/results/BNP_results/experimental data/50000_D421C_mPAINT.csv"
BNP_exp_K560C = "C:/Users/lsega/Documents/Master/Ries Group/results/BNP_results/experimental data/50000_life_K560C_new.csv"
BNP_exp_K560N = "C:/Users/lsega/Documents/Master/Ries Group/results/BNP_results/experimental data/50000_life_K560N.csv"
ASF_exp_mPAINT = "C:/Users/lsega/Documents/Master/Ries Group/results/ASF_results/experimental data/ASF_D421C_mPAINT_FitX.csv"
ASF_exp_K560C = "C:/Users/lsega/Documents/Master/Ries Group/results/ASF_results/experimental data/ASF_life_K560C_new_FitX.csv"
ASF_exp_K560N = "C:/Users/lsega/Documents/Master/Ries Group/results/ASF_results/experimental data/ASF_life_K560N_FitX.csv"



'''
Function to plot ASF and BNP result

Arguments:
    input_data = path to input file (used for BNP-step/AutoStepfinder)
    FitX = path to AutoStepfinder result
    BNP_result = path to BNP-Stepfinder result
    show_gt = set to True for synthetic data and set to False for experimental data (no ground truth)
'''
def plot_data(input_data, FitX_file, BNP_result, show_gt=True):
    # Read the data from the CSV file
    data = pd.read_csv(input_data, delimiter=',', header=None)
    ASF_data = pd.read_csv(FitX_file, delimiter=',', header=None)
    BNP_data = pd.read_csv(BNP_result, delimiter=',', header=0)

    # Extract the data
    time = data.iloc[:, 0]
    position = data.iloc[:, 1]
    ASF_Steps = ASF_data.iloc[:, 0]
    BNP_Steps = BNP_data.iloc[:, 1]

    #plot data
    plt.scatter(time, position, marker='o', s=0.1, color='black')
    plt.plot(time, ASF_Steps, label='AutoStepfinder', color='blue')
    plt.plot(time, BNP_Steps, label='BNP-Step', color='red')

    #if data is synthetic, plot ground truth as well, else set show_gt=False
    if show_gt==True:
        ground_truth = data.iloc[:, 2]
        plt.plot(time, ground_truth, label='Ground Truth', color='black')

    # Add axis names and legend
    plt.xlabel('time in ms')
    plt.ylabel('position in nm')
    plt.legend()
    # Display the plot
    #plt.show()
    # Save plot
    plt.savefig("test_substep_T324C_1mM_sample1_ASFvsBNP.png", bbox_inches='tight')

#plot_data(K560_new, FitX_file, BNP_result, show_gt=False)

'''
Function to calculate the mean squared error

Arguments:
    input_data = path to input file (used for BNP-step/AutoStepfinder)
    FitX = path to AutoStepfinder result
    BNP_result = path to BNP-Stepfinder result
    show_gt = set to True for synthetic data and set to False for experimental data (no ground truth)
'''
def mean_square_error(input_data, FitX_file, BNP_result, show_gt=True):
    #load data
    data = pd.read_csv(input_data, delimiter=',', header=None)
    ASF_data = pd.read_csv(FitX_file, delimiter=',', header=None)
    BNP_data = pd.read_csv(BNP_result, delimiter=',', header=0)

    # Extract the data
    #time = data.iloc[:, 0]
    ASF_Steps = ASF_data.iloc[:, 0].values
    BNP_Steps = BNP_data.iloc[:, 1].values

    # if data is synthetic, plot ground truth, else set show_gt=False
    if show_gt == True:
        ground_truth = data.iloc[:, 2]

    #MSE_ASF = np.mean(np.square(ASF_Steps - ground_truth))
    #MSE_BNP = np.mean(np.square(BNP_Steps - ground_truth))

    MSE_ASF = mean_squared_error(ASF_Steps, ground_truth)
    MSE_BNP = mean_squared_error(BNP_Steps, ground_truth)

    print("MSE for" + input_data)
    print("MSE for ASF: " + str(MSE_ASF))
    print("MSE for BNP: " + str(MSE_BNP))

    #possibly plot MSE
    '''x = np.square(ASF_Steps - ground_truth)
    y = np.square(BNP_Steps - ground_truth)

    plt.plot(x, label='AutoStepfinder', color='blue')
    plt.plot(y, label='BNP-Step', color='red')
    plt.legend()
    plt.show()
    arr = pd.DataFrame({"BNP": y, "ASF": x})
    print(arr)'''

#mean_square_error(input_data, ASF_Steps, BNP_Steps)

'''
Analyse data (average stepsize and dwelltime according to ASF or BNP-Step)

'''

def load_data_for_analysis(data_path, file_type=None):
    # load data
        #file_type = "ASF" for AutoStepfinder Result and "BNP" for BNP-Step results

    if file_type == "ASF":
        data = pd.read_csv(data_path, delimiter=',', header=None)
        return data
    elif file_type == "BNP":
        data = pd.read_csv(data_path, delimiter=',', header=0)
        return data.iloc[:,1].to_frame()
    else:
        data = pd.read_csv(data_path, delimiter=',', header=None)
        ground_truth = data[2]
        return ground_truth.to_frame()


def analyse_data(data_path, file_type=None):
    # load data using inbuild function:
        # typ is either:
        # ASF (for AutoStepfinder result file),
        # BNP (for BNP-Step result),
        # or input_file for analysis of the ground truth

    data = load_data_for_analysis(data_path, file_type)
    results = []
    previous_row = None

    # interater though every row and check for steps by comparing current step hight
    for i in range(1, len(data)):
        current_row = data.iloc[i].values
        previous_row = data.iloc[i - 1].values

        #if there is a difference in between the rows -> step is detected, dwell time, stephight, level before and after step safed, as well as the index of the step in the data
        if not np.array_equal(current_row, previous_row):
            step_position = i
            step_size = current_row - previous_row
            dwell_time = step_position - results[-1][0] if results else i
            level_before = previous_row
            level_after = current_row

            results.append((step_position, step_size.item(), dwell_time, level_before.item(), level_after.item()))

    # results are stored in a pandas dataframe for further analysis
    results_df = pd.DataFrame(results, columns=['step_position', 'step_size', 'dwell_time', 'level_before', 'level_after'])
    return(results_df)


#remove outliers
def remove_outlier(df_in, col_name):
    # Interquartile range
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    #remove outliers from column
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


# preprocessing of data received using analyse_data function and calculation of mean stepsize and dwelltime
def evaluate_data_analysis(ASF_result, BNP_result, flag=False):

    #analyse data
    ASF_analysis_result = analyse_data(ASF_result, file_type="ASF")
    BNP_analysis_result = analyse_data(BNP_result, file_type="BNP")

    #drop negative results / backstepping of kinesin
    BNP_rm_negativevalues = BNP_analysis_result[BNP_analysis_result >= 0].dropna()
    ASF_rm_negativevalues = ASF_analysis_result[ASF_analysis_result >= 0].dropna()

    #remove outliers according to IQR
    BNP_rm_outliers1 = remove_outlier(BNP_rm_negativevalues, 'step_size')
    BNP_rm_outliers = remove_outlier(BNP_rm_outliers1, 'dwell_time')
    ASF_rm_outliers1 = remove_outlier(ASF_rm_negativevalues, 'step_size')
    ASF_rm_outliers = remove_outlier(ASF_rm_outliers1, 'dwell_time')

    if flag == True:
        BNP_rm_outliers = BNP_rm_outliers[BNP_rm_outliers["step_position"]<450]
        ASF_rm_outliers = ASF_rm_outliers[ASF_rm_outliers["step_position"]<450]

    #calculate mean values for both datasets
    ASF_mean_values = pd.Series([ASF_rm_outliers.step_size.mean(), ASF_rm_outliers.dwell_time.mean()],
                                index=['mean stepsize', 'mean dwelltime'])
    BNP_mean_values = pd.Series([BNP_rm_outliers.step_size.mean(), BNP_rm_outliers.dwell_time.mean()],
                                index=['mean stepsize', 'mean dwelltime'])

    #concat results into new table
    frames = [BNP_mean_values, ASF_mean_values]
    concat_results = pd.concat(frames, axis= 1)
    cols = ['BNP', 'ASF']
    concat_results.columns = cols

    print(concat_results)


evaluate_data_analysis(BNP_result=BNP_exp_K560N,ASF_result=ASF_exp_K560N, flag=True)

















######################################################################################################################################
# for synthetic data? But throws together results achieved using different SN ratios...

def analyse_batch_data(directory, file_type=None):
    frames = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            print(f)
            file_path = os.path.abspath(os.path.join(dirpath, f))
            results = analyse_data(file_path, file_type=file_type)
            mean_table = pd.Series([results.step_size.mean(),results.dwell_time.mean()],index=['step_size', 'dwell_time'])
            frames.append(mean_table)
    y = pd.concat(frames,axis=1)
    print(y)

#analyse_batch_data(BNP_TypII, file_type="BNP")

####################################################################################
def extract_substep_data(filepath):
    #load txt as array
    arr = np.loadtxt(filepath, delimiter="\t", skiprows=1)
    #transform into pandas dataframe
    df = pd.DataFrame(arr)
    #kinesin position is probably stored in first column?
    position = df[0]
    t = df[1]
    plt.plot(position, t)

    #find indices of all tracs
    indices_end = []
    indices_start = [0]
    for i in range(len(position)):
        if position[i] == 0 and position[i-1] != 0:
            indices_end.append(i)
        elif i > 0 and position[i] != 0 and position[i-1] == 0:
            indices_start.append(i)

    #print(indices_end)
    #print(indices_start)
    test = position.iloc[indices_start[9]:indices_end[9],]
    test_time = range(len(test))
    #plt.plot(test_time, test)
    #plt.show()
    #test.to_csv('test_substep_T324C_10uM_sample1.txt', sep=' ', header=False, index=False)


#extract_substep_data(data3)



