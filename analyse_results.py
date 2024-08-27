
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FitX_file = "Y:/users/Valerie/results/ASF_results/experimental data/ASF_life_K560N_FitX.csv"
input_data = "Y:/users/Valerie/data/experimental data/life_K560N.csv"
BNP_result = "Y:/users/Valerie/results/BNP_results/experimental data/50000_life_K560N.csv"

exp_gt = "Y:/users/Valerie/data/simulated data/noisy_data_typI/data_typI.txt"
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

    plt.scatter(time, position, marker='o', s=0.1, color='black')
    plt.plot(time, ASF_Steps, label='AutoStepfinder', color='blue')
    plt.plot(time, BNP_Steps, label='BNP-Step', color='red')

    #if data is synthetic, plot ground truth, else set show_gt=False
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
    plt.savefig("data_typI_SN_08_ASFvsBNP.png", bbox_inches='tight')

#plot_data(input_data, FitX_file, BNP_result)

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

    MSE_ASF = np.mean(np.square(ASF_Steps - ground_truth))
    MSE_BNP = np.mean(np.square(BNP_Steps - ground_truth))

    print("MSE for" + input_data)
    print("MSE for ASF: " + str(MSE_ASF))
    print("MSE for BNP: " + str(MSE_BNP))

    '''x = np.square(ASF_Steps - ground_truth)
    y = np.square(BNP_Steps - ground_truth)

    plt.plot(x, label='AutoStepfinder', color='blue')
    plt.plot(y, label='BNP-Step', color='red')
    plt.legend()
    plt.show()
    arr = pd.DataFrame({"BNP": y, "ASF": x})
    print(arr)'''


#mean_square_error(input_data, FitX_file, BNP_result)

def load_data_for_analysis(data_path, file_type=None):
    # load data
        ## include file_type ASF if input is AutoStepfinder FitX result data
        ## Note: Only works if input is in txt format!
    if file_type == "ASF":
        data = pd.read_csv(data_path, delimiter=',', header=None)
        return data
    elif file_type == "BNP":
        data = pd.read_csv(BNP_result, delimiter=',', header=0)
        return data.iloc[:,1].to_frame()
    else:
        data = pd.read_csv(data_path, delimiter=',', header=None)
        ground_truth = data[2]
        return ground_truth.to_frame()

def analyse_data(data_path, file_type=None):
    #load data:
        # typ is either:
        # ASF (for AutoStepfinder result file),
        # BNP (for BNP-Step result),
        # or input_file for analysis of the ground truth
    data = load_data_for_analysis(data_path, file_type)
    results = []
    previous_row = None

    for i in range(1, len(data)):
        current_row = data.iloc[i].values
        previous_row = data.iloc[i - 1].values

        if not np.array_equal(current_row, previous_row):
            step_position = i
            step_size = current_row - previous_row
            dwell_time = step_position - results[-1][0] if results else i
            level_before = previous_row
            level_after = current_row

            results.append((step_position, float(step_size), dwell_time, float(level_before), float(level_after)))


    results_matrix = np.array(results, dtype=object)
    result_df = pd.DataFrame(results_matrix, columns=['step_position', 'step_size', 'dwell_time', 'level_before', 'level_after'])

    print(result_df)


#analyse_data(BNP_result, file_type="BNP")
#analyse_data(input_data)

data = "C:/Users/Valerie/Documents/Master/Ries Group/T324C/1mM/K3241mMresults by dates20210618sample2.txt"
data2 = "C:/Users/Valerie/Documents/Master/Ries Group/T324C/100uM/K324100uMresults by dates20210219sample1.txt"
data3 = "C:/Users/Valerie/Documents/Master/Ries Group/T324C/10uM/K32410uMg27+g44_10uMresults by dates20210909sample1.txt"

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
    plt.show()
    #test.to_csv('test_substep_T324C_10uM_sample1.txt', sep=' ', header=False, index=False)


x = "Y:/users/Valerie/results/ASF_results/experimental data/test_substep_T324C_10uM_sample1_FitX.csv"
y = pd.read_csv(x, delimiter=',', header=None)

extract_substep_data(data3)


#plt.plot(y)
#plt.show()


