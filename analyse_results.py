
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#"Y:/users/Valerie/Loeff-Kerssemakers-et-al-AutoStepFinder/steppy_results/ASF_noisy_data_1_FitX.csv"
#"C:/Users/Valerie/Documents/Master/Ries Group/Stepfinder_Data Simulator/noisy_data.txt"

FitX_file = "Y:/users/Valerie/results/ASF_results/simulated data/ASF_noisy_data_typIV_FitX.csv"
input_data = "Y:/users/Valerie/data/Simulated_data/data_typIV.txt"
BNP_result = "Y:/users/Valerie/results/BNP_results/simulated_data/50000_BNB_noisy_data_typIV.csv"

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
        plt.plot(time, ground_truth, label='Ground Truth', color='yellow')

    # Add axis names and legend
    plt.xlabel('time in ms')
    plt.ylabel('position in nm')
    plt.legend()
    # Display the plot
    plt.show()

plot_data(input_data, FitX_file, BNP_result)

def plot_x(input_data):
    # Read the data from the CSV file
    data = pd.read_csv(input_data, delimiter=',', header=None)

    # Extract the data
    time = data.iloc[:, 0]
    position = data.iloc[:, 1]

    plt.scatter(time, position, marker='o', s=0.1, color='black')

    # Add axis names and legend
    plt.xlabel('time in ms')
    plt.ylabel('position in nm')
    # Display the plot
    plt.show()

def load_data_for_analysis(data_path, file_type=None):
    # load data
        ## include file_type ASF if input is AutoStepfinder FitX result data
        ## Note: Only works if input is in txt format!
    data = pd.read_csv(data_path, delimiter=',', header=None)
    if file_type == "ASF":
        return data
    else:
        ground_truth = data[2]
        return ground_truth.to_frame()

def analyse_data(data_path, file_type=None):
    #load data:
        # typ is either ASF (for AutoStepfinder result file), or input_file for analysis of the ground truth
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

    print(type(result_df))



#analyse_data(FitX, file_type="ASF")
#analyse_data(input_data)




