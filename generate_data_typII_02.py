import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.stats import gamma

'''
Generate Data of typII but with sinus curve (freq: 240 Hz) added to baseline:

Arguments:
stepsize = default set to 8nm
max_lenght = equivalent to total time, default set to 300
SN_ratio = Signal to noise ratio, default set to 1/3 (decision based on AutoStepfinder paper, optimal results expected for that ratio)
'''

def sample_dwelltime(max_length=300, step_size=8):
    # Initialize the list to hold step sizes
    steps = []
    # Keep track of the total length
    total_length = 0
    stepsize = step_size
    while total_length < max_length:
        # Sample a single dwell time from the gamma distribution. Shape and scale were chosen by trial an error... the distribution was supposed to be close to exponential and the selected values around 30 ms
        dwell_time = np.random.gamma(shape=2, scale=14)
        dwelltime = int(dwell_time)

        # Check if adding this dwelltime would exceed the maximum length of the dataset. If it exceeds, than the new dwell time goes until the end of the dataset
        if total_length + dwelltime > max_length:
            dwelltime = max_length - total_length

        # Append the step size for the duration of the dwell time
        steps.extend([stepsize] * dwelltime)

        # Update the total length
        total_length += dwelltime
        # Increase the step size for the next dwell time
        stepsize = stepsize + step_size
    return steps

'''testrun on sample_sinus_dwelltime
x=sample_sinus_dwelltime()
plt.plot(x)
plt.show()'''

# add noise to ground truth
def generate_sinus_noisy_data(max_length=300, step_size=8, SN_ratio=1/3):
    x = np.linspace(0, max_length, num=300)
    raw_data = sample_dwelltime(max_length, step_size)
    sinus_noise = raw_data + + (1.4 * np.sin(2 * np.pi * 0.24 * x))
    noise = np.random.normal(0, SN_ratio * step_size, max_length)
    noisy_data = sinus_noise + noise
    return (noisy_data, raw_data)

'''x = generate_sinus_noisy_data()
plt.plot(x[0])
plt.show()'''

# create txt input_files for ASF, BNP-Step and further analysis
def generate_txt_file(max_length=300, step_size=8, SN_ratio=1/3):
    time = range(max_length)
    noisy_data = generate_sinus_noisy_data(max_length, step_size, SN_ratio)
    for t in time:
        list_noisy = [t, float(round(noisy_data[0][t], 5))]
        list_groundtruth = [t, float(round(noisy_data[1][t], 5))]
        with open("output/BNP_noisy_data_typII_02_SN_05.txt", 'a') as f:
            print(str(list_noisy[0]) + "," + str(list_noisy[1]), file=f)
        f.close()
        with open("output/ASF_noisy_data_typII_02_SN_05.txt", 'a') as f:
            print(list_noisy[1], file=f)
        f.close()
        with open("output/data_typII_02_SN_05.txt", 'a') as f:
            print(str(list_noisy[0]) + "," + str(list_noisy[1]) + "," + str(list_groundtruth[1]), file=f)
        f.close()

generate_txt_file(SN_ratio=0.5)






