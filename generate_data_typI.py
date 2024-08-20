import numpy as np

'''
Generate Data of typI: Normal (step height=8, dwelltime=1/10 of number of datapoints(=300), number of steps=10, S/N ratio = 1/3

Call generate_txt_file to generate input data of TypI for AutoStepfinder, BNP-step and ground truth without noise

Arguments:
stepsize = default set to 8nm
number of steps = default set to 10
number of datapoints = equivalent to to time, default set to 300
SN_ration = Signal to noise ratio, default set to 1/3 (decision based on AutoStepfinder paper, optimal results expected for that ratio)
'''
def generate_goundtruth(stepsize=8, number_of_steps=10, number_of_dp=300):
    dwelltime = number_of_dp/number_of_steps
    time = range(number_of_dp)
    generated_data = []
    for t in time:
        position = int(t/dwelltime) * stepsize
        generated_data.append(position)
    return(generated_data)

# add noise to ground truth
def generate_noisy_data(stepsize=8, number_of_steps=10, number_of_dp=300, SN_ratio=1/3):
    raw_data = generate_goundtruth(stepsize, number_of_steps, number_of_dp)
    noise = np.random.normal(0, SN_ratio * stepsize, number_of_dp)
    noisy_data = raw_data + noise
    return(noisy_data, raw_data)


# create txt input_files for ASF, BNP-Step and further analysis
def generate_txt_file(stepsize=8, number_of_steps=10, number_of_dp=300, SN_ratio=1/3):
    time = range(number_of_dp)
    noisy_data = generate_noisy_data(stepsize, number_of_steps, number_of_dp, SN_ratio)
    for t in time:
        list_noisy = [t, float(round(noisy_data[0][t],5))]
        list_groundtruth = [t, float(round(noisy_data[1][t],5))]
        with open("output/BNB_noisy_data_typI_SN_08.txt", 'a') as f:
            print(str(list_noisy[0])+","+str(list_noisy[1]), file=f)
        f.close()
        with open("output/ASF_noisy_data_typI_08.txt", 'a') as f:
            print(list_noisy[1], file=f)
        f.close()
        with open("output/data_typI_08.txt", 'a') as f:
            print(str(list_noisy[0])+","+str(list_noisy[1])+","+str(list_groundtruth[1]), file=f)
        f.close()


generate_txt_file(SN_ratio=0.8)

#plt.plot(x)
#plt.show()
