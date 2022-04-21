from TICC_solver import TICC
import numpy as np
import time
import os

data_path = os.path.join(os.path.dirname(__file__),'../../data/')

def effect_of_length(time_series_id):
    data = np.loadtxt(data_path+'synthetic_data_for_segmentation/test'+str(time_series_id)+'.csv', delimiter=',')
    data = np.concatenate([data[:,:4] for x in range(15)])

    ticc = TICC(window_size=5, number_of_clusters=5, lambda_parameter=1e-4, beta=2200, maxIters=1, threshold=2e-4,
                write_out_file=False, prefix_string="output_folder/", num_proc=1)

    time_list = []
    for length in range(1,21):
        time_start=time.time()
        ticc.fit_transform(data[:length*10000,:])
        time_end=time.time()
        print(length,time_end-time_start)
        time_list.append(time_end-time_start)
    time_list = np.array(time_list)
    print(time_list.round(2))
    return time_list.round(2)

def effect_of_dimension(time_series_id):
    time_list = []
    data = np.loadtxt(data_path+'../data/synthetic_data_for_segmentation/test'+str(time_series_id)+'.csv', delimiter=',')
    data = np.hstack([data, data, data, data, data])
    data = np.vstack([data, data])
    data = data[:30000,:]
    for i in range(1,21):
        time_start=time.time()
        ticc = TICC(window_size=3, number_of_clusters=5, lambda_parameter=1e-4, beta=2200, maxIters=5, threshold=1e-4,
                write_out_file=False, prefix_string="output_folder/", num_proc=1)
        ticc.fit_transform(data[:, :i])
        time_end=time.time()
        print(i, time_end-time_start)
        time_list.append(time_end-time_start)
    print(np.array(time_list).round(2))
    return np.array(time_list).round(2)

if __name__ == '__main__':
    # run 10 times and get the average performance.
    # for i in range(1,11):
    #     t_list = effect_of_length(i)
    #     np.savetxt(os.path.dirname(__file__)+'/effect_of_length/time'+str(i)+'.txt', t_list)

    # for i in range(5):
    #     t_list = effect_of_dimension(i)
    #     np.savetxt(os.path.dirname(__file__)+'/effect_of_dim/time'+str(i)+'.txt', t_list)
    
    time_list = []
    for i in range(1,2):
        time_list.append(np.loadtxt(os.path.dirname(__file__)+'/effect_of_length/time'+str(i)+'.txt'))
    result = np.vstack(time_list)
    print(result.shape)
    result_list = []
    for i in range(20):
        result_list.append(np.mean(result[:,i]))
    print(np.array(result_list).round(2))
