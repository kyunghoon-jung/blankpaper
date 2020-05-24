import numpy as np
import glob
import sys
sys.path.insert(0, "/home/sonic/Coding/Git/CPMG_Analysis/2nd_automation/")
sys.path.insert(0, "/home/sonic/Coding/Git/CPMG_Analysis/Paper_Draft_Data/")
from decom_utils_side import *
from models import *
init_notebook_mode(connected=True) 
import time
np.set_printoptions(suppress=True)
import itertools
from multiprocessing import Pool 
POOL_PROCESS = 23  
FILE_GEN_INDEX = 2 
pool = Pool(processes=POOL_PROCESS)  
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
from adabound import AdaBound

import matplotlib.pyplot as plt 

print("Generation of datasets excuted.", time.asctime())
tic = time.time()

SAVE_DIR = '/data1/HPC/dataset_N256/'
SAVE_DIR = '/data1/HPC/Models_N256/'

PRE_PROCESS = False
PRE_SCALE = 1

def return_combination_A_lists(chosen_indices, full_chosen_indices, cut_threshold=2):
    total_combination = [] 
    for temp_idx in chosen_indices: 
        indices = [] 
        if type(temp_idx) == np.int64: 
            temp_idx = np.array([temp_idx]) 
        for j in full_chosen_indices: 
            abs_temp = np.abs(temp_idx - j) 
            if len(abs_temp[abs_temp<cut_threshold]) == 0:
                indices.append(j) 
        temp_idx = list(temp_idx) 
        total_combination += [temp_idx+[j] for j in indices]
    return np.array(total_combination) 

def return_total_hier_index_list(A_list, cut_threshold=2):
    total_index_lists = []

    A_list_length = len(A_list) 
    if (A_list_length==3): return np.array([[[1]]])
    if (A_list_length==4): return np.array([[[1]], [[2]]])
    if (A_list_length==5): return np.array([[[1],[2],[3]], [[1,3]]])

    if A_list_length%2 == 0:
        final_idx = A_list_length//2 
    else:
        final_idx = A_list_length//2 + 1

    full_chosen_indices = np.arange(1, A_list_length-1) 
    half_chosen_indices = np.arange(1, final_idx) 
    temp_index = return_combination_A_lists(half_chosen_indices, full_chosen_indices) 

    while 1:
        if (A_list_length>=10) & (A_list_length<12):
            if len(temp_index[0])>=2: total_index_lists.append(temp_index)
        elif (A_list_length>=12) & (A_list_length<15): 
            if len(temp_index[0])>=3: total_index_lists.append(temp_index)
        elif (A_list_length>=15):
            if len(temp_index[0])>=4: total_index_lists.append(temp_index)
        else:
            total_index_lists.append(temp_index)
        temp_index = return_combination_A_lists(temp_index, full_chosen_indices) 
        if len(temp_index) == 0:break 
    return np.array(total_index_lists) 
  
MAGNETIC_FIELD = 403.553                        # Unit: Gauss
GYRO_MAGNETIC_RATIO = 1.0705*1000               # Unit: Herts 
WL_VALUE = MAGNETIC_FIELD*GYRO_MAGNETIC_RATIO*2*np.pi

'''
parser = argparse.ArgumentParser(description='parameter assign')

parser.add_argument('-cuda', required=True, type=int, help='choice of cuda device. type: int')
parser.add_argument('-pulse', required=True, type=int, help='CPMG pulse (N). type: int')
parser.add_argument('-width', required=True, type=int, help='image width. type: int')
parser.add_argument('-time', required=True, type=int, help='number of data points used. type: int')
parser.add_argument('-bmin', required=True, type=int, help='minimum boundary of B (Hz). type: int')
parser.add_argument('-bmax', required=True, type=int, help='maximum boundary of B (Hz). type: int')
parser.add_argument('-noise', required=True, type=float, help='maxmum noise value (scale: M value). type: float')
parser.add_argument('-path', required=True, type=str, help='name of save directory for prediction files. type: float')
parser.add_argument('-multibatch', required=True, type=int, help='the number of batch size per cpu-thread per class. type: int')
parser.add_argument('-modelname', required=True, type=str, help='the name of the model file. type: str')
# ex) -cuda 0 -pulse 32 -width 5 -time 10000 -bmin 11000 -bmax 65000 -noise 0.03 -path temptemp -totalclass 4 -multibatch 65536

args = parser.parse_args()
CUDA_DEVICE = args.cuda                
torch.cuda.set_device(device=CUDA_DEVICE)

N_PULSE = args.pulse
IMAGE_WIDTH = args.width
TIME_RANGE  = args.time
total_class_num = args.totalclass      # CAUTION: total_class_num = 'how many maximum targets' to be considered. ex) if [0, 1, 2]-> class_num=3'

B_init  = args.bmin       
B_final = args.bmax       
noise_scale = args.noise  
SAVE_DIR_NAME = str(args.path)
MODEL_NAME = str(args.modelname)

cpu_num_for_multi = 20    # the number of cpu-thread used for generating datasets
batch_for_multi = args.multibatch # the number of batch size per cpu-thread
class_batch = cpu_num_for_multi*batch_for_multi # the number of batch per class
'''
CUDA_DEVICE = 0
torch.cuda.set_device(device=CUDA_DEVICE)

N_PULSE = 256
N_PULSE_32 = 32
ADD_EXISTING_SPIN = False
IMAGE_WIDTH = 8
TIME_RANGE  = 10000
num_of_summation = 4

B_init  = 1500
B_final = 25000       
noise_scale = 0.1
SAVE_DIR_NAME = 'temptemptemptemp'
MODEL_NAME = 'modelmodeltest'

cpu_num_for_multi = 20    # the number of cpu-thread used for generating datasets
batch_for_multi = 64      # the number of batch size per cpu-thread
class_batch = cpu_num_for_multi*batch_for_multi # the number of batch per class

data_dic = np.load('/home/sonic/Coding/Git/CPMG_Analysis/data/20200329_data_dic.npy').item()
exp_data = data_dic['exp_{}'.format(N_PULSE)]
exp_data_32 = data_dic['exp_{}'.format(N_PULSE_32)]
exp_data_deco = data_dic['exp_coher_{}'.format(N_PULSE)]
exp_data_deco_32 = data_dic['exp_coher_{}'.format(N_PULSE_32)]
time_data = data_dic['time_{}'.format(N_PULSE)]
time_data_32 = data_dic['time_{}'.format(N_PULSE_32)]
spin_bath = data_dic['bath_{}'.format(N_PULSE)]
spin_bath_32 = data_dic['bath_{}'.format(N_PULSE_32)]

# total_indices: [Dictionary (keys: A (Hz))] the set of indices for converting 1D CPMG data into 2D image.  example) total_indices[10000]  
total_indices = np.load('/home/sonic/Coding/Git/CPMG_Analysis/data/CNN_model/indices/total_indices_v4_N{}.npy'.format(N_PULSE)).item() 
total_indices_32 = np.load('/home/sonic/Coding/Git/CPMG_Analysis/data/CNN_model/indices/total_indices_v4_N{}.npy'.format(N_PULSE_32)).item() 
# AB_lists_dic: [Dictionary (keys: A (Hz))] AB values grouped by each local period.  example) AB_lists_dic[10000]
AB_lists_dic = np.load('/home/sonic/Coding/Git/CPMG_Analysis/data/CNN_model/AB_target_cadidates/AB_target_dic_v4.npy').item()

A_target_margin = 100           # the marginal range of the target spins. (Hz)
A_side_margin = 250            # the marginal range of the side spins. (Hz)
A_final_margin = 20            # the marginal range of the final target spins. (Hz)
distance_btw_target_side = 450 # distance_btw_target_side - (A_target_margin+A_side_margin) 
                               # = the final distance between target and side
                               # this distance highly affects the loss during the trainnig
                               # because it gets more difficult as the distance gets narrower      

A_far_side_margin = 5000       # the marginal range of the far_side spins. (Hz)
side_candi_num = 5             # the number of "how many times" to generate 'AB_side_candidate'

A_num = 1    # this determines how many sections will be divided within the whole target range of A 
B_num = 1    # this determines how many sections will be divided within the whole target range of B
class_num = A_num*B_num + 1   # CAUTION: class_num = 'how many classes in a target AB candidates'
image_width = IMAGE_WIDTH

A_hier_margin = 25
A_existing_margin = 150
B_existing_margin = 2500
deno_pred_N32_B12000_above = np.array([
    [-20738.524397906887, 40421.56414091587],
    [-8043.729442048509, 19196.62602543831],
    [36020.12688619586, 26785.71864962578],
    [11463.297802180363, 57308.602420240],
    [-24492.32775241693, 23001.877063512802]
])
ADD_EXISTING_SPIN = True

A_lists = np.load('/data2/tempdir/grouped_A_lists_N256_2.npy')
A_lists = np.array(
      [
       list([-8000, -7950, -7900, -7850, -7800]),
#        list([-5250, -5200, -5150, -5100, -5050, -5000, -4950, -4900, -4850]),
#        list([-3900, -3850, -3800]),
#        list([-3200, -3150, -3100, -3050, -3000, -2950, -2900, -2850, -2800, -2750, -2700, -2650, -2600, -2550, -2500]),
#        list([-1400, -1350, -1300, -1250, -1200, -1150, -1100, -1050, -1000, -950, -900, -850, -800]),
       list([300, 350, 450, 500, 550]),
#        list([1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200]),
       list([2950, 3000, 3050, 3100, 3150]),
       list([3900, 3950, 4000, 4050, 4100]),
       list([4450, 4500, 4550, 4600, 4650]), 
#        list([5500, 5550, 5600]),
#        list([9550, 9600, 9650]), 
#        list([13800, 13850, 13900, 13950]),
       list([19550, 19600, 19650, 19700, 19750]), 
#        list([20150, 20200, 20250]),
#        list([30600, 30650, 30700]), 
#        list([31350, 31400, 31450]),
#        list([35950, 36000, 36050]),
#        list([48400, 48450, 48500])
      ], dtype=object
)

tic = time.time()

width_list = [8, 8, 10, 10, 15, 15]     
time_list = [7500]  
time_32_list = [7500, 10000]
bmax_list = [12000]   
B_init_list = [3000, 4000]  
noise_scale_list = [0.08]  
zero_scale_list = [0.3]
parameter_list = [[width, total_time, bmax, B_init, noise_scale, zero_scale, time_32] for width, total_time, bmax, B_init, noise_scale, zero_scale, time_32 in itertools.product(width_list, time_list, 
                                                                                                                                                        bmax_list, B_init_list, 
                                                                                                                                                        noise_scale_list, zero_scale_list, time_32_list)]

total_results = []
for batch_for_multi in [32, 64, 128]:
    class_batch = cpu_num_for_multi*batch_for_multi # the number of batch per class
    for idx, [width, total_time, B_final, B_init, noise_scale, zero_scale, time_32] in enumerate(parameter_list):
        model_lists = np.array([[A_temp_list[0], A_temp_list[-1], B_init, B_final] for A_temp_list in A_lists])
        image_width = width
        TIME_RANGE = total_time
        TIME_RANGE_32 = time_32

        total_raw_pred_list = []
        total_deno_pred_list = []
        total_A_lists = []
        print("================================================================================")
        print('image_width:{}, TIME_RANGE:{}, B_init:{}, B_end:{}'.format(width, total_time, B_init, B_final))
        print("================================================================================")

        for model_idx, [A_first, A_end, B_first, B_end] in enumerate(model_lists):
            print("========================================================================")
            print('A_first:{}, A_end:{}, B_first:{}, B_end:{}, Bat_mul:{}'.format(A_first, A_end, B_first, B_end, batch_for_multi))
            print("========================================================================")
            A_resol, B_resol = 50, B_end-B_first

            A_idx_list = np.arange(A_first, A_end+A_resol, A_num*A_resol)
            if (B_end-B_first)%B_resol==0:
                B_idx_list = np.arange(B_first, B_first+B_resol, B_num*B_resol)
            else:
                B_idx_list = np.arange(B_first, B_end, B_num*B_resol)
            AB_idx_set = [[A_idx, B_idx] for A_idx, B_idx in itertools.product(A_idx_list, B_idx_list)]

            A_side_num = 8
            A_side_resol = 600
            if N_PULSE==32:
                B_side_min, B_side_max = 6000, 70000
                B_side_gap = 2000
                B_target_gap = 1000  # distance between targets only valid when B_num >= 2.

            elif N_PULSE==256:
                B_side_min, B_side_max = 1000, 14000
                B_side_gap = 0  # distance between target and side (applied for both side_same and side)
                B_target_gap = 0  

            if ((N_PULSE == 32) & (B_first<12000)):
                PRE_PROCESS = True
                PRE_SCALE = 8

            spin_zero_scale = {'same':zero_scale, 'side':0.50, 'mid':0.05, 'far':0.05}  # setting 'same'=1.0 for hierarchical model 

            epochs = 20
            valid_batch = 4096
            valid_mini_batch = 1024

            args = (AB_lists_dic, N_PULSE, A_num, B_num, A_resol, B_resol, A_side_num, A_side_resol, B_side_min,
                        B_side_max, B_target_gap, B_side_gap, A_target_margin, A_side_margin, A_far_side_margin,
                        class_batch, class_num, spin_zero_scale, distance_btw_target_side, side_candi_num) 

            for class_idx in range(num_of_summation):
                TPk_AB_candi, _, temp_hier_target_AB_candi  = gen_TPk_AB_candidates(AB_idx_set, True, *args)
                temp_hier_target_AB_candi[:,:,0] = get_marginal_arr(temp_hier_target_AB_candi[:,:,0], A_hier_margin)
                if class_idx==0:
                    total_hier_target_AB_candi = temp_hier_target_AB_candi[:]
                    target_candidates = TPk_AB_candi[0, :, 0, :]
                    side_candidates   = TPk_AB_candi[1, :, 0, :] 
                    rest_candidates   = TPk_AB_candi[1, :, 1:, :] 
                else:
                    total_hier_target_AB_candi = np.concatenate((total_hier_target_AB_candi, temp_hier_target_AB_candi[:]), axis=1)
                    target_candidates = np.concatenate((target_candidates, TPk_AB_candi[0, :, 0, :]), axis=0)
                    side_candidates   = np.concatenate((side_candidates, TPk_AB_candi[1, :, 0, :]), axis=0)
                    rest_candidates   = np.concatenate((rest_candidates, TPk_AB_candi[1, :, 1:, :]), axis=0) 

            hier_indices = return_total_hier_index_list(A_idx_list, cut_threshold=2)
            total_class_num = hier_indices[-1][0].__len__() + 1

            total_TPk_AB_candidates = np.zeros((total_class_num, num_of_summation*TPk_AB_candi.shape[1], total_class_num+TPk_AB_candi.shape[2]+2, 2))
            indices = np.random.randint(rest_candidates.shape[0], size=(total_class_num, rest_candidates.shape[0]))
            total_TPk_AB_candidates[:, :, (total_class_num-1):-4, :] = rest_candidates[indices]
            indices = np.random.randint(side_candidates.shape[0], size=(total_class_num, side_candidates.shape[0], 2)) 
            total_TPk_AB_candidates[:, :, -4:-2, :] = side_candidates[indices] 
            indices = np.random.randint(side_candidates.shape[0], size=(total_class_num, side_candidates.shape[0], 2)) 
            total_TPk_AB_candidates[:, :, -2:, :] = side_candidates[indices] 
            
            if ADD_EXISTING_SPIN == True:
                total_TPk_AB_candidates = return_existing_spins_wrt_margins(deno_pred_N32_B12000_above, total_TPk_AB_candidates, A_existing_margin, B_existing_margin)

            final_TPk_AB_candidates = total_TPk_AB_candidates[:1]
            for class_idx, hier_index in enumerate(hier_indices): 
                temp_batch = total_TPk_AB_candidates.shape[1] // len(hier_index)
                for idx2, index in enumerate(hier_index):
                    temp = np.swapaxes(total_hier_target_AB_candi[index], 0, 1)
                    if idx2 < (len(hier_index)-1):
                        temp_idx = np.random.randint(total_hier_target_AB_candi.shape[1], size=(temp_batch))
                        total_TPk_AB_candidates[class_idx+1, (idx2)*temp_batch:(idx2+1)*temp_batch, :len(index)] = temp[temp_idx]
                    else:
                        residual_batch = total_TPk_AB_candidates[class_idx+1, (idx2)*temp_batch:, :len(index)].shape[0]
                        temp_idx = np.random.randint(total_hier_target_AB_candi.shape[1], size=(residual_batch))
                        total_TPk_AB_candidates[class_idx+1, (idx2)*temp_batch:, :len(index)] = temp[temp_idx]
                final_TPk_AB_candidates = np.concatenate((final_TPk_AB_candidates, total_TPk_AB_candidates[class_idx+1:class_idx+2, :, :]), axis=0)
#             raise
            median_A_idx = len(AB_idx_set) // 2
    
            # below is for generating N256 datasets
            model_index = get_model_index(total_indices, AB_idx_set[median_A_idx][0], time_thres_idx=TIME_RANGE-20, image_width=image_width) 

            total_class_num = final_TPk_AB_candidates.shape[0]
            X_train_arr = np.zeros((total_class_num, final_TPk_AB_candidates.shape[1], model_index.shape[0], 2*image_width+1))
            Y_train_arr = np.zeros((total_class_num, final_TPk_AB_candidates.shape[1], total_class_num))
            mini_batch = X_train_arr.shape[1] // cpu_num_for_multi
            for class_idx in range(total_class_num):
                Y_train_arr[class_idx, :, class_idx] = 1
                for idx1 in range(cpu_num_for_multi):
                    AB_lists_batch = final_TPk_AB_candidates[class_idx, idx1*mini_batch:(idx1+1)*mini_batch]
                    globals()["pool_{}".format(idx1)] = pool.apply_async(gen_M_arr_batch, [AB_lists_batch, model_index, time_data[:TIME_RANGE], 
                                                                                            WL_VALUE, N_PULSE, PRE_PROCESS, PRE_SCALE, 
                                                                                            noise_scale, spin_bath[:TIME_RANGE]])

                for idx2 in range(cpu_num_for_multi):
                    X_train_arr[class_idx, idx2*mini_batch:(idx2+1)*mini_batch] = globals()["pool_{}".format(idx2)].get(timeout=None) 
                print("class_idx:", class_idx, end=' ') 
                
            # below is for generating N32 datasets
            model_index_32 = get_model_index(total_indices_32, AB_idx_set[median_A_idx][0], time_thres_idx=TIME_RANGE_32-20, image_width=image_width) 

            total_class_num = final_TPk_AB_candidates.shape[0]
            X_train_arr_32 = np.zeros((total_class_num, final_TPk_AB_candidates.shape[1], model_index_32.shape[0], 2*image_width+1))
            Y_train_arr_32 = np.zeros((total_class_num, final_TPk_AB_candidates.shape[1], total_class_num))
            mini_batch = X_train_arr_32.shape[1] // cpu_num_for_multi
            for class_idx in range(total_class_num):
                Y_train_arr_32[class_idx, :, class_idx] = 1
                for idx1 in range(cpu_num_for_multi):
                    AB_lists_batch = final_TPk_AB_candidates[class_idx, idx1*mini_batch:(idx1+1)*mini_batch]
                    globals()["pool_{}".format(idx1)] = pool.apply_async(gen_M_arr_batch, [AB_lists_batch, model_index_32, time_data_32[:TIME_RANGE_32], 
                                                                                            WL_VALUE, N_PULSE_32, PRE_PROCESS, PRE_SCALE, 
                                                                                            noise_scale, spin_bath_32[:TIME_RANGE_32]])

                for idx2 in range(cpu_num_for_multi):
                    X_train_arr_32[class_idx, idx2*mini_batch:(idx2+1)*mini_batch] = globals()["pool_{}".format(idx2)].get(timeout=None) 
                print("class_idx:", class_idx, end=' ') 
            
            X_train_arr = X_train_arr.reshape(-1, model_index.flatten().shape[0]) 
            Y_train_arr = Y_train_arr.reshape(-1, Y_train_arr.shape[2]) 

            X_train_arr_32 = X_train_arr_32.reshape(-1, model_index_32.flatten().shape[0]) 
            Y_train_arr_32 = Y_train_arr_32.reshape(-1, Y_train_arr_32.shape[2]) 

            X_train_arr = np.concatenate((X_train_arr, X_train_arr_32), axis=1)

            model = HPC(X_train_arr.shape[-1], Y_train_arr.shape[-1]).cuda() 
            try:
                model(torch.Tensor(X_train_arr[:16]).cuda()) 
            except:
                raise NameError("The input shape should be revised")

            total_parameter = sum(p.numel() for p in model.parameters()) 
            print('total_parameter: ', total_parameter / 1000000, 'M')

            MODEL_PATH = '/data2/test2243/'
            mini_batch_list = [2048]  
            learning_rate_list = [5e-6] 
            op_list = [['Adabound', [30,15,7,1]]] 
            criterion = nn.BCELoss().cuda()

            hyperparameter_set = [[mini_batch, learning_rate, selected_optim_name] for mini_batch, learning_rate, selected_optim_name in itertools.product(mini_batch_list, learning_rate_list, op_list)]
            print("==================== A_idx: {}, B_idx: {} ======================".format(A_first, B_first))

            total_loss, total_val_loss, total_acc, trained_model = train(MODEL_PATH, MODEL_NAME, N_PULSE, X_train_arr, Y_train_arr, model, hyperparameter_set, criterion, 
                                                                        epochs, valid_batch, valid_mini_batch, exp_data, is_pred=False, is_print_results=False, is_preprocess=PRE_PROCESS, PRE_SCALE=PRE_SCALE,
                                                                        model_index=model_index, exp_data_deno=exp_data_deco, is_regression=False)

            model.load_state_dict(torch.load(trained_model[0][0])) 
            model.eval()
            print("Model loaded as evalutation mode. Model path:", trained_model[0][0])

            exp_data_test = np.hstack((exp_data[model_index.flatten()], exp_data_32[model_index_32.flatten()]))
            exp_data_test = 1-(2*exp_data_test - 1)
            exp_data_test = exp_data_test.reshape(1, -1)
            exp_data_test = torch.Tensor(exp_data_test).cuda()

            pred = model(exp_data_test)
            pred = pred.detach().cpu().numpy()
            total_raw_pred_list.append([pred[0], total_class_num, hier_indices[-1][0].__len__()])
            print("raw", np.argmax(pred), np.max(pred), pred)

            exp_data_test = np.hstack((exp_data_deco[model_index.flatten()], exp_data_deco_32[model_index_32.flatten()]))
            exp_data_test = 1-(2*exp_data_test - 1)
            exp_data_test = exp_data_test.reshape(1, -1)
            exp_data_test = torch.Tensor(exp_data_test).cuda()

            pred = model(exp_data_test)
            pred = pred.detach().cpu().numpy()
            total_deno_pred_list.append([pred[0], total_class_num, hier_indices[-1][0].__len__()])
            print("deno", np.argmax(pred), np.max(pred), pred)
            print("Config:N_PULSE{}, B_init{}, B_final{}, total_time{} w{}_batch{} zero{} time_32{}".format(N_PULSE, B_init, B_final, total_time, width, batch_for_multi, zero_scale, time_32))
            total_results.append([total_raw_pred_list[model_idx], total_deno_pred_list[model_idx], hier_indices, width, total_time, time_32, B_init, B_final, noise_scale, batch_for_multi, zero_scale]) 

        np.save('/data2/test2243/confirm_raw_results_N{}_Bmin{}_Bmax{}_t{}_w{}_batch{}_side_05.npy'.format(N_PULSE, B_init, B_final, total_time, width, batch_for_multi), np.array(total_raw_pred_list))
        np.save('/data2/test2243/confirm_deno_results_N{}_Bmin{}_Bmax{}_t{}_w{}_batch{}_side_05.npy'.format(N_PULSE, B_init, B_final, total_time, width, batch_for_multi), np.array(total_deno_pred_list))


