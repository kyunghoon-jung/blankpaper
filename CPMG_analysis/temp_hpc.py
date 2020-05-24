def HPC_prediction(model, AB_idx_set, total_indices, time_range, image_width, cut_idx, exp_data, exp_data_deno, 
                   total_A_lists, total_raw_pred_list, total_deno_pred_list, save_to_file=False):    
                   
        model.eval()

        raw_pred = []
        deno_pred = []
        A_pred_lists = []
        for idx1, [A_idx, B_idx] in enumerate(AB_idx_set):
            model_index = get_model_index(total_indices, A_idx, time_thres_idx=time_range-20, image_width=image_width)
            model_index = model_index[:cut_idx, :]
            exp_data_test = exp_data[model_index.flatten()]

            exp_data_test = 1-(2*exp_data_test - 1)
            exp_data_test = exp_data_test.reshape(1, -1)
            exp_data_test = torch.Tensor(exp_data_test).cuda()

            pred = model(exp_data_test)
            pred = pred.detach().cpu().numpy()

            A_pred_lists.append(A_idx)
            raw_pred.append(pred[0])

            total_A_lists.append(A_idx)
            total_raw_pred_list.append(pred[0])

            print(A_idx, np.argmax(pred), np.max(pred), pred)
            exp_data_test = exp_data_deno[model_index.flatten()]

            exp_data_test = 1-(2*exp_data_test - 1)
            exp_data_test = exp_data_test.reshape(1, -1)
            exp_data_test = torch.Tensor(exp_data_test).cuda()

            pred = model(exp_data_test)
            pred = pred.detach().cpu().numpy()
            deno_pred.append(pred[0])
            print(A_idx, np.argmax(pred), np.max(pred), pred)
            print() 

            total_deno_pred_list.append(pred[0])
        raw_pred = np.array(raw_pred).T
        deno_pred = np.array(deno_pred).T
        if save_to_file:
            np.save(MODEL_PATH+'A_idx_{}_A{}-{}_B{}-{}'.format(model_idx, A_first, A_end, B_first, B_end), A_pred_lists)
            np.save(MODEL_PATH+'raw_pred_{}_A{}-{}_B{}-{}'.format(model_idx, A_first, A_end, B_first, B_end), raw_pred)
            np.save(MODEL_PATH+'deno_pred_{}_A{}-{}_B{}-{}'.format(model_idx, A_first, A_end, B_first, B_end), deno_pred)

    total_raw_pred_list  = np.array(total_raw_pred_list).T
    total_deno_pred_list = np.array(total_deno_pred_list).T

    return total_A_lists, total_raw_pred_list, total_deno_pred_list
