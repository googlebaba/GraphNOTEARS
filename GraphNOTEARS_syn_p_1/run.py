from locally_connected import LocallyConnected
from lbfgsb_scipy import LBFGSBScipy
from trace_expm import trace_expm
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
import GraphNOTEARS
import notears_torch_version
import lasso
import dynotears
import utils as ut

device = torch.device("cuda:0")


def data_pre(n, d, s0, w_graph_type, p_graph_type, sem_type):

    w_true = ut.simulate_dag(d, s0, w_graph_type)
    w_mat = ut.simulate_parameter(w_true)


    adj1 = ut.generate_adj(n)

    num_step = 5
    Xbase = []

    Xbase1 = ut.simulate_linear_sem(w_mat, n, sem_type, noise_scale=0.5)
    p1_mat, p1_true = ut.generate_tri(d, p_graph_type, low_value=0.0, high_value=2)
 

    for i in range(num_step):
        Xbase1 = ut.simulate_linear_sem_with_P(w_mat, p1_mat, adj1@Xbase1, n, sem_type, noise_scale=1)
        Xbase.append(Xbase1)

    return Xbase, adj1, w_true,w_mat, p1_true, p1_mat


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    import utils as ut
    # ut.set_random_seed(123)

    n_ = [100, 200, 300, 500]

    d_ = [5, 10, 20, 30]

    w_graph_types = ['ER', 'BA'] 
    p_graph_types = ['ER', 'SBM'] 
    sem_types = ['exp']


    re_file = "result/p_1_results.txt"

    for n in n_:
        for d in d_:
            for w_graph_type in w_graph_types:
                for p_graph_type in p_graph_types:
                    for sem_type in sem_types:
                        w_fdr = [[],[],[]]
                        w_tpr = [[],[],[]]
                        w_fpr = [[],[],[]]
                        w_shd = [[],[],[]]
                        w_nnz = [[],[],[]]
                        w_f1 = [[],[],[]]

                        p1_fdr = [[],[],[]]
                        p1_tpr = [[],[],[]]
                        p1_fpr = [[],[],[]]
                        p1_shd = [[],[],[]]
                        p1_nnz = [[],[],[]]
                        p1_f1 = [[],[],[]]

                        s0 = 1 * d
                        for times in range(5):
                            Xlags, adj1, w_true,w_mat, p1_true, p1_mat = data_pre(n, d, s0, w_graph_type,p_graph_type, sem_type)


                            string = " n= "+ str(n) + " d= " + str(d) + " s0= " + str(s0) + ' w_graph_type = ' + w_graph_type + ' p_graph_type = ' + p_graph_type + ' sem_type = ' + sem_type + "\n"
                            print(string)
                            with open(re_file, 'a') as result_file:
                                result_file.write(string)
                            result_file.close()
                            adj1_torch = torch.Tensor(adj1)
                            Xlags_torch = torch.Tensor(np.array(Xlags))

                            model_1 = GraphNOTEARS.model_p1_MLP(dims=[d, n, 1], bias=True)
                            model_1.to(device)
                            W_est_1, P1_est_1 = GraphNOTEARS.linear_model(model_1, Xlags_torch, adj1_torch,  lambda1 = 0.01, lambda2 = 0.01, lambda3 = 0.01)
                            model_notears = notears_torch_version.NotearsMLP(dims=[d, n, 1], bias=True)
                            model_notears.to(device)
                            W_est_2 = notears_torch_version.notears_nonlinear(model_notears, Xlags_torch[1:], lambda1=0.01, lambda2=0.01)

                            model_lasso = lasso.lasso_MLP(dims=[d, n, 1], bias=True)
                            model_lasso.to(device)
                            P1_est_2 = lasso.lasso_model(model_lasso, Xlags_torch, adj1_torch, lambda1=0.01, lambda2=0.01)

                            # run dynotears
                            model_dynotears = dynotears.dynotears_MLP(dims=[d, n, 1], bias=True)
                            model_dynotears.to(device)
                            W_est_3, P1_est_3 = dynotears.dynotears_model(model_dynotears, Xlags_torch, adj1_torch, lambda1 = 0.01, lambda2 = 0.01, lambda3 = 0.01)

                            w_est = [W_est_1, W_est_2, W_est_3] #
                            p1_est = [P1_est_1, P1_est_2, P1_est_3] #


                            threshold = [0.3]

                            for i in range(3):
                                with open(re_file, 'a') as result_file:
                                    if i == 0:
                                        print("-----model_p2-----")
                                        result_file.write("-----model_p2-----"+"\n")
                                    if i == 1:
                                        print("-----notears + lasso-----")
                                        result_file.write("-----notears + lasso-----"+"\n")
                                    if i == 2:
                                        print("-----dynotears_p2-----")
                                        result_file.write("-----dynotears-----"+"\n")
                                W_est_ = w_est[i]
                                P1_est_ = p1_est[i]

                                for thre in threshold:

                                    W_est_[np.abs(W_est_) < thre] = 0


                                    fdr,tpr,fpr,shd,pred_size = ut.count_accuracy(w_true, W_est_ != 0)
                                    w_f1_ = f1_score(w_true, W_est_ != 0, average="micro")

                                    w_fdr[i].append(fdr)
                                    w_tpr[i].append(tpr)
                                    w_fpr[i].append(fpr)
                                    w_f1[i].append(w_f1_)
                                    w_shd[i].append(shd)
                                    w_nnz[i].append(pred_size)

                                    acc = ' fdr = ' + str(fdr) + ' tpr = ' + str(tpr) + ' fpr = ' + str(fpr) + ' shd = ' + str(shd) + ' nnz = ' + str(pred_size)
                                    string = "W_est: threshold = " + str(thre) + str(acc) +  "  f1 = " + str(w_f1_)  + "\n"
                                    print(string)
                                    with open(re_file, 'a') as result_file:
                                        result_file.write(string)
                                    W_est_ = w_est[i]


                                for thre in threshold:
                                    P1_est_[np.abs(P1_est_) < thre] = 0

                                    fdr,tpr,fpr,shd,pred_size = ut.count_accuracy(p1_true, P1_est_ != 0)
                                    p1_f1_ = f1_score(p1_true, P1_est_ != 0, average="micro")

                                    p1_fdr[i].append(fdr)
                                    p1_tpr[i].append(tpr)
                                    p1_fpr[i].append(fpr)
                                    p1_f1[i].append(p1_f1_)
                                    p1_shd[i].append(shd)
                                    p1_nnz[i].append(pred_size)

                                    acc = ' fdr = ' + str(fdr) + ' tpr = ' + str(tpr) + ' fpr = ' + str(fpr) + ' shd = ' + str(shd) + ' nnz = ' + str(pred_size)
                                    string = "P1_est: threshold = " + str(thre) + " acc = " + str(acc) + "  f1 = " + str(p1_f1_) +"\n"
                                    print(string)
                                    with open(re_file, 'a') as result_file:
                                        result_file.write(string)

                               

                        with open(re_file, 'a') as result_file:
                            result_file.write("\n")
                            result_file.write("mean and var for each model with " + ' n = ' +str(n) + ' d = ' +str(d) + ' w_graph_type = ' + w_graph_type + ' p_graph_type = ' + p_graph_type + ' sem_type = ' + sem_type + "\n")
                            result_file.write("\n")
                        for i in range(3):
                            if i == 0:
                                model_name = "model_1: "
                            if i == 1:
                                model_name = "notears + lasso: "
                            if i == 2:
                                model_name = "dynotears: "
                            string = model_name + " w_fdr mean = " + str(np.mean(w_fdr[i])) + " w_fdr var = " + str(np.var(w_fdr[i])) +  \
                                     " | w_tpr mean = " + str(np.mean(w_tpr[i])) + " w_tpr var = " + str(np.var(w_tpr[i])) + \
                                    " | w_fpr mean = " + str(np.mean(w_fpr[i])) + " w_fpr var = " + str(np.var(w_fpr[i])) + \
                                    " | w_f1 mean = " + str(np.mean(w_f1[i])) + " w_f1 var = " + str(np.var(w_f1[i])) + \
                                     " | w_shd mean = " + str(np.mean(w_shd[i])) + " w_shd var = " + str(np.var(w_shd[i])) + "\n"

                            with open(re_file, 'a') as result_file:
                                result_file.write(string)
                                result_file.write("\n")

                        for i in range(3):
                            if i == 0:
                                model_name = "model_1"
                            if i == 1:
                                model_name = "notears + lasso"
                            if i == 2:
                                model_name = "dynotears"
                            string = model_name + " p1_fdr mean = " + str(np.mean(p1_fdr[i])) + " p1_fdr var = " + str(np.var(p1_fdr[i])) +  \
                                     " | p1_tpr mean = " + str(np.mean(p1_tpr[i])) + " p1_tpr var = " + str(np.var(p1_tpr[i])) + \
                                    " | p1_fpr mean = " + str(np.mean(p1_fpr[i])) + " p1_fpr var = " + str(np.var(p1_fpr[i])) + \
                                    " | p1_f1 mean = " + str(np.mean(p1_f1[i])) + " p1_f1 var = " + str(np.var(p1_f1[i])) + \
                                    " | p1_shd mean = " + str(np.mean(p1_shd[i])) + " p1_shd var = " + str(np.var(p1_shd[i])) + "\n"
                            with open(re_file, 'a') as result_file:
                                result_file.write(string)
                                result_file.write("\n")


if __name__ == '__main__':
    main()
