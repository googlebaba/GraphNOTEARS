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
import dynotears_p2
import utils as ut

device = torch.device("cuda:1")

def data_pre(n, d, s0, w_graph_type, p_graph_type, sem_type):
    w_true = ut.simulate_dag(d, s0, w_graph_type)
    w_mat = ut.simulate_parameter(w_true)

    adj1 = ut.generate_adj(n)
    adj2 = ut.generate_adj(n)
    num_step = 5
    Xbase = []

    Xbase1 = ut.simulate_linear_sem(w_mat, n, sem_type, noise_scale=0.3)
    p1_mat, p1_true = ut.generate_tri(d, p_graph_type, low_value=0.0, high_value=2)

    p2_mat, p2_true = ut.generate_tri(d, p_graph_type, low_value=0.5, high_value=1)
    Xbase.append(Xbase1)

    Xbase.append(Xbase1)

 
    for i in range(num_step+1):
        Xbase1 = ut.simulate_linear_sem_with_P(w_mat, p1_mat, p2_mat, adj1@Xbase[-1], adj2@Xbase[-2], n, sem_type, noise_scale=1)
        Xbase.append(Xbase1)

    return Xbase, adj1, adj2, w_true,w_mat, p1_true, p1_mat, p2_true, p2_mat



def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    import utils as ut

    ut.set_random_seed(4)
    n_ = [500]
    d_ = [5]
    w_graph_types = ['ER'] #  'BA'
    p_graph_types = ['ER'] #  'SBM'
    sem_types = ['gauss'] # 'exp'

    re_file = "result/p2_result.txt"

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

                        p2_fdr = [[], [], []]
                        p2_tpr = [[], [], []]
                        p2_fpr = [[], [], []]
                        p2_shd = [[], [], []]
                        p2_nnz = [[], [], []]
                        p2_f1 = [[], [], []]

                        for times in range(1):
                            s0 = 1 * d # 边数等于变量数

                            X, adj1, adj2, w_true,w_mat, p1_true, p1_mat, p2_true, p2_mat = data_pre(n, d, s0, w_graph_type,p_graph_type, sem_type)

                            string = " n= "+ str(n) + " d= " + str(d) + " s0= " + str(s0) + ' w_graph_type = ' + w_graph_type + ' p_graph_type = ' + p_graph_type + ' sem_type = ' + sem_type + "\n"
                            print(string)
                            with open(re_file, 'a') as result_file:
                                result_file.write(string)
                            result_file.close()

                            adj1_torch = torch.Tensor(adj1)

                            adj2_torch = torch.Tensor(adj2)
                            X_torch = torch.Tensor(X)

                            model_1 = GraphNOTEARS.model_p1_MLP(dims=[d, n, 1], bias=True)
                            model_1.to(device)
                            W_est_1, P1_est_1, P2_est_1 = GraphNOTEARS.linear_model(model_1, X_torch, adj1_torch, adj2_torch, lambda1 = 0.01, lambda2 = 0.01, lambda3 = 0.01)
                            print("P1_est_1", P1_est_1, p1_mat)

                            print("P2_est_1", P2_est_1, p2_mat)

                            print("W_est_1", W_est_1, w_true)

                            # run notears 运行 notears
                            model_notears = notears_torch_version.NotearsMLP(dims=[d, n, 1], bias=True)
                            W_est_2 = notears_torch_version.notears_nonlinear(model_notears, X_torch[2:], lambda1=0.01, lambda2=0.01)

                            # run lasso 运行lasso
                            model_lasso = lasso.lasso_MLP(dims=[d, n, 1], bias=True)
                            P1_est_2, P2_est_2 = lasso.lasso_model(model_lasso, X_torch, adj1_torch, adj2_torch, lambda1=0.01, lambda2=0.01)

                            # run dynotears
                            model_dynotears = dynotears_p2.dynotears_MLP(dims=[d, n, 1], bias=True)
                            model_dynotears.to(device)
                            W_est_3, P1_est_3, P2_est_3 = dynotears_p2.dynotears_model(model_dynotears, X_torch, adj1_torch, adj2_torch, lambda1 = 0.01, lambda2 = 0.01, lambda3 = 0.01)

                            w_est = [W_est_1, W_est_2, W_est_3] #
                            p1_est = [P1_est_1, P1_est_2, P1_est_3] #
                            p2_est = [P2_est_1, P2_est_2, P2_est_3]


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
                                P2_est_ = p2_est[i]

                                for thre in threshold:
                                    W_est_[np.abs(W_est_) < thre] = 0
                                    print("************W_mat:***********")
                                    print(w_mat)
                                    print("************W_est_:***********")
                                    print(W_est_)
                                    # assert ut.is_dag(W_est_)# 判断是否是dag


                                    fdr,tpr,fpr,shd,pred_size = ut.count_accuracy(w_true, W_est_ != 0)
                                    w_f1_ = f1_score(w_true, W_est_ != 0, average="micro")

                                    w_fdr[i].append(fdr)
                                    w_tpr[i].append(tpr)
                                    w_fpr[i].append(fpr)
                                    w_f1[i].append(w_f1_)
                                    w_shd[i].append(shd)
                                    w_nnz[i].append(pred_size)

                                    acc = ' fdr = ' + str(fdr) + ' tpr = ' + str(tpr) + ' fpr = ' + str(fpr) + ' shd = ' + str(shd) + ' nnz = ' + str(pred_size)
                                    string = "W_est: threshold = " + str(thre) + " acc = " + str(acc) +  "  f1 = " + str(w_f1_)  + "\n"
                                    print(string)
                                    with open(re_file, 'a') as result_file:
                                        result_file.write(string)
                                    W_est_ = w_est[i]


                                for thre in threshold:
                                    # process p1
                                    P1_est_[np.abs(P1_est_) < thre] = 0
                                    print("************P1_mat:***********")
                                    print(p1_mat)
                                    print("************P1_est_:***********")
                                    print(P1_est_)

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
                                    P1_est_ = p1_est[i]

                                print("P2_est：")
                                for thre in threshold:
                                    P2_est_[np.abs(P2_est_) < thre] = 0

                                    print("************P2_mat:***********")
                                    print(p2_mat)
                                    print("************P2_est_:***********")
                                    print(P2_est_)

                                    fdr, tpr, fpr, shd, pred_size = ut.count_accuracy(p2_true, P2_est_ != 0)
                                    p2_f1_ = f1_score(p2_true, P2_est_ != 0, average="micro")

                                    p2_fdr[i].append(fdr)
                                    p2_tpr[i].append(tpr)
                                    p2_fpr[i].append(fpr)
                                    p2_f1[i].append(p2_f1_)
                                    p2_shd[i].append(shd)
                                    p2_nnz[i].append(pred_size)

                                    acc = ' fdr = ' + str(fdr) + ' tpr = ' + str(tpr) + ' fpr = ' + str(
                                        fpr) + ' shd = ' + str(shd) + ' nnz = ' + str(pred_size)
                                    string = "P2_est: threshold = " + str(thre) + " acc = " + str(acc) + "  f1 = " + str(
                                        p2_f1_) + "\n"
                                    print(string)
                                    with open(re_file, 'a') as result_file:
                                        result_file.write(string)
                                    P2_est_ = p2_est[i]

                        """""
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
                                    " | w_f1 mean = " + str(np.mean(w_f1[i])) + " w_fpr var = " + str(np.var(w_f1[i])) + \
                                     " | w_shd mean = " + str(np.mean(w_shd[i])) + " w_nnz mean = " + str(np.mean(w_nnz[i])) + "\n"
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
                                    " | p1_f1 mean = " + str(np.mean(p1_f1[i])) + " p1_fpr var = " + str(np.var(p1_f1[i])) + \
                                result_file.write(string)
                                result_file.write("\n")

                        for i in range(3):
                            if i == 0:
                                model_name = "model_1"
                            if i == 1:
                                model_name = "notears + lasso"
                            if i == 2:
                                model_name = "dynotears"
                            string = model_name + " p2_fdr mean = " + str(np.mean(p2_fdr[i])) + " p2_fdr var = " + str(np.var(p2_fdr[i])) +  \
                                     " | p2_tpr mean = " + str(np.mean(p2_tpr[i])) + " p2_tpr var = " + str(np.var(p2_tpr[i])) + \
                                    " | p2_fpr mean = " + str(np.mean(p2_fpr[i])) + " p2_fpr var = " + str(np.var(p2_fpr[i])) + \
                                    " | p2_f1 mean = " + str(np.mean(p2_f1[i])) + " p2_fpr var = " + str(np.var(p2_f1[i])) + \
                                     " | p2_shd mean = " + str(np.mean(p2_shd[i])) + " p2_nnz mean = " + str(np.mean(p2_nnz[i])) + "\n"
                            with open(re_file, 'a') as result_file:
                                result_file.write(string)
                                result_file.write("\n")
                        """
if __name__ == '__main__':
    main()
