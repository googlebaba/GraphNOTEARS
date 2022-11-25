from locally_connected import LocallyConnected
# from lbfgsb_scipy_p1 import LBFGSBScipy # >50 可选
from lbfgsb_scipy import LBFGSBScipy
from trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
import utils as ut
import scipy.sparse
device = torch.device("cuda:0")

class model_p1_MLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(model_p1_MLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        nums = dims[1]
        self.dims = dims
        self.node_nums = nums
        self.w_est = nn.Parameter(torch.ones((d, d)))
        self.p1_est = nn.Parameter(torch.ones((d, d)))
        self.p2_est = nn.Parameter(torch.ones((d, d)))

    def forward(self, Xlags, adj1):  # [n, d] -> [n, d]
        # M = XW + A@Xlag1@P1 + A@Xlag2@P2
        M = torch.matmul(Xlags[1:],self.w_est) + torch.matmul(torch.matmul(adj1, Xlags[:-1]),self.p1_est) 
        return M

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        h = trace_expm(self.w_est * self.w_est) - d  # (Zheng et al. 2018)
        return h

    def diag_zero(self):
        diag_loss = torch.trace(self.w_est * self.w_est)
        return diag_loss

def squared_loss(output, target):
    n = target.shape[0] * target.shape[1]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss

def L1Norm(matrix):
    return  torch.abs(matrix).sum()

def dual_ascent_step(model, X_lags, adj1, lambda1, lambda2, lambda3, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    primal = torch.Tensor([0.0]).float().cuda()
    while rho < rho_max:
        def closure():
            print("rho < rho_max:", rho, rho_max)
            optimizer.zero_grad()
            X_hat = model(X_lags, adj1)
            loss = squared_loss(X_hat, X_lags[1:])
            h_val = model.h_func()
            diag_loss = model.diag_zero()
            penalty1 = 0.5 * rho * h_val * h_val + alpha * h_val
            primal_obj = primal + loss + 100 * penalty1 + 1000 * diag_loss +  lambda1 * L1Norm(model.w_est) + lambda2 * L1Norm(model.p1_est) #+ lambda3 * L1Norm(model.p2_est) # l2_reg + l1_reg
            print("h", h)
            primal_obj.backward()
            return primal_obj

        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func()
        if h_new.item() > 0.25 * h:
            rho *= 10
        else: # 50 100 100 < 0.5
            break
    alpha += rho * h_new
    return rho, alpha, h_new

def linear_model(model: nn.Module,
                      Xlags,
                      adj1,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      lambda3: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, Xlags, adj1, lambda1, lambda2, lambda3,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.w_est
    W_est = W_est.detach().numpy()

    P1_est = model.p1_est
    P1_est = P1_est.detach().numpy()

    P2_est = model.p2_est
    P2_est = P2_est.detach().numpy()

    return W_est, P1_est


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    import utils as ut
    ut.set_random_seed(123)

    # 此处修改参数，n样本数，d变量数，s0是DAG矩阵W的边数
    # 50, 5, 2
    # h_new.item() < 0.1 # 500, 50, 25
    # 50 100 100 < 0.5
    n, d, s0, graph_type, sem_type = 500, 100, 100, 'ER', 'gauss'

    # 生成w_mat
    w_true = ut.simulate_dag(d, s0, graph_type)
    w_mat = ut.simulate_parameter(w_true)
    # np.savetxt('W_true.csv', w_mat, delimiter=',')

    X_ = ut.simulate_linear_sem(w_mat, n, sem_type)
    Xlags = ut.simulate_linear_sem(w_mat, n, sem_type)

    # 存w_mat
    print("w_true", w_true)
    print("w_mat", w_mat.tolist())
    np.savetxt('w_true.csv', w_true, delimiter=',')
    np.savetxt('w_mat.csv', w_mat, delimiter=',')

    print("X_", X_)
    print("X_lags", Xlags)

    # 生成p_mat
    p_mat = ut.generate_tri(d)
    p_true = ut.matrix_to_zerone(p_mat)
    print("p_mat", p_mat)

    # 存p_mat
    np.savetxt('p_mat.csv', p_mat, delimiter=',')
    np.savetxt('p_true.csv', p_true, delimiter=',')

    # 生成adj
    adj = ut.generate_adj(n)

    # 存adj
    np.savetxt('adj.csv', adj, delimiter=',')

    # 生成X
    X = X_ + adj @ Xlags @ p_mat

    # 存储X和Xlags
    np.savetxt('X.csv', X, delimiter=',')
    np.savetxt('Xlags.csv', Xlags, delimiter=',')

    # 初始化model
    model = model_p1_MLP(dims=[d, n, 1], bias=True)
    model.to(device)

    adj = torch.Tensor(adj)
    Xlags = torch.Tensor(Xlags)
    X = torch.Tensor(X)

    print("adj ", adj)
    print("X ", X)

    W_est, P_est = linear_model(model, X, Xlags, adj, lambda1 = 0.1, lambda2 = 0.1, lambda3 = 0.1)

    print("W_est:", W_est)
    W_est_ = W_est
    print("****************Model_P1***********")
    print("W_est:")
    w_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    for thre in w_threshold:
        # thre = 0.8 # w的阈值设为0.8时效果最佳
        W_est_[np.abs(W_est_) < thre] = 0
        acc = ut.count_accuracy(w_true, W_est_ != 0)
        # assert ut.is_dag(W_est)
        print("w_threshold = ", thre, acc, ut.is_dag(W_est_))
        W_est_ = W_est

    P_est_ = P_est
    print("****************Model_P1***********")
    print("P_est:")
    for thre in w_threshold:
        # thre = 0.4 # p的阈值设为0.4时效果最佳
        P_est_[np.abs(P_est_) < thre] = 0
        acc = ut.count_accuracy(p_true, P_est_ != 0)
        # assert ut.is_dag(P_est)
        print("p_threshold = ", thre, acc)
        P_est_ = P_est


if __name__ == '__main__':
    main()
