from locally_connected import LocallyConnected
#from lbfgsb_scipy_p1 import LBFGSBScipy # 特征数大于50可用
from lbfgsb_scipy import LBFGSBScipy
from trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np

class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        nums = dims[1]
        self.node_nums = nums
        self.dims = dims
        self.w_est = nn.Parameter(torch.Tensor(np.ones((d, d))))

    def forward(self, X):  # [n, d] -> [n, d]
        # M = X @ W + A @ X @ P
        M = torch.matmul(X, self.w_est)
        return M

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        # fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        # fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        # A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        h = trace_expm(self.w_est * self.w_est) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def diag_zero(self):
        diag_loss = torch.trace(self.w_est * self.w_est)
        return diag_loss


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss

def L1Norm(matrix):
    return  torch.abs(matrix).sum()

def dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X)
            loss = squared_loss(X_hat, X)
            h_val = model.h_func()
            diag_loss = model.diag_zero()
            penalty1 = 0.5 * rho * h_val * h_val + alpha * h_val
            primal_obj = loss + 100 * penalty1 + 1000 * diag_loss + lambda1 * L1Norm(model.w_est)
            primal_obj.backward()
            return primal_obj
        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func()
        if h_new.item() > 0.25 * h:
            rho *= 10
        elif h_new.item() < 5: #
            break
    alpha += rho * h_new
    return rho, alpha,h_new

def notears_nonlinear(model: nn.Module,
                      X,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max :
            break
    W_est = model.w_est
    W_est = W_est.detach().numpy()
    return W_est

def main():

    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    import utils as ut
    ut.set_random_seed(123)

    n, d, s0, graph_type, sem_type = 500, 100, 100, 'ER', 'gauss'

    # 读取存储的X
    X = np.loadtxt("X.csv", delimiter=",")

    # 读取存储的w_mat
    w_mat = np.loadtxt("w_mat.csv", delimiter=",")

    # notears模型
    model = NotearsMLP(dims=[d, n, 1], bias=True)
    W_est  = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01)

    # 输出结果
    W_est_ = W_est
    print("****************Notears***********")
    print("W_est:")
    w_threshold = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,1.0,1.1,1.2,1.3]
    isdag = []
    for thre in w_threshold:
        W_est_[np.abs(W_est_) < thre] = 0
        acc = ut.count_accuracy(w_mat, W_est_ != 0)
        print("w_threshold = ", thre,acc,ut.is_dag(W_est_))
        W_est_ = W_est

if __name__ == '__main__':
    main()
