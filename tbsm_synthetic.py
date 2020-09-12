# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# miscellaneous
from os import path
import sys

# numpy and scikit-learn
import numpy as np

from sklearn.metrics import roc_auc_score

# pytorch
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

# In synthetic experiment we generate the output vectors z of the embedding
# layer directly, therefore we create a custom TBSM, rather than instantiate
# an existing general model.

# Synthetic experiment code
# It generates time series data in D dimensions
# with the property that binary label has some dependency
# on coupling between time series components in pairs of dimensions.
def synthetic_experiment():

    N, Nt, D, T = 50000, 5000, 5, 10

    auc_results = np.empty((0, 5), np.float32)

    def generate_data(N, high):
        H = np.random.uniform(low=-1.0, high=1.0, size=N * D * T).reshape(N, T, D)
        w = np.random.uniform(low=-1.0, high=1.0, size=N * D).reshape(N, 1, D)
        return H, w

    for K in range(0, 31, 10):

        print("num q terms: ", K)
        # ----- train set ------
        H, w = generate_data(N, 1.0)
        wt = np.transpose(w, (0, 2, 1))
        p = np.zeros(D * K, dtype=np.int).reshape(K, D)
        for j in range(K):
            p[j, :] = np.random.permutation(D)
            wt2 = wt[:, p[j], :]
            wt = wt + wt2
        Q = np.matmul(H[:, :, :], wt[:, :, :])  # similarity coefs
        Q = np.squeeze(Q, axis=2)
        R = np.mean(Q, axis=1)
        R = np.sign(R)
        # s1 = np.count_nonzero(R > 0)
        # print(Q.shape)
        # print("num pos, total: ", s1, N)
        R = R + 1
        t_train = R.reshape(N, 1)
        z_train = np.concatenate((H, w), axis=1)

        # ----- test set ------
        H, w = generate_data(Nt, 1.0)
        wt = np.transpose(w, (0, 2, 1))
        for j in range(K):
            wt2 = wt[:, p[j], :]
            wt = wt + wt2
        Q = np.matmul(H[:, :, :], wt[:, :, :])  # dot product
        Q = np.squeeze(Q, axis=2)
        R = np.mean(Q, axis=1)
        R = np.sign(R) + 1
        t_test = R.reshape(Nt, 1)
        z_test = np.concatenate((H, w), axis=1)
        # debug prints
        # print(z_train.shape, t_train.shape)

        class SyntheticDataset:
            def __init__(self, F, y):
                self.F = F
                self.y = y

            def __getitem__(self, index):

                if isinstance(index, slice):
                    return [
                        self[idx] for idx in range(
                            index.start or 0, index.stop or len(self), index.step or 1
                        )
                    ]

                return self.F[index], self.y[index]

            def __len__(self):
                return len(self.y)

        ztraind = SyntheticDataset(z_train, t_train)
        ztestd = SyntheticDataset(z_test, t_test)

        def collate_zfn(list_of_tuples):
            data = list(zip(*list_of_tuples))
            F = torch.tensor(data[0], dtype=torch.float)
            y = torch.tensor(data[1], dtype=torch.float)
            # y = torch.unsqueeze(y, 1)
            return F, y

        ztrain_ld = torch.utils.data.DataLoader(
            ztraind,
            batch_size=128,
            num_workers=0,
            collate_fn=collate_zfn,
            shuffle=True
        )
        ztest_ld = torch.utils.data.DataLoader(
            ztestd,
            batch_size=Nt,
            num_workers=0,
            collate_fn=collate_zfn,
        )

        ### define TBSM in PyTorch ###
        class TBSM_SubNet(nn.Module):
            def __init__(
                    self,
                    mode,
                    num_inner,
                    D,
                    T,
            ):
                super(TBSM_SubNet, self).__init__()

                self.mode = mode
                self.num_inner = num_inner
                if self.mode in ["def", "ind", "dot"]:
                    if self.mode in ["def", "ind"]:
                        self.A = []
                        mean = 0.0
                        std_dev = np.sqrt(2 / (D + D))
                        for _ in range(self.num_inner):
                            E = np.eye(D, dtype=np.float32)
                            W = np.random.normal(mean, std_dev, size=(1, D, D)) \
                                .astype(np.float32)
                            self.A.append(Parameter(torch.tensor(E + W),
                            requires_grad=True))

                    d = self.num_inner * T
                    # d = self.num_inner * D + D
                    ln_mlp = np.array([d, 2 * d, 1])
                    self.mlp = dlrm.DLRM_Net().create_mlp(ln_mlp, ln_mlp.size - 2)
                elif self.mode == "mha":
                    m = D           # dim
                    self.nheads = 8
                    self.emb_m = self.nheads * m  # mha emb dim
                    mean = 0.0
                    std_dev = np.sqrt(2 / (m + m))  # np.sqrt(1 / m) # np.sqrt(1 / n)
                    qm = np.random.normal(mean, std_dev, size=(1, m, self.emb_m)) \
                        .astype(np.float32)
                    self.Q = Parameter(torch.tensor(qm), requires_grad=True)
                    km = np.random.normal(mean, std_dev, size=(1, m, self.emb_m))  \
                        .astype(np.float32)
                    self.K = Parameter(torch.tensor(km), requires_grad=True)
                    vm = np.random.normal(mean, std_dev, size=(1, m, self.emb_m)) \
                        .astype(np.float32)
                    self.V = Parameter(torch.tensor(vm), requires_grad=True)
                    d = self.nheads * m
                    ln_mlp = np.array([d, 2 * d, 1])
                    self.mlp = dlrm.DLRM_Net().create_mlp(ln_mlp, ln_mlp.size - 2)
                else:
                    d = D * (T + 1)
                    ln_mlp = np.array([d, 2 * d, 1])
                    self.mlp = dlrm.DLRM_Net().create_mlp(ln_mlp, ln_mlp.size - 2)

            def forward(self, x):

                # H * w
                H = x[:, :-1, :]
                w = torch.unsqueeze(x[:, -1, :], dim=1)
                w = torch.transpose(w, 1, 2)
                # inner products
                if self.mode in ["def", "ind"]:
                    for j in range(self.num_inner):
                        aw = torch.matmul(self.A[j], w)
                        if self.mode == "def":
                            aw = torch.matmul(self.A[j].permute(0, 2, 1), aw)
                        a1 = torch.bmm(H, aw)
                        if j == 0:
                            z = a1
                        else:
                            z = torch.cat([z, a1], dim=1)
                    z = torch.squeeze(z, dim=2)
                elif self.mode == "dot":
                    z = torch.bmm(H, w)
                    z = torch.squeeze(z, dim=2)
                elif self.mode == "mha":
                    w = torch.transpose(w, 1, 2)
                    # print("mha shapes: ", w.shape, self.Q.shape)
                    Qx = torch.transpose(torch.matmul(w, self.Q), 0, 1)
                    HK = torch.transpose(torch.matmul(H, self.K), 0, 1)
                    HV = torch.transpose(torch.matmul(H, self.V), 0, 1)
                    multihead_attn = nn.MultiheadAttention(self.emb_m, self.nheads)
                    attn_output, _ = multihead_attn(Qx, HK, HV)
                    # print("attn shape: ", attn_output.shape)
                    z = torch.squeeze(attn_output, dim=0)
                else:  # concat
                    H = torch.flatten(H, start_dim=1, end_dim=2)
                    w = torch.flatten(w, start_dim=1, end_dim=2)
                    z = torch.cat([H, w], dim=1)
                # obtain probability of a click as a result of MLP
                p = dlrm.DLRM_Net().apply_mlp(z, self.mlp)

                return p

        def train_inner(znet):

            loss_fn = torch.nn.BCELoss(reduction="mean")
            # loss_fn = torch.nn.L1Loss(reduction="mean")
            optimizer = torch.optim.Adagrad(znet.parameters(), lr=0.05)
            # optimizer = torch.optim.SGD(znet.parameters(), lr=0.05)

            znet.train()
            nepochs = 1
            for _ in range(nepochs):
                TA = 0
                TS = 0
                for _, (X, y) in enumerate(ztrain_ld):

                    batchSize = X.shape[0]

                    # forward pass
                    Z = znet(X)

                    # loss
                    # print("Z, y: ", Z.shape, y.shape)
                    E = loss_fn(Z, y)
                    # compute loss and accuracy
                    # L = E.detach().cpu().numpy()  # numpy array
                    z = Z.detach().cpu().numpy()  # numpy array
                    t = y.detach().cpu().numpy()  # numpy array

                    # rounding t: smooth labels case
                    A = np.sum((np.round(z, 0) == np.round(t, 0)).astype(np.uint16))
                    TA += A
                    TS += batchSize

                    optimizer.zero_grad()
                    # backward pass
                    E.backward(retain_graph=True)
                    # optimizer
                    optimizer.step()
                    # if j % 500 == 0:
                    #     acc = 100.0 * TA / TS
                    #     print("j, acc: ", j, acc)
                    #     TA = 0
                    #     TS = 0

            z_final = np.zeros(Nt, dtype=np.float)
            offset = 0
            znet.eval()
            for _, (X, _) in enumerate(ztest_ld):

                batchSize = X.shape[0]
                Z = znet(X)
                z_final[offset: offset + batchSize] = \
                    np.squeeze(Z.detach().cpu().numpy(), axis=1)
                offset += batchSize
                # E = loss_fn(Z, y)
                # L = E.detach().cpu().numpy()  # numpy array

            # loss_net = L
            # print(znet.num_inner, znet.mode, ": ", loss_net)
            auc_net = 100.0 * roc_auc_score(t_test.astype(int), z_final)
            print(znet.num_inner, znet.mode, ": ", auc_net)

            return auc_net

        dim = T
        znet = TBSM_SubNet("dot", 1, D, dim)  # c or c,w
        res1 = train_inner(znet)
        znet = TBSM_SubNet("def", 1, D, dim)  # c or c,w
        res2 = train_inner(znet)
        znet = TBSM_SubNet("def", 4, D, dim)  # c or c,w
        res3 = train_inner(znet)
        znet = TBSM_SubNet("def", 8, D, dim)  # c or c,w
        res4 = train_inner(znet)
        znet = TBSM_SubNet("mha", 1, D, dim)  # c or c,w
        res5 = train_inner(znet)
        auc_results = np.append(auc_results, np.array([[res1, res2, res3, res4, res5]]),
            axis=0)
    print(auc_results)
    # np.savez_compressed(
    #     'auc_synthetic.npz',
    #     auc_results=auc_results,
    # )

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Synthetic data experiments (TBSM)")

    # path to dlrm
    parser.add_argument("--dlrm-path", type=str, default="")
    args = parser.parse_args()

    if not path.exists(str(args.dlrm_path)):
        sys.exit("Please provide path to DLRM as --dlrm-path")

    sys.path.insert(1, args.dlrm_path)
    import dlrm_s_pytorch as dlrm

    synthetic_experiment()
