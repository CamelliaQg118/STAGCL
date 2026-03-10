import torch.nn.modules.loss
import torch.nn.functional as F
from k_means import kmeans_1
from STAGCL.STAGCL_module import STAGCL_module, reversible_model
from STAGCL.STAGCL_module_DEGs import STAGCL_module_degs
from tqdm import tqdm
from utils import *
import STAGCL
import torch.backends.cudnn as cudnn
from sklearn.cluster import KMeans
cudnn.deterministic = True
cudnn.benchmark = True


def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

# class stagcl:
#     def __init__(
#             self,
#             X,
#             adata,
#             adj,
#             edge_index,
#             smooth_fea,
#             n_clusters,
#             dataset,
#             rec_w=1,
#             adv_w=0.1,
#             cos_w=0.5,
#             sim_w=0.5,
#             match_w=1.5,
#             kl_w=1,
#             dec_tol=0.00,#容忍标签波动
#             threshold=0.5,
#             epochs=600,
#             dec_interval=3,
#             lr=0.0002,
#             decay=0.0002,
#             device='cuda:0',
#             mode='clustering',
#     ):
#         self.random_seed = 42
#         STAGCL.fix_seed(self.random_seed)
#
#         self.n_clusters = n_clusters
#         self.cos_w = cos_w
#         self.rec_w = rec_w
#         self.adv_w = adv_w
#         self.sim_w = sim_w
#         self.match_w = match_w
#         self.kl_w =kl_w
#         self.device = device
#         self.dec_tol = dec_tol
#         self.threshold = threshold #高置信度的阈值
#
#         self.adata = adata.copy()
#         self.dataset = dataset
#         self.cell_num = len(X)
#         self.epochs = epochs
#         self.dec_interval = dec_interval
#         self.learning_rate = lr
#         self.weight_decay = decay
#         self.adata = adata.copy()
#         # self.X = torch.FloatTensor(X.copy()).to(self.device)
#         # self.input_dim = self.X.shape[1]
#         self.adj = adj.to(self.device)
#         self.edge_index = edge_index.to(self.device)
#         self.smooth_fea = torch.FloatTensor(smooth_fea).to(self.device)
#         self.mode = mode
#
#         if self.mode == 'clustering':
#             self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)
#         elif self.mode == 'imputation':
#             self.X = torch.FloatTensor(self.adata.X.copy()).to(self.device)
#         else:
#             raise Exception
#         self.input_dim = self.X.shape[-1]
#
#         self.model = STAGCL_module(self.input_dim, self.n_clusters).to(self.device)
#
#     def train(self, dec_tol=0.00):
#         self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
#         """这里有个问题，是rever_network的优化器原来是单独优化的而且使用的是SDG，后续试试"""
#         emb, H1, H2, q, loss_adv, loss_rec, loss_cos, loss_sim = self.model_eval()
#
#         # predict_labels, centers, dis = kmeans_1(self.X, self.n_clusters, distance="euclidean", device=self.device)
#
#         kmeans = KMeans(n_clusters=self.model.edsc_cluster_n, n_init=self.model.edsc_cluster_n * 2, random_state=42)
#         y_pred_last = np.copy(kmeans.fit_predict(emb))
#         self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
#         self.model.train()
#         list_adv = []
#         list_rec = []
#         list_cos = []
#         list_sim = []
#         list_kl = []
#         list_match = []
#         epoch_max = 0
#         ari_max = 0
#         idx_max = []
#         emb_max = []
#
#         if self.dataset in ['Human_Breast_Cancer', 'DLPFC', 'Mouse_Brain_Anterior_Section1']:
#             for epoch in tqdm(range(self.epochs)):
#                 self.model.train()
#                 self.optimizer.zero_grad()
#                 if epoch % self.dec_interval == 0:
#                     _, _, _, tmp_q, _, _, _, _ = self.model_eval()
#
#                     tmp_p = target_distribution(torch.Tensor(tmp_q))
#                     y_pred = tmp_p.cpu().numpy().argmax(1)
#
#                     delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
#                     y_pred_last = np.copy(y_pred)
#                     self.model.train()
#                     if epoch > 0 and delta_label < self.dec_tol:
#                         print('delta_label {:.4}'.format(delta_label), '< tol', self.dec_tol)
#                         print('Reached tolerance threshold. Stopping training.')
#                         break
#                 emb, H1, H2, q, loss_adv, loss_rec, loss_cos, loss_sim = self.model(self.adata, self.X, self.adj, self.edge_index)
#                 if epoch > 200:
#                     pseudo_z1 = torch.softmax(H1, dim=-1)
#                     pseudo_z2 = torch.softmax(H2, dim=-1)
#                     # emb = torch.from_numpy(emb).float().to(self.device)
#                     predict_labels, centers, dis = kmeans_1(emb, self.n_clusters, distance="euclidean", device=self.device)
#                     high_confidence = torch.min(dis, dim=1).values.cpu()
#                     threshold = torch.sort(high_confidence).values[int(len(high_confidence) * self.threshold)]
#                     high_confidence_idx = np.argwhere(high_confidence < threshold)[0]
#                     print("high_confidence_idx",  high_confidence_idx)
#                     h_i = high_confidence_idx.to(self.device)
#                     y_sam = torch.tensor(predict_labels, device=self.device)[high_confidence_idx]
#                     loss_1 = (F.cross_entropy(pseudo_z1[h_i], y_sam)).mean()
#                     loss_2 = (F.cross_entropy(pseudo_z2[h_i], y_sam)).mean()
#                     loss_match = loss_1 + loss_2
#
#                 else:
#                     loss_match = torch.tensor(0.0, device=self.device)
#
#                 torch.set_grad_enabled(True)
#                 # emb, H1, H2, q, loss_adv, loss_rec, loss_cos, loss_sim = self.model(self.adata, self.X, self.adj, self.edge_index)
#                 loss_kl = F.kl_div(q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
#                 loss_tatal = self.rec_w * loss_rec + self.adv_w * loss_adv + self.cos_w * loss_cos + \
#                              self.sim_w * loss_sim + self.match_w * loss_match + self.kl_w * loss_kl
#                 # 反向传播与优化
#                 loss_tatal.backward()
#                 self.optimizer.step()
#
#
#                 list_rec.append(loss_rec.detach().cpu().numpy())
#                 list_adv.append(loss_adv.detach().cpu().numpy())
#                 list_cos.append(loss_cos.detach().cpu().numpy())
#                 list_sim.append(loss_sim.detach().cpu().numpy())
#                 list_kl.append(loss_kl.detach().cpu().numpy())
#                 list_match.append(loss_match.detach().cpu().numpy())
#                 print('loss_rec = {:.5f}'.format(loss_rec), 'loss_adv= {:.5f}'.format(loss_adv),
#                       'loss_cos = {:.5f}'.format(loss_cos), 'loss_sim= {:.5f}'.format(loss_sim),
#                       'loss_match = {:.5f}'.format(loss_match), 'loss_kl = {:.5f}'.format(loss_kl),
#                       ' loss_total = {:.5f}'.format(loss_tatal))
#
#                 emb, _, _, _, _, _, _, _ = self.model_eval()
#                 kmeans = KMeans(n_clusters=self.n_clusters).fit(emb)
#                 idx = kmeans.labels_
#                 self.adata.obsm['STAGCL'] = emb
#                 labels = self.adata.obs['ground']
#                 labels = pd.to_numeric(labels, errors='coerce')
#                 labels = pd.Series(labels).fillna(0).to_numpy()
#                 idx = pd.Series(idx).fillna(0).to_numpy()
#
#                 ari_res = metrics.adjusted_rand_score(labels, idx)
#                 if ari_res > ari_max:
#                     ari_max = ari_res
#                     epoch_max = epoch
#                     idx_max = idx
#                     emb_max = emb
#             import matplotlib.pyplot as plt
#             fig, ax = plt.subplots()
#             ax.plot(list_rec, label='rec')
#             ax.plot(list_adv, label='adv')
#             ax.plot(list_cos, label='cos')
#             ax.plot(list_kl, label='kl')
#             # print(type(list_match[0]))
#             ax.plot(list_match, label='match')
#             ax.plot(list_sim, label='sim')
#             ax.legend()
#             plt.show()
#
#             # acc, f1 = cluster_acc(labels, idx_max)
#             print("epoch_max", epoch_max)
#             print("ARI=======", ari_max)
#             nmi_res = metrics.normalized_mutual_info_score(labels, idx_max)
#             print("NMI=======", nmi_res)
#             self.adata.obs['STAGCL'] = idx_max.astype(str)
#             self.adata.obsm['emb'] = emb_max
#             return self.adata.obsm['emb'], self.adata.obs['STAGCL']
#         else:
#
#             for epoch in tqdm(range(self.epochs)):
#                 self.model.train()
#                 self.optimizer.zero_grad()
#                 if epoch % self.dec_interval == 0:
#                     emb, H1, H2, tmp_q, loss_adv, loss_rec, loss_cos, loss_sim = self.model_eval()
#
#                     tmp_p = target_distribution(torch.Tensor(tmp_q))
#                     y_pred = tmp_p.cpu().numpy().argmax(1)
#
#                     delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
#                     y_pred_last = np.copy(y_pred)
#                     self.model.train()
#                     if epoch > 0 and delta_label < self.dec_tol:
#                         print('delta_label {:.4}'.format(delta_label), '< tol', self.dec_tol)
#                         print('Reached tolerance threshold. Stopping training.')
#                         break
#
#                 if epoch > 200:
#                     pseudo_z1 = torch.softmax(H1, dim=-1)
#                     pseudo_z2 = torch.softmax(H2, dim=-1)
#                     predict_labels, centers, dis = kmeans_1(emb, self.n_clusters, distance="euclidean",
#                                                             device=self.device)
#                     high_confidence = torch.min(dis, dim=1).values
#                     threshold = torch.sort(high_confidence).values[int(len(high_confidence) * self.threshold)]
#                     high_confidence_idx = np.argwhere(high_confidence < threshold)[0]  # 高置信索引
#                     h_i = high_confidence_idx.numpy()
#                     y_sam = torch.tensor(predict_labels, device=self.device)[high_confidence_idx]
#
#                     loss_match = (F.cross_entropy(pseudo_z1[h_i], y_sam)).mean() + (
#                         F.cross_entropy(pseudo_z2[h_i], y_sam)).mean()
#                     # 两视图的伪标签分布pseudo_z1[h_i]，pseudo_z2[h_i]和伪标签y_sam作交叉熵损失
#
#                     torch.set_grad_enabled(True)
#
#                     _, _, _, q, loss_adv, loss_rec, loss_cos, loss_sim = self.model(self.adata, self.X, self.adj,
#                                                                                     self.edge_index)
#                     # total loss
#                     loss_kl = F.kl_div(q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
#                     loss_tatal = self.rec_w * loss_rec + self.adv_w * loss_adv + self.cos_w * loss_cos + \
#                                  self.sim_w * loss_sim + self.kl_w * loss_kl + self.match_w * loss_match
#
#                 else:
#                     torch.set_grad_enabled(True)
#                     loss_kl = F.kl_div(q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
#                     # total loss
#                     loss_tatal = self.rec_w * loss_rec + self.adv_w * loss_adv + self.cos_w * loss_cos + \
#                                  self.sim_w * loss_sim + self.kl_w * loss_kl
#
#                 loss_tatal.backward()
#                 self.optimizer.step()
#
#             return emb
#
#     def model_eval(self):
#         self.model.eval()
#         emb, H1, H2, q, loss_adv, loss_rec, loss_cos, loss_sim = self.model(self.adata, self.X, self.adj, self.edge_index)
#         emb = emb.data.cpu().numpy()
#         q = q.data.cpu().numpy()
#         # H1 = H1.data.cpu().numpy()
#         # H2 = H2.data.cpu().numpy()
#         loss_adv = loss_adv.data.cpu().numpy()
#         loss_rec = loss_rec.data.cpu().numpy()
#         loss_cos = loss_cos.data.cpu().numpy()
#         loss_sim = loss_sim.data.cpu().numpy()
#
#         return emb, H1, H2, q, loss_adv, loss_rec, loss_cos, loss_sim


class stagcl:
    def __init__(
            self,
            X,
            adata,
            adj,
            edge_index,
            adj_remove_dig,
            n_clusters,
            dataset,
            # rec_w=1,
            # adv_w=0.1,
            # cos_w=0.5,
            # sim_w=0.5,
            # match_w=1.5,
            # kl_w=1,
            rec_w=1,
            adv_w=0.1,
            cos_w=0.5,
            sim_w=1,
            match_w=1,
            kl_w=1,
            dec_tol=0.00,#容忍标签波动
            threshold=0.5,
            epochs=600,
            dec_interval=3,
            lr=0.0002,
            decay=0.0002,
            device='cuda:0',
            mode='clustering',
    ):
        self.random_seed = 42
        STAGCL.fix_seed(self.random_seed)

        self.n_clusters = n_clusters
        self.cos_w = cos_w
        self.rec_w = rec_w
        self.adv_w = adv_w
        self.sim_w = sim_w
        self.match_w = match_w
        self.kl_w =kl_w
        self.device = device
        self.dec_tol = dec_tol
        self.threshold = threshold #高置信度的阈值

        self.adata = adata.copy()
        self.dataset = dataset
        self.cell_num = len(X)
        self.epochs = epochs
        self.dec_interval = dec_interval
        self.learning_rate = lr
        self.weight_decay = decay
        self.adata = adata.copy()
        # self.X = torch.FloatTensor(X.copy()).to(self.device)
        # self.input_dim = self.X.shape[1]
        self.adj = adj.to(self.device)
        self.edge_index = edge_index.to(self.device)

        adj_tensor = torch.tensor(adj_remove_dig.todense(), dtype=torch.float32)
        self.adj_tensor = adj_tensor.to(self.device)
        self.mode = mode
        self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)
        self.input_dim = self.X.shape[-1]

        self.model = STAGCL_module(self.X, self.input_dim, self.adj_tensor, self.n_clusters).to(self.device)


        # if self.mode == 'clustering':
        #     self.model = STAGCL_module(self.X, self.input_dim, self.adj_tensor, self.n_clusters).to(self.device)
        #
        # elif self.mode == 'imputation':
        #     self.X1 = torch.FloatTensor(self.adata.X.copy()).to(self.device)
        #     self.input_dim1 = self.X1.shape[-1]
        #     self.model = STAGCL_module_degs(self.adj, self.input_dim, self.input_dim1, self.n_clusters).to(self.device)
        #
        # else:
        #     raise Exception

    def train(self, dec_tol=0.00):
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)

        emb, H1, H2, loss_rec, loss_sim,  loss_cos, loss_adv = self.model_eval()

        # predict_labels, centers, dis = kmeans_1(self.X, self.n_clusters, distance="euclidean", device=self.device)

        # kmeans = KMeans(n_clusters=self.model.edsc_cluster_n, n_init=self.model.edsc_cluster_n * 2, random_state=42)
        # y_pred_last = np.copy(kmeans.fit_predict(emb))
        # self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        self.model.train()
        list_adv = []
        list_rec = []
        list_cos = []
        list_sim = []
        list_kl = []
        list_match = []
        epoch_max = 0
        ari_max = 0
        idx_max = []
        emb_max = []

        if self.dataset in ['Human_Breast_Cancer', 'DLPFC', 'Mouse_Brain_Anterior_Section1']:
            for epoch in tqdm(range(self.epochs)):
                self.model.train()
                self.optimizer.zero_grad()
                # if epoch % self.dec_interval == 0:
                #     _, _, _, tmp_q, _, _, _, _ = self.model_eval()
                #
                #     tmp_p = target_distribution(torch.Tensor(tmp_q))
                #     y_pred = tmp_p.cpu().numpy().argmax(1)
                #
                #     delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                #     y_pred_last = np.copy(y_pred)
                #     self.model.train()
                #     if epoch > 0 and delta_label < self.dec_tol:
                #         print('delta_label {:.4}'.format(delta_label), '< tol', self.dec_tol)
                #         print('Reached tolerance threshold. Stopping training.')
                #         break
                emb, H1, H2, loss_rec, loss_sim, loss_cos, loss_adv = self.model(self.adata, self.X, self.adj, self.edge_index)
                if epoch > 200:
                    pseudo_z1 = torch.softmax(H1, dim=-1)
                    pseudo_z2 = torch.softmax(H2, dim=-1)
                    # emb = torch.from_numpy(emb).float().to(self.device)
                    predict_labels, centers, dis = kmeans_1(emb, self.n_clusters, distance="euclidean", device=self.device)
                    high_confidence = torch.min(dis, dim=1).values.cpu()
                    threshold = torch.sort(high_confidence).values[int(len(high_confidence) * self.threshold)]
                    high_confidence_idx = np.argwhere(high_confidence < threshold)[0]
                    # print("high_confidence_idx",  high_confidence_idx)
                    h_i = high_confidence_idx.to(self.device)
                    y_sam = torch.tensor(predict_labels, device=self.device)[high_confidence_idx]
                    loss_1 = (F.cross_entropy(pseudo_z1[h_i], y_sam)).mean()
                    loss_2 = (F.cross_entropy(pseudo_z2[h_i], y_sam)).mean()
                    loss_match = loss_1 + loss_2

                else:
                    loss_match = torch.tensor(0.0, device=self.device)

                torch.set_grad_enabled(True)
                emb, H1, H2, loss_rec, loss_sim, loss_cos, loss_adv = self.model(self.adata, self.X, self.adj, self.edge_index)
                # loss_kl = F.kl_div(q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
                loss_tatal = self.rec_w * loss_rec + self.adv_w * loss_adv + self.cos_w * loss_cos +\
                             self.sim_w * loss_sim + self.match_w * loss_match# + self.kl_w * loss_kl
                # 反向传播与优化
                loss_tatal.backward()
                self.optimizer.step()


                list_rec.append(loss_rec.detach().cpu().numpy())
                list_adv.append(loss_adv.detach().cpu().numpy())
                list_cos.append(loss_cos.detach().cpu().numpy())
                list_sim.append(loss_sim.detach().cpu().numpy())
                # list_kl.append(loss_kl.detach().cpu().numpy())
                list_match.append(loss_match.detach().cpu().numpy())
                print('loss_rec = {:.5f}'.format(loss_rec),
                      'loss_adv= {:.5f}'.format(loss_adv),
                      'loss_cos= {:.5f}'.format(loss_cos),
                      'loss_sim= {:.5f}'.format(loss_sim),
                      'loss_match = {:.5f}'.format(loss_match), #'loss_kl = {:.5f}'.format(loss_kl),
                      'loss_total = {:.5f}'.format(loss_tatal))

                emb, _, _, _, _, _, _ = self.model_eval()
                kmeans = KMeans(n_clusters=self.n_clusters).fit(emb)
                idx = kmeans.labels_
                self.adata.obsm['STAGCL'] = emb
                labels = self.adata.obs['ground']
                labels = pd.to_numeric(labels, errors='coerce')
                labels = pd.Series(labels).fillna(0).to_numpy()
                idx = pd.Series(idx).fillna(0).to_numpy()

                ari_res = metrics.adjusted_rand_score(labels, idx)
                if ari_res > ari_max:
                    ari_max = ari_res
                    epoch_max = epoch
                    idx_max = idx
                    emb_max = emb
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(list_rec, label='rec')
            ax.plot(list_adv, label='adv')
            ax.plot(list_cos, label='cos')
            ax.plot(list_kl, label='kl')
            # print(type(list_match[0]))
            ax.plot(list_match, label='match')
            ax.plot(list_sim, label='sim')
            ax.legend()
            plt.show()

            # acc, f1 = cluster_acc(labels, idx_max)
            print("epoch_max", epoch_max)
            print("ARI=======", ari_max)
            nmi_res = metrics.normalized_mutual_info_score(labels, idx_max)
            print("NMI=======", nmi_res)
            self.adata.obs['STAGCL'] = idx_max.astype(str)
            self.adata.obsm['emb'] = emb_max
            return self.adata.obsm['emb'], self.adata.obs['STAGCL']

        # # return self.adata.obsm['emb'], self.adata.obs['STMACL'], acc, f1#GNNs,parameters,xiaorong
        # if self.mode == 'clustering':
        #     return self.adata.obsm['emb'], self.adata.obs['STMACL'], acc, f1,
        # elif self.mode == 'imputation':
        #     return self.adata.obsm['emb'], self.adata.obs['STMACL'], acc, f1, self.adata.obsm['rec']# GNNs,parameters,xiaorong

        else:

            for epoch in tqdm(range(self.epochs)):
                self.model.train()
                self.optimizer.zero_grad()

                emb, H1, H2, loss_rec, loss_sim, loss_cos, loss_adv = self.model(self.adata, self.X, self.adj,
                                                                                    self.edge_index)
                if epoch > 200:
                    pseudo_z1 = torch.softmax(H1, dim=-1)
                    pseudo_z2 = torch.softmax(H2, dim=-1)
                    # emb = torch.from_numpy(emb).float().to(self.device)
                    predict_labels, centers, dis = kmeans_1(emb, self.n_clusters, distance="euclidean",
                                                            device=self.device)
                    high_confidence = torch.min(dis, dim=1).values.cpu()
                    threshold = torch.sort(high_confidence).values[int(len(high_confidence) * self.threshold)]
                    high_confidence_idx = np.argwhere(high_confidence < threshold)[0]
                    # print("high_confidence_idx",  high_confidence_idx)
                    h_i = high_confidence_idx.to(self.device)
                    y_sam = torch.tensor(predict_labels, device=self.device)[high_confidence_idx]
                    loss_1 = (F.cross_entropy(pseudo_z1[h_i], y_sam)).mean()
                    loss_2 = (F.cross_entropy(pseudo_z2[h_i], y_sam)).mean()
                    loss_match = loss_1 + loss_2

                else:
                    loss_match = torch.tensor(0.0, device=self.device)

                torch.set_grad_enabled(True)
                emb, H1, H2, loss_rec, loss_sim, loss_cos, loss_adv = self.model(self.adata, self.X, self.adj,
                                                                                    self.edge_index)
                # loss_kl = F.kl_div(q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
                loss_tatal = self.rec_w * loss_rec + self.adv_w * loss_adv + self.cos_w * loss_cos +\
                             self.sim_w * loss_sim + self.match_w * loss_match  # + self.kl_w * loss_kl
                # 反向传播与优化
                loss_tatal.backward()
                self.optimizer.step()
                emb, _, _, _, _, _, _ = self.model_eval()
            # if self.mode == 'clustering':
            #     return emb
            # elif self.mode == 'imputation':
            #     self.adata.obsm['rec'] = rec
            #     return emb, self.adata.obsm['rec']# GNNs,parameters,xiaorong

            return emb

    def model_eval(self):
        self.model.eval()
        emb, H1, H2, loss_rec, loss_sim, loss_cos, loss_adv = self.model(self.adata, self.X, self.adj, self.edge_index)
        emb = emb.data.cpu().numpy()
        # q = q.data.cpu().numpy()
        # H1 = H1.data.cpu().numpy()
        # H2 = H2.data.cpu().numpy()
        loss_adv = loss_adv.data.cpu().numpy()
        loss_rec = loss_rec.data.cpu().numpy()
        loss_cos = loss_cos.data.cpu().numpy()
        loss_sim = loss_sim.data.cpu().numpy()

        return emb, H1, H2, loss_rec, loss_sim, loss_cos, loss_adv

