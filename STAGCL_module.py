from functools import partial
from STAGCL.layers import *
from torch.nn import Linear, LeakyReLU


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def adv_loss(x, y):
    loss = -F.mse_loss(x, y)
    return loss


# def cos_loss(x, x_aug):
#     T = 1.0
#     batch_size, _ = x.size()
#     x_abs = x.norm(dim=1)
#     x_aug_abs = x_aug.norm(dim=1)
#     sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
#     sim_matrix = torch.exp(sim_matrix / T)
#     pos_sim = sim_matrix[range(batch_size), range(batch_size)]
#     loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
#     loss = - torch.log(loss).mean()
#     return loss
def cos_loss(x1, x2, temperature=0.2):
    """
    x1: Tensor of shape [batch_size, dim]
    x2: Tensor of shape [batch_size, dim]
    Returns scalar loss.
    """
    # Step 1: Normalize
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)

    # Step 2: Concatenate both views
    batch_size = x1.size(0)
    x_all = torch.cat([x1, x2], dim=0)  

    # Step 3: Similarity matrix
    sim_matrix = torch.matmul(x_all, x_all.T) / temperature  

    # Step 4: Remove diagonal self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=x1.device)
    sim_matrix.masked_fill_(mask, -float('inf'))

    # Step 5: Positive similarity: i<->i+N
    positives = torch.cat([torch.arange(batch_size, device=x1.device) + batch_size,
                           torch.arange(batch_size, device=x1.device)])
    labels = positives

    # Step 6: Cross-entropy loss
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class STAGCL_module(nn.Module):
    def __init__(
            self,
            X,
            input_dim,
            adj_tensor,
            nclass,
            latent_dim=128,
            output_dim=64,
            train_dim=128,
            latdim=1000,
            p_drop=0.2,
            dorp_code=0.2,
            dropout=0.2,
            mask_rate=0.8,
            remask_rate=0.1,
            drop_edge_rate=0.1,
            alpha=0.1,
            d=64,
            down=True,
            up=False,
            decode_type='GCN',
            encoder_type='GCN',
            aug_type='GCN',
            use_bn='true',
            device='cuda:0'
    ):
        super(STAGCL_module, self).__init__()
        self.adj_tensor = adj_tensor
        self.X = X
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.latent_hidden = latent_dim
        self.edin_dim = input_dim
        self.edlatent_dim = latent_dim
        self.edout_dim = output_dim
        self.dein_dim = output_dim
        self.deout_dim = latent_dim
        self.cluster_dim = output_dim
        self.latent_p =output_dim
        self.train_dim = train_dim
        self.ende_dim = output_dim
        self.mask_dim = output_dim
        self.input_latent = output_dim
        self.mlp_indim = X.shape[0]
        self.mlp_latdim = latdim
        self.mlp_outdim = input_dim
        self.edsc_cluster_n = nclass
        self.emb_dim = output_dim*3

        self.nclass = nclass
        self.dropout = dropout
        self.p_drop = p_drop
        self.dorp_code = dorp_code

        self.device = device
        self.use_bn = use_bn
        self.decode_type = decode_type
        self.encoder_type = encoder_type
        self.aug_type = aug_type
        self.down = down
        self.up = up
        self.alpha = alpha
        self.mask_rate = mask_rate
        self.remask_rate = remask_rate
        self.drop_edge_rate = drop_edge_rate

        self.d = d
        self.d_dim = output_dim

        self.encoder = Encodeer_Model(self.input_dim, self.latent_hidden, self.output_dim, self.p_drop, self.device)
        self.decoder = self.Code(self.decode_type, self.dein_dim, self.deout_dim, self.input_dim, self.dorp_code)
        self.encode_aug = self.Code(self.aug_type, self.input_dim, self.latent_dim, self.output_dim, self.dorp_code)
        self.encode_latent = self.Code(self.encoder_type, self.input_latent, self.latent_dim, self.output_dim, self.dorp_code)
        self.rev1 = reversible_model(self.input_dim)
        self.rev2 = reversible_model(self.output_dim)
        self.att = Atten_Model(self.X, self.latent_dim, self.adj_tensor, self.nclass)
        self. MLP = MLP_model(self.mlp_indim, self.mlp_latdim, self.mlp_outdim)

        self.loss_type1 = self.setup_loss_fn(loss_fn='adver')
        self.loss_type2 = self.setup_loss_fn(loss_fn='cos')
        self.loss_type3 = self.setup_loss_fn(loss_fn='sce')


        self.cluster_layer = Parameter(torch.Tensor(self.nclass, output_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.input_dim)).to(self.device)
        self.dec_mask_token = nn.Parameter(torch.zeros(1, self.mask_dim)).to(self.device)
        self.encoder_to_decoder = nn.Linear(self.ende_dim, self.cluster_dim, bias=False)
        self.reset_parameters_for_token()

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)

    def forward(self, adata, X, adj, edge_index):
        adj, X1, (mask_nodes, keep_nodes) = self.encoding_mask_noise(adj, X, self.mask_rate)
        Zf = self.encoder(X1)
        X_rev = self.rev1(X, self.down)

        # ident = torch.eye(X.shape[0]).to(self.device)
        # X_att = self.att(X) + ident
        # X_att = ident - X_att

        ########adver loss
        loss_adv = self.loss_type1(X_rev, X)#
        # loss_adv = self.loss_type1(X_att, self.adj_tensor)#

        #adj_augmention
        edge_drop = dropout_edge(edge_index, self.drop_edge_rate)
        num_nodes = adj.shape[0]
        adj_drop = edge_index_to_sparse_adj(edge_drop, num_nodes, device='cuda')

        #learn embedding
        Hfea = self.encode_latent(Zf, adj_drop)
        H1 = self.encode_aug(X, adj)
        H2 = self.encode_aug(X_rev, adj)

        # Hfea = self.encode_latent(Zf, adj)
        # H1 = self.encode_aug(X, adj_drop)
        # H2 = self.encode_aug(X_rev, adj)

        # H1 = self.encode_aug(X_rev, adj_drop)
        # X_att = self.MLP(X_att)
        # H2 = self.encode_aug(X_att, adj)



        ############fuse embedding
        emb1 = torch.cat([Hfea, H1, H2], dim=1).to(self.device)
        linear = nn.Linear(self.emb_dim, self.output_dim).to(self.device)
        emb = linear(emb1).to(self.device)

        #############adv_embedding
        Z1 = self.rev2(H1, self.down)
        # print("Z1", Z1.size())
        Z2 = self.rev2(H2, self.up)
        # print("Z2", Z2.size())

        ###############loss_cos
        loss1 = self.loss_type2(H1, Z2)
        loss2 = self.loss_type2(H2, Z1)
        loss_cos = loss1+loss2
        ##############loss_sim
        cross_sim_ori_pro = H1 * Z1
        cross_sim_re_ori = Z2 * H1
        loss_sim = -self.loss_type1(cross_sim_ori_pro, cross_sim_re_ori)

        ############## feature reconstruction
        H = Hfea.clone()
        H_remask, _, _ = self.random_remask(adj, H, self.remask_rate)#
        H_rec = self.decoder(H_remask, adj)
        x_init1 = X[mask_nodes]
        x_rec1 = H_rec[mask_nodes]
        loss_rec = self.loss_type3(x_init1, x_rec1)


        ###################cluster guide
        # q = 1.0 / ((1.0 + torch.sum((emb.unsqueeze(1) - self.cluster_layer) ** 2, dim=2) / self.alpha))
        # q = q.pow((self.alpha + 1.0) / 2.0)
        # q = q ** (self.alpha + 1.0) / 2.0
        # q = q / torch.sum(q, dim=1, keepdim=True)  

        return emb, H1, H2, loss_rec, loss_sim, loss_cos, loss_adv



    def setup_loss_fn(self, loss_fn):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=3)
        elif loss_fn == "adver":
            criterion = partial(adv_loss)
        elif loss_fn == "cos":
            criterion = partial(cos_loss)
        else:
            raise NotImplementedError
        return criterion

    def cos_d_loss(self, x, x_neg):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_neg.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_neg) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


    def Code(self, m_type, in_dim, num_hidden, out_dim, dropout) -> nn.Module:
        if m_type == "GCN":
            mod = GCN(in_dim, num_hidden, out_dim, dropout)
        elif m_type in ['gcn', 'gat', 'gat2', 'gin', 'sage']:
            mod = GNNEncoder(in_dim, num_hidden, out_dim, num_layers=2, dropout=dropout, bn=False, layer=m_type)
        elif m_type == "mlp":
            mod = nn.Sequential(nn.Linear(in_dim, num_hidden * 2), nn.PReLU(), nn.Dropout(0.2), nn.Linear(num_hidden * 2, out_dim))
        elif m_type == "linear":
            mod = nn.Linear(in_dim, out_dim)
        else:
            raise NotImplementedError
        return mod

    def encoding_mask_noise(self, adj, x, mask_rate=0.3):
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[token_nodes] += self.enc_mask_token
        use_adj = adj.clone()
        return use_adj, out_x, (mask_nodes, keep_nodes)

    def random_remask(self, adj, rep, remask_rate=0.5):
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes:]
        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token
        return rep, remask_nodes, rekeep_nodes

    def attention(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = F.relu(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = self.conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.use_residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)

    # @torch.no_grad()
    # def evaluate(self, x, edge_index):
    #     enc_rep = self.online_encoder(x, edge_index)
    #     rep = self.encoder_to_decoder(enc_rep)
    #     rep = self.projector(rep, edge_index)
    #     recon = self.decoder(rep, edge_index)
    #     return enc_rep, recon


def dropout_edge(edge_index, p=0.5, force_undirected=False):
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    row, col = edge_index#

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:#
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()#

    return edge_index


class reversible_model(nn.Module):
    def __init__(self, dims):
        super(reversible_model, self).__init__()

        self.down1 = nn.Linear(dims, dims//2)
        self.down2 = nn.Linear(dims//2, dims)#

        self.up1 = nn.Linear(dims, dims * 2)#
        self.up2 = nn.Linear(dims * 2, dims)#


    def forward(self, x, flag):

        if flag:
            x = self.down1(x)
            down_feature = self.down2(x)
            down_feature = F.normalize(down_feature, dim=1, p=2)
            return down_feature

        else:
            up_feature = self.up2(self.up1(x))
            up_feature = F.normalize(up_feature, dim=1, p=2)
            return up_feature


class projetor1(nn.Module):
    def __init__(self, input_dim, cluster_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.LeakyReLU(),
            nn.Linear(int(input_dim/2), cluster_dim),
            nn.LeakyReLU(),
        )
        self.linear_shortcut = nn.Linear(input_dim, cluster_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class projector2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),

            nn.ReLU(),
            nn.Linear(input_dim, input_dim),

            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)




class Encodeer_Model(nn.Module):
    def __init__(self, input_dim, intermediate_dim, kan_dim, p_drop, device):
        super(Encodeer_Model, self).__init__()
        self.device = device
        self.full_block = full_block(input_dim, intermediate_dim,  p_drop).to(self.device)
        self.KAN = KANLinear(intermediate_dim, kan_dim).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.full_block(x)
        feat = self.KAN(x)
        return feat


def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )


def edge_index_to_sparse_adj(edge_index, num_nodes, device='cuda'):
    edge_weight = torch.ones(edge_index.size(1), device=device) 
    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_weight,
        size=(num_nodes, num_nodes),
        device=device
    )
    return adj


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


class Atten_Model(torch.nn.Module):
    def __init__(self, fea, nhidden, edge_indices_no_diag, nclass):
        super(Atten_Model, self).__init__()

        self.edge_indices_no_diag = edge_indices_no_diag
        self.in_features = fea.shape[1]
        self.out_features = nhidden
        self.num_classes = nclass
        self.W = Linear(self.in_features, self.out_features, bias=False)
        self.a = Parameter(torch.Tensor(2 * self.out_features, 1))
        self.W1 = Linear(self.in_features, self.num_classes, bias=False)
        self.num1 = fea.shape[0]
        self.leakyrelu = LeakyReLU(0.1)
        self.tmp = []
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W.weight)
        glorot(self.a)
        glorot(self.W1.weight)

    def forward(self, x):
        Wx = self.W(x)
        self.A_ds_no_diag = self.CalAttenA(Wx)
        return self.A_ds_no_diag

    def CalAttenA(self, Wx):
        indices = self.edge_indices_no_diag.clone()
        indices = torch.nonzero(indices).t()
        fea1 = Wx[indices[0, :], :]
        fea2 = Wx[indices[1, :], :]#
        fea12 = torch.cat((fea1, fea2), 1)#
        atten_coef = torch.exp(self.leakyrelu(torch.mm(fea12, self.a))).flatten()
        A_atten = torch.zeros([self.num1, self.num1]).cuda()
        A_atten[indices[0, :], indices[1, :]] = atten_coef
        s1 = A_atten.sum(1)
        pos1 = torch.where(s1 == 0)[0]
        A_atten[pos1, pos1] = 1#
        A_atten = A_atten.t() / A_atten.sum(1)
        return A_atten.t()


class MLP_model(nn.Module):
    def __init__(self, inputdim, latentdim, outpudim):
        super(MLP_model, self).__init__()
        self.layers1 = nn.Linear(inputdim, latentdim)
        self.layers2 = nn.Linear(latentdim, outpudim)

    def forward(self, x):
        out1 = self.layers1(x)
        out2 = self.layers2(out1)
        out2 = F.normalize(out2, dim=1, p=2)
        return out2


