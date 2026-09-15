from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Function
from units.datasets import load_data
from units.unit import set_requires_grad, GRL_coeff, clustering_metrics
from model import ACGRL
from units.loss import Loss


def setup_seed(seed): 
    torch.manual_seed(seed+1)
    torch.cuda.manual_seed_all(seed+2)
    np.random.seed(seed+3)
    random.seed(seed+4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def reconstruction(model, data_list, view_num, Criterion, optimizer):
    x_hat, *_ = model(
        data_list)  # type is list overall
    loss_list = []
    loss_1 = 0
    optimizer.zero_grad()
    for view in range(view_num):
        loss_list.append(Criterion.forward_mse(x_hat[view], data_list[view]))
    loss = sum(loss_list)  # Reconstruction Loss only, now
    loss_1 += loss.item() / view_num
 
    loss.backward()
    optimizer.step()

    return loss_1
def UCRM(model, data_list, view_num, criterion, optimizer, lambda_clu):
    loss_list = []
    _, pseudo_list, _, _ = model(data_list)
    loss_3 = 0
    optimizer.zero_grad()
    for i in range(view_num):
        for j in range(i + 1, view_num):
            q_m = pseudo_list[i]
            q_n = pseudo_list[j]
            loss_align = criterion.pseudoAlignLoss(q_m, q_n)
            loss_list.append(loss_align)

    loss = lambda_clu * sum(loss_list) / view_num
    loss_3 += loss.item()


    loss.backward()
    optimizer.step()


    return loss_3


def GRCAM_Bi(model, data_list, view_num, Criterion, args, device, optimizer_CRMN, optimizer_ViewDisc, iter_idx, num_iters):


    '''Train the view discriminator.'''
    set_requires_grad(model.view_discriminator, True)
    set_requires_grad(model.MLP, False)
    model.view_discriminator.train()
    model.MLP.eval()
    _, _, _, z_share = model(data_list)
    for _ in range(args.n_critic):
        optimizer_ViewDisc.zero_grad()
        loss_list_class = []
        for view in range(view_num):
            shared_z = z_share[view].detach()

            specific_label = torch.full((shared_z.shape[0],), view, dtype=torch.long, device=device)
            specific_logit = model.view_discriminator(shared_z)

            loss_cla = Criterion.forward_CrossEntropy(specific_logit, specific_label)
            loss_list_class.append(loss_cla)
        loss_classifier = sum(loss_list_class) / view_num

        loss_1 = loss_classifier.item()
        loss_classifier.backward()
        optimizer_ViewDisc.step()

    '''Training CRMN'''
    optimizer_CRMN.zero_grad()
    set_requires_grad(model.view_discriminator, False)
    set_requires_grad(model.MLP, True)
    model.view_discriminator.eval()
    model.MLP.train()
    _, _, _, z_share = model(data_list)
    loss_list_con = []
    for view in range(view_num):
        shared_z = z_share[view]
        # lambd = 1
        beta = args.beta
        lambd = GRL_coeff(iter_idx,beta, num_iters)    # best lambda is 3
        # lambd = linear_lambda(iter_idx / float(num_iters), max_lambda=1.0, ramp_up_end=0.5)
        shared_z_rev = GRL(shared_z, lambd = lambd)
        specific_label = torch.full((shared_z.shape[0],), view, dtype=torch.long, device=device)
        shared_logit = model.view_discriminator(shared_z_rev)

        loss_con = Criterion.forward_CrossEntropy(shared_logit, specific_label)
        loss_list_con.append(loss_con)

    loss_consistency = sum(loss_list_con) / view_num
    loss_2 = loss_consistency.item()
    loss_consistency.backward()
    optimizer_CRMN.step()

    return loss_1, loss_2

def CRFN_Adversarial(model, data_list, view_num, Criterion, args, device, optimizer_CRFN, optimizer_Disc):
    loss_list_disc = []

    loss_1, loss_2 = 0, 0
    '''Training Discriminator'''
    set_requires_grad(model.discriminator, True)
    set_requires_grad(model.filter, False)
    model.filter.eval()
    model.discriminator.train()
    _, _, z_con, z_share = model(data_list)
    optimizer_Disc.zero_grad()

    share_label = torch.full((z_share[0].shape[0], ), 0, dtype=torch.long, device=device)
    aplha = args.epsilon
    eta = args.eta
    for view in range(view_num):
        specific_label = torch.full((z_share[0].shape[0],), view + 1, dtype=torch.long, device=device)
        specific_logit = model.discriminator(z_con[view].detach())
        shared_logit = model.discriminator(z_share[view].detach())
        loss_ce = aplha * Criterion.forward_CrossEntropy(specific_logit, specific_label) + (1 - aplha)*Criterion.forward_CrossEntropy(shared_logit, share_label)  + \
                  eta * Criterion.forward_Entropy(specific_logit)
        loss_list_disc.append(loss_ce)

    loss_disc = sum(loss_list_disc)

    loss_1 += loss_disc.item() / view_num
    loss_disc.backward()
    optimizer_Disc.step()


    '''Training Filter'''
    set_requires_grad(model.discriminator, False)
    set_requires_grad(model.filter, True)
    model.filter.train()
    model.discriminator.eval()
    _, _, z_con, z_share = model(data_list)
    optimizer_CRFN.zero_grad()
    loss_list_crfn = []

    for view in range(view_num):
        specific_label = torch.full((z_share[0].shape[0],), view + 1, dtype=torch.long, device=device)
        specific_logit = model.discriminator(z_con[view])
        shared_logit = model.discriminator(z_share[view].detach())
        loss_filter = aplha * Criterion.forward_CrossEntropy(specific_logit, specific_label) + (
                1 - aplha) * Criterion.forward_CrossEntropy(shared_logit, share_label) + \
                  eta * Criterion.forward_Entropy(specific_logit)
        loss_list_crfn.append(loss_filter)
    loss_filter = sum(loss_list_crfn)
    loss_2 += loss_filter.item() / view_num
    loss_filter.backward()
    optimizer_CRFN.step()

    return loss_1, loss_2


def main_train(args, dataset_name, config, device):
    setup_seed(args.seed)
    view_num = config['view_num']
    train_loader, test_loader, n_cluster = data_preprocess(dataset_name, args, view_num)




    epochs = args.epochs
    dim = config['architectures']
    model = ACGRL(auto_dim=dim, view_num=view_num, cluster_n=n_cluster)


    model = model.to(device) 
    '''Training Discriminator and Generator, respectively'''



    t_max_epoch = epochs
    lr = 1e-3
    optimizer_CRMN = optim.Adam([
        {'params': model.MLP.parameters()},
        {'params': model.Encoder_c.parameters()}], lr = args.lr)
    scheduler_CRMN = CosineAnnealingLR(optimizer_CRMN, T_max = t_max_epoch, eta_min=1e-6)

    optimizer_CRFN = optim.Adam([{'params': model.filter.parameters()}], lr = args.lr)  # , {'params': model.Encoder_s.parameters()}
    scheduler_CRFN = CosineAnnealingLR(optimizer_CRFN, T_max = t_max_epoch, eta_min=1e-6)

    optimizer_Disc = optim.Adam(model.discriminator.parameters(), lr = lr)
    scheduler_Disc = CosineAnnealingLR(optimizer_Disc, T_max = t_max_epoch, eta_min=1e-6)

    optimizer_Rec = optim.Adam([ 
        {'params': model.Encoder_s.parameters()},
        {'params': model.Decoder.parameters()}
    ], lr=args.lr)
    optimizer_UCM = optim.Adam(model.pseudo_mlp.parameters(), lr = lr)

    optimizer_ViewDisc = optim.Adam(model.view_discriminator.parameters(), lr=lr)
    scheduler_ViewDisc = CosineAnnealingLR(optimizer_ViewDisc, T_max=t_max_epoch, eta_min=1e-6)

    ''' Training '''
    pre_train = args.pre_train
    num_iters = len(train_loader) * pre_train
    iter_idx = 0
    criterion = Loss(n_cluster)

    for epoch in range(epochs):
        model.train()
        loop = tqdm(enumerate(zip(*train_loader)), desc=f'Training Processing:  {epoch + 1} / {epochs} ',total=len(train_loader[0]), # if you want bar, you can copy total=len(train_loader[0]),
                    leave=True, dynamic_ncols=True)
        all_loss, loss_consis, loss_rec, loss_adver_disc, loss_adver_cla, loss_Clu, loss_comple = 0, 0, 0, 0, 0, 0, 0 # loss_1 is L_res

        '''Datasets Initialization'''
        for batch_idx, batch in loop:

            data_list, target_list = zip(*batch)
            data_list = list(data_list)



            for view in range(view_num):
                data_list[view] = data_list[view].to(device)


            '''Training Consistency Adversarial Module'''
            if epoch < pre_train:

                loss_1, loss_2 = GRCAM_Bi(model, data_list, view_num, criterion, args, device, optimizer_CRMN, optimizer_ViewDisc, iter_idx, num_iters)

                iter_idx += 1
                loss_adver_cla += loss_1
                loss_consis += loss_2

            else:
                set_requires_grad(model.view_discriminator, False)
                set_requires_grad(model.MLP, False)
                set_requires_grad(model.Encoder_c, False)
                model.view_discriminator.eval()
                model.MLP.eval()
                model.Encoder_c.eval()
                loss_1, loss_2 = CRFN_Adversarial(model, data_list, view_num, criterion, args, device, optimizer_CRFN, optimizer_Disc)
     
                loss_adver_disc += loss_1
                loss_comple += loss_2
 

                loss_rec += reconstruction(model, data_list, view_num, criterion, optimizer_Rec)
                loss_Clu += UCRM(
                    model,
                    data_list,
                    view_num,
                    criterion,
                    optimizer_UCM,
                    args.lambda_clu,
                )

            all_loss = loss_consis + loss_adver_cla + loss_adver_disc + loss_rec + loss_Clu + loss_comple
            loop.set_postfix(Recon_Loss=f"{loss_rec: .6f}",
                             All_Loss=f'{all_loss: .6f}',
                             Con_Loss = f"{loss_consis: .6f}",
                             Comple_Loss = f"{loss_comple: .6f}",
                             zDiscri_Loss = f"{loss_adver_disc: .6f}",
                             ViewDiscriminator_Loss=f"{loss_adver_cla: .6f}",
                             Pseudo_Loss = f"{loss_Clu: .6f}")
        if epoch < pre_train:
            scheduler_CRMN.step()
            scheduler_ViewDisc.step()
        else:
            scheduler_Disc.step()
            scheduler_CRFN.step()
    metrics = Evaluation(model, test_loader, device, view_num)
    print(f'Clustering performance: ACC={metrics["acc"]:.4f}, '
          f'ARI={metrics["ari"]:.4f}, '
          f'NMI={metrics["nmi"]:.4f}')
    return metrics

def Evaluation(model, test_loader, device, view_num):

    all_labels_true, all_labels_pred = [], []
    with torch.no_grad():  # if you want bar, you can copy total=len(train_loader[0]) in loop,
        model.eval()
        data = enumerate(zip(*test_loader))
        for batch_idx, batch in data:
            data_list, target_list = zip(*batch)
            data_list = list(data_list)
            for i in range(view_num): data_list[i] = data_list[i].to(device)
            _, pseudo_list, _, _ = model(data_list)

            '''Pseudo-Clustering Result'''
            p = torch.stack(pseudo_list)
            max_probs = torch.mean(p, dim=0)
            final_p, final_c = torch.max(max_probs, dim = 1)
            y = final_c.detach().cpu().numpy().ravel().astype(int)
            label = target_list[0].numpy().ravel()
            '''Clustering Result Recording'''
            all_labels_true.extend(label)
            all_labels_pred.extend(y)

    return clustering_metrics(all_labels_pred, all_labels_true)



class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def GRL(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

def data_preprocess(dataset_name, args, view_num):
    '''
        :param dataset_name: Training dataset name
        :param args: args object
        :return: *A list, called train_loader and test_loader, include each view *
    '''
    data_train, y_list = load_data(dataset_name)
    # mask = (np.random.rand(args.batch_size, 128) > 0.7).astype(np.float32)
    np.random.seed(args.seed)
    train_loader, test_loader = [], []
    index_list = np.arange(len(data_train[0]))
    index_train = np.random.choice(index_list, size=int(len(index_list) * args.train_rate), replace=False)

    index_test = np.setdiff1d(index_list, index_train)  # The shuffle operation on the next line won't shuffle the data; that will happen later.
    np.random.shuffle(index_test)
    n_cluster = len(np.unique(y_list))
    for i in range(view_num):
        x_train, y_train = data_train[i][index_train] , y_list[index_train]

        x_test, y_test = data_train[i][index_test], y_list[index_test]

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.int32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.int32)
        dataset_train = TensorDataset(x_train_tensor, y_train_tensor)
        dataset_test = TensorDataset(x_test_tensor, y_test_tensor)
        data_train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, drop_last=False)
        data_test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)
        train_loader.append(data_train_loader)
        test_loader.append(data_test_loader)


    return train_loader, test_loader, n_cluster
