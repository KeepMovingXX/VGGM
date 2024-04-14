import argparse
from model import DeepVGAE
from untils import *
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool
from sklearn.mixture import GaussianMixture
from evaluation import Evaluation
from tqdm.auto import tqdm


parser = argparse.ArgumentParser(description='VGGM')
parser.add_argument('--dataset', default='./data/mit.mtx', help='Dataset name')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay')
parser.add_argument('--batch_size', type=int, default=45, help='batch size of training')
parser.add_argument('--device', type=str, default='cpu', help='device')
parser.add_argument('--liner_dim',  type=int, default=94)
parser.add_argument('--enc_in_channels', type=int, default=32, help='input dim of encoder ')
parser.add_argument('--enc_hidden_channels', type=int, default=16, help='hidden dims of encoder ')
parser.add_argument('--enc_out_channels', type=int, default=8, help='output dims of encoder ')
parser.add_argument('--iter_times', type=int, default=3, help='joint train times ')
parser.add_argument('--components', type=int, default=3, help='n_components of GMM ')
parser.add_argument('--covariance_type', type=str, default='diag', help='covariance_type of GMM ')
parser.add_argument('--snapshots', type=int, default=45, help='number of snapshots ')
parser.add_argument('--nodes', type=int, default=94, help='number of nodes ')
args = parser.parse_args()


groundEventmit = set([1, 7, 8, 10, 11, 12, 14, 15, 16, 18, 20, 22,
                       26, 27, 29, 30, 36, 37, 38, 39])
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
datalist = load_graph(args.dataset, args.device, args.enc_in_channels)
loader = DataLoader(datalist, batch_size= args.batch_size, shuffle=True)


def Pretrainer():
    for _ in tqdm(range(args.epochs)):
        for data in loader:
            model.train()
            optimizer.zero_grad()
            loss = model.loss(data.x, data.edge_index)
            loss.backward()
            optimizer.step()
    nodes_emb = model.encode(data.x, data.edge_index)
    graph_emb = global_max_pool(nodes_emb, batch=data.batch).detach().cpu().numpy()
    return graph_emb


def results(graph_emb):
    pre, reca, f1, fpr = [], [], [], []
    for i in range(2, 10):
        gmm = GaussianMixture(n_components=i, covariance_type=args.covariance_type, reg_covar=1e-5)
        gmm_result = gmm.fit_predict(graph_emb)
        change_set = predictResult(gmm_result)
        precision, recall, fvalue, fpr = Evaluation(change_set, groundEventmit, 0, args.snapshots)
        pre.append(precision)
        reca.append(recall)
        f1.append(fvalue)
    # pre_bestidx = pre.index(max(pre))
    # reca_bestidx = reca.index(max(reca))
    f1_bestidx = f1.index(max(f1))
    return pre[f1_bestidx], reca[f1_bestidx], f1[f1_bestidx]


def init_gmm(graph_emb):
    gmm = GaussianMixture(n_components=args.components, covariance_type=args.covariance_type)
    gmm.fit(graph_emb)
    model.pi.data = torch.FloatTensor(gmm.weights_).to(device)
    model.mu.data = torch.FloatTensor(gmm.means_).to(device)
    model.logvar.data = torch.log(torch.FloatTensor(gmm.covariances_)).to(device)


if __name__ == '__main__':

    train_precision, train_recall, train_f1 = [], [], []
    eval_precision, eval_recall, eval_f1 = [], [],[]
    model = DeepVGAE(args)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    GMM = GaussianMixture(n_components=args.components, covariance_type=args.covariance_type, reg_covar=1e-5)
    print('VGAE begin to pretrain...')
    graph_embedding = Pretrainer()
    print('GMM begin to fit...')
    train_result = GMM.fit_predict(graph_embedding)
    change_set = predictResult(train_result)
    precision, recall, fvalue, fpr = Evaluation(change_set, groundEventmit, 0, args.snapshots)
    train_precision.append(precision)
    train_recall.append(recall)
    train_f1.append(fvalue)

    # get the evaluation results
    eval_pre, eval_re, eval_f = results(graph_embedding)
    eval_precision.append(eval_pre)
    eval_recall.append(eval_re)
    eval_f1.append(eval_f)

    # joint training
    for j in range(0, args.iter_times):
        print('joint training', '\t', str(j+1),'\t', 'times')
        model.pi.data = torch.FloatTensor(GMM.weights_).to(device)
        model.mu.data = torch.FloatTensor(GMM.means_).to(device)
        model.logvar.data = torch.log(torch.FloatTensor(GMM.covariances_)).to(device)
        for epo in tqdm(range(args.epochs)):
            for indata in loader:
                model.train()
                optimizer.zero_grad()
                loss1 = model.loss_function(args.nodes * args.snapshots, indata.x, indata.edge_index)
                loss1.backward()
                optimizer.step()
        nodes_emb = model.encode(indata.x, indata.edge_index)
        graph_emb = global_max_pool(nodes_emb, batch=indata.batch).detach().cpu().numpy()

        # train result this joint training
        train_result = GMM.fit_predict(graph_emb)
        change_set = predictResult(train_result)
        precision, recall, fvalue, fpr = Evaluation(change_set, groundEventmit, 0, args.snapshots)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(fvalue)

        # eval result this joint training
        eval_pre, eval_re, eval_f = results(graph_emb)
        eval_precision.append(eval_pre)
        eval_recall.append(eval_re)
        eval_f1.append(eval_f)

    # get the final result
    f1_bestidx = train_f1.index(max(train_f1))
    print('train: the results is \n', f'precision: {train_precision[f1_bestidx]:.4f}', '\t', f'recall: {train_recall[f1_bestidx]}', '\t', f'F1: {train_f1[f1_bestidx]:.4f}')
    eval_f1_bestidx = eval_f1.index(max(eval_f1))
    print('eval: the results is \n', f'precision: {eval_precision[eval_f1_bestidx]:.4f}', '\t', f'recall: {eval_recall[eval_f1_bestidx]}', '\t',
          f'F1: {eval_f1[eval_f1_bestidx]:.4f}')





