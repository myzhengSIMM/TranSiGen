from dataset import TranSiGenDataset
from model import TranSiGen
from utils import *
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training TranSiGen")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--molecule_path", type=str)

    parser.add_argument("--dev", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=364039)
    parser.add_argument("--molecule_feature", type=str, default='KPGT', help='molecule_feature(KPGT, ECFP4)')
    parser.add_argument("--initialization_model", type=str, default='pretrain_shRNA', help='molecule_feature(pretrain_shRNA, random)')
    parser.add_argument("--split_data_type", type=str, default='smiles_split', help='split_data_type(random_split, smiles_split, cell_split)')
    parser.add_argument("--train_cell_count", type=str, default='None', help='if cell_split, train_cell_count=10,50,all, else None')

    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--n_latent", type=int, default=100)
    parser.add_argument("--molecule_feature_embed_dim", nargs='+', type=int, default=[400])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)


    args = parser.parse_args()
    return args

def evaluate_TranSiGen(args):

    print(args)
    random_seed = args.seed
    setup_seed(random_seed)
    dev = torch.device(args.dev if torch.cuda.is_available() else 'cpu')

    cell_count = 164
    # print('cell count:', cell_count)

    with open(args.molecule_path, 'rb') as f:
        idx2smi = pickle.load(f)

    # print('all data:', len(data['canonical_smiles']), len(set(data['canonical_smiles'])))


    init_mode = args.initialization_model
    feat_type = args.molecule_feature

    split_type = args.split_data_type
    train_cell_count = args.train_cell_count

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    beta = args.beta
    dropout = args.dropout
    weight_decay = args.weight_decay

    print('feature:', feat_type, 'init_mode:', init_mode, 'split_type:', split_type, 'learning_rate:', learning_rate, 'dropout:', dropout, 'beta:', beta, 'weight decay:', weight_decay)

    if split_type != 'cells_split':
        print('train_cell_count used for cell_split')
        assert (train_cell_count == 'None')

    # Out dir
    if split_type == 'cells_split':
        local_out = '../results/trained_models_{}_cell_{}/{}/feature_{}_init_{}_{}/'.format(cell_count, split_type, random_seed, feat_type, init_mode, train_cell_count)
        print(local_out)
    else:
        local_out = '../results/trained_models_{}_cell_{}/{}/feature_{}_init_{}/'.format(cell_count, split_type, random_seed, feat_type, init_mode)
    isExists = os.path.exists(local_out)
    if not isExists:
        os.makedirs(local_out)
        print('Directory created successfully')
    else:
        print('Directory already exists')

    pairt = load_from_HDF('../data/LINCS2020/test.h5')

    test = TranSiGenDataset(
        LINCS_index=pairt['LINCS_index'],
        mol_feature_type=feat_type,
        mol_id=pairt['canonical_smiles'],
        cid=pairt['cid']
    )

    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=seed_worker)


    filename = local_out + 'best_model.pt'
    model = torch.load(filename, map_location='cpu')
    model.dev = torch.device(dev)
    model.to(dev)

    save_dir = local_out + 'predict'
    isExists = os.path.exists(save_dir)
    print(save_dir)
    if not isExists:
        os.makedirs(save_dir)
        print('Directory created successfully')
    else:
        print('Directory already exists')


    print('===============Evaluate model performance==============')
    setup_seed(random_seed)
    _, test_metrics_dict, test_metrics_dict_ls = model.test_model(loader=test_loader, metrics_func=['pearson', 'rmse', 'precision100'])

    df_metrics = pd.DataFrame(columns=['data'] + list(test_metrics_dict.keys()))
    for name, dict_value in zip(['test'], [test_metrics_dict]):
        df_metrics.loc[df_metrics.shape[0]] = [name] + list(dict_value.values())

    print('restruction evaluation:', df_metrics)
    round(df_metrics, 3).to_csv(save_dir + '/restruction_result.csv', index=False)

    for name, rec_dict_value in zip(['test'], [test_metrics_dict_ls]):
        df_rec = pd.DataFrame.from_dict(rec_dict_value)
        smi_ls = []
        for smi_id in df_rec['cp_id']:
            smi_ls.append(idx2smi[smi_id])
        df_rec['canonical_smiles'] = smi_ls
        df_rec.to_csv(save_dir + '/{}_restruction_result_all_samples.csv'.format(name), index=False)


if __name__ == "__main__":
    args = parse_args()
    evaluate_TranSiGen(args)

