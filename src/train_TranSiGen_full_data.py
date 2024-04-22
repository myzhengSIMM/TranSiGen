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

    parser.add_argument("--train_flag", type=bool, default=False)
    parser.add_argument("--eval_metric", type=bool, default=False)
    parser.add_argument("--predict_profile", type=bool, default=False)

    args = parser.parse_args()
    return args

def train_TranSiGen(args):

    print(args)
    # torch.set_num_threads(1)
    # Set model
    # random_seed = int(''.join([str(random.randint(0, 9)) for _ in range(4)]))
    random_seed = args.seed
    setup_seed(random_seed)
    dev = torch.device(args.dev if torch.cuda.is_available() else 'cpu')

    data = load_from_HDF(args.data_path)
    cell_count = len(set(data['cid']))
    print('cell count:', cell_count)
    # data = subsetDict(data, np.arange(10000))

    with open(args.molecule_path, 'rb') as f:
        idx2smi = pickle.load(f)

    print('all data:', len(data['canonical_smiles']), len(set(data['canonical_smiles'])))

    train_flag = args.train_flag
    eval_metric = args.eval_metric
    predict_profile = args.predict_profile

    init_mode = args.initialization_model
    feat_type = args.molecule_feature
    if feat_type == 'KPGT':
        features_dim = 2304
    elif feat_type == 'ECFP4':
        features_dim = 2048
    split_type = args.split_data_type
    train_cell_count = args.train_cell_count
    n_folds = 5

    n_epochs = args.n_epochs
    n_latent = args.n_latent
    features_embed_dim = args.molecule_feature_embed_dim
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

    if split_type in ['random_split', 'smiles_split']:
        pair, pairv, pairt = split_data(data, n_folds=n_folds, split_type=split_type, rnds=random_seed)
    elif split_type == 'cells_split':
        pair, pairv, pairt = split_data_cid(data, train_cell_count=train_cell_count)
    print('===============', split_type, '================')
    print('train', len(set(pair['cid'])), len(pair['canonical_smiles']), len(set(pair['canonical_smiles'])), )
    print('valid', len(set(pairv['cid'])), len(pairv['canonical_smiles']), len(set(pairv['canonical_smiles'])), )
    print('test', len(set(pairt['cid'])), len(pairt['canonical_smiles']), len(set(pairt['canonical_smiles'])), )

    for name, pair_data in zip(['train', 'valid', 'test'], [pair, pairv, pairt]):
        df = pd.DataFrame(pair_data['canonical_smiles'], columns=['canonical_smiles'])
        df.drop_duplicates(inplace=True)
        # df.to_csv(local_out + '{}_{}_data_canonical_smiles.csv'.format(split_type, name), index=False)


    train = TranSiGenDataset(
        LINCS_index=pair['LINCS_index'],
        mol_feature_type=feat_type,
        mol_id=pair['canonical_smiles'],
        cid=pair['cid']
    )

    valid = TranSiGenDataset(
        LINCS_index=pairv['LINCS_index'],
        mol_feature_type=feat_type,
        mol_id=pairv['canonical_smiles'],
        cid=pairv['cid']
    )

    test = TranSiGenDataset(
        LINCS_index=pairt['LINCS_index'],
        mol_feature_type=feat_type,
        mol_id=pairt['canonical_smiles'],
        cid=pairt['cid']
    )


    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=seed_worker)
    valid_loader = torch.utils.data.DataLoader(dataset=valid, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=seed_worker)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=seed_worker)


    if train_flag:
        model = TranSiGen(n_genes=978, n_latent=n_latent, n_en_hidden=[1200], n_de_hidden=[800],
                            features_dim=features_dim, features_embed_dim=features_embed_dim,
                            init_w=True, beta=beta, device=dev, dropout=dropout,
                            path_model=local_out, random_seed=random_seed)
        model_dict = model.state_dict()

        if init_mode == 'pretrain_shRNA':
            print('=====load vae for x1 and x2=======')
            filename = '../results/trained_model_shRNA_vae_x1/best_model.pt'
            model_base_x1 = torch.load(filename, map_location='cpu')
            model_base_x1_dict = model_base_x1.state_dict()
            for k in model_dict.keys():
                if k in model_base_x1_dict.keys():
                    model_dict[k] = model_base_x1_dict[k]
            filename = '../results/trained_model_shRNA_vae_x2/best_model.pt'
            model_base_x2 = torch.load(filename, map_location='cpu')
            model_base_x2_dict = model_base_x2.state_dict()
            for k in model_dict.keys():
                if k in model_base_x2_dict.keys():
                    model_dict[k] = model_base_x2_dict[k]
            model.load_state_dict(model_dict)
            del model_base_x1, model_base_x2
        model.to(dev)
        print(model)

        epoch_hist, best_epoch = model.train_model(train_loader=train_loader, test_loader=valid_loader,
                                                   n_epochs=n_epochs, learning_rate=learning_rate, weight_decay=weight_decay, save_model=True)

        epoch_result = pd.DataFrame.from_dict(epoch_hist)
        epoch_result['epoch'] = np.arange(n_epochs)
        # epoch_result.to_csv(local_out + 'epoch{}_lr{}.csv'.format(n_epochs, learning_rate), index=False)

        epoch = epoch_result['valid_loss'].idxmin()
        if epoch == best_epoch:
            print('best valid epoch:', epoch)
        else:
            print('warning: inconsistent best valid')


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

    if eval_metric:
        print('===============Evaluate model performance==============')
        setup_seed(random_seed)
        _, _, test_metrics_dict_ls = model.test_model(loader=test_loader, metrics_func=['pearson', 'rmse', 'precision100'])


        for name, rec_dict_value in zip(['test'], [test_metrics_dict_ls]):
            df_rec = pd.DataFrame.from_dict(rec_dict_value)
            smi_ls = []
            for smi_id in df_rec['cp_id']:
                smi_ls.append(idx2smi[smi_id])
            df_rec['canonical_smiles'] = smi_ls
            df_rec.to_csv(save_dir + '/{}_restruction_result_all_samples.csv'.format(name), index=False)


    if predict_profile:
        print('===============Predict profile==============')
        for name, loader in zip(['test'], [test_loader]):
            x1_array, x2_array, x1_rec_array, x2_rec_array, x2_pred_array, _, mol_id_array, cid_array, sig_array = model.predict_profile(loader=loader)
            ddict_data = dict()
            ddict_data['x1'] = x1_array
            ddict_data['x2'] = x2_array
            ddict_data['x2_rec'] = x2_rec_array
            ddict_data['x2_pred'] = x2_pred_array
            ddict_data['cp_id'] = mol_id_array
            ddict_data['cid'] = cid_array
            ddict_data['sig'] = sig_array

            for k in ddict_data.keys():
                print(type(ddict_data[k][0]), ddict_data[k].shape)
            save_to_HDF(save_dir + '/{}_prediction_profile.h5'.format(name), ddict_data)


if __name__ == "__main__":
    args = parse_args()
    train_TranSiGen(args)

