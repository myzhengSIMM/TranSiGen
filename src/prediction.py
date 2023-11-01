from dataset import TranSiGenDataset_screening
from utils import *
from cmapPy.pandasGEXpress.parse import parse
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for prediction")
    parser.add_argument("--model_path", type=str, default='../results/trained_models_164_cell_smiles_split/364039/feature_KPGT_init_pretrain_shRNA/best_model.pt')
    parser.add_argument("--data_path", type=str, default='../data/PRISM/screening_compound.csv')
    parser.add_argument("--molecule_feature_embed_path", type=str, default='../data/PRISM/KPGT_emb2304.pickle')
    parser.add_argument("--cell", type=str, default='YAPC')
    parser.add_argument("--seed", type=int, default=364039)
    parser.add_argument("--dev", type=str, default='cuda:0')

    args = parser.parse_args()
    return args

def prediction_profiles(args):
    df_gene = pd.read_csv('../data/LINCS2020/geneinfo_processed.csv')
    df_landmark_gene = df_gene[(df_gene['pr_is_bing'] == 1) & (df_gene['pr_is_lm']==1)]
    df_best_infer_gene = df_gene[(df_gene['pr_is_bing'] == 1) & (df_gene['pr_is_lm']==0)]
    landmark_ids = df_landmark_gene['pr_id'].tolist()
    best_infer_ids = df_best_infer_gene['pr_id'].tolist()
    weight_path = '../data/LINCS2020/infer_weight.gctx'
    infer_weight = parse(weight_path, cid=['OFFSET']+landmark_ids, rid=best_infer_ids)
    infer_weight_df_tmp = infer_weight.data_df
    infer_weight_df = infer_weight_df_tmp[['OFFSET'] + landmark_ids]
    infer_weight_df = infer_weight_df.loc[best_infer_ids]

    with open('../data/LINCS2020/modz_x1.pickle', 'rb') as f:
        dict_modz_x1_all_cid = pickle.load(f)


    dev = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model_path, map_location='cpu')
    model.dev = torch.device(dev)
    model.to(dev)
    print(model)

    selected_cid = args.cell
    random_seed = args.seed
    with open(args.molecule_feature_embed_path, 'rb') as f:
        smi2emb = pickle.load(f)
    df_screening = pd.read_csv(args.data_path)

    emb_array = []
    smi_idx_array = []
    for idx, row in df_screening.iterrows():
        smi = row['canonical_smiles']
        emb_array.append(smi2emb[smi])
        smi_idx_array.append(row['cp_id'])
    emb_array = np.array(emb_array)
    smi_idx_array = np.array(smi_idx_array)
    cid_array = np.array([selected_cid] * emb_array.shape[0])
    x1_array = dict_modz_x1_all_cid[selected_cid]
    x1_array = np.repeat(x1_array, emb_array.shape[0], axis=0).astype(np.float32)

    test = TranSiGenDataset_screening(x1=x1_array, mol_feature=emb_array, mol_id=smi_idx_array, cid=cid_array)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=64, shuffle=False, drop_last=False, num_workers=4, worker_init_fn=seed_worker)

    setup_seed(random_seed)
    x2_pred_array, cp_id_array, cid_array = model.predict_profile_for_x1(test_loader)
    ddict_data = dict()
    ddict_data['x1'] = x1_array
    ddict_data['x2_pred'] = x2_pred_array
    ddict_data['cp_id'] = cp_id_array
    ddict_data['cid'] = cid_array

    sig_ls = []
    for idx in range(ddict_data['cid'].shape[0]):
        sig_ls.append(str(int(ddict_data['cp_id'][idx])) + '_' + ddict_data['cid'][idx])
    ddict_data['sig'] = np.array(sig_ls)

    ## infer gene
    for data_type in ['x1', 'x2_pred']:
        x = ddict_data[data_type]
        inferred = np.dot(x, np.array(infer_weight_df.T[1:])) + np.array(
            infer_weight_df.T.loc['OFFSET'])  # (18539, 9196)
        x_tmp = np.concatenate((x, inferred), axis=1)
        ddict_data['{}_inferred'.format(data_type)] = x_tmp

    for k in ddict_data.keys():
        print(k, ddict_data[k].shape)
    save_to_HDF('../results/6.Phenotype_based_drug_repurposing/prediction_profile_{}_{}.h5'.format(selected_cid, random_seed), ddict_data)




if __name__ == "__main__":
    args = parse_args()
    prediction_profiles(args)