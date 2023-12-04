# TranSiGen
code for "TranSiGen: Deep representation learning of chemical-induced transcriptional profile"

![Image text](https://github.com/myzheng-SIMM/TranSiGen/blob/main/data/Fig1.tif)
Fig. 1 TranSiGen’s architecture and application. a The data processing flow for TranSiGen. b The architecture and inference process of TranSiGen. c The applications of TranSiGen-derived representation.

### Train model
```
python train_TranSiGen_full_data.py --data_path ../dataLINCS2020/processed_data_id.h5
                                    --molecule_path ../data/LINCS2020/idx2smi.pickle
                                    --molecule_feature KPGT
                                    --initialization_model pretrain_shRNA
                                    --split_data_type smiles_split
                                    --n_epochs 300
                                    --n_latent 100
                                    --molecule_feature_embed_dim [400]
                                    --batch_size 64
                                    --learning_rate 1e-3
                                    --beta 0.1
                                    --dropout 0.1
                                    --weight_decay 1e-5
                                    --train_flag True
                                    --eval_metric True
```

### Infer profile
```
python train_TranSiGen_full_data.py --data_path ../dataLINCS2020/processed_data_id.h5
                                    --molecule_path ../data/LINCS2020/idx2smi.pickle
                                    --molecule_feature KPGT
                                    --initialization_model pretrain_shRNA
                                    --split_data_type smiles_split
                                    --n_epochs 300
                                    --n_latent 100
                                    --molecule_feature_embed_dim [400]
                                    --batch_size 64
                                    --learning_rate 1e-3
                                    --beta 0.1
                                    --dropout 0.1
                                    --weight_decay 1e-5
                                    --train_flag True
                                    --eval_metric True
```

### Setup and dependencies
requirements.txt contains environment of this project.

### Requirements
python = 3.6.13  
pytorch = 1.5.1  
cmappy = 4.0.1  
scikit-learn = 0.24.2  
numpy = 1.19.5  
rdkit = 2020.09.1  
