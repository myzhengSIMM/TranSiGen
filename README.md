# TranSiGen
code for "Deep Representation Learning of Chemical-induced Transcriptional Profile for Phenotype-Based Drug Discovery"

![Image text](https://github.com/myzheng-SIMM/TranSiGen/blob/main/data/TranSiGen.jpg)  
Fig. 1 TranSiGen’s architecture and application. a The data processing flow for TranSiGen. b The architecture and inference process of TranSiGen. c The applications of TranSiGen-derived representation.

### Train model
```
python train_TranSiGen_full_data.py --data_path ../data/LINCS2020/data_example/processed_data_id.h5
                                    --molecule_path ../data/LINCS2020/idx2smi.pickle
                                    --molecule_feature KPGT
                                    --initialization_model pretrain_shRNA
                                    --split_data_type smiles_split
                                    --n_epochs 300
                                    --n_latent 100
                                    --molecule_feature_embed_dim 400
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
python prediction.py --model_path   ../results/trained_models_164_cell_smiles_split/364039/feature_KPGT_init_pretrain_shRNA/best_model.pt
                     --data_path ../data/PRISM/screening_compound.csv
                     --molecule_feature_embed_path ../data/PRISM/KPGT_emb2304.pickle
                     --cell YAPC
                     --seed 100
```
(infer_weight.gctx for inference has been uploaded in Release)

### Setup and dependencies
requirements.txt contains environment of this project.

### Requirements
python = 3.6.13  
pytorch = 1.5.1  
cmappy = 4.0.1  
scikit-learn = 0.24.2  
numpy = 1.19.5  
rdkit = 2020.09.1  

### Update
The following required files can be downloaded from the Release:  
processed_data.h5  
infer_weight.gctx  
ligand_based_virtual_screening.rar
