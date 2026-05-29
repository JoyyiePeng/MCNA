import random
from models.detector import *
from dgl.data.utils import load_graphs
import os
import json
import pandas
import torch


class Dataset:
    def __init__(self, name='tfinance', prefix='datasets/'):
        graph = load_graphs(prefix + name)[0][0]
        if graph.idtype == torch.int32:
            graph = graph.long()
        self.name = name
        self.graph = graph

    def split(self, semi_supervised=True, trial_id=0):
        if semi_supervised:
            trial_id += 10
        self.graph.ndata['train_mask'] = self.graph.ndata['train_masks'][:,trial_id]
        self.graph.ndata['val_mask'] = self.graph.ndata['val_masks'][:,trial_id]
        self.graph.ndata['test_mask'] = self.graph.ndata['test_masks'][:,trial_id]
        print(self.graph.ndata['train_mask'].sum(), self.graph.ndata['val_mask'].sum(), self.graph.ndata['test_mask'].sum())


model_detector_dict = {
    # Classic Methods
    'MLP': BaseGNNDetector,
    'KNN': KNNDetector,
    'SVM': SVMDetector,
    'RF': RFDetector,
    'XGBoost': XGBoostDetector,
    'XGBOD': XGBODDetector,
    'NA': XGBNADetector,

    # Standard GNNs
    'GCN': BaseGNNDetector,
    'SGC': BaseGNNDetector,
    'GIN': BaseGNNDetector,
    'GraphSAGE': BaseGNNDetector,
    'GAT': BaseGNNDetector,
    'GT': BaseGNNDetector,
    'PNA': BaseGNNDetector,
    'BGNN': BGNNDetector,

    # Specialized GNNs
    'GAS': GASDetector,
    'BernNet': BaseGNNDetector,
    'AMNet': BaseGNNDetector,
    'BWGNN': BaseGNNDetector,
    'GHRN': GHRNDetector,
    'GATSep': BaseGNNDetector,
    'PCGNN': PCGNNDetector,
    'DCI': DCIDetector,

    # Heterogeneous GNNs
    'RGCN': HeteroGNNDetector,
    'HGT': HeteroGNNDetector,
    'CAREGNN': CAREGNNDetector,
    'H2FD': H2FDetector, 

    # Tree Ensembles with Neighbor Aggregation
    'RFGraph': RFGraphDetector,
    'XGBGraph': XGBGraphDetector,
    
    # Custom Methods
    'MySAGE': BaseGNNDetector,
    'GAGA': GAGADetector,
    'ConsisGAD': ConsisGADDetector,
    'KYCGCN': BaseGNNDetector,
    
    # Geometric GNNs
    'SpaceGNN': SpaceGNNDetector,
    
    # Dynamic Grouping GNNs
    'DGAGNN': DGAGNNDetector,

    # Extened SAGEs
    'GraphSAGEMean': BaseGNNDetector,
    'GraphSAGEPool': BaseGNNDetector,
    'GraphSAGELSTM': BaseGNNDetector,
    'GraphSAGEGCN': BaseGNNDetector,
    'MySAGEMean': BaseGNNDetector,
    'MySAGEPool': BaseGNNDetector,
    'MySAGELSTM': BaseGNNDetector,
    'MySAGEGCN': BaseGNNDetector,

    # Adaptive Message Passing
    'GADAM': GADAMDetector,
}


def save_results(results, file_id):
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if file_id is None:
        file_id = 0
        while os.path.exists('results/{}.xlsx'.format(file_id)):
            file_id += 1
    results.transpose().to_excel('results/{}.xlsx'.format(file_id))
    print('save to file ID: {}'.format(file_id))
    return file_id

def better_save_results(results, file_id):
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if file_id is None:
        file_id = 0
        while os.path.exists('results/{}.xlsx'.format(file_id)):
            file_id += 1
    
    # Create Excel writer with multiple sheets
    with pandas.ExcelWriter(f'results/{file_id}.xlsx', engine='openpyxl') as writer:
        # Sheet 0 - All results (same as original function)
        results.transpose().to_excel(writer, sheet_name='All Results')
        
        # Extract model names and dataset names
        models = results['name'].tolist()
        datasets = []
        for col in results.columns:
            if '-AUROC mean' in col:
                datasets.append(col.split('-AUROC mean')[0])
        
        # Create separate sheets for each metric with datasets as rows and methods as columns
        for metric in ['AUROC', 'AUPRC', 'RecK', 'F1']:
            metric_df = pandas.DataFrame(index=datasets, columns=models)
            for dataset in datasets:
                for i, model in enumerate(models):
                    metric_df.loc[dataset, model] = results.iloc[i][f'{dataset}-{metric} mean']
            metric_df.to_excel(writer, sheet_name=f'{metric} Mean')
    
    # Create markdown table
    md_content = "# Results\n\n"
    md_content += "| Dataset | Metric |"
    for model in models:
        md_content += f" {model} |"
    md_content += "\n"
    
    # Header separator
    md_content += "| --- | --- |"
    for _ in models:
        md_content += " --- |"
    md_content += "\n"
    
    # Content rows
    for dataset in datasets:
        # First row for dataset has AUROC
        md_content += f"| {dataset} | AUROC |"
        for i, model in enumerate(models):
            mean = results.iloc[i][f'{dataset}-AUROC mean']
            std = results.iloc[i][f'{dataset}-AUROC std']
            if not pandas.isna(mean) and not pandas.isna(std):
                md_content += f" {mean:.4f} ±{std:.4f} |"
            else:
                md_content += " N/A |"
        md_content += "\n"
        
        # Second row for AUPRC
        md_content += f"| | AUPRC |"
        for i, model in enumerate(models):
            mean = results.iloc[i][f'{dataset}-AUPRC mean']
            std = results.iloc[i][f'{dataset}-AUPRC std']
            if not pandas.isna(mean) and not pandas.isna(std):
                md_content += f" {mean:.4f} ±{std:.4f} |"
            else:
                md_content += " N/A |"
        md_content += "\n"
        
        # Third row for RecK
        md_content += f"| | RecK |"
        for i, model in enumerate(models):
            mean = results.iloc[i][f'{dataset}-RecK mean']
            std = results.iloc[i][f'{dataset}-RecK std']
            if not pandas.isna(mean) and not pandas.isna(std):
                md_content += f" {mean:.4f} ±{std:.4f} |"
            else:
                md_content += " N/A |"
        md_content += "\n"

        # Forth row for F1-score
        md_content += f"| | F1-score |"
        for i, model in enumerate(models):
            mean = results.iloc[i][f'{dataset}-F1 mean']
            std = results.iloc[i][f'{dataset}-F1 std']
            if not pandas.isna(mean) and not pandas.isna(std):
                md_content += f" {mean:.4f} ±{std:.4f} |"
            else:
                md_content += " N/A |"
        md_content += "\n"
        
        # # Fourth row for Time
        # md_content += f"| | Time |"
        # for i, model in enumerate(models):
        #     time_val = results.loc[i, f'{dataset}-Time']
        #     if not pandas.isna(time_val):
        #         md_content += f" {time_val:.2f}s |"
        #     else:
        #         md_content += " N/A |"
        # md_content += "\n"
    
    # Save markdown to file
    with open(f'results/{file_id}.md', 'w') as f:
        f.write(md_content)
        
    print(f'Results saved to file ID: {file_id} (Excel and Markdown)')
    return file_id

def sample_param(model, dataset, t=0):
    model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}
    if t == 0:
        return model_config
    for k, v in param_space[model].items():
        model_config[k] = random.choice(v)
    # Avoid OOM in Random Search
    if model in ['GAT', 'GATSep', 'GT'] and dataset in ['tfinance', 'dgraphfin', 'tsocial']:
        model_config['h_feats'] = 16
        model_config['num_heads'] = 2
    if dataset == 'tsocial':
        model_config['h_feats'] = 16
    if dataset in ['dgraphfin', 'tsocial']:
        if 'k' in model_config:
            model_config['k'] = min(5, model_config['k'])
        if 'num_cluster' in model_config:
            model_config['num_cluster'] = 2
        # if 'num_layers' in model_config:
        #     model_config['num_layers'] = min(2, model_config['num_layers'])
    return model_config


def sample_trial_param(trial, model, dataset, t=0):
    model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}
    if t == 0:
        return model_config
    for k, v in trial_space[model].items():
        choice_method = v[0]
        choice_args = v[1] if len(v) >= 2 else tuple()
        choice_kwargs = v[2] if len(v) >= 3 else dict()
        model_config[k] = eval(f"trial.suggest_{choice_method}(k, *choice_args, **choice_kwargs)")
    # Avoid OOM in Random Search
    if model in ['GAT', 'GATSep', 'GT'] and dataset in ['tfinance', 'dgraphfin', 'tsocial']:
        model_config['h_feats'] = 16
        model_config['num_heads'] = 2
    if dataset == 'tsocial':
        model_config['h_feats'] = 16
    if dataset in ['dgraphfin', 'tsocial']:
        # if 'k' in model_config:
        #     model_config['k'] = min(5, model_config['k'])
        if 'num_cluster' in model_config:
            model_config['num_cluster'] = 2
    return model_config


param_space = {}
trial_space = {}

param_space['MLP'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['GCN'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['SGC'] = {
    'h_feats': [16, 32, 64],
    'k': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'mlp_layers': [1, 2],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['GIN'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'agg': ['sum', 'max', 'mean'],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['GraphSAGE'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'agg': ['mean', 'gcn', 'pool'],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}
trial_space['GraphSAGE'] = {
    'h_feats': ('categorical', ([16, 32, 64], )),
    'num_layers': ('int', (1, 3)),
    'agg': ('categorical', (['mean', 'gcn', 'pool'], )),
    'drop_rate': ('categorical', ([0, 0.1, 0.2, 0.3], )),
    'lr': ("float", (10e-3, 10e-1)),
    'activation': ('categorical', (['ReLU', 'LeakyReLU', 'Tanh'], ))
}

param_space['MySAGE'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'agg': ['mean', 'gcn', 'pool'],
    'dropout': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}
trial_space['MySAGE'] = {
    'h_feats': ('categorical', ([16, 32, 64], )),
    'num_layers': ('int', (1, 3)),
    'agg': ('categorical', (['mean', 'gcn', 'pool'], )),
    'dropout': ('categorical', ([0, 0.1, 0.2, 0.3], )),
    'lr': ("float", (10e-3, 10e-1)),
    'activation': ('categorical', (['ReLU', 'LeakyReLU', 'Tanh'], ))
}

param_space['ChebNet'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'mlp_layers': [1, 2],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['BernNet'] = {
    'h_feats': [16, 32, 64],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'mlp_layers': [1, 2],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'orders': [2, 3, 4, 5],
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['AMNet'] = {
    'h_feats': [16, 32, 64],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'orders': [2, 3],
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['BWGNN'] = {
    # Fixed original params (defaults)
    'h_feats': [32],
    'num_layers': [2],
    'lr': [0.01],
    'mlp_layers': [2],
    'activation': ['ReLU'],
    # Only search GatedFusion params
    'min_gcn_weight': [0.1, 0.2, 0.3, 0.5],
    'max_gcn_weight': [0.7, 0.8, 0.9, 1.0],
    'gcn_init_weight': [0.3, 0.5, 0.7, 0.9],
    'gate_dropout': [0.1, 0.2, 0.3, 0.5],
}

param_space['GAS'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'k': range(3, 51),
    'dist': ['euclidean', 'cosine'],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['GHRN'] = {
    'h_feats': [16, 32, 64],
    'del_ratio': 10 ** np.linspace(-2, -1, 1000),
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'mlp_layers': [1, 2],
}

param_space['KNNGCN'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'k': list(range(3, 51)),
    'dist': ['euclidean', 'cosine'],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['XGBoost'] = {
    'n_estimators': list(range(10, 201)),
    'eta': 0.5 * 10 ** np.linspace(-1, 0, 1000),
    'lambda': [0, 1, 10],
    'subsample': [0.5, 0.75, 1]
}
trial_space['XGBoost'] = {
    'n_estimators': ("int", (10, 200)),
    'eta': ("float", (0.05, 0.5)),
    'lambda': ("categorical", ([0, 1, 10], )),
    'subsample': ("categorical", ([0.5, 0.75, 1], ))
}

param_space['XGBGraph'] = {
    'n_estimators': list(range(10, 201)),
    'eta': 0.5 * 10 ** np.linspace(-1, 0, 1000),
    'lambda': [0, 1, 10],
    # 'alpha': [0, 0.5, 1],
    'subsample': [0.5, 0.75, 1],
    'num_layers': [1, 2, 3, 4],
    'agg': ['sum', 'max', 'mean'],
    'booster': ['gbtree', 'dart']
}
trial_space['XGBGraph'] = {
    'n_estimators': ("int", (10, 200)),
    'eta': ("float", (0.05, 0.5)),
    'lambda': ("categorical", ([0, 1, 10], )),
    # 'alpha': [0, 0.5, 1],
    'subsample': ("categorical", ([0.5, 0.75, 1], )),
    'num_layers': ("categorical", ([1, 2, 3, 4], )),
    'agg': ("categorical", (['sum', 'max', 'mean'], )),
    'booster': ("categorical", (['gbtree', 'dart'], ))
}

param_space['RF'] = {
    'n_estimators': list(range(10, 201)),
    'criterion': ['gini', 'entropy'],
    'max_samples': list(np.linspace(0.1, 1, 1000)),
}

param_space['RFGraph'] = {
    'n_estimators': list(range(10, 201)),
    'criterion': ['gini', 'entropy'],
    'max_samples': [0.5, 0.75, 1],
    'max_features': ['sqrt', 'log2', None],
    'num_layers': [1, 2, 3, 4],
    'agg': ['sum', 'max', 'mean'],
}

param_space['SVM'] = {
    'weights': ['uniform', 'distance'],
    'C': list(10 ** np.linspace(-1, 1, 1000))
}

param_space['KNN'] = {
    'k': list(range(3, 51)),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

param_space['XGBOD'] = {
    'n_estimators': list(range(10, 201)),
    'learning_rate': 0.5 * 10 ** np.linspace(-1, 0, 1000),  # [0.05, 0.1, 0.2, 0.3, 0.5],
    'lambda': [0, 1, 10],
    'subsample': [0.5, 0.75, 1],
    'booster': ['gbtree', 'dart']
}

param_space['GAT'] = {
    'h_feats': [16, 32],
    'num_heads': [1, 2, 4, 8],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['GATSep'] = {
    'h_feats': [16, 32],
    'num_heads': [1, 2, 4, 8],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['GT'] = {
    'h_feats': [16, 32],
    'num_heads': [1, 2, 4, 8],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['PCGNN'] = {
    'h_feats': [16, 32, 64],
    'del_ratio': np.linspace(0.01, 0.8, 1000),
    'add_ratio': np.linspace(0.01, 0.8, 1000),
    'dist': ['euclidean', 'cosine'],
    # 'k': list(range(3, 10)),
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['DCI'] = {
    'h_feats': [16, 32, 64],
    'pretrain_epochs': [20, 50, 100],
    'num_cluster': list(range(2,31)),
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['BGNN'] = {
    'depth': [4,5,6,7],
    'iter_per_epoch': [2,5,10,20],
    'gbdt_lr': 10 ** np.linspace(-2, -0.5, 1000),
    'normalize_features': [True, False],
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['NA'] = {
    'n_estimators': list(range(10, 201)),
    'eta': 0.5 * 10 ** np.linspace(-1, 0, 1000),
    'lambda': [0, 1, 10],
    'subsample': [0.5, 0.75, 1],
    'k': list(range(0, 51)),
}

param_space['PNA'] = {
    'h_feats': [16, 32, 64],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh'],
}

param_space['SpaceGNN'] = {
    'h_feats': [32, 64, 128],
    'layer_num': [2, 3, 4],
    'drop_rate': [0, 0.05, 0.1],
    'alpha': [0.3, 0.5, 0.7],
    'beta': [0.3, 0.5, 0.7],
    'batch_size': [32, 50, 64],
    'stdneg': [0.05, 0.1, 0.15],
    'stdpos': [0.05, 0.1, 0.15],
    'lr': 10 ** np.linspace(-4, -2, 1000),
}

param_space['DGAGNN'] = {
    'h_feats': [32, 64, 128],
    'n_hidden': [64, 128, 256],
    'p': [0.1, 0.3, 0.5],
    'n_head': [1, 2, 4],
    'unclear_up': [0.05, 0.1, 0.15],
    'unclear_down': [0.05, 0.1, 0.15],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-4, -2, 1000),
}