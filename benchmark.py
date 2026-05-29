import argparse
import time
from utils import *
import pandas
import os
import warnings
import traceback
warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))

def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--semi_supervised', type=int, default=0)
parser.add_argument('--inductive', type=int, default=0)
parser.add_argument('--models', type=str, default=None)
parser.add_argument('--datasets', type=str, default=None)
parser.add_argument('--better_output', choices=['True', 'False'], default='True')
parser.add_argument('--no_mcna', action='store_true',
                    help='Disable the MCNA branch (use the bare backbone only)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate (paper: tuned from {1e-4, 5e-4, 1e-3, 5e-3})')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Adam weight decay (paper: tuned from {1e-5, 1e-4, 1e-3})')
parser.add_argument('--epochs', type=int, default=10000,
                    help='Max training epochs (random_search.py uses 100)')
parser.add_argument('--patience', type=int, default=100,
                    help='Early-stopping patience (random_search.py uses 20)')
parser.add_argument('--cn_top_m', type=int, default=50,
                    help='Top-M truncation for the CN computation (default 50)')
parser.add_argument('--cn_semantic', type=str, default='shell_set',
                    choices=['shell_set', 'shell_walk'],
                    help='shell_set = |N^k(i) cap N^k(j)|; '
                         'shell_walk = walk-count A^k masked to k-hop shell')
parser.add_argument('--moe_top_k', type=int, default=2,
                    help='Number of hops each node activates in the MoE')
parser.add_argument('--moe_noise', type=float, default=0.0,
                    help='Gaussian noise added to router logits during training')
parser.add_argument('--moe_min_temp', type=float, default=1.0,
                    help='Minimum router temperature (smooths logits)')
parser.add_argument('--moe_router_dropout', type=float, default=0.0,
                    help='Dropout applied on router_probs during training')
args = parser.parse_args()

better_result = args.better_output == 'True'

columns = ['name']
new_row = {}
datasets = ['reddit', 'weibo', 'amazon', 'yelp', 'tolokers',                        # 0-4
            'questions', 'tfinance', 'elliptic', 'dgraphfin', 'tsocial',            # 5-9
            # 'hetero/amazon', 'hetero/yelp'
            'alpha_homora', 'cryptopia_hacker', 'plus_token_ponzi', 'upbit_hack',   # 10-13
            ]
models = model_detector_dict.keys()

if args.datasets is not None:
    if '-' in args.datasets:
        st, ed = args.datasets.split('-')
        datasets = datasets[int(st):int(ed)+1]
    else:
        datasets = [datasets[int(t)] for t in args.datasets.split(',')]
print('Evaluated Datasets: ', datasets)

if args.models is not None:
    models = args.models.split('-')
    print('Evaluated Baselines: ', models)

for dataset in datasets:
    for metric in ['AUROC mean', 'AUROC std', 'AUPRC mean', 'AUPRC std',
                   'RecK mean', 'RecK std', 'Time']:
        columns.append(dataset+'-'+metric)

results = pandas.DataFrame(columns=columns)
file_id = None
for model in models:
    model_result = {'name': model}
    for dataset_name in datasets:
        if model in ['CAREGNN', 'H2FD'] and 'hetero' not in dataset_name:
            continue
        time_cost = 0
        train_config = {
            'device': 'cuda',
            'epochs': args.epochs,
            'patience': args.patience,
            'metric': 'AUROC',
            'inductive': bool(args.inductive)
        }
        data = Dataset(dataset_name)
        model_config = {'model': model, 'lr': args.lr, 'drop_rate': 0,
                        'weight_decay': args.weight_decay,
                        'cn_top_m': args.cn_top_m,
                        'cn_semantic': args.cn_semantic,
                        'moe_top_k': args.moe_top_k,
                        'moe_noise_std': args.moe_noise,
                        'moe_min_temperature': args.moe_min_temp,
                        'moe_router_dropout': args.moe_router_dropout}
        if args.no_mcna:
            model_config['use_ncn'] = False
            model_config['use_multihop'] = False
            model_config['use_multihop_moe'] = False
            model_config['use_moe'] = False
        if dataset_name == 'tsocial':
            model_config['h_feats'] = 16
            # if model in ['GHRN', 'KNNGCN', 'AMNet', 'GT', 'GAT', 'GATv2', 'GATSep', 'PNA']:   # require more than 24G GPU memory
                # continue

        auc_list, pre_list, rec_list, f1_list = [], [], [], []
        for t in range(args.trials):
            torch.cuda.empty_cache()
            print("Dataset {}, Model {}, Trial {}".format(dataset_name, model, t))
            data.split(args.semi_supervised, t)
            seed = seed_list[t]
            set_seed(seed)
            train_config['seed'] = seed
            try:
                detector = model_detector_dict[model](train_config, model_config, data)
                st = time.time()
                # print(detector.model)
                test_score = detector.train()  # if no F1-score printed to stdout, check detector! the eval() in super class has been modified to return F1-score so in results.
            except torch.cuda.OutOfMemoryError:
                test_score = {'AUROC': 0, 'AUPRC': 0, 'RecK': 0, 'F1': 0}
                print(f"Out of memory error for {model} on {dataset_name} at trial {t}. OG traceback: \n{traceback.format_exc()}")
            auc_list.append(test_score['AUROC']), pre_list.append(test_score['AUPRC']), rec_list.append(test_score['RecK']), f1_list.append(test_score['F1'])
            ed = time.time()
            time_cost += ed - st
        del detector, data

        model_result[dataset_name+'-AUROC mean'] = np.mean(auc_list, where=np.array(auc_list) > 0)
        model_result[dataset_name+'-AUROC std'] = np.std(auc_list, where=np.array(auc_list) > 0)
        model_result[dataset_name+'-AUPRC mean'] = np.mean(pre_list, where=np.array(pre_list) > 0)
        model_result[dataset_name+'-AUPRC std'] = np.std(pre_list, where=np.array(pre_list) > 0)
        model_result[dataset_name+'-RecK mean'] = np.mean(rec_list, where=np.array(rec_list) > 0)
        model_result[dataset_name+'-RecK std'] = np.std(rec_list, where=np.array(rec_list) > 0)
        model_result[dataset_name+'-F1 mean'] = np.mean(f1_list, where=np.array(f1_list) > 0)
        model_result[dataset_name+'-F1 std'] = np.std(f1_list, where=np.array(f1_list) > 0)
        model_result[dataset_name+'-Time'] = time_cost/args.trials
    model_result = pandas.DataFrame(model_result, index=[0])
    results = pandas.concat([results, model_result])
    if better_result:
        file_id = better_save_results(results, file_id)
    else:
        file_id = save_results(results, file_id)
    print(results)
