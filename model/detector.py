from models.gnn import *
from models.attention import *
from models.spacegnn import SpaceGNNOriginal
from models.dgagnn import DGA
from models.arc_model import ARC, normalize_adj, sparse_mx_to_torch_sparse_tensor
from models.gadam import LocalModel, GlobalModel
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.cluster import KMeans
from dgl.nn.pytorch.factory import KNNGraph
import dgl
import numpy as np
import pandas as pd
import itertools
import psutil, os
from catboost import Pool, CatBoostClassifier, CatBoostRegressor, sum_models
from torch.utils.data import DataLoader
from typing import Iterable
import traceback
import time
import scipy.sparse as sp

class BaseDetector(object):
    def __init__(self, train_config, model_config, data):
        self.model_config = model_config
        self.train_config = train_config
        self.data = data
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        graph = self.data.graph.to(self.train_config['device'])
        self.labels = graph.ndata['label']
        self.train_mask = graph.ndata['train_mask'].bool()
        self.val_mask = graph.ndata['val_mask'].bool()
        self.test_mask = graph.ndata['test_mask'].bool()
        self.weight = (1 - self.labels[self.train_mask]).sum().item() / self.labels[self.train_mask].sum().item()
        self.source_graph = graph
        print(train_config['inductive'])
        if train_config['inductive'] == False:
            self.train_graph = graph
            self.val_graph = graph
        else:
            self.train_graph = graph.subgraph(self.train_mask)
            self.val_graph = graph.subgraph(self.train_mask+self.val_mask)
        self.best_score = -1
        self.patience_knt = 0
        
    def train(self):
        pass

    def eval(self, labels, probs):
        score = {}
        with torch.no_grad():
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            if torch.is_tensor(probs):
                probs = probs.cpu().numpy()
            score['AUROC'] = roc_auc_score(labels, probs)
            score['AUPRC'] = average_precision_score(labels, probs)
            score['F1'] = f1_score(labels, probs > 0.5)

            labels = np.array(labels)
            k = labels.sum()
        score['RecK'] = sum(labels[probs.argsort()[-k:]]) / sum(labels)
        return score


class BaseGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        gnn = globals()[model_config['model']]
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = gnn(**model_config).to(train_config['device'])

    def train(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model_config['lr'],
            weight_decay=self.model_config.get('weight_decay', 0.0),
        )
        train_labels, val_labels, test_labels = self.labels[self.train_mask], self.labels[self.val_mask], self.labels[self.test_mask]
        final_epoch = 0
        best_expert_stats = None
        for e in range(self.train_config['epochs']):

            self.model.train()
            logits = self.model(self.train_graph)
            main_loss = F.cross_entropy(logits[self.train_graph.ndata['train_mask']], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
                                   
            aux_loss_weight = self.model_config.get('aux_loss_weight', 1.0)
            aux_loss = self.model.get_aux_loss() if hasattr(self.model, 'get_aux_loss') else 0.0
            loss = main_loss + aux_loss_weight*aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # The following code is used to record the memory usage
            # py_process = psutil.Process(os.getpid())
            # print(f"CPU Memory Usage: {py_process.memory_info().rss / (1024 ** 3)} GB")
            # print(f"GPU Memory Usage: {torch.cuda.memory_reserved() / (1024 ** 3)} GB")
            if self.model_config['drop_rate'] > 0 or self.train_config['inductive']:
                self.model.eval()
                logits = self.model(self.val_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_graph.ndata['val_mask']])
            if val_score[self.train_config['metric']] > self.best_score:
                final_epoch = e
                if self.train_config['inductive']:
                    logits = self.model(self.source_graph)
                    probs = logits.softmax(1)[:, 1]
                best_expert_stats = self._get_expert_stats()
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'], val_score['F1'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK'], test_score['F1']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        # ========== 新增：训练结束后打印路由权重 ==========
        self._print_routing_weights(final_epoch, best_expert_stats)
        # ================================================
        return test_score
    def _get_expert_stats(self):
        """获取当前 MoE 统计（深拷贝）"""
        for name, module in self.model.named_modules():
            if hasattr(module, 'get_expert_stats'):
                return module.get_expert_stats().copy()  # 深拷贝
        return None
    
    def _print_routing_weights(self, epoch, stats=None):
        """打印MoE路由权重统计"""
        if stats is None:
            # 兼容旧逻辑：重新跑一次
            self.model.eval()
            with torch.no_grad():
                _ = self.model(self.source_graph if self.train_config['inductive'] else self.train_graph)
                stats = self._get_expert_stats()
        
        if stats is None:
            print("No MoE layer found.")
            return
            
        print(f"\n{'='*60}")
        print(f"Best Epoch {epoch} Routing Weights:")
        print(f"  1-hop weight: {stats.get('hop1_avg_weight', 0):.4f}")
        print(f"  2-hop weight: {stats.get('hop2_avg_weight', 0):.4f}")
        print(f"  3-hop weight: {stats.get('hop3_avg_weight', 0):.4f}")
        print(f"  GCN gate:     {stats.get('gcn_weight', 0):.4f}")
        print(f"  MoE gate:     {stats.get('moe_weight', 0):.4f}")
        print(f"{'='*60}\n")

# RGCN, HGT
class HeteroGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        hgnn = globals()[model_config['model']]
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        model_config['etypes'] = self.source_graph.canonical_etypes
        self.model = hgnn(**model_config).to(train_config['device'])
        
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], self.labels[self.val_mask], self.labels[self.test_mask]
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(self.train_graph)
            loss = F.cross_entropy(logits[self.train_graph.ndata['train_mask']], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0 or self.train_config['inductive']:
                self.model.eval()
                logits = self.model(self.val_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_graph.ndata['val_mask']])
            if val_score[self.train_config['metric']] > self.best_score:
                if self.train_config['inductive']:
                    logits = self.model(self.source_graph)
                    probs = logits.softmax(1)[:, 1]
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'], val_score['F1'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK'], test_score['F1']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class CAREGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = CAREGNN(**model_config).to(train_config['device'])

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]
        rl_idx = torch.nonzero(self.train_mask & self.labels, as_tuple=False).squeeze(1)
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(self.train_graph, e)
            loss = F.cross_entropy(logits[self.train_graph.ndata['train_mask']], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.model.RLModule(self.train_graph, e, rl_idx)
            if self.model_config['drop_rate'] > 0 or self.train_config['inductive']:
                self.model.eval()
                logits = self.model(self.val_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_graph.ndata['val_mask']])
            if val_score[self.train_config['metric']] > self.best_score:
                if self.train_config['inductive']:
                    logits = self.model(self.source_graph)
                    probs = logits.softmax(1)[:, 1]
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class NAGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = BWGNN(**model_config).to(train_config['device'])
        self.aggregate = dglnn.GINConv(None, activation=None, init_eps=0,
                                 aggregator_type='mean').to(self.train_config['device'])

    def train(self):
        k = 5 if 'k' not in self.model_config else self.model_config['k']
        dist = 'cosine' if 'dist' not in self.model_config else self.model_config['dist']
        feat = self.data.graph.ndata['feature'].to(self.train_config['device'])
        if k > 0:
            knn_graph = KNNGraph(k)
            knn_g = knn_graph(feat, algorithm="bruteforce-sharemem", dist=dist)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(self.source_graph)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            if k > 0:
                # neighbor smoothing
                probs = self.aggregate(knn_g, probs)
            
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class SVMDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        penalty = 'l2' if 'penalty' not in self.model_config else self.model_config['penalty']
        loss = 'squared_hinge' if 'loss' not in self.model_config else self.model_config['loss']
        C = 1 if 'C' not in self.model_config else self.model_config['C']
        self.model = svm.LinearSVC(penalty=penalty, loss=loss, C=C)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        self.model.fit(train_X, train_y)
        pred_val_y = self.model.decision_function(val_X)
        pred_y = self.model.decision_function(test_X)
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class KNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        k = 5 if 'k' not in self.model_config else self.model_config['k']
        weights = 'uniform' if 'weights' not in self.model_config else self.model_config['weights']
        p = 2 if 'p' not in self.model_config else self.model_config['p']
        self.model = KNeighborsClassifier(n_neighbors=k, weights=weights, p=p, n_jobs=32)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        self.model.fit(train_X, train_y)
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class XGBODDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        from pyod.models.xgbod import XGBOD
        super().__init__(train_config, model_config, data)
        self.model = XGBOD(n_jobs=32, **model_config)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        if self.train_mask.sum() > 100000: # avoid out of time
            train_X = train_X[:2000]
            train_y = train_y[:2000]
        print(train_X.shape, train_y.shape)
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        self.model.fit(train_X, train_y)
        pred_val_y = self.model.decision_function(val_X)
        pred_y = self.model.decision_function(test_X)
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class XGBoostDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        import xgboost as xgb
        # from xgboost import XGBClassifier
        eval_metric = roc_auc_score if train_config['metric'] == "AUROC" else average_precision_score
        self.model = xgb.XGBClassifier(tree_method='gpu_hist', eval_metric=eval_metric, **model_config)
        # self.model = XGBClassifier(tree_method='gpu_hist', eval_metric=eval_metric, **model_config)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)

        self.model.fit(train_X, train_y, sample_weight=weights, eval_set=[(val_X, val_y)], verbose=False)
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class XGBNADetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        import xgboost as xgb
        # from xgboost import XGBClassifier
        eval_metric = roc_auc_score if train_config['metric'] == "AUROC" else average_precision_score
        self.model = xgb.XGBClassifier(tree_method='gpu_hist', eval_metric=eval_metric, **model_config)
        # self.model =XGBClassifier(tree_method='gpu_hist', eval_metric=eval_metric, **model_config)
        self.aggregate = dglnn.GINConv(None, activation=None, init_eps=0,
                                 aggregator_type='mean').to(self.train_config['device'])

    def train(self):
        k = 5 if 'k' not in self.model_config else self.model_config['k']
        dist = 'cosine' if 'dist' not in self.model_config else self.model_config['dist']
        feat = self.data.graph.ndata['feature'].to(self.train_config['device'])
        if k > 0:
            knn_graph = KNNGraph(k)
            knn_g = knn_graph(feat, algorithm="bruteforce-sharemem", dist=dist)

        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)

        self.model.fit(train_X, train_y, sample_weight=weights, eval_set=[(val_X, val_y)], verbose=False)
        X = self.source_graph.ndata['feature'].cpu().numpy()
        probs = torch.tensor(self.model.predict_proba(X)[:, 1]).cuda()
        if k > 0:
            probs = self.aggregate(knn_g, probs)
        pred_val_y = probs[self.val_mask]
        pred_y = probs[self.test_mask]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class XGBGraphDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        import xgboost as xgb
        # from xgboost import XGBClassifier
        eval_metric = roc_auc_score if train_config['metric'] == "AUROC" else average_precision_score
        self.model = xgb.XGBClassifier(tree_method='gpu_hist', eval_metric=eval_metric, verbose=2, **model_config)
        # self.model = XGBClassifier(tree_method='gpu_hist', eval_metric=eval_metric, verbose=2, **model_config)
        gnn = GIN_noparam(**model_config).to(self.source_graph.device)
        new_feat = gnn(self.source_graph)
        if self.train_config['inductive'] == True:
            new_feat[self.train_mask] = gnn(self.source_graph.subgraph(self.train_mask))
            val_graph = self.source_graph.subgraph(self.train_mask+self.val_mask)
            new_feat[self.val_mask] = gnn(val_graph)[val_graph.ndata['val_mask']]
        ##
        self.source_graph.ndata['feature'] = new_feat.detach()

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)

        self.model.fit(train_X, train_y, sample_weight=weights, eval_set=[(val_X, val_y)])  # early_stopping_rounds =20
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class RFDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        n_estimators = 100 if 'n_estimators' not in model_config else model_config['n_estimators']
        criterion = 'gini' if 'criterion' not in model_config else model_config['criterion']
        max_samples = None if 'max_samples' not in model_config else model_config['max_samples']
        max_features = 'sqrt' if 'max_features' not in model_config else model_config['max_features']
        self.model = RandomForestClassifier(n_jobs=32, n_estimators=n_estimators, criterion=criterion,
                                            max_samples=max_samples, max_features=max_features)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)
        self.model.fit(train_X, train_y, sample_weight=weights)
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class RFGraphDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        n_estimators = 100 if 'n_estimators' not in model_config else model_config['n_estimators']
        criterion = 'gini' if 'criterion' not in model_config else model_config['criterion']
        max_samples = None if 'max_samples' not in model_config else model_config['max_samples']
        max_features = 'sqrt' if 'max_features' not in model_config else model_config['max_features']
        self.model = RandomForestClassifier(n_jobs=32, n_estimators=n_estimators, criterion=criterion,
                                            max_samples=max_samples, max_features=max_features)
        gnn = GIN_noparam(**model_config).to(self.source_graph.device)
        new_feat = gnn(self.source_graph)
        if self.train_config['inductive'] == True:
            new_feat[self.train_mask] = gnn(self.source_graph.subgraph(self.train_mask))
            val_graph = self.source_graph.subgraph(self.train_mask+self.val_mask)
            new_feat[self.val_mask] = gnn(val_graph)[val_graph.ndata['val_mask']]
        self.source_graph.ndata['feature'] = new_feat

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)
        self.model.fit(train_X, train_y, sample_weight=weights)
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class GASDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        model_config['mlp_layers'] = 0
        self.model = GCN(**model_config).to(train_config['device'])
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.model.h_feats * 2, self.model.h_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(self.model.h_feats, 2)).to(train_config['device'])

    def train(self):
        k = 5 if 'k' not in self.model_config else self.model_config['k']
        dist = 'cosine' if 'dist' not in self.model_config else self.model_config['dist']
        feat = self.data.graph.ndata['feature'].to(self.train_config['device'])
        knn_graph = KNNGraph(k)
        knn_g = knn_graph(feat, algorithm="bruteforce-sharemem", dist=dist)
        knn_g.ndata["feature"] = feat

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]

        for e in range(self.train_config['epochs']):
            self.model.train()
            h_origin = self.model(self.source_graph)
            h_knn = self.model(knn_g)
            h_all = torch.cat([h_origin, h_knn], -1)
            logits = self.mlp(h_all)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class KNNGCNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = GCN(**model_config).to(train_config['device'])

    def train(self):
        k = 5 if 'k' not in self.model_config else self.model_config['k']
        dist = 'cosine' if 'dist' not in self.model_config else self.model_config['dist']
        feat = self.data.graph.ndata['feature'].to(self.train_config['device'])
        knn_graph = KNNGraph(k)
        knn_g = knn_graph(feat, algorithm="bruteforce-sharemem", dist=dist)
        new_g = dgl.merge([knn_g, self.source_graph])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]

        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(new_g)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model(self.source_graph)
            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class GHRNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = BWGNN(**model_config).to(train_config['device'])

    def random_walk_update(self, delete_ratio):
        graph = self.source_graph
        edge_weight = torch.ones(graph.num_edges()).to(self.train_config['device'])
        norm = dgl.nn.pytorch.conv.EdgeWeightNorm(norm='both')
        graph.edata['w'] = norm(graph, edge_weight)
        aggregate_fn = fn.u_mul_e('h', 'w', 'm')
        reduce_fn = fn.sum(msg='m', out='ay')

        graph.ndata['h'] = graph.ndata['feature']
        graph.update_all(aggregate_fn, reduce_fn)
        graph.ndata['ly'] = graph.ndata['feature'] - graph.ndata['ay']
        graph.apply_edges(self.inner_product_black)
        black = graph.edata['inner_black']
        threshold = int(delete_ratio * graph.num_edges())
        edge_to_move = set(black.sort()[1][:threshold].tolist())
        graph_new = dgl.remove_edges(graph, list(edge_to_move))
        return graph_new

    def inner_product_black(self, edges):
        inner_black = (edges.src['ly'] * edges.dst['ly']).sum(axis=1)
        return {'inner_black': inner_black}

    def train(self):
        del_ratio = 0.015 if 'del_ratio' not in self.model_config else self.model_config['del_ratio']
        if del_ratio != 0.:
            graph = self.random_walk_update(del_ratio)
            graph = dgl.add_self_loop(dgl.remove_self_loop(graph))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(graph)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class PCGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = ChebNet(**model_config).to(train_config['device'])

    def process(self, del_ratio=0.7, add_ratio=0.3, k_max=5, dist='cosine', **kwargs):
        graph = self.source_graph.long()
        features = graph.ndata['feature']
        edges = graph.adj().coalesce()
        edges_num = edges.indices().shape[1]
        dd = torch.zeros([edges_num], device=features.device)
        step = 5000000
        idx = 0
        while idx < edges_num:  # avoid OOM on large datasets
            st = idx
            idx += step
            ed = idx if idx < edges_num else edges_num
            f1 = features[edges.indices()[0, st:ed]]
            f2 = features[edges.indices()[1, st:ed]]
            dd[st:ed] = (f1 - f2).norm(1, dim=1).detach().clone()

        # the choose step: remove edges
        selected_edges = (dd).topk(int(edges_num * del_ratio)).indices.long()
        graph = dgl.remove_edges(graph, selected_edges)
        selected_nodes = (graph.ndata['label'] == 1) & (graph.ndata['train_mask'] == 1)

        # the choose step: add edges
        g_id = selected_nodes.nonzero().squeeze(-1)
        ave_degree = graph.in_degrees(g_id).float().mean() * add_ratio
        k = min(int(ave_degree), k_max) + 1

        knn_g = dgl.knn_graph(graph.ndata['feature'][selected_nodes], algorithm="bruteforce-sharemem",
                              k=k, dist=dist)
        u, v = g_id[knn_g.edges()[0]], g_id[knn_g.edges()[1]]
        graph = dgl.add_edges(graph, u.long(), v.long())
        return graph

    def train(self):
        graph = self.process(**self.model_config)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(graph)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device), reduce=False)
            degree_weight = graph.in_degrees()
            loss = (loss * degree_weight[self.train_mask]).mean() / degree_weight.max()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class DCIDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        gnn = globals()[model_config['model']]
        print(gnn)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = gnn(**model_config).to(train_config['device'])
        self.num_cluster = 2 if 'num_cluster' not in model_config else model_config['num_cluster']
        self.pretrain_epochs = 100 if 'pretrain_epochs' not in model_config else model_config['pretrain_epochs']

        self.kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(self.data.graph.ndata['feature'])
        self.ss_label = self.kmeans.labels_
        self.cluster_info = [list(np.where(self.ss_label == i)[0]) for i in range(self.num_cluster)]

    def train(self):
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for e in range(1, self.pretrain_epochs):
            self.model.train()
            loss = self.model(self.source_graph, self.cluster_info, self.num_cluster)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                print(loss)
                optimizer.step()
            # re-clustering
            if e % 20 == 0:
                self.model.eval()
                emb = self.model.get_emb(self.source_graph)
                kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(emb.detach().cpu().numpy())
                ss_label = kmeans.labels_
                self.cluster_info = [list(np.where(ss_label == i)[0]) for i in range(self.num_cluster)]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model.encoder(self.source_graph, use_mlp=True)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model.encoder(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])

            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print(
                    'Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                        loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                        test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score
    

class BGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        # gnn = globals()[model_config['model']]
        self.depth = 6 if 'depth' not in model_config else model_config['depth']
        self.iter_per_epoch = 10 if 'iter_per_epoch' not in model_config else model_config['iter_per_epoch']
        self.gbdt_alpha = 1 if 'gbdt_alpha' not in model_config else model_config['gbdt_alpha']
        self.gbdt_lr = 0.1 if 'gbdt_lr' not in model_config else model_config['gbdt_lr']
        self.train_non_gbdt = False if 'train_non_gbdt' not in model_config else model_config['train_non_gbdt']
        self.only_gbdt = False if 'only_gdbt' not in model_config else model_config['only_gdbt']
        self.normalize_features = False if 'nomarlize_features' not in model_config else model_config['normalize_features']

        if not self.only_gbdt:
            model_config['in_feats'] = self.source_graph.ndata['feature'].shape[1] + self.labels.unique().shape[0]
        else:
            model_config['in_feats'] = self.labels.unique().size(0)

        self.model = GCN(**model_config).to(train_config['device'])
        self.gbdt_model = None
    
    def preprocess(self):
        gbdt_X_train = pd.DataFrame(self.source_graph.ndata['feature'][self.train_mask].cpu().numpy())
        gbdt_y_train = pd.DataFrame(self.labels[self.train_mask].cpu().numpy()).astype(float)

        raw_X = pd.DataFrame(self.source_graph.ndata['feature'].clone().cpu().numpy())
        encoded_X = self.source_graph.ndata['feature'].clone()
        if not self.only_gbdt and self.normalize_features:
            min_vals, _ = torch.min(encoded_X[self.train_mask], dim=0, keepdim=True)
            max_vals, _ = torch.max(encoded_X[self.train_mask], dim=0, keepdim=True)
            encoded_X[self.train_mask] = (encoded_X[self.train_mask] - min_vals) / (max_vals - min_vals)
            encoded_X[self.val_mask | self.test_mask] = (encoded_X[self.val_mask | self.test_mask] - min_vals) / (max_vals - min_vals)
            if encoded_X.isnan().any():
                row, col = torch.where(encoded_X.isnan())
                encoded_X[row, col] = self.source_graph.ndata['feature'][row, col]
            if encoded_X.isinf().any():
                row, col = torch.where(encoded_X.isinf())
                encoded_X[row, col] = self.source_graph.ndata['feature'][row, col]

        node_features = torch.empty(encoded_X.shape[0], self.model_config['in_feats'], requires_grad=True, device=self.labels.device)
        if not self.only_gbdt:
            node_features.data[:, :-2] = self.source_graph.ndata['feature'].clone()
        self.source_graph.ndata['feature'] = node_features
        return gbdt_X_train, gbdt_y_train, raw_X, encoded_X

    def train_gbdt(self, gbdt_X_train, gbdt_y_train, epoch):
        pool = Pool(gbdt_X_train, gbdt_y_train)
        if epoch == 0:
            catboost_model_obj = CatBoostClassifier
            catboost_loss_fn = 'MultiClass'
        else:
            catboost_model_obj = CatBoostRegressor
            catboost_loss_fn = 'MultiRMSE'
        
        epoch_gbdt_model = catboost_model_obj(iterations=self.iter_per_epoch,
                                              depth=self.depth,
                                              learning_rate=self.gbdt_lr,
                                              loss_function=catboost_loss_fn,
                                              random_seed=0,
                                              nan_mode='Min')
        epoch_gbdt_model.fit(pool, verbose=False)
        
        if epoch == 0:
            self.base_gbdt = epoch_gbdt_model
        else:
            if self.gbdt_model is None:
                self.gbdt_model = epoch_gbdt_model
            else:
                self.gbdt_model = sum_models([self.gbdt_model, epoch_gbdt_model], weights=[1, self.gbdt_alpha])
                # self.gbdt_model = self.append_gbdt_model(epoch_gbdt_model, weights=[1, self.gbdt_alpha])

    def update_node_features(self, X, encoded_X):
        predictions = self.base_gbdt.predict_proba(X)
        # predictions = self.base_gbdt.predict(X, prediction_type='RawFormulaVal')
        if self.gbdt_model is not None:
            predictions_after_one = self.gbdt_model.predict(X)
            predictions += predictions_after_one

        predictions = torch.tensor(predictions, device=self.labels.device)
        node_features = self.source_graph.ndata['feature']
        if not self.only_gbdt:
            if self.train_non_gbdt:
                predictions = torch.concat((node_features.detach().data[:, :-2], predictions), dim=1)
            else:
                predictions = torch.concat((encoded_X, predictions), dim=1)
        node_features.data = predictions.float().data

    def train(self):
        gbdt_X_train, gbdt_y_train, raw_X, encoded_X = self.preprocess()
        optimizer = torch.optim.Adam(
            itertools.chain(*[self.model.parameters(), [self.source_graph.ndata['feature']]]), lr=self.model_config['lr']
        )
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]

        for e in range(self.train_config['epochs']):
            self.train_gbdt(gbdt_X_train, gbdt_y_train, e)
            self.update_node_features(raw_X, encoded_X)
            node_features_before = self.source_graph.ndata['feature'].clone()
            
            self.model.train()
            for _ in range(self.iter_per_epoch):
                logits = self.model(self.source_graph)
                loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            logits = self.model(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
            
            # Update GBDT target
            gbdt_y_train = (self.source_graph.ndata['feature'] - node_features_before)[self.train_mask, -2:].detach().cpu().numpy()
            
            # Check if update is frozen
            if np.isclose(gbdt_y_train.sum(), 0.):
                print('Nodes do not change anymore. Stopping...')
                break
        return test_score


class H2FDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]

        g = self.source_graph
        canon = g.canonical_etypes
        new_g = dgl.heterograph({
            canon[0]: g[canon[0]].edges(),
            canon[1]: g[canon[1]].edges(),
            canon[2]: g[canon[2]].edges(),
            (canon[0][0], 'homo', canon[0][0]): dgl.to_homogeneous(g).edges()
            })
        homo_edges = new_g.edges(etype='homo')
        for feat in g.ndata:
            new_g.ndata[feat] = g.ndata[feat].clone()
        
        homo_labels, homo_train_mask = self.generate_edges_labels(homo_edges, new_g.ndata['label'].cpu().tolist(), new_g.ndata['train_mask'].nonzero().squeeze(1).tolist())

        new_g.edges['homo'].data['label'] = homo_labels.cuda()
        new_g.edges['homo'].data['train_mask'] = homo_train_mask.cuda()
        for ntype in g.ntypes:
            for key in g.ndata.keys():
                new_g.nodes[ntype].data[key] = g.nodes[ntype].data[key].clone()
        # dgl.save_graphs('new_amazon', [new_g])
        # new_g = dgl.load_graphs('new_amazon')[0][0].to(train_config['device'])
        self.source_graph = new_g
        model_config['graph'] = self.source_graph
        self.model = H2FD(**model_config).to(train_config['device'])

    def generate_edges_labels(self, edges, labels, train_idx):
        row, col = edges[0].cpu(), edges[1].cpu()
        edge_labels = []
        edge_train_mask = []
        for i, j in zip(row, col):
            i = i.item()
            j = j.item()
            if labels[i] == labels[j]:
                edge_labels.append(1)
            else:
                edge_labels.append(-1)
            if i in train_idx and j in train_idx:
                edge_train_mask.append(1)
            else:
                edge_train_mask.append(0)
        edge_labels = torch.Tensor(edge_labels).long()
        edge_train_mask = torch.Tensor(edge_train_mask).bool()
        return edge_labels, edge_train_mask

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]
        for e in range(self.train_config['epochs']):
            self.model.train()
            loss, logits = self.model(self.source_graph)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_graph.ndata['val_mask']])
            if val_score[self.train_config['metric']] > self.best_score:
                if self.train_config['inductive']:
                    loss, logits = self.model(self.source_graph)
                    probs = logits.softmax(1)[:, 1]
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


def infinite_iter(data_list: Iterable):
    it = iter(data_list)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(data_list)


class GAGADetector(BaseDetector):
    def __init__(self, train_config, model_config, data, cache=True, cache_dir="./gaga_%s.npz"):
        super().__init__(train_config, model_config, data)
        gnn = GAGA
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        model_config['num_edge_types'] = len(data.graph.etypes)
        device = train_config['device']
        self.model = gnn(**model_config).to(device)

        dataset_name = data.name
        etypes = data.graph.etypes
        cache_name = cache_dir % (dataset_name.replace("/", "_"))

        if not cache or not os.path.exists(cache_name):  # FIXME random mask may cause recall=0
            self.masked_train_label = torch.clone(self.labels)  # label 0 = benign, 1 = fraud, 2 = masked
            self.masked_train_label[~self.train_mask.bool()] = 2
            while True:  # Loop until each type of label will be sampled
                mask_rate = 0.8
                random_mask = torch.rand(self.train_mask.sum()) < mask_rate
                label_count = (torch.sum(self.masked_train_label[self.train_mask][~random_mask] == 0).item(), torch.sum(self.masked_train_label[self.train_mask][~random_mask] == 1).item())
                label_balance = 1 - (abs(label_count[0] - label_count[1]) / (label_count[0] + label_count[1] + 1e-8))
                dataset_count = (torch.sum(self.masked_train_label[self.train_mask] == 0).item(), torch.sum(self.masked_train_label[self.train_mask] == 1).item())
                dataset_balance = 1 - (abs(dataset_count[0] - dataset_count[1]) / (dataset_count[0] + dataset_count[1] + 1e-8))
                if label_count[0] != 0 and label_count[1] != 0 and label_balance > dataset_balance:  # Ensure both labels are present and balanced
                    break
            print("Label balance: ", label_balance, "Dataset balance:", dataset_balance, "Label count: ", label_count)
            self.masked_train_label[self.train_mask][random_mask] = 2

            self.sampled_train_feature = self.model.pre_feature_sample(self.train_graph, self.masked_train_label)
            self.sampled_val_feature = self.model.pre_feature_sample(self.val_graph, self.masked_train_label)
            self.sampled_test_feature = self.model.pre_feature_sample(self.source_graph, self.masked_train_label)
        else:  # cache == True and cache file exists
            data = np.load(cache_name)
            self.masked_train_label = torch.from_numpy(data['train_label']).to(device)

            train_feature = torch.from_numpy(data['train_feature']).to(device)
            val_feature = torch.from_numpy(data['val_feature']).to(device)
            test_feature = torch.from_numpy(data['test_feature']).to(device)

            self.sampled_train_feature = {etypes[i]: train_feature[i] for i in range(len(etypes))}
            self.sampled_val_feature = {etypes[i]: val_feature[i] for i in range(len(etypes))}
            self.sampled_test_feature = {etypes[i]: test_feature[i] for i in range(len(etypes))}

            # if 'train_mask' in data and 'val_mask' in data and 'test_mask' in data:
            #     self.train_mask = torch.from_numpy(data['train_mask'])
            #     self.val_mask = torch.from_numpy(data['val_mask'])
            #     self.test_mask = torch.from_numpy(data['test_mask'])

        if cache and not os.path.exists(cache_name):
            np.savez(cache_name,
                     train_label=self.masked_train_label.detach().cpu().numpy(),
                     train_feature=torch.stack([self.sampled_train_feature[k] for k in etypes], dim=0).detach().cpu().numpy(),
                     val_feature=torch.stack([self.sampled_val_feature[k] for k in etypes], dim=0).detach().cpu().numpy(),
                     test_feature=torch.stack([self.sampled_val_feature[k] for k in etypes], dim=0).detach().cpu().numpy()
                     )

        batch_size = model_config.get('batch_size', 256)
        self.weight_decay = model_config.get('weight_decay', 1e-4)

        data_length = list(self.sampled_train_feature.items())[0][1].shape[0]
        self.train_loader = infinite_iter(DataLoader(torch.arange(data_length)[self.train_mask.cpu()], batch_size = batch_size, shuffle=True, drop_last=False, num_workers=8))
        self.val_loader = infinite_iter(DataLoader(torch.arange(data_length)[self.val_mask.cpu()], batch_size = batch_size, shuffle=True, drop_last=False, num_workers=8))
        self.test_loader = infinite_iter(DataLoader(torch.arange(data_length)[self.test_mask.cpu()], batch_size = batch_size, shuffle=True, drop_last=False, num_workers=8))

    def get_data(self, type='train'):
        assert type in ['train', 'val', 'test']
        loader = {
            'train': self.train_loader,
            'val': self.val_loader,
            'test': self.test_loader,
        }[type]
        features = {
            'train': self.sampled_train_feature,
            'val': self.sampled_val_feature,
            'test': self.sampled_test_feature,
        }[type]

        data_idx = next(loader)
        res = dict()
        for k in features.keys():
            res[k] = features[k][data_idx]
        return res, data_idx

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'], weight_decay=self.weight_decay)
        train_labels, val_labels, test_labels = self.labels[self.train_mask], self.labels[self.val_mask], self.labels[self.test_mask]

        for e in range(self.train_config['epochs']):
            self.model.train()
            data, idx = self.get_data('train')
            logits = self.model(self.train_graph, data)
            loss = F.cross_entropy(logits, self.masked_train_label[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0 or self.train_config['inductive']:
                self.model.eval()
                data, idx = self.get_data('val')
                logits = self.model(self.val_graph, data)
            probs = logits[:, 0]
            try:
                val_score = self.eval(self.labels[idx], probs)
            except ValueError:
                print("ValueError: ", traceback.format_exc())
            if val_score[self.train_config['metric']] > self.best_score:
                if self.train_config['inductive']:
                    data, idx = self.get_data('test')
                    logits = self.model(self.source_graph, data)
                    probs = logits.softmax(1)[:, 1]
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(self.labels[idx], probs)
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score

class ConsisGADDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]

        model_args = {
            'to-homo': False,
            'shuffle-train': True,
            'hidden-dim': 64,
            'num-layers': 1,
            'weight-decay': 0.00001,
            'training-ratio': 1,
            'train-procedure': 'CT',
            'mlp-drop': 0.4,
            'input-drop': 0.0,
            'hidden-drop': 0.0,
            'mlp12-dim': 128,
            'mlp3-dim': 128,
            'bn-type': 2,
            'optim': 'adam',
            'store-model': True,
            'trainable-consis-weight': 1.5,
            'trainable-temp': 0.0001,
            'trainable-eps': 0.000000000001,
            'trainable-drop-rate': 0.2,
            'trainable-warm-up': -1,
            'trainable-model': 'mlp',
            'trainable-optim': 'adam',
            'trainable-lr': 0.005,
            'trainable-weight-decay': 0.00001,
            'topk-mode': 4,
            'diversity-type': 'cos',
            'unlabel-ratio': 4,
            'normal-th': 7,
            'fraud-th': 88,
            'trainable-detach-y': True,
            'trainable-div-eps': True,
            'trainable-detach-mask': False,
            'batch-size': 128,
            'train-iterations': 10
        }

        model_args['device'] = 'cuda:0'
        model_args.update(model_config)

        print("[INFO]", "current model args =", model_args)

        self.train_nids = torch.nonzero(self.data.graph.ndata['train_mask']).squeeze()
        self.valid_nids = torch.nonzero(self.data.graph.ndata['val_mask']).squeeze()
        self.test_nids = torch.nonzero(self.data.graph.ndata['test_mask']).squeeze()

        self.labeled_nids = self.train_nids
        self.unlabeled_nids = torch.concatenate([self.train_nids, self.valid_nids, self.test_nids])

        power = 10 if data.name == 'tfinance' else 16
        self.valid_loader = DataLoader(self.valid_nids, batch_size = 2 ** power, shuffle=False, drop_last=False, num_workers=4)
        self.test_loader = DataLoader(self.test_nids, batch_size = 2 ** power, shuffle=False, drop_last=False, num_workers=4)
        self.labeled_loader = DataLoader(self.labeled_nids, batch_size = model_args['batch-size'], shuffle=model_args['shuffle-train'], drop_last=True, num_workers=0)
        self.unlabeled_loader = DataLoader(self.labeled_nids, batch_size = model_args['batch-size'] * model_args['unlabel-ratio'], shuffle=model_args['shuffle-train'], drop_last=True, num_workers=0)

        self.model = ConsisGADGNN(
            model_config['in_feats'],
            64,
            2,
            self.source_graph.etypes,
            128,
            128,
            model_args['input-drop'],
            model_args['hidden-drop'],
            model_args['mlp-drop'],
            model_args['num-layers']
        ).to(train_config['device'])

        if model_args['optim'] == 'rmpprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=model_args['lr'])
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_args['lr'], weight_decay=0.0)
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model_args['num-layers'])
        self.argumentator = LearnableDataArugmentation(
            self.model,
            drop_rate=model_args['trainable-drop-rate'],
            lr=model_args['trainable-lr'],
            eps=model_args['trainable-eps'],
            temp=model_args['trainable-temp'],
            weight_decay=model_args['trainable-weight-decay'],
        )

        self.task_loss = ConsisCombinedLoss()

        self.args = model_args
    
    @staticmethod
    def sample_blocks(graph: dgl.DGLGraph, seed_nodes: torch.Tensor, sampler: dgl.dataloading.MultiLayerFullNeighborSampler):
        seed_nodes = seed_nodes.to(graph.idtype)
        input_nodes, output_nodes, blocks = sampler.sample_blocks(graph, seed_nodes)
        return input_nodes, output_nodes, blocks

    @staticmethod
    def UDA_train_epoch(epoch, model, arugmentator, graph, loader, optimizer, sampler, task_loss, args):
        model.train()
        num_iters = args['train-iterations']
        device = "cuda:0"
        
        label_loader, unlabel_loader = loader
        unlabel_loader_iter = infinite_iter(unlabel_loader)
        label_loader_iter = infinite_iter(label_loader)

        for idx in range(num_iters):
            label_idx = next(label_loader_iter)
            unlabel_idx = next(unlabel_loader_iter)
            assert label_idx is not None and unlabel_idx is not None

            if epoch > args['trainable-warm-up']:
                _, u, u_blocks = ConsisGADDetector.sample_blocks(graph, unlabel_idx.to(device), sampler)
                p_h_u, y_w_u = arugmentator(u_blocks)
            else:
                p_h_u, y_w_u = torch.tensor(1.0, requires_grad=False), torch.tensor(1.0, requires_grad=False)

            _, _, s_blocks = ConsisGADDetector.sample_blocks(graph, label_idx.to(device), sampler)
            p_v = model(s_blocks)
            y_v = s_blocks[-1].dstdata['label']

            loss = task_loss(p_v, y_v, y_w_u, p_h_u) + args['weight-decay'] * l2_regularization(model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return p_v
    
    @staticmethod
    def get_model_pred(model, graph, data_loader, sampler, args):
        model.eval()

        pred_list = []
        target_list = []
        with torch.no_grad():
            for index, node_index in enumerate(data_loader):  # 1/4s pre 100 samples
                _, _, blocks = sampler.sample_blocks(graph, node_index.to(args['device']).to(graph.idtype))
                pred = model(blocks)
                target = blocks[-1].dstdata['label']

                pred.detach()
                pred[pred.isnan()] = 0

                pred_list.append(pred)
                target_list.append(target.detach())

            pred_list = torch.cat(pred_list, dim=0)
            target_list = torch.cat(target_list, dim=0)
            pred_list = pred_list.exp()[:, 1]
        
        return pred_list, target_list 

    def train(self):
        # train_labels, val_labels, test_labels = self.labels[self.train_mask], \
        #                                         self.labels[self.val_mask], self.labels[self.test_mask]
        for epoch in range(self.train_config['epochs']):
            # print("[INFO]", f"current epoch={epoch}")
            self.UDA_train_epoch(
                epoch, self.model, self.argumentator, self.source_graph,
                (self.labeled_loader, self.unlabeled_loader),
                self.optimizer, self.sampler, self.task_loss, self.args
            )
            
            val_probs, val_labels = self.get_model_pred(self.model, self.source_graph, self.valid_loader, self.sampler, self.args)
            test_probs, test_labels = self.get_model_pred(self.model, self.source_graph, self.test_loader, self.sampler, self.args)

            val_results, test_results = \
                self.eval(val_labels, val_probs), self.eval(test_labels, test_probs)

            # print("[DEBUG]", "current evaluate result =", val_results, test_results)
            
            if val_results[self.train_config['metric']] > self.best_score:
                self.best_score = val_results[self.train_config['metric']]
                test_in_best_val = test_results
                print("[RES]", test_results, "@", epoch)

                if self.args['store-model']:
                    ...  # TODO

        return test_in_best_val

# class SpaceGNNDetector(BaseDetector):
#     def __init__(self, train_config, model_config, data):
#         super().__init__(train_config, model_config, data)
        
#         # SpaceGNN specific parameters
#         self.alpha = model_config.get('alpha', 0.5)
#         self.beta = model_config.get('beta', 0.5)
#         self.layer_num = model_config.get('layer_num', 2)
#         self.batch_size = model_config.get('batch_size', 512)  # Restore original batch_size for better stability
#         # Timing/verbose switch
#         self.debug_timing = model_config.get('debug_timing', True)
        
#         # Initialize curvature parameters
#         stdneg = model_config.get('stdneg', 0.1)
#         stdpos = model_config.get('stdpos', 0.1)
#         cneg = torch.FloatTensor(self.layer_num).normal_(-0.1, stdneg)
#         cpos = torch.FloatTensor(self.layer_num).normal_(0.1, stdpos)
        
#         # Precompute multi-layer features like original SpaceGNN (on complete graph)
#         self._precompute_multilayer_features()
        
#         # Initialize SpaceGNN model
#         model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
#         self.model = SpaceGNNOriginal(
#             in_dim=model_config['in_feats'],
#             hid_dim=model_config.get('h_feats', 32),
#             out_dim=2,
#             layer_num=self.layer_num,
#             drop_rate=model_config.get('drop_rate', 0),
#             cneg=cneg,
#             cpos=cpos
#         ).to(train_config['device'])
        
#         # Initialize sampler for batch training
#         self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)


#         self.moe_layer = SparseMultiHopMoE(
#             gcn_layer=nn.Identity(),
#             in_feats=model_config['in_feats'],
#             out_feats=model_config['in_feats'],
#             num_hops=self.num_hops,
#             top_k=1,
#         ).to(train_config['device'])
        
#         # 预计算
#         feat = self.source_graph.ndata['feature_0'].to(train_config['device'])
#         self.moe_layer.precompute_all(self.source_graph, feat)

    
#     def _precompute_multilayer_features(self):
#         """Precompute multi-layer features like original SpaceGNN on complete graph"""
#         t0 = time.perf_counter()
#         graph = self.source_graph
#         graph.ndata['feature_0'] = graph.ndata['feature'].float()
#         h = graph.ndata['feature_0']
        
#         # Precompute all layers at once on complete graph (like original SpaceGNN)
#         for i in range(1, self.layer_num):
#             # Propagate features using mean aggregation on complete graph
#             graph.ndata['h'] = h
#             graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
#             h = graph.ndata.pop('h')
#             graph.ndata[f'feature_{i}'] = h
        
#         # 性能优化: 预先将所有特征移到GPU
#         device = self.train_config['device']
#         print(f"🖥️  Using device: {device}")
#         print(f"🔧 CUDA available: {torch.cuda.is_available()}")
#         if torch.cuda.is_available():
#             print(f"🎮 CUDA device count: {torch.cuda.device_count()}")
#             print(f"🎯 Current CUDA device: {torch.cuda.current_device()}")
        
#         for i in range(self.layer_num):
#             feature_key = f'feature_{i}'
#             graph.ndata[feature_key] = graph.ndata[feature_key].to(device)
#             print(f"📊 Feature {feature_key} moved to {graph.ndata[feature_key].device}")
        
#         t1 = time.perf_counter()
#         print(f"✅ Precomputed {self.layer_num} layers of features on complete graph and moved to {device} in {(t1-t0):.3f}s")
    
#     def train(self):
#         # Create DGL DataLoaders: parallel sampling and GPU blocks
#         build_t0 = time.perf_counter()

#         def _mask_to_indices(mask):
#             """Convert boolean mask to graph idtype to satisfy DGL sampler."""
#             idx = mask.nonzero(as_tuple=False).view(-1)
#             return idx.to(self.source_graph.idtype)

#         train_indices = _mask_to_indices(self.train_mask)
#         val_indices = _mask_to_indices(self.val_mask)
#         test_indices = _mask_to_indices(self.test_mask)

#         device = self.train_config['device']
#         # If graph/indices are on CUDA, DGL requires num_workers == 0
#         # Otherwise, allow parallel workers on CPU-side sampling
#         if device != 'cpu':
#             num_workers = 0
#         else:
#             num_workers = self.model_config.get('num_workers', 4)

#         train_loader = dgl.dataloading.DataLoader(
#             self.source_graph,
#             train_indices,
#             self.sampler,
#             batch_size=self.batch_size,
#             shuffle=True,
#             drop_last=True,
#             num_workers=num_workers,
#             device=device,
#         )
#         val_loader = dgl.dataloading.DataLoader(
#             self.source_graph,
#             val_indices,
#             self.sampler,
#             batch_size=self.batch_size,
#             shuffle=False,
#             drop_last=False,
#             num_workers=num_workers,
#             device=device,
#         )
#         test_loader = dgl.dataloading.DataLoader(
#             self.source_graph,
#             test_indices,
#             self.sampler,
#             batch_size=10000,
#             shuffle=False,
#             drop_last=False,
#             num_workers=num_workers,
#             device=device,
#         )
#         build_t1 = time.perf_counter()
#         print(f"🧱 Built DataLoaders in {(build_t1-build_t0):.3f}s (num_workers={num_workers}, device={device})")

#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
#         train_labels, val_labels, test_labels = self.labels[self.train_mask], self.labels[self.val_mask], self.labels[self.test_mask]

#         # 设备信息
#         print(f"🚀 Starting training on device: {device}")
#         print(f"📈 Model device: {next(self.model.parameters()).device}")
#         print(f"🎯 Source graph device: {self.source_graph.ndata['feature'].device}")
#         if hasattr(torch.cuda, 'memory_allocated') and device != 'cpu':
#             print(f"💾 GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
#             print(f"💽 GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")

#         for e in range(self.train_config['epochs']):
#             ep_t0 = time.perf_counter()
#             # print(f"[Epoch {e}] ▶️ start")
#             self.model.train()
#             last_yield_end = time.perf_counter()
#             for bidx, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
#                 t_batch_get = time.perf_counter()
#                 t_wait = t_batch_get - last_yield_end
#                 # Attach precomputed features to the GPU blocks (optimized)
#                 t_attach0 = time.perf_counter()
#                 # 优化1: 预先转换input_nodes类型，避免在循环中重复转换
#                 input_nodes_long = input_nodes.long() if input_nodes.dtype != torch.long else input_nodes
                
#                 # 优化2: 批量索引所有特征，减少GPU操作次数
#                 all_features = []
#                 for i in range(self.layer_num):
#                     feature_key = f'feature_{i}'
#                     all_features.append(self.source_graph.ndata[feature_key][input_nodes_long])
                
#                 # 优化3: 批量赋值
#                 for i, features in enumerate(all_features):
#                     blocks[0].srcdata[f'feature_{i}'] = features
#                 t_attach1 = time.perf_counter()

#                 t_fwd0 = time.perf_counter()

#                 # ===== 调用 forward_minibatch =====
#                 original_feat = blocks[0].srcdata['feature_0']
#                 enhanced_feat = self.moe_layer.forward_minibatch(original_feat, input_nodes_long)
#                 blocks[0].srcdata['feature_0'] = enhanced_feat
#                 # ==================================


#                 probs1, probs2, probs3 = self.model(blocks)
#                 t_fwd1 = time.perf_counter()
#                 target = blocks[-1].dstdata['label']

#                 probs = (1 - self.beta) * ((1 - self.alpha) * probs1 + self.alpha * probs2) + self.beta * probs3
#                 loss = F.nll_loss(probs, target)

#                 t_bwd0 = time.perf_counter()
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 t_bwd1 = time.perf_counter()

#                 if self.debug_timing and (e == 0 and bidx < 5 or bidx % 100 == 0):
#                     try:
#                         nsrc, ndst, nedges = blocks[0].num_src_nodes(), blocks[0].num_dst_nodes(), blocks[0].num_edges()
#                     except Exception:
#                         nsrc = ndst = nedges = -1
#                     # print(
#                     #     f"[Epoch {e}][Batch {bidx}] wait {t_wait*1000:.1f}ms, attach {(t_attach1-t_attach0)*1000:.1f}ms, "
#                     #     f"fwd {(t_fwd1-t_fwd0)*1000:.1f}ms, bwd {(t_bwd1-t_bwd0)*1000:.1f}ms, "
#                     #     f"block(src={nsrc}, dst={ndst}, edges={nedges}), loss={loss.item():.4f}"
#                     # )
#                 last_yield_end = time.perf_counter()

#             # Validation phase
#             # test_epoch = self.model_config.get('test_epoch', 5)
#             # if e % test_epoch == 0:
#             #     val_score = self._evaluate(val_loader, val_labels)
#             #     if val_score[self.train_config['metric']] > self.best_score:
#             #         self.patience_knt = 0
#             #         self.best_score = val_score[self.train_config['metric']]
#             #         test_score = self._evaluate(test_loader, test_labels)
#             #         print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
#             #             e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
#             #             test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
#             #         if device != 'cpu' and hasattr(torch.cuda, 'memory_allocated'):
#             #             print(f"💾 GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB cached")
#             #     else:
#             #         self.patience_knt += test_epoch
#             #         if self.patience_knt > self.train_config['patience']:
#             #             break

#             val_score = self._evaluate(val_loader, val_labels)
#             if val_score[self.train_config['metric']] > self.best_score:
#                 self.patience_knt = 0
#                 self.best_score = val_score[self.train_config['metric']]
#                 test_score = self._evaluate(test_loader, test_labels)
#                 print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
#                     e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
#                     test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
#                 ep_t1 = time.perf_counter()
#                 print(f"[Epoch {e}] ⏱️ {(ep_t1-ep_t0):.3f}s")
#                 if device != 'cpu' and hasattr(torch.cuda, 'memory_allocated'):
#                     pass
#                     # print(f"💾 GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB cached")
#             else:
#                 self.patience_knt += 1
#                 if self.patience_knt > self.train_config['patience']:
#                     break


#         return test_score
    
#     def _evaluate(self, data_loader, true_labels):
#         """Evaluate model using DGL DataLoader that yields GPU blocks"""
#         self.model.eval()
#         probs1_list, probs2_list, probs3_list, targets_list = [], [], [], []

#         with torch.no_grad():
#             last_yield_end = time.perf_counter()
#             eval_batch = 0
#             t_eval0 = time.perf_counter()
#             for input_nodes, output_nodes, blocks in data_loader:
#                 t_batch_get = time.perf_counter()
#                 t_wait = t_batch_get - last_yield_end
#                 # Attach precomputed features (already on GPU) - optimized
#                 t_attach0 = time.perf_counter()
#                 # 优化: 预先转换input_nodes类型并批量处理
#                 input_nodes_long = input_nodes.long() if input_nodes.dtype != torch.long else input_nodes
                
#                 # 批量索引和赋值
#                 for i in range(self.layer_num):
#                     feature_key = f'feature_{i}'
#                     blocks[0].srcdata[feature_key] = self.source_graph.ndata[feature_key][input_nodes_long]
#                 t_attach1 = time.perf_counter()

#                 targets = blocks[-1].dstdata['label']
#                 t_fwd0 = time.perf_counter()

#                 # ===== 调用 forward_minibatch =====
#                 original_feat = blocks[0].srcdata['feature_0']
#                 enhanced_feat = self.moe_layer.forward_minibatch(original_feat, input_nodes_long)
#                 blocks[0].srcdata['feature_0'] = enhanced_feat
#                 # ==================================


#                 probs1, probs2, probs3 = self.model(blocks)
#                 t_fwd1 = time.perf_counter()

#                 targets_list.append(targets)
#                 probs1_list.append(probs1.detach())
#                 probs2_list.append(probs2.detach())
#                 probs3_list.append(probs3.detach())

#                 if self.debug_timing and eval_batch < 3:
#                     try:
#                         nsrc, ndst, nedges = blocks[0].num_src_nodes(), blocks[0].num_dst_nodes(), blocks[0].num_edges()
#                     except Exception:
#                         nsrc = ndst = nedges = -1
#                     print(
#                         f"[Eval][Batch {eval_batch}] wait {t_wait*1000:.1f}ms, attach {(t_attach1-t_attach0)*1000:.1f}ms, fwd {(t_fwd1-t_fwd0)*1000:.1f}ms, block(src={nsrc}, dst={ndst}, edges={nedges})"
#                     )
#                 last_yield_end = time.perf_counter()
#                 eval_batch += 1

#             targets = torch.cat(targets_list, dim=0)
#             probs1 = torch.cat(probs1_list, dim=0).exp()[:, 1]
#             probs2 = torch.cat(probs2_list, dim=0).exp()[:, 1]
#             probs3 = torch.cat(probs3_list, dim=0).exp()[:, 1]
#             t_eval1 = time.perf_counter()
#             print(f"[Eval] ⏱️ {(t_eval1-t_eval0):.3f}s over {eval_batch} batches")

#         probs = (1 - self.beta) * ((1 - self.alpha) * probs1 + self.alpha * probs2) + self.beta * probs3

#         return self.eval(true_labels, probs)


# class SpaceGNNDetector(BaseDetector):
#     def __init__(self, train_config, model_config, data):
#         super().__init__(train_config, model_config, data)
        
#         # SpaceGNN specific parameters
#         self.alpha = model_config.get('alpha', 0.5)
#         self.beta = model_config.get('beta', 0.5)
#         self.layer_num = model_config.get('layer_num', 2)
#         self.batch_size = model_config.get('batch_size', 10240)  # Restore original batch_size for better stability
#         # Timing/verbose switch
#         self.debug_timing = model_config.get('debug_timing', True)
        

#         # ===== 【修复1】添加 MoE 参数 =====
#         self.use_moe = model_config.get('use_moe', True)
#         self.num_hops = model_config.get('num_hops', 3)
#         # ==================================


#         # Initialize curvature parameters
#         stdneg = model_config.get('stdneg', 0.1)
#         stdpos = model_config.get('stdpos', 0.1)
#         cneg = torch.FloatTensor(self.layer_num).normal_(-0.1, stdneg)
#         cpos = torch.FloatTensor(self.layer_num).normal_(0.1, stdpos)
        
#         # Precompute multi-layer features like original SpaceGNN (on complete graph)
#         self._precompute_multilayer_features()
        
#         # Initialize SpaceGNN model
#         model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
#         self.model = SpaceGNNOriginal(
#             in_dim=model_config['in_feats'],
#             hid_dim=model_config.get('h_feats', 32),
#             out_dim=2,
#             layer_num=self.layer_num,
#             drop_rate=model_config.get('drop_rate', 0),
#             cneg=cneg,
#             cpos=cpos
#         ).to(train_config['device'])
        
#         # Initialize sampler for batch training
#         self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)


#         # ===== 【修复2】MoE 层初始化 =====
#         if self.use_moe:
#             self.moe_layer = SparseMultiHopMoE(
#                 gcn_layer=nn.Identity(),
#                 in_feats=model_config['in_feats'],
#                 out_feats=model_config['in_feats'],
#                 num_hops=self.num_hops,
#                 top_k=1,
#             ).to(train_config['device'])
            
#             # 预计算（feature_0 已在 GPU 上）
#             feat = self.source_graph.ndata['feature_0']
#             self.moe_layer.precompute_all(self.source_graph, feat)
#         # ==================================

    
#     def _precompute_multilayer_features(self):
#         """Precompute multi-layer features like original SpaceGNN on complete graph"""
#         t0 = time.perf_counter()
#         graph = self.source_graph
#         graph.ndata['feature_0'] = graph.ndata['feature'].float()
#         h = graph.ndata['feature_0']
        
#         # Precompute all layers at once on complete graph (like original SpaceGNN)
#         for i in range(1, self.layer_num):
#             # Propagate features using mean aggregation on complete graph
#             graph.ndata['h'] = h
#             graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
#             h = graph.ndata.pop('h')
#             graph.ndata[f'feature_{i}'] = h
        
#         # 性能优化: 预先将所有特征移到GPU
#         device = self.train_config['device']
#         print(f"🖥️  Using device: {device}")
#         print(f"🔧 CUDA available: {torch.cuda.is_available()}")
#         if torch.cuda.is_available():
#             print(f"🎮 CUDA device count: {torch.cuda.device_count()}")
#             print(f"🎯 Current CUDA device: {torch.cuda.current_device()}")
        
#         for i in range(self.layer_num):
#             feature_key = f'feature_{i}'
#             graph.ndata[feature_key] = graph.ndata[feature_key].to(device)
#             print(f"📊 Feature {feature_key} moved to {graph.ndata[feature_key].device}")
        
#         t1 = time.perf_counter()
#         print(f"✅ Precomputed {self.layer_num} layers of features on complete graph and moved to {device} in {(t1-t0):.3f}s")
    
#     def train(self):
#         # Create DGL DataLoaders: parallel sampling and GPU blocks
#         build_t0 = time.perf_counter()

#         def _mask_to_indices(mask):
#             """Convert boolean mask to graph idtype to satisfy DGL sampler."""
#             idx = mask.nonzero(as_tuple=False).view(-1)
#             return idx.to(self.source_graph.idtype)

#         train_indices = _mask_to_indices(self.train_mask)
#         val_indices = _mask_to_indices(self.val_mask)
#         test_indices = _mask_to_indices(self.test_mask)

#         device = self.train_config['device']
#         # If graph/indices are on CUDA, DGL requires num_workers == 0
#         # Otherwise, allow parallel workers on CPU-side sampling
#         if device != 'cpu':
#             num_workers = 0
#         else:
#             num_workers = self.model_config.get('num_workers', 4)

#         train_loader = dgl.dataloading.DataLoader(
#             self.source_graph,
#             train_indices,
#             self.sampler,
#             batch_size=self.batch_size,
#             shuffle=True,
#             drop_last=True,
#             num_workers=num_workers,
#             device=device,
#         )
#         val_loader = dgl.dataloading.DataLoader(
#             self.source_graph,
#             val_indices,
#             self.sampler,
#             batch_size=self.batch_size,
#             shuffle=False,
#             drop_last=False,
#             num_workers=num_workers,
#             device=device,
#         )
#         test_loader = dgl.dataloading.DataLoader(
#             self.source_graph,
#             test_indices,
#             self.sampler,
#             batch_size=10000,
#             shuffle=False,
#             drop_last=False,
#             num_workers=num_workers,
#             device=device,
#         )
#         build_t1 = time.perf_counter()
#         print(f"🧱 Built DataLoaders in {(build_t1-build_t0):.3f}s (num_workers={num_workers}, device={device})")

#         # ===== 【修复3】优化器包含 moe_layer 参数 =====
#         if self.use_moe:
#             optimizer = torch.optim.Adam(
#                 list(self.model.parameters()) + list(self.moe_layer.parameters()),
#                 lr=self.model_config['lr']
#             )
#         else:
#             optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
#         # ===============================================


#         train_labels, val_labels, test_labels = self.labels[self.train_mask], self.labels[self.val_mask], self.labels[self.test_mask]

#         # 设备信息
#         print(f"🚀 Starting training on device: {device}")
#         print(f"📈 Model device: {next(self.model.parameters()).device}")
#         print(f"🎯 Source graph device: {self.source_graph.ndata['feature'].device}")
#         if hasattr(torch.cuda, 'memory_allocated') and device != 'cpu':
#             print(f"💾 GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
#             print(f"💽 GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")

#         for e in range(self.train_config['epochs']):
#             ep_t0 = time.perf_counter()
#             # print(f"[Epoch {e}] ▶️ start")
#             self.model.train()
#             # ===== 【修复4】moe_layer 也要 train 模式 =====
#             if self.use_moe:
#                 self.moe_layer.train()
#             # =============================================
#             last_yield_end = time.perf_counter()
#             for bidx, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
#                 t_batch_get = time.perf_counter()
#                 t_wait = t_batch_get - last_yield_end
#                 # Attach precomputed features to the GPU blocks (optimized)
#                 t_attach0 = time.perf_counter()
#                 # 优化1: 预先转换input_nodes类型，避免在循环中重复转换
#                 input_nodes_long = input_nodes.long() if input_nodes.dtype != torch.long else input_nodes
                
#                 # 优化2: 批量索引所有特征，减少GPU操作次数
#                 all_features = []
#                 for i in range(self.layer_num):
#                     feature_key = f'feature_{i}'
#                     all_features.append(self.source_graph.ndata[feature_key][input_nodes_long])
                
#                 # 优化3: 批量赋值
#                 for i, features in enumerate(all_features):
#                     blocks[0].srcdata[f'feature_{i}'] = features
#                 t_attach1 = time.perf_counter()

#                 t_fwd0 = time.perf_counter()

#                 # ===== 【修复5】条件调用 =====
#                 if self.use_moe:
#                     original_feat = blocks[0].srcdata['feature_0']
#                     enhanced_feat = self.moe_layer.forward_minibatch(original_feat, input_nodes_long)
#                     blocks[0].srcdata['feature_0'] = enhanced_feat
#                 # =============================


#                 probs1, probs2, probs3 = self.model(blocks)
#                 t_fwd1 = time.perf_counter()
#                 target = blocks[-1].dstdata['label']

#                 probs = (1 - self.beta) * ((1 - self.alpha) * probs1 + self.alpha * probs2) + self.beta * probs3
#                 loss = F.nll_loss(probs, target)

#                 # ===== 【修复6】加入 MoE 辅助损失 =====
#                 if self.use_moe:
#                     loss = loss + self.moe_layer.get_aux_loss()
#                 # =====================================
#                 t_bwd0 = time.perf_counter()
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 t_bwd1 = time.perf_counter()

#                 if self.debug_timing and (e == 0 and bidx < 5 or bidx % 100 == 0):
#                     try:
#                         nsrc, ndst, nedges = blocks[0].num_src_nodes(), blocks[0].num_dst_nodes(), blocks[0].num_edges()
#                     except Exception:
#                         nsrc = ndst = nedges = -1
#                     # print(
#                     #     f"[Epoch {e}][Batch {bidx}] wait {t_wait*1000:.1f}ms, attach {(t_attach1-t_attach0)*1000:.1f}ms, "
#                     #     f"fwd {(t_fwd1-t_fwd0)*1000:.1f}ms, bwd {(t_bwd1-t_bwd0)*1000:.1f}ms, "
#                     #     f"block(src={nsrc}, dst={ndst}, edges={nedges}), loss={loss.item():.4f}"
#                     # )
#                 last_yield_end = time.perf_counter()

#             # Validation phase
#             # test_epoch = self.model_config.get('test_epoch', 5)
#             # if e % test_epoch == 0:
#             #     val_score = self._evaluate(val_loader, val_labels)
#             #     if val_score[self.train_config['metric']] > self.best_score:
#             #         self.patience_knt = 0
#             #         self.best_score = val_score[self.train_config['metric']]
#             #         test_score = self._evaluate(test_loader, test_labels)
#             #         print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
#             #             e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
#             #             test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
#             #         if device != 'cpu' and hasattr(torch.cuda, 'memory_allocated'):
#             #             print(f"💾 GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB cached")
#             #     else:
#             #         self.patience_knt += test_epoch
#             #         if self.patience_knt > self.train_config['patience']:
#             #             break

#             val_score = self._evaluate(val_loader, val_labels)
#             if val_score[self.train_config['metric']] > self.best_score:
#                 self.patience_knt = 0
#                 self.best_score = val_score[self.train_config['metric']]
#                 test_score = self._evaluate(test_loader, test_labels)
#                 print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
#                     test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
#                 ep_t1 = time.perf_counter()
#                 print(f"[Epoch {e}] ⏱️ {(ep_t1-ep_t0):.3f}s")
#                 if device != 'cpu' and hasattr(torch.cuda, 'memory_allocated'):
#                     pass
#                     # print(f"💾 GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB cached")
#             else:
#                 self.patience_knt += 1
#                 if self.patience_knt > self.train_config['patience']:
#                     break


#         return test_score
    
#     def _evaluate(self, data_loader, true_labels):
#         """Evaluate model using DGL DataLoader that yields GPU blocks"""
#         self.model.eval()
#         # ===== 【修复7】moe_layer 也要 eval 模式 =====
#         if self.use_moe:
#             self.moe_layer.eval()
#         # =============================================
#         probs1_list, probs2_list, probs3_list, targets_list = [], [], [], []

#         with torch.no_grad():
#             last_yield_end = time.perf_counter()
#             eval_batch = 0
#             t_eval0 = time.perf_counter()
#             for input_nodes, output_nodes, blocks in data_loader:
#                 t_batch_get = time.perf_counter()
#                 t_wait = t_batch_get - last_yield_end
#                 # Attach precomputed features (already on GPU) - optimized
#                 t_attach0 = time.perf_counter()
#                 # 优化: 预先转换input_nodes类型并批量处理
#                 input_nodes_long = input_nodes.long() if input_nodes.dtype != torch.long else input_nodes
                
#                 # 批量索引和赋值
#                 for i in range(self.layer_num):
#                     feature_key = f'feature_{i}'
#                     blocks[0].srcdata[feature_key] = self.source_graph.ndata[feature_key][input_nodes_long]
#                 t_attach1 = time.perf_counter()

#                 targets = blocks[-1].dstdata['label']
#                 t_fwd0 = time.perf_counter()

#                 # ===== 【修复8】评估时也用 MoE =====
#                 if self.use_moe:
#                     original_feat = blocks[0].srcdata['feature_0']
#                     enhanced_feat = self.moe_layer.forward_minibatch(original_feat, input_nodes_long)
#                     blocks[0].srcdata['feature_0'] = enhanced_feat
#                 # ===================================


#                 probs1, probs2, probs3 = self.model(blocks)
#                 t_fwd1 = time.perf_counter()

#                 targets_list.append(targets)
#                 probs1_list.append(probs1.detach())
#                 probs2_list.append(probs2.detach())
#                 probs3_list.append(probs3.detach())

#                 if self.debug_timing and eval_batch < 3:
#                     try:
#                         nsrc, ndst, nedges = blocks[0].num_src_nodes(), blocks[0].num_dst_nodes(), blocks[0].num_edges()
#                     except Exception:
#                         nsrc = ndst = nedges = -1
#                     # print(
#                     #     f"[Eval][Batch {eval_batch}] wait {t_wait*1000:.1f}ms, attach {(t_attach1-t_attach0)*1000:.1f}ms, fwd {(t_fwd1-t_fwd0)*1000:.1f}ms, block(src={nsrc}, dst={ndst}, edges={nedges})"
#                     # )
#                 last_yield_end = time.perf_counter()
#                 eval_batch += 1

#             targets = torch.cat(targets_list, dim=0)
#             probs1 = torch.cat(probs1_list, dim=0).exp()[:, 1]
#             probs2 = torch.cat(probs2_list, dim=0).exp()[:, 1]
#             probs3 = torch.cat(probs3_list, dim=0).exp()[:, 1]
#             t_eval1 = time.perf_counter()
#             # print(f"[Eval] ⏱️ {(t_eval1-t_eval0):.3f}s over {eval_batch} batches")

#         probs = (1 - self.beta) * ((1 - self.alpha) * probs1 + self.alpha * probs2) + self.beta * probs3

#         return self.eval(true_labels, probs)

class SpaceGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        
        # SpaceGNN specific parameters
        self.alpha = model_config.get('alpha', 0.5)
        self.beta = model_config.get('beta', 0.5)
        self.layer_num = model_config.get('layer_num', 2)
        self.batch_size = model_config.get('batch_size',2048)  # Restore original batch_size for better stability
        # Timing/verbose switch
        self.debug_timing = model_config.get('debug_timing', True)
        

        # ===== 【修复1】添加 MoE 参数 =====
        self.use_moe = model_config.get('use_moe', True)
        self.num_hops = model_config.get('num_hops', 3)
        # ==================================


        # Initialize curvature parameters
        stdneg = model_config.get('stdneg', 0.1)
        stdpos = model_config.get('stdpos', 0.1)
        cneg = torch.FloatTensor(self.layer_num).normal_(-0.1, stdneg)
        cpos = torch.FloatTensor(self.layer_num).normal_(0.1, stdpos)
        
        # Precompute multi-layer features like original SpaceGNN (on complete graph)
        self._precompute_multilayer_features()
        # ===== 【修复】先定义这些变量 =====
        device = train_config['device']
        in_feats = self.data.graph.ndata['feature'].shape[1]
        hid_dim = model_config.get('h_feats', 32)
        model_config['in_feats'] = in_feats  # 保存到 config 供后续使用

        # ===== 创建 MoE 层 =====
        if self.use_moe:
            # 注意：预计算用 in_feats，输出用 hid_dim
            self.moe_layer = SparseMultiHopMoE(
                gcn_layer=nn.Identity(),
                in_feats=in_feats,    # 预计算用原始特征维度
                out_feats=hid_dim,    # 输出匹配 CurvLayer 内部维度
                num_hops=self.num_hops
            ).to(device)
            
            # 预计算（用原始特征）
            feat = self.source_graph.ndata['feature_0']
            self.moe_layer.precompute_all(self.source_graph, feat)
        else:
            self.moe_layer = None


        self.model = SpaceGNNOriginal(
            in_dim=in_feats,
            hid_dim=hid_dim,
            out_dim=2,
            layer_num=self.layer_num,
            drop_rate=model_config.get('drop_rate', 0),
            cneg=cneg,
            cpos=cpos,
            moe_layer=self.moe_layer,  # 传入 MoE
        ).to(device)
        
        # Initialize sampler for batch training
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)

    
    def _precompute_multilayer_features(self):
        """Precompute multi-layer features like original SpaceGNN on complete graph"""
        t0 = time.perf_counter()
        graph = self.source_graph
        graph.ndata['feature_0'] = graph.ndata['feature'].float()
        h = graph.ndata['feature_0']
        
        # Precompute all layers at once on complete graph (like original SpaceGNN)
        for i in range(1, self.layer_num):
            # Propagate features using mean aggregation on complete graph
            graph.ndata['h'] = h
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
            h = graph.ndata.pop('h')
            graph.ndata[f'feature_{i}'] = h
        
        # 性能优化: 预先将所有特征移到GPU
        device = self.train_config['device']
        print(f"🖥️  Using device: {device}")
        print(f"🔧 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 CUDA device count: {torch.cuda.device_count()}")
            print(f"🎯 Current CUDA device: {torch.cuda.current_device()}")
        
        for i in range(self.layer_num):
            feature_key = f'feature_{i}'
            graph.ndata[feature_key] = graph.ndata[feature_key].to(device)
            print(f"📊 Feature {feature_key} moved to {graph.ndata[feature_key].device}")
        
        t1 = time.perf_counter()
        print(f"✅ Precomputed {self.layer_num} layers of features on complete graph and moved to {device} in {(t1-t0):.3f}s")
    
    def train(self):
        # Create DGL DataLoaders: parallel sampling and GPU blocks
        build_t0 = time.perf_counter()
        best_expert_stats = None
        def _mask_to_indices(mask):
            """Convert boolean mask to graph idtype to satisfy DGL sampler."""
            idx = mask.nonzero(as_tuple=False).view(-1)
            return idx.to(self.source_graph.idtype)

        train_indices = _mask_to_indices(self.train_mask)
        val_indices = _mask_to_indices(self.val_mask)
        test_indices = _mask_to_indices(self.test_mask)

        device = self.train_config['device']
        # If graph/indices are on CUDA, DGL requires num_workers == 0
        # Otherwise, allow parallel workers on CPU-side sampling
        if device != 'cpu':
            num_workers = 0
        else:
            num_workers = self.model_config.get('num_workers', 4)

        train_loader = dgl.dataloading.DataLoader(
            self.source_graph,
            train_indices,
            self.sampler,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            device=device,
        )
        val_loader = dgl.dataloading.DataLoader(
            self.source_graph,
            val_indices,
            self.sampler,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            device=device,
        )
        test_loader = dgl.dataloading.DataLoader(
            self.source_graph,
            test_indices,
            self.sampler,
            batch_size=10000,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            device=device,
        )
        build_t1 = time.perf_counter()
        print(f"🧱 Built DataLoaders in {(build_t1-build_t0):.3f}s (num_workers={num_workers}, device={device})")

        # ===== 【修复3】优化器包含 moe_layer 参数 =====
        if self.use_moe:
            optimizer = torch.optim.Adam(
                list(self.model.parameters()) + list(self.moe_layer.parameters()),
                lr=self.model_config['lr']
            )
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        # ===============================================


        train_labels, val_labels, test_labels = self.labels[self.train_mask], self.labels[self.val_mask], self.labels[self.test_mask]

        # 设备信息
        print(f"🚀 Starting training on device: {device}")
        print(f"📈 Model device: {next(self.model.parameters()).device}")
        print(f"🎯 Source graph device: {self.source_graph.ndata['feature'].device}")
        if hasattr(torch.cuda, 'memory_allocated') and device != 'cpu':
            print(f"💾 GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"💽 GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        final_epoch=0
        for e in range(self.train_config['epochs']):
            
            ep_t0 = time.perf_counter()
            # print(f"[Epoch {e}] ▶️ start")
            self.model.train()
            # ===== 【修复4】moe_layer 也要 train 模式 =====
            if self.use_moe:
                self.moe_layer.train()
            # =============================================
            last_yield_end = time.perf_counter()
            for bidx, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
                # 附加特征
                input_nodes_long = input_nodes.long()
                for i in range(self.layer_num):
                    blocks[0].srcdata[f'feature_{i}'] = self.source_graph.ndata[f'feature_{i}'][input_nodes_long]

                # ===== 【关键】获取 output_nodes 作为 node_indices =====
                output_nodes_long = output_nodes.long()
                # 【新增】把 feature_0 附加到 dstdata，供 MoE 使用
                if self.use_moe:
                    blocks[0].dstdata['feature_0'] = self.source_graph.ndata['feature_0'][output_nodes_long]
                                
                probs1, probs2, probs3 = self.model(blocks, node_indices=output_nodes_long)
                target = blocks[-1].dstdata['label']

                probs = (1 - self.beta) * ((1 - self.alpha) * probs1 + self.alpha * probs2) + self.beta * probs3
                loss = F.nll_loss(probs, target)

                # ===== 【修复6】加入 MoE 辅助损失 =====
                if self.use_moe:
                    loss = loss + self.moe_layer.get_aux_loss()
                # =====================================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.debug_timing and (e == 0 and bidx < 5 or bidx % 100 == 0):
                    try:
                        nsrc, ndst, nedges = blocks[0].num_src_nodes(), blocks[0].num_dst_nodes(), blocks[0].num_edges()
                    except Exception:
                        nsrc = ndst = nedges = -1
                    # print(
                    #     f"[Epoch {e}][Batch {bidx}] wait {t_wait*1000:.1f}ms, attach {(t_attach1-t_attach0)*1000:.1f}ms, "
                    #     f"fwd {(t_fwd1-t_fwd0)*1000:.1f}ms, bwd {(t_bwd1-t_bwd0)*1000:.1f}ms, "
                    #     f"block(src={nsrc}, dst={ndst}, edges={nedges}), loss={loss.item():.4f}"
                    # )

            # Validation phase
            # test_epoch = self.model_config.get('test_epoch', 5)
            # if e % test_epoch == 0:
            #     val_score = self._evaluate(val_loader, val_labels)
            #     if val_score[self.train_config['metric']] > self.best_score:
            #         self.patience_knt = 0
            #         self.best_score = val_score[self.train_config['metric']]
            #         test_score = self._evaluate(test_loader, test_labels)
            #         print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
            #             e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
            #             test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            #         if device != 'cpu' and hasattr(torch.cuda, 'memory_allocated'):
            #             print(f"💾 GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB cached")
            #     else:
            #         self.patience_knt += test_epoch
            #         if self.patience_knt > self.train_config['patience']:
            #             break

            val_score = self._evaluate(val_loader, val_labels)
            if val_score[self.train_config['metric']] > self.best_score:
                final_epoch=e
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self._evaluate(test_loader, test_labels)
                # ===== 新增：保存最好 epoch 的统计 =====
                best_expert_stats = self.moe_layer.get_expert_stats().copy()
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
                ep_t1 = time.perf_counter()
                print(f"[Epoch {e}] ⏱️ {(ep_t1-ep_t0):.3f}s")
                if device != 'cpu' and hasattr(torch.cuda, 'memory_allocated'):
                    pass
                    # print(f"💾 GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB cached")
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        # ========== 新增：训练结束后打印路由权重 ==========
        self._print_routing_weights(final_epoch, best_expert_stats)
        # ================================================
        return test_score
    def _print_routing_weights(self, epoch, stats=None):
        """打印MoE路由权重统计"""
        if stats is None:
            # 如果没传入，用当前的（兼容旧代码）
            self.model.eval()
            with torch.no_grad():
                stats = self.moe_layer.get_expert_stats()
        
        gcn_w = stats.get('gcn_weight', 0)
        moe_w = stats.get('moe_weight', 0)
        h_proj_w = 1 - gcn_w - moe_w
        print(f"\n{'='*60}")
        print(f"Best Epoch {epoch} Routing Weights:")  # 改成 Best Epoch
        print(f"  1-hop weight: {stats.get('hop1_avg_weight', 0):.4f}")
        print(f"  2-hop weight: {stats.get('hop2_avg_weight', 0):.4f}")
        print(f"  3-hop weight: {stats.get('hop3_avg_weight', 0):.4f}")
        print(f"  GCN gate:     {gcn_w:.4f}")
        print(f"  MoE gate:     {moe_w:.4f}")
        print(f"  h_proj gate:  {h_proj_w:.4f}")
        print(f"{'='*60}\n")
    
    def _evaluate(self, data_loader, true_labels):
        """Evaluate model using DGL DataLoader that yields GPU blocks"""
        self.model.eval()
        # ===== 【修复7】moe_layer 也要 eval 模式 =====
        if self.use_moe:
            self.moe_layer.eval()
        # =============================================
        probs1_list, probs2_list, probs3_list, targets_list = [], [], [], []

        with torch.no_grad():
            last_yield_end = time.perf_counter()
            eval_batch = 0
            t_eval0 = time.perf_counter()
            for input_nodes, output_nodes, blocks in data_loader:
                input_nodes_long = input_nodes.long()
                for i in range(self.layer_num):
                    blocks[0].srcdata[f'feature_{i}'] = self.source_graph.ndata[f'feature_{i}'][input_nodes_long]

                # ===== 【关键】传入 node_indices =====
                output_nodes_long = output_nodes.long()
                # 【新增】把 feature_0 附加到 dstdata
                if self.use_moe:
                    blocks[0].dstdata['feature_0'] = self.source_graph.ndata['feature_0'][output_nodes_long]
                
                probs1, probs2, probs3 = self.model(blocks, node_indices=output_nodes_long)
                
                probs1_list.append(probs1.detach())
                probs2_list.append(probs2.detach())
                probs3_list.append(probs3.detach())
                

                if self.debug_timing and eval_batch < 3:
                    try:
                        nsrc, ndst, nedges = blocks[0].num_src_nodes(), blocks[0].num_dst_nodes(), blocks[0].num_edges()
                    except Exception:
                        nsrc = ndst = nedges = -1
                    # print(
                    #     f"[Eval][Batch {eval_batch}] wait {t_wait*1000:.1f}ms, attach {(t_attach1-t_attach0)*1000:.1f}ms, fwd {(t_fwd1-t_fwd0)*1000:.1f}ms, block(src={nsrc}, dst={ndst}, edges={nedges})"
                    # )
                last_yield_end = time.perf_counter()
                eval_batch += 1

            # targets = torch.cat(targets_list, dim=0)
            probs1 = torch.cat(probs1_list, dim=0).exp()[:, 1]
            probs2 = torch.cat(probs2_list, dim=0).exp()[:, 1]
            probs3 = torch.cat(probs3_list, dim=0).exp()[:, 1]
            t_eval1 = time.perf_counter()
            # print(f"[Eval] ⏱️ {(t_eval1-t_eval0):.3f}s over {eval_batch} batches")

        probs = (1 - self.beta) * ((1 - self.alpha) * probs1 + self.alpha * probs2) + self.beta * probs3

        return self.eval(true_labels, probs)



class DGAGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        # 先初始化基础属性，但不调用父类构造函数以避免图被移到GPU
        self.model_config = model_config
        self.train_config = train_config
        self.data = data
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        
        # 保持原始图在CPU上用于UVA采样
        cpu_graph = self.data.graph
        self.labels = cpu_graph.ndata['label']
        self.train_mask = cpu_graph.ndata['train_mask'].bool()
        self.val_mask = cpu_graph.ndata['val_mask'].bool()
        self.test_mask = cpu_graph.ndata['test_mask'].bool()
        self.weight = (1 - self.labels[self.train_mask]).sum().item() / self.labels[self.train_mask].sum().item()
        
        # source_graph保持在CPU用于UVA采样
        self.source_graph = cpu_graph
        
        # 创建GPU版本的图用于其他操作（如果需要）
        if train_config['inductive'] == False:
            self.train_graph = cpu_graph
            self.val_graph = cpu_graph
        else:
            self.train_graph = cpu_graph.subgraph(self.train_mask)
            self.val_graph = cpu_graph.subgraph(self.train_mask+self.val_mask)
            
        self.best_score = -1
        self.patience_knt = 0
        
        # DGA-GNN specific parameters
        self.n_hidden = model_config.get('n_hidden', 128)
        self.p = model_config.get('p', 0.3)
        self.n_head = model_config.get('n_head', 1)
        self.unclear_up = model_config.get('unclear_up', 0.1)
        self.unclear_down = model_config.get('unclear_down', 0.1)
        
        # Initialize DGA model - 添加多跳参数
        self.model = DGA(
            in_feats=model_config['in_feats'],
            n_hidden=self.n_hidden,
            num_nodes=self.data.graph.num_nodes(),
            n_classes=2,
            n_etypes=1,
            p=self.p,
            n_head=self.n_head,
            unclear_up=self.unclear_up,
            unclear_down=self.unclear_down,
            # ===== 新增 =====
            use_multihop=model_config.get('use_multihop', True),
            num_hops=model_config.get('num_hops', 3),
            top_k=model_config.get('top_k', 1),
        ).to(train_config['device'])
        
        # Initialize tracking variables like original
        self.ps = []  # Store probability history
        
        # Setup DataLoader like original - simplified for GADBench
        self.batch_size = train_config.get('batch_size', 4096)
        
        # Create samplers
        trn_sampler = dgl.dataloading.NeighborSampler([-1])
        val_sampler = dgl.dataloading.NeighborSampler([-1])
        
        idx_dtype = self.source_graph.idtype 

        # Get indices (保持在CPU，强制使用int32类型以兼容DGL)
        def mask_to_index(mask):
            return torch.nonzero(mask, as_tuple=False).squeeze(1)
            # return torch.nonzero(mask, as_tuple=False).squeeze(1).to(torch.int32)
            # return torch.nonzero(mask, as_tuple=False).squeeze(1).to(idx_dtype)
        
        self.trn_idx = mask_to_index(self.train_mask)
        self.val_idx = mask_to_index(self.val_mask) 
        self.tst_idx = mask_to_index(self.test_mask)
        
        # 确保所有图张量在内存中连续存储，避免DGL DataLoader的pin_memory错误
        for key in self.source_graph.ndata:
            if isinstance(self.source_graph.ndata[key], torch.Tensor):
                self.source_graph.ndata[key] = self.source_graph.ndata[key].contiguous()
        
        # Create DataLoaders - 现在source_graph在CPU上，可以使用UVA
        self.trn_dataloader = dgl.dataloading.DataLoader(
            self.source_graph,  # CPU graph
            self.trn_idx,  # CPU indices
            trn_sampler,
            device=train_config['device'],  # 数据会被传输到GPU
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            use_uva=True,
            num_workers=0,
        )
        
        self.val_dataloader = dgl.dataloading.DataLoader(
            self.source_graph,  # CPU graph
            torch.arange(self.source_graph.num_nodes()),
            # torch.arange(self.source_graph.num_nodes(), dtype=torch.int32),  # 强制使用int32
            # torch.arange(self.source_graph.num_nodes(), dtype=idx_dtype),  # 强制使用int32
            val_sampler,
            device=train_config['device'],  # 数据会被传输到GPU
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            use_uva=True,
            num_workers=0,
        )
        
        # Weight for loss (class balancing)
        w = [1, 1]  # Default weights like original
        self.w = torch.FloatTensor(w).to(train_config['device'])

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'], weight_decay=self.model_config.get('weight_decay', 5e-4))
        
        # Add learning rate scheduler like original - 监控验证性能
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=100, verbose=True)
        
        train_labels, val_labels, test_labels = self.labels[self.train_mask], self.labels[self.val_mask], self.labels[self.test_mask]
    
        # ===== 预计算多跳：使用原始特征 =====
        if self.model.use_multihop:
            full_feat = self.source_graph.ndata['feature']  # 不需要 .to(device)，方法内部会处理
            self.model.precompute_multihop(self.source_graph, full_feat, self.train_config['device'])  # ✅ 调用模型方法

        final_epoch = 0  
        for e in range(self.train_config['epochs']):
            self.model.train()
            final_epoch = e
            # Training step with DataLoader like original
            epoch_loss = 0
            for batch_idx, (input_nodes, output_nodes, blocks) in enumerate(self.trn_dataloader):
                x = blocks[0].srcdata["feature"]  # 使用'feature'而不是'feat'
                y = blocks[-1].dstdata["label"]
                
                logits, emb_logits = self.model(blocks, x)
                loss = F.cross_entropy(logits, y, self.w)
                # Could add emb_loss like original: loss = loss + 0.5*emb_loss
                # ===== 新增：加上辅助损失 =====
                if self.model.use_multihop:
                    aux_loss = self.model.multihop_layer.get_aux_loss()
                    loss = loss + aux_loss
                # ==============================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation step - inference on full graph like original
            if e % 1 == 0:  # Validate every epoch
                self.model.eval()
                with torch.no_grad():
                    y_list = []
                    logits_list = []
                    
                    for input_nodes, output_nodes, blocks in self.val_dataloader:
                        x = blocks[0].srcdata["feature"]  # 使用'feature'而不是'feat'
                        y = blocks[-1].dstdata["label"]
                        logits, emb_logits = self.model(blocks, x)
                        
                        y_list.append(y)
                        logits_list.append(logits)
                    
                    y_all = torch.cat(y_list, dim=0).cpu().numpy()
                    logits_all = torch.cat(logits_list, dim=0)
                    prob = logits_all.softmax(-1).cpu().numpy()[:, 1]
                    
                    # Update super_mask like original validation_epoch_end
                    self.ps.append(logits_all.softmax(-1).cpu().numpy())
                    if len(self.ps) > 1:
                        # Update super_mask with moving average of predictions
                        avg_probs = np.mean(self.ps[-10:], axis=0)  # Last 10 epochs average
                        self.model.super_mask.copy_(torch.FloatTensor(avg_probs).to(self.model.super_mask.device))
                
                # Calculate metrics on validation set
                val_score = self.eval(val_labels, prob[self.val_idx.cpu()])
                scheduler.step(val_score['AUROC'])  # 监控验证AUC，AUC越高越好
                
                if val_score[self.train_config['metric']] > self.best_score:
                    self.patience_knt = 0
                    self.best_score = val_score[self.train_config['metric']]
                    test_score = self.eval(test_labels, prob[self.tst_idx.cpu()])
                    print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}'.format(
                        e, epoch_loss/len(self.trn_dataloader), val_score['AUROC'], val_score['AUPRC'], val_score['RecK'], val_score['F1'],
                        test_score['AUROC'], test_score['AUPRC'], test_score['RecK'], test_score['F1']))
                else:
                    self.patience_knt += 1
                    if self.patience_knt > self.train_config['patience']:
                        break
        

        # ========== 新增：训练结束后打印路由权重 ==========
        self._print_routing_weights(final_epoch)
        # ================================================
        return test_score
    def _print_routing_weights(self, epoch):
        """打印MoE路由权重统计"""
        if not self.model.use_multihop:
            print("多跳模块未启用，跳过路由权重打印")
            return
            
        self.model.eval()
        with torch.no_grad():
            # 直接从 multihop_layer 获取统计，不需要再跑前向
            if hasattr(self.model, 'multihop_layer'):
                stats = self.model.multihop_layer.get_expert_stats()
                print(f"\n{'='*60}")
                print(f"Final Routing Weights (Epoch {epoch}):")
                print(f"  1-hop weight: {stats.get('hop1_avg_weight', 0):.4f}")
                print(f"  2-hop weight: {stats.get('hop2_avg_weight', 0):.4f}")
                print(f"  3-hop weight: {stats.get('hop3_avg_weight', 0):.4f}")
                print(f"  GCN gate:     {stats.get('gcn_weight', 0):.4f}")
                print(f"  MoE gate:     {stats.get('moe_weight', 0):.4f}")
                print(f"{'='*60}\n")             


class ARCDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        
        # ARC specific parameters - using original ARC defaults
        self.num_hops = model_config.get('num_hops', 2)      # 原始默认值
        self.num_prompt = model_config.get('num_prompt', 10)  # 原始默认值
        
        # ===== 新增：多跳 MoE 参数 =====
        self.use_multihop_moe = model_config.get('use_multihop_moe', True)
        
        # # 如果没有指定，使用原始ARC的默认参数
        # if 'h_feats' not in model_config:
        #     model_config['h_feats'] = 1024


        # 获取节点数量，根据图大小自动调整 h_feats
        n_nodes = self.data.graph.number_of_nodes()
        # 根据图大小动态调整隐藏层维度，避免OOM
        if 'h_feats' not in model_config:
            if n_nodes > 500000:
                model_config['h_feats'] = 256
                print(f"Large graph detected ({n_nodes} nodes). Reducing h_feats to 256.")
            elif n_nodes > 200000:
                model_config['h_feats'] = 512
                print(f"Medium-large graph detected ({n_nodes} nodes). Reducing h_feats to 512.")
            else:
                model_config['h_feats'] = 1024  # 原始默认值

        if 'num_layers' not in model_config:
            model_config['num_layers'] = 4
        if 'activation' not in model_config:
            model_config['activation'] = 'ELU'
        if 'lr' not in model_config:
            model_config['lr'] = 1e-5
        if 'weight_decay' not in model_config:
            model_config['weight_decay'] = 5e-5
        
        # Precompute multi-hop features
        self._precompute_multilayer_features()
        
        # Initialize ARC model
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = ARC(**model_config).to(train_config['device'])

        # ===== 新增：保存图用于 MoE =====
        if self.use_multihop_moe:
            self.graph_for_moe = self.source_graph.to(train_config['device'])
        else:
            self.graph_for_moe = None
    
    # def _precompute_multilayer_features(self):
    #     """Precompute multi-hop propagation features like original ARC"""
    #     graph = self.source_graph
        
    #     # Get adjacency matrix with proper index type handling
    #     try:
    #         adj = graph.adj().to_dense().cpu().numpy()
    #     except IndexError as e:
    #         if "tensors used as indices must be long, byte or bool tensors" in str(e):
    #             # Fix index tensor type issue for certain datasets like tolokers
    #             # Use alternative approach: convert to scipy sparse first
    #             adj_sparse = graph.adj()
    #             # Get the underlying sparse tensor data
    #             row, col = adj_sparse.coo()
    #             # Create adjacency matrix manually
    #             import scipy.sparse as sp
    #             n_nodes = graph.number_of_nodes()
    #             adj_scipy = sp.coo_matrix((torch.ones(len(row)), (row.cpu().numpy(), col.cpu().numpy())), 
    #                                     shape=(n_nodes, n_nodes))
    #             adj = adj_scipy.toarray().astype(float)
    #         else:
    #             raise e
        
    #     # Normalize adjacency matrix
    #     adj_norm = normalize_adj(adj + np.eye(adj.shape[0]))  # Add self-loops
    #     adj_norm_tensor = sparse_mx_to_torch_sparse_tensor(adj_norm).to(self.train_config['device'])
        
    #     # Multi-hop feature propagation - exactly like original ARC
    #     x = graph.ndata['feature'].float()
    #     h_list = [x]
    #     for _ in range(self.num_hops - 1):
    #         h_list.append(torch.spmm(adj_norm_tensor, h_list[-1]))
        
    #     # Store in a custom data structure that mimics original ARC's graph object
    #     class GraphData:
    #         def __init__(self, x_list):
    #             self.x_list = x_list
        
    #     self.graph_data = GraphData(h_list)
    

    def _precompute_multilayer_features(self):
        """Precompute multi-hop propagation features like original ARC"""
        graph = self.source_graph
        n_nodes = graph.number_of_nodes()
        
        # Get edges directly from DGL graph - avoid dense conversion entirely
        src, dst = graph.edges()
        src = src.cpu().numpy()
        dst = dst.cpu().numpy()
        
        # Create sparse adjacency matrix directly
        adj = sp.coo_matrix((np.ones(len(src)), (src, dst)), 
                            shape=(n_nodes, n_nodes))
        
        # Add self-loops and normalize (all in sparse format)
        adj_with_self_loops = adj + sp.eye(n_nodes)
        adj_norm = normalize_adj(adj_with_self_loops)
        adj_norm_tensor = sparse_mx_to_torch_sparse_tensor(adj_norm).to(self.train_config['device'])
        
        # Multi-hop feature propagation - exactly like original ARC
        x = graph.ndata['feature'].float().to(self.train_config['device'])
        h_list = [x]
        for _ in range(self.num_hops - 1):
            h_list.append(torch.spmm(adj_norm_tensor, h_list[-1]))
        
        # 清理邻接矩阵
        del adj_norm_tensor
        torch.cuda.empty_cache()
        # Store in a custom data structure that mimics original ARC's graph object
        class GraphData:
            def __init__(self, x_list):
                self.x_list = x_list
        
        self.graph_data = GraphData(h_list)

    def _create_few_shot_mask(self, labels):
        """Create few-shot support set mask from normal training nodes"""
        train_labels = labels[self.train_mask]
        train_indices = torch.nonzero(self.train_mask).squeeze(1)
        
        # Find normal nodes in training set
        normal_train_indices = train_indices[train_labels == 0]
        
        # Randomly select support nodes
        if len(normal_train_indices) < self.num_prompt:
            raise ValueError(f"Not enough normal training nodes ({len(normal_train_indices)}) to select {self.num_prompt} support nodes.")
        
        perm = torch.randperm(len(normal_train_indices))
        selected_indices = normal_train_indices[perm[:self.num_prompt]]
        
        # Create mask
        few_shot_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
        few_shot_mask[selected_indices] = True
        
        return few_shot_mask
    
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                   lr=self.model_config['lr'], 
                                   weight_decay=self.model_config.get('weight_decay', 5e-5))
        
        train_labels = self.labels[self.train_mask]
        val_labels = self.labels[self.val_mask] 
        test_labels = self.labels[self.test_mask]
        final_epoch = 0  # 新增：记录最好的 epoch
        best_expert_stats = None  # 新增：保存最好 epoch 的统计
        
        for e in range(self.train_config['epochs']):
            self.model.train()
            
            # # Get residual embeddings for all nodes
            # residual_embed = self.model(self.graph_data)
            # ===== 修改：传入图 =====
            residual_embed = self.model(self.graph_data, graph=self.graph_for_moe)
            
            # Only use training data for loss computation - following GADBench standard
            train_residual_embed = residual_embed[self.train_mask]
            train_labels_for_loss = self.labels[self.train_mask]
            
            # Compute ARC training loss using only training data
            loss = self.model.cross_attn.get_train_loss(train_residual_embed, train_labels_for_loss, self.num_prompt)
            # ===== 新增：辅助损失 =====
            if self.use_multihop_moe:
                aux_loss = self.model.multihop_layer.get_aux_loss()
                loss = loss + aux_loss
            # ==========================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation every 5 epochs or when dropout is used
            if self.model_config['drop_rate'] > 0 or e % 1 == 0:
                self.model.eval()
                with torch.no_grad():
                    # residual_embed = self.model(self.graph_data)
                    # ===== 修改：传入图 =====
                    residual_embed = self.model(self.graph_data, graph=self.graph_for_moe)
                    
                    # Create few-shot mask from training data only
                    few_shot_mask = self._create_few_shot_mask(self.labels)
                    
                    # Get test scores using cross attention
                    query_scores = self.model.cross_attn.get_test_score(
                        residual_embed, few_shot_mask, self.labels)
                    
                    # Map scores back to all nodes (query nodes only get scores)
                    all_scores = torch.zeros(self.labels.shape[0], device=self.labels.device)
                    query_indices = torch.nonzero(few_shot_mask == False).squeeze(1)
                    all_scores[query_indices] = query_scores
                    
                    val_score = self.eval(val_labels, all_scores[self.val_mask])
                    
                    if val_score[self.train_config['metric']] > self.best_score:
                        final_epoch = e  # 记录最好的 epoch
                        self.patience_knt = 0
                        self.best_score = val_score[self.train_config['metric']]
                        test_score = self.eval(test_labels, all_scores[self.test_mask])
                        # ===== 新增：保存最好 epoch 的统计 =====
                        if self.use_multihop_moe:
                            best_expert_stats = self.model.multihop_layer.get_expert_stats().copy()
                       
                        print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}'.format(
                            e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'], val_score['F1'],
                            test_score['AUROC'], test_score['AUPRC'], test_score['RecK'], test_score['F1']))
                    else:
                        self.patience_knt += 1 #if e % 5 == 0 else 1
                        if self.patience_knt > self.train_config['patience']:
                            break
        self._print_routing_weights(final_epoch, best_expert_stats)
        
        return test_score
    def _print_routing_weights(self, epoch, stats=None):
        """打印MoE路由权重统计"""
        if not self.use_multihop_moe:
            print("多跳 MoE 模块未启用，跳过路由权重打印")
            return
        
        if stats is None:
            # 如果没有保存的统计，用当前的
            self.model.eval()
            with torch.no_grad():
                _ = self.model(self.graph_data, graph=self.graph_for_moe)
                stats = self.model.multihop_layer.get_expert_stats()
        
        print(f"\n{'='*60}")
        print(f"Best Epoch {epoch} Routing Weights:")
        print(f"  1-hop weight: {stats.get('hop1_avg_weight', 0):.4f}")
        print(f"  2-hop weight: {stats.get('hop2_avg_weight', 0):.4f}")
        print(f"  3-hop weight: {stats.get('hop3_avg_weight', 0):.4f}")
        print(f"  GCN gate:     {stats.get('gcn_weight', 0):.4f}")
        print(f"  MoE gate:     {stats.get('moe_weight', 0):.4f}")
        print(f"{'='*60}\n")




# ============================================================
# GADAM Detector
# ============================================================

class GADAMDetector(BaseDetector):
    """
    GADAM: Boosting Graph Anomaly Detection with Adaptive Message Passing
    Two-stage detector: LocalModel (LIM) -> GlobalModel (adaptive message passing)
    """
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        self.out_dim = model_config.get('h_feats', 64)
        self.local_epochs = model_config.get('local_epochs', 100)
        self.global_epochs = model_config.get('global_epochs', 50)
        self.local_lr = model_config.get('local_lr', 1e-3)
        self.global_lr = model_config.get('global_lr', 5e-4)
        self.ano_topk = model_config.get('ano_topk', 0.05)
        self.nor_topk = model_config.get('nor_topk', 0.3)

    def train(self):
        device = self.train_config['device']
        graph = self.source_graph.to(device)
        feats = graph.ndata['feature'].float().to(device)
        in_feats = feats.shape[1]
        labels = self.labels.cpu().numpy()
        train_labels = self.labels[self.train_mask]
        val_labels = self.labels[self.val_mask]
        test_labels = self.labels[self.test_mask]

        # ===== Stage 1: Local Model (LIM) =====
        local_net = LocalModel(graph, in_feats, self.out_dim, nn.PReLU()).to(device)
        local_opt = torch.optim.Adam(local_net.parameters(), lr=self.local_lr)

        def init_xavier(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

        local_net.apply(init_xavier)

        best_local_loss = float('inf')
        best_local_state = None

        for e in range(self.local_epochs):
            local_net.train()
            local_opt.zero_grad()
            loss, l1, l2 = local_net(feats)
            loss.backward()
            local_opt.step()

            if loss.item() < best_local_loss:
                best_local_loss = loss.item()
                best_local_state = {k: v.clone() for k, v in local_net.state_dict().items()}

            if e % 20 == 0:
                print(f"[GADAM Local] Epoch {e}, Loss {loss.item():.4f}")

        # Restore best local model
        local_net.load_state_dict(best_local_state)
        local_net.eval()

        with torch.no_grad():
            h, mean_h = local_net.encoder(feats)
            pos = graph.ndata['pos']

        # ===== Select high-confidence normal/anomaly nodes =====
        scores_local = -pos.detach()
        num_nodes = graph.num_nodes()

        num_ano = int(num_nodes * self.ano_topk)
        _, ano_idx = torch.topk(scores_local, num_ano)

        num_nor = int(num_nodes * self.nor_topk)
        _, nor_idx = torch.topk(-scores_local, num_nor)

        center = h[nor_idx].mean(dim=0).detach()

        # ===== Stage 2: Global Model (Adaptive Message Passing) =====
        global_net = GlobalModel(
            graph, in_feats, self.out_dim, nn.PReLU(),
            nor_idx, ano_idx, center
        ).to(device)
        global_opt = torch.optim.Adam(global_net.parameters(), lr=self.global_lr)
        global_net.apply(init_xavier)

        best_val_score = -1
        test_score = None

        for e in range(self.global_epochs):
            global_net.train()
            global_opt.zero_grad()
            loss, scores = global_net(feats, e)
            loss.backward()
            global_opt.step()

            # Compute mixed anomaly score
            with torch.no_grad():
                mix_score = -(scores + pos).detach()
                probs = mix_score

                # Normalize to [0, 1] for evaluation
                probs_normalized = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)

                val_score = self.eval(val_labels, probs_normalized[self.val_mask])

                if val_score[self.train_config['metric']] > best_val_score:
                    self.patience_knt = 0
                    best_val_score = val_score[self.train_config['metric']]
                    test_score = self.eval(test_labels, probs_normalized[self.test_mask])

                    if e % 10 == 0:
                        print(f"[GADAM Global] Epoch {e}, Loss {loss.item():.4f}, "
                              f"Val AUC {val_score['AUROC']:.4f}, Test AUC {test_score['AUROC']:.4f}")
                else:
                    self.patience_knt += 1
                    if self.patience_knt > self.train_config['patience']:
                        break

        if test_score is None:
            with torch.no_grad():
                mix_score = -(scores + pos).detach()
                probs_normalized = (mix_score - mix_score.min()) / (mix_score.max() - mix_score.min() + 1e-8)
                test_score = self.eval(test_labels, probs_normalized[self.test_mask])

        return test_score
