import os
import copy
import json
import numpy as np
from pathlib import Path
#from parameter_parser import parameter_parser
import torch
from collections import defaultdict
from lib_utils.utils import fix_seed,result_printer,mean_std_metrics
from lib_utils.train_agent import Trainer
from lib_utils.eval_agent import Evaluator
from lib_models.HNN import HCHA,HyperGCN,HNHN,SetGNN,UniGNN,UniGCNII,LEGCN,HyperND,EquivSetGNN,\
                            PlainUnigencoder,HJRL,SheafHyperGNN,EHNN,TMPHN,PhenomNN,PhenomNNS,DPHGNN,TFHNN,PlainMLP,HyperGT,CEGCN,CEGAT

from lib_dataset.data_perturbation import perturbation
from lib_dataset.edge_loaders import generate_edge_loaders,generate_split_hyperedges,generate_ind_split_hyperedges,\
                                    generate_observed_split_hyperedges,generate_observed_ind_split_hyperedges,\
                                    build_observed_train_data,observed_edge_split_dir,observed_ehnn_cache_path
from lib_dataset.hg_loaders import generate_split_hypergraphs,generate_hg_loaders
from lib_models.HNN.preprocessing import algo_preprocessing
from lib_utils.aggregator import EdgePredictor,MeanAggregator,MaxminAggregator,MaxAggregator,HyperGPredictor
from lib_utils.metrics import aggr_metrics,avg_result_printer_edge


def _to_serializable(obj):
    """Convert numpy / defaultdict objects into plain JSON-serializable data."""
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _is_true(value):
    return str(value).lower() in ('1', 'true', 'yes', 't')


def save_results(args, task, metrics, extra=None):
    """Persist per-seed metrics to results/<task>/<dataset>/<method>[_...].json."""
    out_dir = Path(getattr(args, 'result_dir', './results')) / task / args.dname
    out_dir.mkdir(parents=True, exist_ok=True)

    name = args.method
    if task == 'edge_pred':
        name += f'_{args.edge_pred_protocol}_{args.edge_split_mode}'
    if _is_true(getattr(args, 'is_perturbed', False)):
        name += f'_pert-{args.pert_mode}-{args.pert_p}'

    payload = {
        'task': task,
        'dataset': args.dname,
        'method': args.method,
        'num_seeds': args.num_seeds,
        'metrics': _to_serializable(metrics),
        'config': {k: v for k, v in vars(args).items()
                   if isinstance(v, (int, float, str, bool, type(None)))},
        **(extra or {}),
    }
    path = out_dir / f'{name}.json'
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[Saved] {path}')
    return path


class ExpAgent:
    
    def __init__(self,args,**kwargs):
        """
        Overall pipline for different kinds of models
        """
        self.args = args
        self.device=args.device
        self.trainer=Trainer(args)
        self.evaluator=Evaluator(args)
        self.train_times = []

    def edge_pred_train_eval(self,data):
        if self.args.edge_pred_protocol == 'observed' and self.args.edge_split_mode == 'ind':
            raise ValueError(
                "The observed edge prediction protocol currently supports only "
                "--edge_split_mode=trand. Node-disjoint observed splits require "
                "inductive encoder support and are not implemented."
            )
        
        metrics_dict = {'train':defaultdict(list),'val':defaultdict(list),'test':defaultdict(list)}
        self.histories = []

        for seed in range(self.args.num_seeds):
            
            fix_seed(seed) 
            
            if self.args.edge_pred_protocol == 'observed':
                dir_path = observed_edge_split_dir(self.args)
            else:
                dir_path = os.path.join(
                    str(self.args.edge_save_dir),
                    self.args.edge_split_mode,
                    self.args.dname,
                )
            
            if self.args.edge_split_mode == 'ind':

                file_path = os.path.join(dir_path, f"split_{seed}.pt")
                if not os.path.exists(file_path):
                    os.makedirs(dir_path, exist_ok=True)
                    if self.args.edge_pred_protocol == 'observed':
                        generate_observed_ind_split_hyperedges(data,self.args,seed)
                    else:
                        generate_ind_split_hyperedges(data,self.args,seed)

            elif self.args.edge_split_mode == 'trand':
                
                file_path = os.path.join(dir_path, f"split_{seed}.pt")
                if not os.path.exists(file_path):
                    os.makedirs(dir_path, exist_ok=True)
                    if self.args.edge_pred_protocol == 'observed':
                        generate_observed_split_hyperedges(data,self.args,seed)
                    else:
                        generate_split_hyperedges(data,self.args,seed)

            else:

                raise NotImplementedError
                
            data_dict = torch.load(file_path, weights_only=False)
            if self.args.edge_pred_protocol == 'observed':
                data_for_edge_pred = build_observed_train_data(data,data_dict,self.args)
                ehnn_cache_path = None
                if self.args.method == 'EHNN':
                    ehnn_cache_path = observed_ehnn_cache_path(
                        self.args,
                        seed,
                    )
                data_for_edge_pred = algo_preprocessing(
                    data_for_edge_pred,
                    self.args,
                    ehnn_cache_path=ehnn_cache_path,
                )
            else:
                data_for_edge_pred = data
            batch_loaders = generate_edge_loaders(data_dict,self.args)
            
            self.args.embedding_mode = True 
            encoder = parse_model(self.args,data_for_edge_pred)
            
            if self.args.aggr_mode=='maxmin':
                aggregator = MaxminAggregator(self.args) 
            elif self.args.aggr_mode=='mean':
                aggregator = MeanAggregator(self.args) 
            elif self.args.aggr_mode=='max':
                aggregator = MaxAggregator(self.args) 
            
            model = EdgePredictor(encoder,aggregator,self.args)
            if self.args.method == 'TMPHN':
                model.aggregator = model.aggregator.to(self.args.device)
            else:
                model = model.to(self.args.device)
            
            model = self.trainer.training(model,data_for_edge_pred,self.args,seed_split=batch_loaders,task_type='edge_pred')
            self.histories.append({'seed': seed, 'history': self.trainer.history})

            if self.args.eval_verbose:
                print(f'------------------------------[Seed {seed}]-----------------------------------')
                result=self.evaluator.evaluate(model,data_for_edge_pred,seed_split=batch_loaders,task_type='edge_pred',verbose=True)
                metrics_dict = aggr_metrics(metrics_dict,result) 
                print(f'------------------------------------------------------------------------------')
            else:
                result=self.evaluator.evaluate(model,data_for_edge_pred,seed_split=batch_loaders,task_type='edge_pred',verbose=True)
                metrics_dict = aggr_metrics(metrics_dict,result) 

        print(f'---------------------------------[Final]--------------------------------------')
        avg_result_printer_edge(metrics_dict)
        print(f'------------------------------------------------------------------------------')
        save_results(self.args, 'edge_pred',
                     {k: dict(v) for k, v in metrics_dict.items()},
                     {'histories': self.histories})

    def node_cls_train_eval(self,data):

        metrics_dict=defaultdict(list)
        self.histories = []

        for seed in range(self.args.num_seeds):
            
            fix_seed(seed) 
            
            masks=data.generate_random_split(train_ratio=self.args.train_prop,val_ratio=self.args.valid_prop,seed=seed)

            if self.args.is_perturbed:
                if self.args.pert_mode in ['spar_label','flip_label']:
                    if self.args.pert_mode == 'spar_label':
                        masks = perturbation(data,mode=self.args.pert_mode,p=self.args.pert_p,masks=masks)
                    elif self.args.pert_mode == 'flip_label':
                        data = perturbation(data,mode=self.args.pert_mode,p=self.args.pert_p,masks=masks)
                    else:
                        raise ValueError('Unimplemented perturbation mode for label robustness')

            model = parse_model(self.args,data)
            if self.args.method == 'TMPHN':
                pass
            else:
                model = model.to(self.args.device)

            self.trainer.training(model,data,self.args,seed_split=masks,task_type='node_cls')
            self.histories.append({'seed': seed, 'history': self.trainer.history})

            self.train_times.append(self.trainer.train_time)

            # Evasion Attack
            if self.args.is_perturbed and not self.args.is_poison:
                test_data = data.evasion_data
            else:
                test_data = data

            if self.args.eval_verbose:
                print(f'------------------------------[Seed {seed}]-----------------------------------')
                result=self.evaluator.evaluate(model,test_data,seed_split=masks,task_type='node_cls',verbose=True)
                print(f'------------------------------------------------------------------------------')
            else:
                result=self.evaluator.evaluate(model,test_data,seed_split=masks,task_type='node_cls',verbose=False)
            
            for m in result:
                metrics_dict[m].append(result[m])
            
        print(f'---------------------------------[Final]--------------------------------------')
        self.test_dict = defaultdict(list) 
        for m in metrics_dict:
            result_printer(metrics_dict[m],m)
            metrics_mean, metrics_std = mean_std_metrics(metrics_dict[m])
            self.test_dict[m].extend([metrics_mean[-1],metrics_std[-1]])
        print(f'Avg Training Time: {np.mean(self.train_times):2f}')
        print(f'------------------------------------------------------------------------------')
        save_results(self.args, 'node_cls', metrics_dict,
                     {'summary': self.test_dict,
                      'train_time': float(np.mean(self.train_times)),
                      'histories': self.histories})

    def hg_cls_train_eval(self,data):

        metrics_dict=defaultdict(list)
        self.histories = []

        for seed in range(self.args.num_seeds):
            
            fix_seed(seed) 
            
            train_set,val_set,test_set = generate_split_hypergraphs(data,self.args.train_prop,self.args.valid_prop,seed)
            batch_loaders = generate_hg_loaders(train_set,val_set,test_set,self.args)
            
            self.args.embedding_mode = True 
            encoder = parse_model(self.args,data)

            model = HyperGPredictor(encoder,data.num_classes,self.args)
            if self.args.method == 'TMPHN':
                model.classifer = model.aggregator.to(self.args.device)
            else:
                model = model.to(self.args.device)
                     
            model = self.trainer.training(model,data,self.args,seed_split=batch_loaders,task_type='hg_cls')
            self.histories.append({'seed': seed, 'history': self.trainer.history})

            if self.args.eval_verbose:
                print(f'------------------------------[Seed {seed}]-----------------------------------')
                result=self.evaluator.evaluate(model,data,seed_split=batch_loaders,task_type='hg_cls',verbose=True)
                print(f'------------------------------------------------------------------------------')
            else:
                result=self.evaluator.evaluate(model,data,seed_split=batch_loaders,task_type='hg_cls',verbose=False)

            for m in result:
                metrics_dict[m].append(result[m])

        print(f'---------------------------------[Final]--------------------------------------')
        for m in metrics_dict:
            result_printer(metrics_dict[m],m)
        print(f'------------------------------------------------------------------------------')
        save_results(self.args, 'hg_cls', metrics_dict,
                     {'histories': self.histories})
        
    def running(self,task_type,data):
        
        if task_type == 'node_cls':
            self.node_cls_train_eval(data)
        elif task_type == 'edge_pred':
            self.edge_pred_train_eval(data)
        elif task_type == 'hg_cls':
            self.hg_cls_train_eval(data)
        else:
            raise NotImplementedError

def parse_model(args, data):
    
    if args.embedding_mode:
        num_targets=args.embedding_hidden
    else:
        num_targets=data.num_classes
    
    # --------- Hypergraph Semi-supervised Models --------------------
    
    if args.method == 'AllSetformer':
        if args.LearnMask:
            model = SetGNN(data.num_features, num_targets, args, data.norm)
        else:
            model = SetGNN(data.num_features, num_targets, args)
    elif args.method == 'AllDeepSets':
        args.PMA = False 
        args.aggregate = 'add'
        if args.LearnMask:
            model = SetGNN(data.num_features, num_targets, args, data.norm)
        else:
            model = SetGNN(data.num_features, num_targets, args)
    elif args.method in ['HGNN','HCHA']:
        model = HCHA(data.num_features, num_targets, args)
    elif args.method == 'HNHN':
        model = HNHN(data.num_features, num_targets, args)
    elif args.method in ['UniGIN']:
        model = UniGNN(data.num_features, num_targets, args)
    elif args.method == 'UniGCNII':
        model = UniGCNII(data.num_features, num_targets, args)
    elif args.method == 'HyperGCN':
        model = HyperGCN(data.num_features, num_targets, args)
    elif args.method == 'LEGCN':
        model = LEGCN(data.num_features, num_targets, args)
    elif args.method == 'HJRL':
        model = HJRL(data.num_features, num_targets, args)
    elif args.method == 'HyperND':
        model = HyperND(data.num_features, num_targets, args)
    elif args.method == 'EDHNN':
        model = EquivSetGNN(data.num_features, num_targets, args)
    elif args.method == 'SheafHyperGNN':
        model = SheafHyperGNN(data.num_features,num_targets,args)
    elif args.method == 'EHNN':
        model = EHNN(data.num_features,num_targets,args,data.ehnn_cache)
    elif args.method == 'TMPHN':
        model = TMPHN(data.num_features,num_targets,data.x,data.neig_dict,args)
    elif args.method == 'PhenomNNS':
        model = PhenomNNS(data.num_features,num_targets,args)
    elif args.method == 'PhenomNN':
        model = PhenomNN(data.num_features,num_targets,args)
    elif args.method == 'DPHGNN':
        model = DPHGNN(data.num_features,num_targets,args)
    elif args.method == 'PlainUnigencoder':
        model = PlainUnigencoder(data.num_features, num_targets, args)
    elif args.method == 'TFHNN':
        model = TFHNN(data.num_features,num_targets,args)
    elif args.method == 'MLP':
        model = PlainMLP(data.num_features,num_targets,args)
    elif args.method == 'HyperGT':
        model = HyperGT(data.num_features,num_targets,args)
    elif args.method == 'CEGCN':
        model = CEGCN(data.num_features,num_targets,args)
    elif args.method == 'CEGAT':
        model = CEGAT(data.num_features,num_targets,args)
    else:
        raise ValueError('Unimplemented model')

    return model
