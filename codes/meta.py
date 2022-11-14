import copy
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from dataset import get_pos_neg_triplets
from models import EncoderDecoder
from utils import trainable_params_stats, EarlyStopping
import logging
logger = logging.getLogger('log')


class MetaLearning(nn.Module):
    def __init__(self, args,
                 inner_lr,
                 outer_lr,
                 update_step,
                 update_step_test
                 ):
        super(MetaLearning, self).__init__()
        self.args = args
        self.outer_lr = outer_lr  # outer-loop lr
        self.inner_lr = inner_lr  # inner-loop lr
        self.update_step = update_step  # meta update steps
        self.update_step_test = update_step_test  # finetuning update steps
        self.device = self.args.device

        if self.args.binary:
            num_rels = 3
        else:
            if self.args.tgt_flag:
                num_rels = self.args.num_rels + 2
            else:
                num_rels = self.args.num_rels + 1

        self.model = EncoderDecoder(args=self.args,
                                    dim=self.args.dim,
                                    num_nodes=self.args.num_nodes,
                                    num_rels=num_rels,
                                    num_layers=self.args.num_layers,
                                    reg_param=self.args.reg_param,
                                    aux_param=self.args.aux_param,
                                    pretrained_node_feats=None)
        # meta-level optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr, weight_decay=1e-4)

        trainable_params_stats(self.model)

    def run(self, train_loader, test_loader):
        save_path = '{}/checkpoint_{}.pt'.format(self.args.ckpt_dir, self.args.save_postfix)
        start = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, metrics=self.args.metrics,
                                       save_path=save_path)
        best_log = None
        for step in range(self.args.max_steps):
            if step % self.args.num_batches == 0:
                t0 = time.time()
                train_loader.dataset.sample()
                logger.info('Data  | Sample {} tasks in {:.04} s'.format(self.args.num_batches, time.time() - t0))

            # train
            t1 = time.time()
            self.model.train()
            input_graphs, tgt_graphs, support_edges, query_edges, tgt_events = next(iter(train_loader))
            batch_support_loss, batch_support_metrics, batch_query_loss, batch_query_metrics = self.update(input_graphs,
                                                                                                           tgt_graphs,
                                                                                                           support_edges,
                                                                                                           query_edges,
                                                                                                           tgt_events)

            if (step + 1) % 5 == 0:
                logger.info('Train | Step {:04d} | Qry Loss:{:.04f}={:.04f}+{:.04f}+{:.04f} | [{:.04f},{:.04f},{:.04f},'
                      '{:.04f},{:.04f},{:.04f}] | Spt Loss:{:.04f}={:.04f}+{:.04f}+{:.04f} | [{:.04f},{:.04f},{:.04f},'
                      '{:.04f},{:.04f},{:.04f}] | Time(s):{:.04f}'.
                      format(step, batch_query_loss['loss'], batch_query_loss['tgt_loss'], batch_query_loss['aux_loss'],
                             batch_query_loss['reg_loss'], batch_query_metrics['auc'], batch_query_metrics['ap'],
                             batch_query_metrics['f1'], batch_query_metrics['p'], batch_query_metrics['r'],
                             batch_query_metrics['acc'], batch_support_loss['loss'], batch_support_loss['tgt_loss'],
                             batch_support_loss['aux_loss'], batch_support_loss['reg_loss'],
                             batch_support_metrics['auc'], batch_support_metrics['ap'], batch_support_metrics['f1'],
                             batch_support_metrics['p'], batch_support_metrics['r'], batch_support_metrics['acc'],
                             time.time() - t1))
            # validation
            if (step + 1) % self.args.valid_freq == 0:
                log = self.test(loader=test_loader, mode='Test', step=step)
                early_stopping(log=log, model=self.model)
                logger.info('----------End Validation At Episode {}----------'.format(step))
                if early_stopping.counter == 0 or best_log is None:
                    best_log = log
                if early_stopping.early_stop:
                    logger.info('Early Stop at Step {}'.format(step))
                    logger.info('Best Metrics | [{:.04f},{:.04f},{:.04f},{:.04f},{:.04f},{:.04f}] '.format(
                        best_log['auc'], best_log['ap'], best_log['f1'], best_log['p'], best_log['r'], best_log['acc']))
                    break
        end = time.time()
        logger.info('Time  | Total Time {:.6f}s'.format(end - start))

    def test(self, loader, mode, step):
        t = time.time()

        support_loss_list = []
        test_query_loss = {'loss': [], 'tgt_loss': [], 'aux_loss': [], 'reg_loss': []}
        test_query_metrics = {'auc': [], 'f1': [], 'ap': [], 'p': [], 'r': [], 'acc': []}

        for idx, data in enumerate(loader):
            input_graphs, tgt_graphs, support_edges, valid_edges, query_edges, tgt_events = data

            support_loss, batch_query_loss, batch_query_metrics = self.finetuning(input_graphs, tgt_graphs,
                                                                                  support_edges, valid_edges,
                                                                                  query_edges, tgt_events)
            test_query_loss['loss'].append(batch_query_loss['loss'])
            test_query_loss['tgt_loss'].append(batch_query_loss['tgt_loss'])
            test_query_loss['aux_loss'].append(batch_query_loss['aux_loss'])
            test_query_loss['reg_loss'].append(batch_query_loss['reg_loss'])

            test_query_metrics['auc'].append(batch_query_metrics['auc'])
            test_query_metrics['ap'].append(batch_query_metrics['ap'])
            test_query_metrics['f1'].append(batch_query_metrics['f1'])
            test_query_metrics['p'].append(batch_query_metrics['p'])
            test_query_metrics['r'].append(batch_query_metrics['r'])
            test_query_metrics['acc'].append(batch_query_metrics['acc'])

            support_loss_list.append(support_loss)

        for k, v in test_query_loss.items():
            test_query_loss[k] = np.mean(v)
        for k, v in test_query_metrics.items():
            test_query_metrics[k] = (np.mean(v), np.std(v))
        support_loss = np.mean(support_loss_list)
        logger.info('{} | Step {:04d} | Qry Loss:{:.04f}={:.04f}+{:.04f}+{:.04f} | AUC:{:.04f}±{:.04f} | AP:{:.04f}±{:.04f}'
              ' | F1:{:.04f}±{:.04f} | P:{:.04f}±{:.04f} | R:{:.04f}±{:.04f}'
              ' | ACC:{:.04f}±{:.04f} | Spt Loss:{:.04f} | Time(s):{:.04f}'.
              format(mode, step, test_query_loss['loss'], test_query_loss['tgt_loss'], test_query_loss['aux_loss'],
                     test_query_loss['reg_loss'], test_query_metrics['auc'][0], test_query_metrics['auc'][1],
                     test_query_metrics['ap'][0], test_query_metrics['ap'][1], test_query_metrics['f1'][0],
                     test_query_metrics['f1'][1], test_query_metrics['p'][0], test_query_metrics['p'][1],
                     test_query_metrics['r'][0], test_query_metrics['r'][1], test_query_metrics['acc'][0],
                     test_query_metrics['acc'][1], support_loss, time.time() - t))

        log = {'auc': test_query_metrics['auc'][0], 'ap': test_query_metrics['ap'][0],
               'f1': test_query_metrics['f1'][0], 'p': test_query_metrics['p'][0], 'r': test_query_metrics['r'][0],
               'acc': test_query_metrics['acc'][0]}
        return log

    def update(self, input_graphs, tgt_graphs, support_edges, query_edges, tgt_events):
        update_step = self.update_step
        num_tasks = len(input_graphs)

        episode_loss = 0

        batch_support_loss = {'loss': 0, 'tgt_loss': 0, 'aux_loss': 0, 'reg_loss': 0}
        batch_support_metrics = {'auc': 0, 'f1': 0, 'ap': 0, 'p': 0, 'r': 0, 'acc': 0}

        batch_query_loss = {'loss': 0, 'tgt_loss': 0, 'aux_loss': 0, 'reg_loss': 0}
        batch_query_metrics = {'auc': 0, 'f1': 0, 'ap': 0, 'p': 0, 'r': 0, 'acc': 0}
        torch.autograd.set_detect_anomaly(True)

        for i in range(num_tasks):
            _input_graph = input_graphs[i].to(self.device, non_blocking=True)
            _tgt_graph = tgt_graphs[i].to(self.device, non_blocking=True)
            _support_edges = support_edges[i]
            _query_edges = query_edges[i]
            _tgt_event = tgt_events[i]

            _support_triplets = _support_edges[0].to(self.device, non_blocking=True)
            _support_labels = _support_edges[1].to(self.device, non_blocking=True)

            _query_triplets = _query_edges[0].to(self.device, non_blocking=True)
            _query_labels = _query_edges[1].to(self.device, non_blocking=True)

            fast_weights = OrderedDict(self.model.named_parameters())  # original params from the model

            for k in range(0, update_step + 1):
                # on the SUPPORT set
                support_loss, support_loss_dict, support_tgt_scores, support_tgt_labels = self.model.forward(
                    input_graphs=_input_graph, feature_graphs=_tgt_graph, triplets=_support_triplets,
                    labels=_support_labels, tgt_event=_tgt_event, vars=fast_weights)

                support_metrics = self.model.evaluate(scores=support_tgt_scores, labels=support_tgt_labels)

                grads = torch.autograd.grad(support_loss, fast_weights.values(), allow_unused=True,
                                            retain_graph=True)

                # update parameters manually with the gradients from SUPPORT set
                if self.args.clip_weight_val > 0:
                    fast_weights = OrderedDict(
                        (name, torch.clamp((param - self.inner_lr * grad),
                                           -self.args.clip_weight_val, self.args.clip_weight_val))
                        for ((name, param), grad) in zip(fast_weights.items(), grads)
                    )
                else:
                    fast_weights = OrderedDict(
                        (name, param - self.inner_lr * grad) for ((name, param), grad) in
                        zip(fast_weights.items(), grads)
                    )

                if k == update_step:
                    batch_support_loss['loss'] += support_loss_dict['loss'] / num_tasks
                    batch_support_loss['aux_loss'] += support_loss_dict['aux_loss'] / num_tasks
                    batch_support_loss['tgt_loss'] += support_loss_dict['tgt_loss'] / num_tasks
                    batch_support_loss['reg_loss'] += support_loss_dict['reg_loss'] / num_tasks

                    batch_support_metrics['auc'] += support_metrics['auc'] / num_tasks
                    batch_support_metrics['ap'] += support_metrics['ap'] / num_tasks
                    batch_support_metrics['f1'] += support_metrics['f1'] / num_tasks
                    batch_support_metrics['p'] += support_metrics['p'] / num_tasks
                    batch_support_metrics['r'] += support_metrics['r'] / num_tasks
                    batch_support_metrics['acc'] += support_metrics['acc'] / num_tasks

                # on QUERY set with updated params
                if k == update_step:
                    query_loss, query_loss_dict, query_tgt_scores, query_tgt_labels = self.model.forward(
                        input_graphs=_input_graph, feature_graphs=_tgt_graph, triplets=_query_triplets,
                        labels=_query_labels, tgt_event=_tgt_event, vars=fast_weights)

                    query_metrics = self.model.evaluate(scores=query_tgt_scores, labels=query_tgt_labels)

                    # if k == update_step:
                    batch_query_loss['loss'] += query_loss_dict['loss'] / num_tasks
                    batch_query_loss['aux_loss'] += query_loss_dict['aux_loss'] / num_tasks
                    batch_query_loss['tgt_loss'] += query_loss_dict['tgt_loss'] / num_tasks
                    batch_query_loss['reg_loss'] += query_loss_dict['reg_loss'] / num_tasks

                    batch_query_metrics['auc'] += query_metrics['auc'] / num_tasks
                    batch_query_metrics['ap'] += query_metrics['ap'] / num_tasks
                    batch_query_metrics['f1'] += query_metrics['f1'] / num_tasks
                    batch_query_metrics['p'] += query_metrics['p'] / num_tasks
                    batch_query_metrics['r'] += query_metrics['r'] / num_tasks
                    batch_query_metrics['acc'] += query_metrics['acc'] / num_tasks

                    episode_loss += query_loss
                    break

        episode_loss = episode_loss / num_tasks
        self.optimizer.zero_grad()
        episode_loss.backward()
        self.optimizer.step()

        return batch_support_loss, batch_support_metrics, batch_query_loss, batch_query_metrics

    def finetuning(self, input_graphs, tgt_graphs, support_edges, valid_edges, query_edges, tgt_events):
        if self.args.fast_adaption_flag:
            update_step = self.update_step_test
            valid_freq = 1
            patience = 1000
        else:
            update_step = self.args.max_fine_tuning_steps
            valid_freq = 60
            patience = 20
        assert len(input_graphs) == 1

        batch_query_curve = {'auc':[], 'ap':[], 'acc':[]}

        start = time.time()

        batch_query_loss = {'loss': 0, 'tgt_loss': 0, 'aux_loss': 0, 'reg_loss': 0}
        batch_query_metrics = {'auc': 0, 'f1': 0, 'ap': 0, 'p': 0, 'r': 0, 'acc': 0}

        torch.autograd.set_detect_anomaly(True)

        _input_graph = input_graphs[0].to(self.device, non_blocking=True)
        _tgt_graph = tgt_graphs[0].to(self.device, non_blocking=True)
        _support_pos_triplets = support_edges[0]
        _valid_pos_triplets = valid_edges[0]
        _query_pos_triplets = query_edges[0]
        _tgt_event = tgt_events[0]

        if self.args.binary:
            _tgt_event_idx = 1
        else:
            if self.args.tgt_flag:
                _tgt_event_idx = self.args.num_rels + 1
            else:
                _tgt_event_idx = self.args.event2idx[_tgt_event]

        _valid_triplets, _valid_labels = get_pos_neg_triplets(args=self.args, mode='valid',
                                                              triplets=_valid_pos_triplets,
                                                              tgt_event_idx=_tgt_event_idx)
        _valid_triplets, _valid_labels = _valid_triplets.to(self.device), _valid_labels.to(self.device)

        _query_triplets, _query_labels = get_pos_neg_triplets(args=self.args, mode='query',
                                                              triplets=_query_pos_triplets,
                                                              tgt_event_idx=_tgt_event_idx)
        _query_triplets, _query_labels = _query_triplets.to(self.device), _query_labels.to(self.device)

        model = copy.deepcopy(self.model)  # DO NOT operate on the original model!

        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.outer_lr, weight_decay=1e-4)
        fast_weights = OrderedDict(model.named_parameters())  # original params from the model

        test_save_path = '{}/test_ckpt_{}.pt'.format(self.args.ckpt_dir, self.args.save_postfix)
        verbose = True if not self.args.fast_adaption_flag else False
        task_early_stopping = EarlyStopping(patience=patience, verbose=verbose, metrics=self.args.metrics,
                                            save_path=test_save_path)
        logger.info('----------Start Fine-Tuning for Event {}----------'.format(_tgt_event))

        for k in range(0, update_step + 1):
            model.train()

            _support_triplets, _support_labels = get_pos_neg_triplets(args=self.args, mode='support',
                                                                      triplets=_support_pos_triplets,
                                                                      tgt_event_idx=_tgt_event_idx)
            _support_triplets, _support_labels = _support_triplets.to(self.device), _support_labels.to(
                self.device)

            support_loss, support_loss_dict, support_tgt_scores, support_tgt_labels = model(input_graphs=_input_graph,
                                                                                            feature_graphs=_tgt_graph,
                                                                                            triplets=_support_triplets,
                                                                                            labels=_support_labels,
                                                                                            tgt_event=_tgt_event,
                                                                                            vars=fast_weights)

            support_metrics = model.evaluate(scores=support_tgt_scores, labels=support_tgt_labels)

            optimizer.zero_grad()
            support_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip gradients
            optimizer.step()

            # on VALID set with updated params
            if k % valid_freq == 0 or k == update_step:
                model.eval()
                with torch.no_grad():
                    _, valid_loss_dict, valid_tgt_scores, valid_tgt_labels = model.forward(input_graphs=_input_graph,
                                                                                           feature_graphs=_tgt_graph,
                                                                                           triplets=_valid_triplets,
                                                                                           labels=_valid_labels,
                                                                                           tgt_event=_tgt_event,
                                                                                           vars=fast_weights)

                    valid_metrics = model.evaluate(scores=valid_tgt_scores, labels=valid_tgt_labels)

                with torch.no_grad():
                    _, query_loss_dict, query_tgt_scores, query_tgt_labels = model.forward(input_graphs=_input_graph,
                                                                                           feature_graphs=_tgt_graph,
                                                                                           triplets=_query_triplets,
                                                                                           labels=_query_labels,
                                                                                           tgt_event=_tgt_event,
                                                                                           vars=fast_weights)

                    query_metrics = model.evaluate(scores=query_tgt_scores, labels=query_tgt_labels)

                    if self.args.fast_adaption_flag:
                        batch_query_curve['auc'].append(query_metrics['auc'])
                        batch_query_curve['ap'].append(query_metrics['ap'])
                        batch_query_curve['acc'].append(query_metrics['acc'])

                task_early_stopping(log=valid_metrics, model=model)
                if task_early_stopping.counter == 0:
                    for _k, _v in query_loss_dict.items():
                        batch_query_loss[_k] = _v
                    for _k, _v in query_metrics.items():
                        batch_query_metrics[_k] = _v
                if not self.args.fast_adaption_flag:
                    logger.info('Fine-Tuning | k={:05d} | Valid Loss:{:.04f} |[{:.04f},{:.04f},{:.04f},{:.04f},{:.04f},{:.04f}] | Query Loss:'
                      '{:.04f}={:.04f}+{:.04f}+{:.04f} |[{:.04f},{:.04f},{:.04f},{:.04f},{:.04f},{:.04f}] |'
                      ' Spt Loss:{:.04f}={:.04f}+{:.04f}+{:.04f} |[{:.04f},{:.04f},{:.04f},{:.04f},{:.04f},{:.04f}]'.
                    format(
                    k, valid_loss_dict['loss'], valid_metrics['auc'], valid_metrics['ap'], valid_metrics['f1'],
                    valid_metrics['p'], valid_metrics['r'], valid_metrics['acc'],
                    query_loss_dict['loss'], query_loss_dict['tgt_loss'], query_loss_dict['aux_loss'],
                    query_loss_dict['reg_loss'], query_metrics['auc'], query_metrics['ap'], query_metrics['f1'],
                    query_metrics['p'], query_metrics['r'], query_metrics['acc'],
                    support_loss_dict['loss'], support_loss_dict['tgt_loss'], support_loss_dict['aux_loss'],
                    support_loss_dict['reg_loss'], support_metrics['auc'], support_metrics['ap'], support_metrics['f1'],
                    support_metrics['p'], support_metrics['r'], support_metrics['acc']))

                if task_early_stopping.early_stop:
                    break
        del model
        del optimizer
        logger.info('Split Test | k={:05d} | Qry Loss:{:.04f}={:.04f}+{:.04f}+{:.04f} | AUC:{:.04f} | AP:{:.04f} | '
              'F1:{:.04f} | Spt Loss:{:.04f} | Time(s):{:.04f}'.
              format(k, batch_query_loss['loss'], batch_query_loss['tgt_loss'], batch_query_loss['aux_loss'],
                     batch_query_loss['reg_loss'], batch_query_metrics['auc'], batch_query_metrics['ap'],
                     batch_query_metrics['f1'], batch_query_metrics['p'], batch_query_metrics['r'],
                     batch_query_metrics['acc'], support_loss_dict['loss'], time.time() - start))
        if self.args.fast_adaption_flag:
            for _k,_v in batch_query_curve.items():
                logger.info('Metrics:{}, Length:{}'.format(_k,len(_v)))
                logger.info(_v)
            step_array = [10,20,50,100]
            for _k,_v in batch_query_curve.items():
                logger.info('Metrics:{}, Array:{}'.format(_k, [_v[_] for _ in step_array]))

        logger.info('----------End Fine-Tuning for Event {}----------'.format(_tgt_event))

        return support_loss_dict['loss'], batch_query_loss, batch_query_metrics
