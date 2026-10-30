import torch
import torch.nn.functional as F
import logging
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from data.base import get_dataloader
from utils.metrics import AverageMeter, Metrics
from .loss import LWFFTLoss
from .model import LW_FFT

__all__ = ['LW_FFT_manager']


class LW_FFT_manager:

    def __init__(self, args, data):
        self.args = args
        self.logger = logging.getLogger(args.logger_name)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        args.device = self.device
        self.model = LW_FFT(args)
        self.model.to(self.device)
        self.optimizer, self.scheduler = self._set_optimizer(args, self.model)
        self.grad_clip_norm = getattr(args, 'grad_clip_norm', 1.0)

        mm_dataloader = get_dataloader(args, data.mm_data)
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            mm_dataloader['train'], mm_dataloader['dev'], mm_dataloader['test']

        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = Metrics(args)

        self.loss_lw = LWFFTLoss(args)

        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)

    def _set_optimizer(self, args, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)

        num_train_optimization_steps = int(args.num_train_examples / args.train_batch_size) * args.num_train_epochs
        num_warmup_steps = int(
            args.num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        return optimizer, scheduler

    
    def _train(self, args):
        early_stopping = EarlyStopping(args)
        grad_clip_norm = getattr(args, 'grad_clip', 1.0)
        if isinstance(grad_clip_norm, (list, tuple)):
            grad_clip_norm = grad_clip_norm[0] if len(grad_clip_norm) > 0 else 1.0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()
            acc_record = AverageMeter()
            route_stats = {
                'text_weight': AverageMeter(),
                'video_weight': AverageMeter(),
                'audio_weight': AverageMeter(),
            }

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device).long()

                with torch.set_grad_enabled(True):
                    outputs = self.model(text_feats, video_feats, audio_feats)
                    logits = outputs['logits']

                    loss = self.loss_lw(outputs, label_ids)

                   
                    if torch.isnan(loss):
                        self.logger.warning(f"NaN loss detected at step {step}, skipping...")
                        continue

                    self.optimizer.zero_grad()
                    loss.backward()

                   
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)

                    preds = logits.argmax(dim=1)
                    acc = (preds == label_ids).float().mean()
                    acc_record.update(acc.item(), label_ids.size(0))
                    loss_record.update(loss.item(), label_ids.size(0))

                    self.optimizer.step()
                    self.scheduler.step()



            outputs = self._get_outputs(args, self.eval_dataloader)
            eval_score = outputs[args.eval_monitor]
            eval_acc = outputs.get('accuracy', 0.0)

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'train_acc': round(acc_record.avg, 4),
                'eval_acc': round(eval_acc, 4),
                'eval_score': round(eval_score, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))

            early_stopping(eval_score, self.model)

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)

    def _get_outputs(self, args, dataloader, show_results=False):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):
            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(text_feats, video_feats, audio_feats)
                logits = outputs['logits']

                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))

        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim=1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_prob = total_maxprobs.cpu().numpy()
        y_logit = torch.sigmoid(total_logits.detach()).cpu().numpy()

        outputs = self.metrics(y_true, y_pred, show_results=show_results)
        outputs['accuracy'] = (total_preds == total_labels).float().mean().item()
        outputs.update({
            'y_pred': y_pred,
            'y_true': y_true,
            'y_prob': y_prob,
            'y_logit': y_logit,
        })

        return outputs

    def _test(self, args):
        test_results = {}
        ind_test_results = self._get_outputs(args, self.test_dataloader, show_results=True)
        if args.train:
            test_results['best_eval_score'] = round(self.best_eval_score, 4)
        test_results.update(ind_test_results)
        return test_results
