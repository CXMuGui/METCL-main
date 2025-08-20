import torch
import torch.nn.functional as F
import logging
from torch import nn
from utils.util import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from data_process.utils import get_dataloader
from utils.metrics import AverageMeter, Metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from .model import METCL
from .loss import SupConLoss
import numpy as np
from src.DDER.DDER_model import DDER
from src.DDER.loss_function import get_loss

__all__ = ['METCL_manager']

class EvidenceProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EvidenceProjector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        return self.fc2(self.relu(x))
    
    

class METCL_manager:

    def __init__(self, args, data):
             
        self.logger = logging.getLogger(args.logger_name)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        args.device = self.device
        self.model = METCL(args)
        self.model.to(self.device)
        

        mm_dataloader = get_dataloader(args, data.mm_data)
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            mm_dataloader['train'], mm_dataloader['dev'], mm_dataloader['test']
            
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.cons_criterion = SupConLoss(temperature=args.temperature)
        self.metrics = Metrics(args)
        
        
        
        self.evidence_projector = EvidenceProjector(input_dim=20, hidden_dim=20, output_dim=768).to(self.device)
        self.lambda_param = nn.Parameter(torch.tensor(0.5))  # 初始值 0.5
        self.projector_evidence = nn.Linear(20, 768).to(self.device)
        

        if args.dataset == "MIntRec":

            out_dims = np.array([[37],[768],[256]])
            self.num_classes = 20
        if args.dataset == "MELD":

            out_dims = np.array([[76],[768],[256]])
            self.num_classes = 12
        self.num_classes = 20
        self.annealing_step = 80
        self.beta = 1
        self.gamma = 1
        self.delta = 1
        
        self.DDER = DDER(3, out_dims, self.num_classes, self.device)
        

        self.combined_model = nn.ModuleList([self.model, self.DDER])

        self.optimizer, self.scheduler = self._set_optimizer(args, self.combined_model)
        
        
        
        
        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)
            
    def _set_optimizer(self, args, model):
        
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, correct_bias=False)
        
        num_train_optimization_steps = int(args.num_train_examples / args.train_batch_size) * args.num_train_epochs
        num_warmup_steps= int(args.num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        
        return optimizer, scheduler
    
    
    

    def evidence_guided_contrastive_loss(self, condition, cons_condition, evidence_a, label_ids):

        condition_norm = F.normalize(condition, p=2, dim=1)  # [B, D]
        cons_condition_norm = F.normalize(cons_condition, p=2, dim=1)  # [B, D]
        

        pos_sim = torch.sum(condition_norm * cons_condition_norm, dim=1)  # [B]
        

        true_evidence = evidence_a[torch.arange(evidence_a.size(0)), label_ids]  # [B]

        evidence_sum = evidence_a.sum(dim=1) + 1e-8   # [B]
        weight = true_evidence / evidence_sum  # [B]
        

        loss = weight * (1.0 - pos_sim)  # [B]
        return loss.mean()

    
    
    

    def _train(self, args): 
        
        early_stopping = EarlyStopping(args)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()
            cons_loss_record = AverageMeter()
            cls_loss_record = AverageMeter()
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                cons_text_feats = batch['cons_text_feats'].to(self.device)
                condition_idx = batch['condition_idx'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)
                
                X = {
                    0: text_feats.float().to(self.device),
                    1: audio_feats.float().to(self.device),
                    2: video_feats.float().to(self.device)
                }
                

                with torch.set_grad_enabled(True):
                    
                    evidences, evidence_a, evidence_con, evidence_div =\
                        self.DDER(X, self.beta)
                    evidence_loss = get_loss(
                        evidences, 
                        evidence_a, 
                        evidence_con, 
                        evidence_div,
                        label_ids, 
                        epoch + 1, 
                        self.num_classes, 
                        self.annealing_step,
                        self.gamma, 
                        self.delta, 
                        self.device)
                    annealing_coef = min(1.0, float(epoch + 1) / self.annealing_step)

                    logits, _, condition, cons_condition = self.model(text_feats, video_feats, audio_feats, cons_text_feats, condition_idx)
                    
                    cons_feature = torch.cat((condition.unsqueeze(1), cons_condition.unsqueeze(1)), dim=1)

                    cons_loss = self.cons_criterion(cons_feature)

                    pro_evi = self.projector_evidence(evidence_a)
                    pro_evi = F.normalize(pro_evi, dim=1) 
                    new_cons_feature = torch.stack([
                        F.normalize(condition,    dim=1),           
                        F.normalize(cons_condition, dim=1),        
                        pro_evi                                   
                    ], dim=1)                                       

                    cons_loss = self.cons_criterion(new_cons_feature)
                    
                    
                    egcl_loss = self.evidence_guided_contrastive_loss(condition, cons_condition, evidence_a, label_ids,
                                             )

                    cls_loss = self.criterion(logits, label_ids)
                    loss = cls_loss + egcl_loss + annealing_coef * evidence_loss

                    self.optimizer.zero_grad()

                    
                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))
                    cons_loss_record.update(cons_loss.item(), label_ids.size(0))
                    cls_loss_record.update(cls_loss.item(), label_ids.size(0))

                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    self.optimizer.step()
                    self.scheduler.step()
                    
            
            outputs = self._get_outputs(args, self.eval_dataloader)
            eval_score = outputs[args.eval_monitor]

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'cons_loss': round(cons_loss_record.avg, 4),
                'cls_loss': round(cls_loss_record.avg, 4),
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

    def _get_outputs(self, args, dataloader, show_results = False):

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        total_features = torch.empty((0, args.feat_size)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            cons_text_feats = batch['cons_text_feats'].to(self.device)
            condition_idx = batch['condition_idx'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            X = {
                0: text_feats.float().to(self.device),
                1: audio_feats.float().to(self.device),
                2: video_feats.float().to(self.device)
            }
            evidences, evidence_a, evidence_con, evidence_div = self.DDER(X, self.beta)
            
            evidence_weight = F.softmax(evidence_a, dim=1)  

                
                
            with torch.set_grad_enabled(False):
                
                logits, features, condition, cons_condition = self.model(text_feats, video_feats, audio_feats, cons_text_feats, condition_idx)
                
                lambda_val = torch.sigmoid(self.lambda_param)  
                logits = lambda_val * logits + (1 - lambda_val) * (logits * evidence_weight)
                
                
                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, features))
                

                
                
                

        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_logit = total_logits.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_prob = total_maxprobs.cpu().numpy()
        y_feat = total_features.cpu().numpy()
        
        outputs = self.metrics(y_true, y_pred, show_results = show_results)
        
        if args.save_pred and show_results:
            np.save('y_true_' + str(args.seed) + '.npy', y_true)
            np.save('y_pred_' + str(args.seed) + '.npy', y_pred)

        outputs.update(
            {
                'y_prob': y_prob,
                'y_logit': y_logit,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_feat': y_feat
            }
        )

        return outputs

    def _test(self, args):
        
        test_results = {}
        
        ind_outputs = self._get_outputs(args, self.test_dataloader, show_results = True)
        if args.train:
            ind_outputs['best_eval_score'] = round(self.best_eval_score, 4)
        
        test_results.update(ind_outputs)
        
        return test_results