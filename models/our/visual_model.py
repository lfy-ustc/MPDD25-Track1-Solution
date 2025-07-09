import torch
import os
import json
from collections import OrderedDict
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.lstm import LSTMEncoder
from models.networks.classifier import FcClassifier
from models.utils.config import OptConfig
import math
import torch.nn as nn


class ourModel(BaseModel, nn.Module):

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        nn.Module.__init__(self)
        super().__init__(opt)

        
        self.loss_names = []
        self.model_names = []

        # visual model
        self.netEmoV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        self.model_names.append('EmoV')

        # Transformer Fusion model
        emo_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=opt.embd_size_v, nhead=int(opt.Transformer_head), batch_first=True)
        self.netEmoFusion = torch.nn.TransformerEncoder(emo_encoder_layer, num_layers=opt.Transformer_layers)
        self.model_names.append('EmoFusion')

        # Classifier
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))

        # cls_input_size = 5*opt.hidden_size, same with max-len
        cls_input_size = opt.feature_max_len * opt.embd_size_v + 1024  # with personalized feature

                                            
        self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoC')
        self.loss_names.append('emo_CE')

        self.netEmoCF = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoCF')
        self.loss_names.append('EmoF_CE')

        self.temperature = opt.temperature


        # self.device = 'cpu'
        # self.netEmoA = self.netEmoA.to(self.device)
        # self.netEmoV = self.netEmoV.to(self.device)
        # self.netEmoFusion = self.netEmoFusion.to(self.device)
        # self.netEmoC = self.netEmoC.to(self.device)
        # self.netEmoCF = self.netEmoCF.to(self.device)

        self.criterion_ce = torch.nn.CrossEntropyLoss()

        if self.isTrain:
            if not opt.use_ICL:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                self.criterion_focal = torch.nn.CrossEntropyLoss() 
            else:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                self.criterion_focal = Focal_Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.ce_weight = opt.ce_weight
            self.focal_weight = opt.focal_weight

        # modify save_dir
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)  


    def post_process(self):
        # called after model.setup()
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

        if self.isTrain:
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            self.netEmoA.load_state_dict(f(self.pretrained_encoder.netEmoA.state_dict()))
            self.netEmoV.load_state_dict(f(self.pretrained_encoder.netEmoV.state_dict()))
            self.netEmoFusion.load_state_dict(f(self.pretrained_encoder.netEmoFusion.state_dict()))

    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

    def set_input(self, input):

        self.acoustic = input['A_feat'].float().to(self.device)
        self.visual = input['V_feat'].float().to(self.device)

        self.emo_label = input['emo_label'].to(self.device)

        if 'personalized_feat' in input:
            self.personalized = input['personalized_feat'].float().to(self.device)
        else:
            self.personalized = None  # if no personalized features given
            

    def forward(self, acoustic_feat=None, visual_feat=None):
        if acoustic_feat is not None:
            self.acoustic = acoustic_feat.float().to(self.device)
            self.visual = visual_feat.float().to(self.device)
        
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        emo_feat_V = self.netEmoV(self.visual)

        '''insure time dimension modification'''
        emo_fusion_feat = emo_feat_V # (batch_size, seq_len, 2 * embd_size)
        
        emo_fusion_feat = self.netEmoFusion(emo_fusion_feat)
        
        '''dynamic acquisition of bs'''
        batch_size = emo_fusion_feat.size(0)

        emo_fusion_feat = emo_fusion_feat.permute(1, 0, 2).reshape(batch_size, -1)  # turn into [batch_size, feature_dim] 1028

        if self.personalized is not None:
            emo_fusion_feat = torch.cat((emo_fusion_feat, self.personalized), dim=-1)  # [batch_size, seq_len * feature_dim + 1024]
        self.hidden_state = emo_fusion_feat
        '''for back prop'''
        self.emo_logits_fusion, _ = self.netEmoCF(emo_fusion_feat)
        """-----------"""

        self.emo_logits, _ = self.netEmoC(emo_fusion_feat)
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)

    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_emo_CE = self.criterion_ce(self.emo_logits, self.emo_label) 
        self.loss_EmoF_CE = self.focal_weight * self.criterion_focal(self.emo_logits_fusion, self.emo_label)
        loss = self.loss_emo_CE + self.loss_EmoF_CE

        loss.backward()

        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        # backward
        self.optimizer.zero_grad()
        self.backward()

        self.optimizer.step()


class ActivateFun(torch.nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.7, gamma=3.0, reduction='mean'):
        super(Focal_Loss, self).__init__()
        # alpha 可以是标量或向量
        if isinstance(alpha, (list, tuple, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = float(alpha)  # 标量
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, targets):
        # preds: [batch_size, num_classes], logits
        # targets: [batch_size], 类别索引
        num_classes = preds.size(1)
        
        # 如果 alpha 是向量，确保其长度与类别数匹配
        if isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha.to(preds.device)
            assert self.alpha.size(0) == num_classes, f"alpha length {self.alpha.size(0)} must match num_classes {num_classes}"
        
        # 计算 softmax 概率
        probs = F.softmax(preds, dim=1)  # [batch_size, num_classes]
        
        # 获取目标类别的概率 p_t
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()  # [batch_size, num_classes]
        pt = (probs * targets_one_hot).sum(dim=1)  # [batch_size], 目标类别概率
        
        # 计算交叉熵损失（逐样本）
        ce_loss = F.cross_entropy(preds, targets, reduction='none')  # [batch_size]
        
        # 处理 alpha
        if isinstance(self.alpha, torch.Tensor):
            # 获取每个样本对应的 alpha 值
            alpha_t = self.alpha[targets]  # [batch_size]
        else:
            alpha_t = self.alpha  # 标量，直接广播
        
        # 计算 Focal Loss
        focal_weight = alpha_t * (1 - pt) ** self.gamma  # [batch_size]
        focal_loss = focal_weight * ce_loss  # [batch_size]
        
        # 根据 reduction 返回
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")