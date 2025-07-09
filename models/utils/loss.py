import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class Focal_Loss(torch.nn.Module):
    def __init__(self, weight=0.5, gamma=3, reduction='mean'):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = weight
        self.reduction = reduction

    def forward(self, preds, targets):
        """
        preds:softmax output
        labels:true values
        """
        ce_loss = F.cross_entropy(preds, targets, reduction='mean')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            raise NotImplementedError("Invalid reduction mode. Please choose 'none', 'mean', or 'sum'.")
        
def compute_qf(class_counts):
    """
    计算 QF Quantity Factor
    :param class_counts: Tensor (C,) -> 每个类别的样本数量
    :return: QF Tensor (C,)
    """
    min_count = class_counts.min().clamp(min=1)  # 避免除以零
    qf = 1 / torch.log(class_counts / min_count + 1)  # 计算 QF
    return qf

def compute_df_from_logits(logits, labels):
    """
    直接从 logits 计算 Difficulty Factor (DF)
    
    :param logits: Tensor of shape (B, num_classes)  -> 归一化后的 logits,相当于 cos(theta_ij)
    :param labels: Tensor of shape (B,)  -> 样本真实类别索引
    :return: DF Tensor of shape (B,)
    """
    # 选取真实类别的 cos(theta_iy)
    # print(logits)
    cos_theta_iy = logits.gather(1, labels.view(-1, 1)).squeeze(1)  # (B,)
    # cos_theta_iy = torch.clamp(cos_theta_iy, min=-1.0, max=1.0)  # 避免超出 [-1,1]
    # 计算 DF = (1 - cos_theta_iy) / 2
    df = (1 - cos_theta_iy) / 2
    return df

class ALALoss(torch.nn.Module):
    def __init__(self, s=30.0):
        """
        Adaptive Logit Adjustment (ALA) Loss
        :param s: Scaling factor for logits, similar to LDAM
        """
        super(ALALoss, self).__init__()
        self.s = s  # Logit scaling factor

    def forward(self, logits, labels, qf):
        """
        :param logits: Tensor of shape (B, C) -> model's output logits
        :param labels: Tensor of shape (B,) -> ground truth labels
        :param qf: Tensor of shape (C,) -> quantity factor for each class
        :return: Loss value
        """
        # 获取 batch_size 和 类别数
        B, C = logits.shape
        # 获取目标类别的 logits
        logit_y = logits.gather(1, labels.view(-1, 1)).squeeze(1)  # (B,)
        df = compute_df_from_logits(logits,labels)
        # 计算 AALA[y]，即 logit 调整量
        AALA_y = df * qf[labels]  # (B,)

        # 计算调整后的目标类别 logit
        adjusted_logit_y = logit_y - AALA_y  # (B,)

        # 计算 Softmax 分子：e^(s * adjusted_logit_y)
        numerator = torch.exp(self.s * adjusted_logit_y)  # (B,)

        # 计算 Softmax 分母：sum(e^(s * logits[j]))
        denominator = torch.exp(self.s * logits).sum(dim=1)  # (B,)

        # 计算 ALA Loss
        loss = -torch.log(numerator / denominator).mean()

        return loss

class RIDELoss(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True, reweight_epoch=-1, 
        base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        self.focal_loss = Focal_Loss()
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0
            
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        print("device:",device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)
        print(self.per_cls_weights_enabled_diversity)
        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0
        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)
            
            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature
            
            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)
            
            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')
        
        return loss

if __name__ == "__main__":
    input = torch.tensor([235,68,33])
    print(compute_qf(input))