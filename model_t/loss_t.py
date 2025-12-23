import torch.nn as nn
import torch.nn.functional as F
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class coding_feat_epoch_entropy_2(nn.Module):
    def __init__(self, alpha=1.0, use_cross_entropy_l1_loss_epochs=400):
        super(coding_feat_epoch_entropy_2, self).__init__()
        self.l1_loss = nn.L1Loss()  # L1 损失
        self.cross_entropy_loss = nn.CrossEntropyLoss()  # 交叉熵损失
        self.alpha = alpha  # L1 损失的权重
        self.use_cross_entropy_l1_loss_epochs = use_cross_entropy_l1_loss_epochs  # 控制何时使用 cross_entropy_l1_loss
        self.mse_loss = nn.MSELoss()
    def kl_div(self, teacher_logits, student_logits, T = 3.0):
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)
        log_soft_student = F.log_softmax(student_logits / T, dim=-1)


        # KL散度损失
        kld_loss = F.kl_div(log_soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
        return  kld_loss
    def count_entropy_loss(self, indices):
        # probabilities = F.softmax(logits, dim=-1)
        # probabilities = torch.clamp(probabilities, min=1e-10, max=1.0)
        unique, counts = torch.unique(indices, return_counts=True)
        probs = counts.float() / counts.sum()
        entropy = -torch.sum(probs * torch.log2(probs + 1e-9))
        return entropy.item()

    def forward(self, indices, z_likelihoods, label, classify_result_before, classify_result_after, feat,
                feat_hat, epoch):
        # 计算 L1 损失
        l1_loss = self.l1_loss(feat, feat_hat)
        cross_entropy = self.cross_entropy_loss(classify_result_after, label)
        kld_loss = 0
        entropy_loss = self.count_entropy_loss(indices)
        rate_cross_entropy = 2
        rate_l1_loss = 20
        rate_kld_loss = 1
        rate_entropy_loss = 1
        total_loss = (
                    cross_entropy * rate_cross_entropy + rate_l1_loss * l1_loss + kld_loss * rate_kld_loss + rate_entropy_loss * entropy_loss)

        return {
            "total_loss": total_loss,
            "cross_entropy_clsresult": cross_entropy * rate_cross_entropy,
            "l1_loss": l1_loss * rate_l1_loss,
            "kld_loss": kld_loss * rate_kld_loss,
            "entropy_loss": rate_entropy_loss * entropy_loss
            # "cross_entropy_l1_loss": cross_entropy_l1_loss * rate_cross_entropy_l1_loss,
            # "KL_div": rate_KL_div * KL_div
        }