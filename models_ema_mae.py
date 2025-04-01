import torch
import torch.nn as nn
from models_mae import MaskedAutoencoderViT, mae_vit_tiny_patch4

class EMAMAE(nn.Module):
    def __init__(self, model_name='mae_vit_tiny_patch4', ema_decay=0.999, norm_pix_loss=False):
        super().__init__()
        # 创建学生模型（原始MAE）
        self.student_model = mae_vit_tiny_patch4(norm_pix_loss=norm_pix_loss)
        
        # 创建目标模型（EMA版本）
        self.target_model = mae_vit_tiny_patch4(norm_pix_loss=norm_pix_loss)
        
        # 初始化目标模型参数与学生模型相同
        self.target_model.load_state_dict(self.student_model.state_dict())
        
        # 设置EMA衰减率
        self.ema_decay = ema_decay
        
        # 冻结目标模型的参数
        for param in self.target_model.parameters():
            param.requires_grad = False
            
    def update_target_model(self):
        """使用EMA更新目标模型参数"""
        for target_param, student_param in zip(self.target_model.parameters(), 
                                             self.student_model.parameters()):
            target_param.data.mul_(self.ema_decay).add_(student_param.data, 
                                                       alpha=1 - self.ema_decay)
    
    def forward(self, imgs, mask_ratio=0.75):
        # 使用学生模型进行预测
        latent, mask, ids_restore = self.student_model.forward_encoder(imgs, mask_ratio)
        pred = self.student_model.forward_decoder(latent, ids_restore)
        
        # 计算与原始图像的重建损失
        loss = self.student_model.forward_loss(imgs, pred, mask)
        
        # 更新目标模型
        self.update_target_model()
        
        return loss, pred, mask

def ema_mae_vit_tiny_patch4(**kwargs):
    model = EMAMAE(
        model_name='mae_vit_tiny_patch4',
        ema_decay=0.999,
        **kwargs)
    return model