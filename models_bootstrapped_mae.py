import torch
import torch.nn as nn
from models_mae import MaskedAutoencoderViT, PatchEmbed, Block, get_2d_sincos_pos_embed

class FeatureMAE(nn.Module):
    """用于特征重建的MAE模型"""
    def __init__(self, img_size=32, patch_size=4, in_chans=3,
                 embed_dim=192, depth=12, num_heads=3,
                 decoder_embed_dim=96, decoder_depth=4, decoder_num_heads=3,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        
        # 编码器部分
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # 解码器部分
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        # 输出层改为预测特征
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        # 初始化位置编码
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # 初始化其他参数
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
        
    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore
        
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        x = x + self.decoder_pos_embed
        
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        
        return x
        
    def forward_loss(self, target_features, pred_features, mask):
        #print("pred_features.shape, target_features.shape", pred_features.shape, target_features.shape)
        loss = (pred_features - target_features) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss
        
    def forward(self, imgs, target_features, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred_features = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(target_features, pred_features, mask)
        return loss, pred_features, mask

class BootstrappedMAE(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3,
                 embed_dim=192, depth=12, num_heads=3,
                 decoder_embed_dim=96, decoder_depth=4, decoder_num_heads=3,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 num_mae=2, current_mae_idx=0):
        super().__init__()
        
        # 根据索引选择模型类型
        if current_mae_idx == 0:
            # 第一个MAE使用原始结构
            self.model = MaskedAutoencoderViT(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth,
                decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
                norm_layer=norm_layer, norm_pix_loss=norm_pix_loss
            )
        else:
            # 后续MAE使用特征重建结构
            self.model = FeatureMAE(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth,
                decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
                norm_layer=norm_layer, norm_pix_loss=norm_pix_loss
            )
        
        # 记录当前MAE的索引
        self.current_mae_idx = current_mae_idx
        self.num_mae = num_mae
        
    def forward(self, imgs, target_features=None, mask_ratio=0.75):
        if self.current_mae_idx == 0:
            # 第一个MAE使用像素重建
            loss, pred, mask = self.model(imgs, mask_ratio)
            return loss, pred, mask
        else:
            # 后续MAE使用特征重建
            loss, pred_features, mask = self.model(imgs, target_features, mask_ratio)
            return loss, pred_features, mask

def bootstrapped_mae_tiny_patch4_dec96d4b(**kwargs):
    model = BootstrappedMAE(
        img_size=32, patch_size=4, in_chans=3,
        embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=96, decoder_depth=4, decoder_num_heads=3,
        mlp_ratio=4, norm_layer=nn.LayerNorm, **kwargs)
    return model 