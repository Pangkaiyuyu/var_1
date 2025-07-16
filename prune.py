import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn,SelfAttention
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn*pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            print("si = ",si)
            print("pn = ",pn)
            print("next_token_map.shape = ",next_token_map.shape)
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    
    
    from typing import List, Optional
    def autoregressive_infer_cfg2_1(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,gt_idx_Bl: Optional[List[torch.Tensor]]=None,
    ) -> 'VAR':   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        #先把梯度都打开
        self.train()
        for p in self.parameters():
            p.requires_grad_(True)
            
        import torch_pruning as tp
        imp = tp.importance.GroupTaylorImportance()
        
        for blk in self.blocks:
            blk.attn.using_flash = False
            blk.attn.using_xform = False
            
        #打印初始模型    
        for m in self.modules(): 
            print(m)# model 就是你的 VAR
            break
        
        B_ex = 2
        label_ex = torch.randint(0, self.num_classes, (B_ex,), device=self.lvl_1L.device)
        x_ex     = torch.randn(B_ex, self.L - self.first_l, self.Cvae, device=self.lvl_1L.device)
        example_inputs = (label_ex, x_ex)
        ignored_layers = [self.head]+[self.head_nm]
        for b in self.blocks: b.attn.kv_caching(False)
        pruner = tp.pruner.BasePruner(
            self,
            example_inputs,
            global_pruning=False,
            importance=imp,
            pruning_ratio=0.20,
            ignored_layers=ignored_layers,
            round_to=8,
        ) 

        print("tp.pruner.MagnitudePruner")
        for b in self.blocks: b.attn.kv_caching(False)
            
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)
        #设置加权参数
        loss_weight_self = [0.15,0.15,0.15,0.15,0.15,0.05,0.05,0.05,0.05,0.05]
        # loss_weight_self = [0.05,0.05,0.05,0.05,0.05,0.15,0.15,0.15,0.15,0.15]
        # loss_weight_self = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
        total = sum(loss_weight_self)
        normalized_weights = [w / total for w in loss_weight_self]
        loss_weight_self = normalized_weights
        loss = 0
        print("loss_weight_self = ",loss_weight_self)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn*pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            #计算加权损失
            criterion = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
            loss_this = criterion(logits_BlV.squeeze(0).view(-1, 4096), gt_idx_Bl[si].view(-1))
            print("loss_this = ",loss_this)
            lw = torch.ones(1, pn*pn, device=self.lvl_1L.device) / (pn*pn) * loss_weight_self[si]
            loss += loss_this.mul(lw).sum(dim=-1).mean()
            print("loss_this.shape = ",loss_this.shape)
            print("lw.shape = ",lw.shape)
            if si == 9:
                self.zero_grad()
                loss.backward()
                print("✅ backward成功")
                print("Before pruning FFN only:")
                print(sum(p.numel() for p in self.parameters()))
                for g in pruner.step(interactive=True):     # interactive=True → 返回可迭代
                    g.prune()
                print("✅ pruner.step()")
                #打印剪枝模型    
                for m in self.modules(): 
                    print(m)# model 就是你的 VAR
                    break
                print("After pruning FFN only:")
                print(sum(p.numel() for p in self.parameters()))
            
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)
        
        
        # # 👇 剪每个 block 的 ada_lin[1]，用梯度平方均值做 importance

        # round_to = 6
        # pruning_ratio = 0.1

        # for i, blk in enumerate(self.blocks):
        #     layer = blk.ada_lin[1]
        #     grad = layer.weight.grad
        #     if grad is None:
        #         print(f"⚠️ block[{i}].ada_lin[1] 的梯度是 None，跳过")
        #         continue

        #     imp = grad.pow(2).mean(dim=1)
        #     out_ch = layer.out_features
        #     keep_ch = int((out_ch * (1 - pruning_ratio)) // round_to * round_to)  # ✅ 对齐 round_to
        #     _, idxs = torch.topk(imp, k=keep_ch)

        #     print(f"✂️ 剪 block[{i}].ada_lin[1]: {out_ch} → {keep_ch}，保留比例={keep_ch / out_ch:.3f}")

        #     # 替换成新的 Linear 层（结构上真正减小了）
        #     old_linear = blk.ada_lin[1]
        #     new_linear = nn.Linear(old_linear.in_features, keep_ch, bias=old_linear.bias is not None)
        #     new_linear.weight.data = old_linear.weight.data[idxs].clone()
        #     if old_linear.bias is not None:
        #         new_linear.bias.data = old_linear.bias.data[idxs].clone()

        #     blk.ada_lin[1] = new_linear
        # print("After pruning ada_lin only:")
        # print(sum(p.numel() for p in self.parameters()))
        # for m in self.modules(): 
        #     print(m)# model 就是你的 VAR
        #     break
        # def pruned_forward(self, x, cond_BD, attn_bias):
        #     if self.shared_aln:
        #         gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2)
        #     else:
        #         ada_out = self.ada_lin(cond_BD)
        #         B, sixC = ada_out.shape
        #         assert sixC % 6 == 0
        #         C_eff = sixC // 6
        #         gamma1, gamma2, scale1, scale2, shift1, shift2 = ada_out.view(-1, 1, 6, C_eff).unbind(2)

        #     x_ln1 = self.ln_wo_grad(x).clone()
        #     x_ln1[:, :, :C_eff] = x_ln1[:, :, :C_eff] * (scale1[:, :, :C_eff] + 1) + shift1[:, :, :C_eff]
        #     attn_out = self.attn(x_ln1, attn_bias=attn_bias).clone()
        #     attn_out[:, :, :C_eff] *= gamma1[:, :, :C_eff]
        #     x = x + self.drop_path(attn_out)

        #     x_ln2 = self.ln_wo_grad(x).clone()
        #     x_ln2[:, :, :C_eff] = x_ln2[:, :, :C_eff] * (scale2[:, :, :C_eff] + 1) + shift2[:, :, :C_eff]
        #     ffn_out = self.ffn(x_ln2).clone()
        #     ffn_out[:, :, :C_eff] *= gamma2[:, :, :C_eff]
        #     x = x + self.drop_path(ffn_out)
            
        #     # 构造 mask: 1 for keep, 0~1 for prune
        #     mask = torch.ones(self.C, device=x.device)
        #     mask[C_eff:] = torch.linspace(1.0, 0.0, self.C - C_eff, device=x.device)

        #     # 应用 mask
        #     x = x * mask.view(1, 1, -1)

        #     return x
        # import types

        # for blk in self.blocks:
        #     blk.forward = types.MethodType(pruned_forward, blk)
        # # 构建依赖图（老版本接口）
        # for i, blk in enumerate(self.blocks):
        #     if hasattr(blk, 'ada_lin'):
        #         blk.ada_lin[1]._trace_mode = True
        #         blk.train()
        #         print(f"[SET] blk[{i}] id={id(blk)}  training={blk.training}  _trace_mode={getattr(blk, '_trace_mode', None)}")
        # for p in self.parameters():
        #     p.requires_grad_(True)
        # self.train()
        # DG = tp.DependencyGraph()
        # DG.build_dependency(self, example_inputs=example_inputs)
        # DG.ignored_params = set() 
        # # 再手动剪每个 block 的 ada_lin[1]
        # for i, blk in enumerate(self.blocks):
        #     ada_lin_layer = blk.ada_lin[1]
        #     # 检查是否在 DependencyGraph 中被追踪到
        #     if ada_lin_layer in DG.module2node:
        #         print(f"✅ Block {i} ada_lin[1] 被追踪到")
        #     else:
        #         print(f"❌ Block {i} ada_lin[1] 未被追踪，计算图中不存在")
        #     out_ch = ada_lin_layer.out_features
        #     keep_ch = int((out_ch * (1 - pruner.pruning_ratio)) // pruner.round_to * pruner.round_to)
        #     idxs = torch.arange(keep_ch, device=ada_lin_layer.weight.device)
        #     from torch_pruning import function

        #     groups = DG.get_pruning_group(ada_lin_layer, function.prune_linear_out_channels, idxs=idxs)
        #     target_groups = [
        #         g for g in groups
        #         if hasattr(g, "dep") and getattr(g.dep, "module", None) is ada_lin_layer
        #     ]
            
        #     print("groups = ",groups)
        #     print("target_groups = ",target_groups)
        #     print("GroupItem fields:", dir(groups[0]))
        #     print(f"🔍 blk[{i}].ada_lin[1]: {ada_lin_layer}, id={id(ada_lin_layer)}")
        #     for g in groups:
        #         print(f"👉 group dep.module: {getattr(g.dep, 'module', None)} id={id(getattr(g.dep, 'module', None))}")

            

        #     for g in groups:
        #         if g.nodes is None:
        #             raise RuntimeError("group.nodes 为 None")
        #         print(f"✂️ Pruning group with {len(g.nodes)} nodes")
        #         g.prune()
            
        #     if group is None:
        #         raise RuntimeError(f"[Block {i}] get_pruning_group 返回 None")

        #     if not hasattr(group, "nodes") or group.nodes is None:
        #         raise RuntimeError(f"[Block {i}] pruning_group.nodes 为 None，依赖链不完整")

        #     if not hasattr(group, "main_target") or group.main_target is None:
        #         raise RuntimeError(f"[Block {i}] pruning_group.main_target 为 None")
        #     print("ada_lin_group = ",group)
        #     group.prune()
        #     print(f"✅ Block {i} ada_lin[1] 剪枝成功: {out_ch} → {keep_ch}")
        #     # 构图后关闭
        # for blk in self.blocks:
        #     if hasattr(blk, 'ada_lin'):
        #         blk._trace_mode = False
        return self   # de-normalize, from [-1, 1] to [0, 1]
    def rebuild_scale_mul_after_head_pruning(self):
        """
        重构每个 block 中的 SelfAttention 的 scale_mul_1H11 和 num_heads。
        依据 self.head_pruning_indices 中记录的被剪掉的 head 索引。
        """
        assert hasattr(self, "head_pruning_indices"), "self.head_pruning_indices 不存在，请确认是否剪枝过"

        for i, blk in enumerate(self.blocks):
            attn = blk.attn
            if not attn.attn_l2_norm:
                continue  # 如果该层不使用 l2 norm，无需 scale_mul

            total_heads = attn.scale_mul_1H11.shape[1]
            pruned = self.head_pruning_indices[i]
            all_heads = torch.arange(total_heads)
            keep = torch.tensor([h.item() for h in all_heads if h.item() not in pruned.tolist()], device=attn.scale_mul_1H11.device)

            # ✅ 重新构建 scale_mul_1H11
            new_scale = attn.scale_mul_1H11.data[:, keep, :, :]  # shape: [1, new_heads, 1, 1]
            attn.scale_mul_1H11 = torch.nn.Parameter(new_scale.clone())

            # ✅ 同时更新 num_heads
            attn.num_heads = len(keep)

            # ✅ debug 信息
            print(f"[Rebuild] Block {i}: num_heads={attn.num_heads}, scale_mul_1H11.shape={attn.scale_mul_1H11.shape}")



    def autoregressive_infer_cfg2_5(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,gt_idx_Bl: Optional[List[torch.Tensor]]=None,
        more_smooth=False,
    ) -> 'VAR':
        self.train()
        for p in self.parameters():
            p.requires_grad_(True)
    # ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        
        import torch_pruning as tp
        imp = tp.importance.GroupTaylorImportance()

        def vit_forward(self, x, attn_bias=None):
            # print(f"[Debug] Executing SelfAttention block {self.block_idx}")
            B, N, C = x.shape
            qkv = self.mat_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim
                    ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            x = self.proj(x)
            # print(f"🔥 [ATTN] proj executed for block {self.block_idx}")
            x = self.proj_drop(x)
            return x

        import types
        for blk in self.blocks:
            blk.attn.using_flash = False
            blk.attn.using_xform = False
            blk.attn.forward = types.MethodType(vit_forward, blk.attn)
        
        # def aln_no_inplace(self, x, cond_BD, attn_bias):
        #     if self.shared_aln:
        #         gamma1, gamma2, scale1, scale2, shift1, shift2 = \
        #             (self.ada_gss + cond_BD).unbind(2)
        #     else:
        #         gamma1, gamma2, scale1, scale2, shift1, shift2 = \
        #             self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        #     # --------- 关键：把 “_” 版本换掉 ----------
        #     # print("x.shape",x.shape)
        #     # print("self.C = ",self.C)
        #     # print("scale1.shape = ",scale1.shape)
        #     # print("shift1.shape = ",shift1.shape)
        #     # print("gamma1.shape = ",gamma1.shape)
        #     x_norm1 = self.ln_wo_grad(x).mul(scale1.add(1)).add(shift1)   # add_ → add
        #     attn_out = self.attn(x_norm1, attn_bias=attn_bias).mul(gamma1)  # mul_ → mul
        #     x = x + self.drop_path(attn_out)

        #     x_norm2 = self.ln_wo_grad(x).mul(scale2.add(1)).add(shift2)
        #     ffn_out = self.ffn(x_norm2).mul(gamma2)
        #     x = x + self.drop_path(ffn_out)
        #     return x
        
        def aln_no_inplace_chunk(self, x, cond_BD, attn_bias):
            """ no in-place, chunk-based, shape-safe """
            if self.shared_aln:
                y = self.ada_gss + cond_BD          # (B, 1, 6C)
            else:
                y = self.ada_lin(cond_BD)           # (B,     6C)

            # === 新：chunk 并在 dim=1 插单元维 ===
            gamma1, gamma2, scale1, scale2, shift1, shift2 = torch.chunk(y, 6, dim=-1)
            scale1 = scale1.unsqueeze(1)            # (B,1,C)
            scale2 = scale2.unsqueeze(1)
            shift1 = shift1.unsqueeze(1)
            shift2 = shift2.unsqueeze(1)
            gamma1 = gamma1.unsqueeze(1)
            gamma2 = gamma2.unsqueeze(1)

            x1 = self.ln_wo_grad(x).mul(scale1.add(1)).add(shift1)
            x  = x + self.drop_path(self.attn(x1, attn_bias=attn_bias).mul(gamma1))

            x2 = self.ln_wo_grad(x).mul(scale2.add(1)).add(shift2)
            x  = x + self.drop_path(self.ffn(x2).mul(gamma2))
            return x
                
        for blk in self.blocks:
            blk.forward = types.MethodType(aln_no_inplace_chunk, blk)
        
        # def aln_before_head_no_inplace(self, x_BLC, cond_BD):
        #     """
        #     Args:
        #         x_BLC : (B, L, C)
        #         cond_BD: (B, D)
        #     Returns:
        #         (B, L, C)   # 与原实现一致
        #     """
        #     scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        #     x = self.ln_wo_grad(x_BLC)
        #     x = x.mul(scale.add(1))   # 非 in-place
        #     x = x.add(shift)          # 非 in-place
        #     return x
        def aln_before_head_no_inplace_chunk(self, x_BLC, cond_BD):
            scale, shift = torch.chunk(self.ada_lin(cond_BD), 2, dim=-1)  # (B,C)
            scale = scale.unsqueeze(1)     # (B,1,C)
            shift = shift.unsqueeze(1)
            return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add(shift)
        for m in self.modules(): 
            print(m)# model 就是你的 VAR
            break
        
        for m in self.modules(): 
            # print(m)# model 就是你的 VAR
            if isinstance(m, AdaLNBeforeHead):
                m.forward = types.MethodType(aln_before_head_no_inplace_chunk, m)
                
        proj_drop_backup = {}
        for blk in self.blocks:
            attn = blk.attn
            proj_drop_backup[attn] = attn.proj_drop  # 保存
            # 临时替换为非inplace Dropout，确保能构建依赖图
            attn.proj_drop = nn.Dropout(p=0.1, inplace=False)
        
        
        B_ex = 2
        label_ex = torch.randint(0, self.num_classes, (B_ex,), device=self.lvl_1L.device)
        x_ex     = torch.randn(B_ex, self.L - self.first_l, self.Cvae, device=self.lvl_1L.device)
        example_inputs = (label_ex, x_ex)
        # base_macs, base_params = tp.utils.count_ops_and_params(self, example_inputs)
        

        ignored_layers = [self.head] + [blk.ffn.fc2 for blk in self.blocks] +[blk.ffn.fc1 for blk in self.blocks]+ [blk.attn.proj for blk in self.blocks]


        num_heads = {}
        out_channel_groups = {}
        # print("---------------------this is var module------------------------")
        for name,m in self.named_modules():
            # print(m)
            if isinstance(m, AdaLNSelfAttn):
                # m.attn.forward = vit_forward.__get__(m.attn, m.attn.__class__)
                num_heads[m.attn.mat_qkv] = m.attn.num_heads

        


        
        
        print("ignored_layer = ",ignored_layers)

        in_channel_groups = {
            blk.attn.mat_qkv: blk.attn.head_dim
            for blk in self.blocks
        }
        print("in_channel_groups = ",in_channel_groups)
        for b in self.blocks: b.attn.kv_caching(False)
        
        
        
            
        pruner = tp.pruner.BasePruner(
            self,
            example_inputs,
            global_pruning=False,
            importance=imp,
            pruning_ratio=0.15,
            prune_num_heads=True,
            head_pruning_ratio=0.15,
            ignored_layers=ignored_layers,
            prune_head_dims=False,
            num_heads=num_heads,
            out_channel_groups=out_channel_groups, 
            round_to=8,
        ) 

        print("tp.pruner.MagnitudePruner")
        


        print("pruner.prune_num_heads = ",pruner.prune_num_heads)
        print("pruner.head_pruning_ratio = ",pruner.head_pruning_ratio)
        print("pruner.num_heads = ",pruner.num_heads)
        
 
        for b in self.blocks: b.attn.kv_caching(False)
        self.train()
        for p in self.parameters():
            p.requires_grad_(True)
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        print("cond_BD.shape = ",cond_BD.shape)
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        print("lvl_pos.shape = ",lvl_pos.shape)
        print("next_token_map.shape = ",next_token_map.shape)
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        print("f_hat.shape = ",f_hat.shape)
        for b in self.blocks: b.attn.kv_caching(True)
        loss_weight_mbq=[]
        loss_weight_self = [0.15,0.15,0.15,0.15,0.15,0.05,0.05,0.05,0.05,0.05]
        # loss_weight_self = [0.05,0.05,0.05,0.05,0.05,0.15,0.15,0.15,0.15,0.15]
        # loss_weight_self = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
        total = sum(loss_weight_self)
        normalized_weights = [w / total for w in loss_weight_self]
        loss_weight_self = normalized_weights
        print("loss_weight_self = ",loss_weight_self)
        
        
        loss = 0
        
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn*pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            print("si = ",si)
            print("logits_BlV.shape = ",logits_BlV.shape)
            print("gt_idx_Bl[si].shape = ",gt_idx_Bl[si].shape)
            
            criterion = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
            loss_this = criterion(logits_BlV.squeeze(0).view(-1, 4096), gt_idx_Bl[si].view(-1))
            print("loss_this = ",loss_this)
            lw = torch.ones(1, pn*pn, device=self.lvl_1L.device) / pn*pn * loss_weight_self[si]
            loss += loss_this.mul(lw).sum(dim=-1).mean()
            print("loss_this.shape = ",loss_this.shape)
            print("lw.shape = ",lw.shape)
            if si == 9:
                self.zero_grad()
                loss.backward()
                print("✅ backward成功")
                
                print("Before pruning FFN only:")
                print(sum(p.numel() for p in self.parameters()))
                # for name, module in self.named_modules():
                #     print(f"{name}: {module}")
                # pruner.step()
                for g in pruner.step(interactive=True):     # interactive=True → 返回可迭代
                    g.prune()
                print("✅ pruner.step()")
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        
        
        
    
        self.rebuild_scale_mul_after_head_pruning()
        print("✅ rebuild_scale_mul_after_head_pruning")
        

        for m in self.modules():
            if isinstance(m, AdaLNSelfAttn):  # 或者你自定义的 block 类型
                print("mat_qkv.out_features = ", m.attn.mat_qkv.out_features)
                print("current num_heads = ", pruner.num_heads[m.attn.mat_qkv])
        
        head_id = 0
        for m in self.modules():
            if isinstance(m, AdaLNSelfAttn):
                print("Head #%d"%head_id)
                print("[Before Pruning] Num Heads: %d, Head Dim: %d =>"%(m.attn.num_heads, m.attn.head_dim))
                # m.attn.num_heads = pruner.num_heads[m.attn.mat_qkv]
                # m.attn.head_dim = m.attn.mat_qkv.out_features // (m.attn.num_heads*3)
                
                m.attn.head_dim = 64
                m.attn.num_heads = m.attn.mat_qkv.out_features // (m.attn.head_dim*3)
        #         m.attn.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, m.attn.num_heads, 1, 1), fill_value=4.0).log().to(attn.mat_qkv.weight.device), requires_grad=True)
        #         embed_dim = m.attn.num_heads*m.attn.head_dim
        #         print("[After Pruning] Num Heads: %d, Head Dim: %d"%(m.attn.num_heads, m.attn.head_dim))
        #         print()
                head_id+=1
                self.num_heads =   m.attn.num_heads
                self.head_dim =   m.attn.num_heads

        
        for m in self.modules():
            if isinstance(m, AdaLNBeforeHead):
                m.forward = types.MethodType(AdaLNBeforeHead.forward, m)
        import types
        for blk in self.blocks:
            blk.attn.using_flash = False
            blk.attn.using_xform = False
            blk.attn.forward = types.MethodType(SelfAttention.forward, blk.attn)        
        for blk in self.blocks:
            attn = blk.attn
            # 还原成剪枝前的 Dropout对象
            attn.proj_drop = proj_drop_backup[attn]

        
        for m in self.modules():
            print(m)
            break
        
        print("After pruning FFN only:")
        # for name, module in self.named_modules():
        #     print(f"{name}: {module}")
        print(sum(p.numel() for p in self.parameters()))
        print("loss_weight_self = ",loss_weight_self)
        print("self.head_pruning_indices = ",self.head_pruning_indices)
        print("self.head_imp = ",self.head_imp)
        
                

        return self   # de-normalize, from [-1, 1] to [0, 1]
    


    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(VAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )
