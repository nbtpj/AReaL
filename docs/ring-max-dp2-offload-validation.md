# Ring-Max 993B SFT:dp2 + Optimizer CPU Offload 验证报告

日期:2026-06-11 ~ 06-12
作者:chucai.dzq(协同 Claude Code)
目标读者:hcy 及 ring-max SFT 相关同学

## TL;DR

针对 `hcy-ring-sft` 当前 dp1 配置(`attn:d1p16t4c4|ffn:d1p16e16`,289s/step),给出经过验证的 dp2 提速方案:

```bash
"allocation_mode='megatron:(attn:d2p8t4c4|ffn:d2p8e16)'" \
+actor.megatron.optimizer_cpu_offload=true \
+actor.megatron.optimizer_offload_fraction=0.3 \
```

- 显存:估算 ~121G/140G(AutoParallel 修正版,误差 ±1-2GB),解决 0606 dp2 三连 OOM。
- 性能:预期 train_step 289s → ~150-170s(约 1.8×);offload fraction=0.3 的开销实测 <1%。
- 代码路径已在单机 8 卡用同构缩小模型全链路验证(allocation 解析、HybridDeviceOptimizer、
  训练循环、save),配置语法逐项踩坑修正过,可直接提交。
- 注意 hydra 覆盖必须带 `+` 前缀(base yaml 未声明这些 key),allocation_mode 要双层引号。

---

## 1. 背景与基线

ring-max25(BailingMoeV2_5,993B total,80 层,256 experts,MLA + linear-attn hybrid)
SFT,32 节点 × 8 × L20X-140GB,AReaL swe/main,bs=2560,131k max_tokens_per_mb。

基线 trial `0608_ring_max_bs2560_g256_lr1.5e-4_1t_p16t8c2_stepfun_no_aegntic_align_flash`
(实际 allocation 为 d1p16t4c4|e16,trial 名中的 p16t8c2 已过时)实测(528 步全量):

| 指标 | 值 |
|---|---|
| train_step | mean 288.9s,median 285.0s |
| load_bcast | 16.3s/step(5.3%) |
| checkpoint_for_recover | 摊销 6.0s/step(~2%) |
| 吞吐 | 53.2k tok/s(208 tok/s/GPU) |
| microbatch/step | 128(PP=16 bubble ≈ 10%,非瓶颈) |

瓶颈是纯计算时间,提速主要靠 DP=2;另有五个未开启的 Megatron fusion 优化
(overlap_param_gather / moe_shared_expert_overlap / moe_permute_fusion /
moe_router_fusion / cross_entropy_loss_fusion,预估 +10-20%,建议 dp2 稳定后单独验证)。

## 2. 为什么之前 dp2 全部 OOM

0606 三次 dp2 尝试与结局:

| allocation | 结局 | 根因 |
|---|---|---|
| attn:d2p16t1c8\|ffn:d2p16e8 | NCCL CUDA error | EP16→8 专家权重翻倍 |
| attn:d2p16t4c2\|ffn:d2p16e8 | OOM(127G+ 再要 2-4G) | 同上 + CP4→2 长序列激活翻倍 |
| attn:d2p8t4c4\|ffn:d2p8e16 | OOM(差 ~4GB) | **最接近可行,只缺 optimizer 显存** |
| attn:d1p16t4c4\|ffn:d1p16e16 | ✓(当前 0608) | 退回 dp1 |

硬约束(数据集 p99.9=101k tokens,max=129k,257 条 >110k):
- **EP 必须 ≥16**:e8 时每 rank 32 个专家,权重 +15GB。
- **CP 必须 ≥4**:c2 时单 rank 最长 65k tokens,attention 激活翻倍。

`d2p8t4c4|e16` 保住 EP16/CP4,用 PP 16→8 换 DP=2,只差 ~10GB —— 这正好是
optimizer states 的量级,所以解法是 optimizer CPU offload。

## 3. 显存账(AutoParallel 修正版)

> 重要修正:AutoParallel / areal-friend parallel 此前默认 `optimizer_cpu_offload=True`,
> 导致估算比实际低 ~50GB/rank("低估 50%"是工具默认值错误,不是模型不准)。
> 已修复(AutoParallel commit f70f2a5,默认改 False),修正后与实测误差 ±1-2GB:

| allocation | 不开 offload | 开 offload(f=1.0) | 实测对照 |
|---|---|---|---|
| d1p16t4c4\|e16(基线) | 120.2G | 66.7G | 跑通 ✓ |
| **d2p8t4c4\|e16(推荐)** | **136.2G(贴红线)** | **84.6G** | OOM @ ~137G ✓ |
| d2p16t4c2\|e8 | 155.6G | — | OOM ✓ |

fraction=0.3 时挪走 ~15GB → ~121G/140G,余量 ~19GB,足够扛长尾 batch 激活尖峰。
CPU 侧:节点 1.5TB RAM,即使 f=1.0 也只占 ~467G/node,无压力。

## 4. 单机烟雾测试(全链路验证)

**方法**:构造 0.92B 同构缩小模型 `/storage/openpsi/users/chucai.dzq/models/ring-max25-mini`
(8 层/H1024/64 专家,保留 MoE+MLA+group_norm+first_k_dense_replace 全部结构,
model_type=bailing_hybrid),单机 8 卡按比例镜像生产并行(attn:d2p2t2|ffn:d2p2e2),
跑 12 步对照。脚本:`AReaL/examples/swe/smoke_sft_dp2_offload.sbatch`(+ `_inner.sh`)。

```bash
sbatch --reservation=swe-rl \
  --export=ALL,OFFLOAD=on,FRACTION=0.3,ALLOC='attn:d2p2t2|ffn:d2p2e2' \
  examples/swe/smoke_sft_dp2_offload.sbatch
```

### 4.1 offload fraction 扫描(同 seed 同数据,逐步 token 数完全一致)

| 配置 | job | 稳态 train_step 均值 | vs off |
|---|---|---|---|
| offload=off | 931227 | 4.59s | 基线 |
| **fraction=0.3** | 931238 | **4.63s** | **+0.9%(基本免费)** |
| fraction=0.5 | 931239 | 4.72s | +2.8% |
| fraction=1.0 | 931213 | 4.94s | +7.6% |
| f=1.0 + overlap_d2h_h2d | 931240 | 4.94s | 无收益 |

结论:
- `optimizer_offload_fraction` 链路有效(引擎日志确认传入,开销随 fraction 单调降)。
- **推荐 fraction=0.3**;若实跑显存余量紧张可升到 0.5/1.0,开销上限 ~7.6%
  (且这是 4.6s 短 step 放大的占比,993B 真实 step 150s+ 时占比会显著更低)。
- `overlap_cpu_optimizer_d2h_h2d` 无可测收益,不必开。
- `use_torch_optimizer_for_cpu_offload` 保持 false(走 fused TE/apex 路径)。

### 4.2 dp1 vs dp2 并行对照(按生产比例 d1p16→d2p8 镜像为 d1p4→d2p2)

| allocation(8 卡) | 稳态均值 | 对比 |
|---|---|---|
| attn:d1p4t2\|e2(镜像 dp1 基线) | 5.99s | — |
| attn:d2p2t2\|e2(镜像 dp2 方案) | 4.59s | **dp2 快 ~23%** |

方向与生产预期一致(小模型上 PP bubble 占比更高,生产 1.8× 的预估主要来自
DP 翻倍 + microbatch/rank 减半,见 §5)。

### 4.3 Megatron 侧机制确认

- AReaL `actor.megatron.optimizer_cpu_offload` → MCoreOptimizerConfig →
  Megatron `HybridDeviceOptimizer`(NVIDIA + 阿里 PAI 官方实现,
  megatron/core/optimizer/cpu_offloading/),`offload_fraction` 为一等参数。
- DistributedOptimizer 对 HybridDeviceOptimizer 有显式适配(10+ 处 isinstance 分支)。
- 历史先例:5/21 `swe-sft/test_glm5_async_on,off` 用 f=1.0 在 GLM-5 上完整训练成功。
  f<1.0 的混合路径此前无人跑过,本次烟雾测试补上了这个验证。

## 5. 给 hcy 的提交命令(基于 0608 原命令,改动已标注)

```bash
python -m areal.infra.launcher.slurm examples/swe/train_sft.py \
    --config examples/swe/swe_sft_flash_moe_v2_128g_align_ling.yaml \
    scheduler.type=null \
    "allocation_mode='megatron:(attn:d2p8t4c4|ffn:d2p8e16)'" \          # 改:dp1→dp2
    +actor.megatron.optimizer_cpu_offload=true \                         # 新增
    +actor.megatron.optimizer_offload_fraction=0.3 \                     # 新增
    experiment_name=hcy-ring-sft \
    trial_name=06xx_ring_max_bs2560_d2p8t4c4e16_optoffload03 \
    ... # 其余参数(lr/数据/saver/recover/cluster)与 0608 完全一致
```

预期:train_step 289s → ~150-170s(DP=2,每 rank microbatch 128→64,p8 bubble 持平
~10%,DP 梯度 reduce-scatter + offload 开销合计 <5%)。**起跑后先看前 20 步的
train_step 和显存峰值再放量**;若 OOM(预估余量 19GB,概率低)依次升 fraction
0.5 → 1.0,不要动 EP/CP。

## 6. 已知坑(提交前务必核对)

1. **hydra `+` 前缀**:`optimizer_cpu_offload` / `optimizer_offload_fraction` /
   `total_train_steps` / `saver.freq_steps` 在 base yaml 没有声明,覆盖必须写
   `+actor.megatron.xxx=...`,否则启动即报 `ConfigCompositionException`。
2. **allocation_mode 引号**:含 `(|)`,要双层引号 `"allocation_mode='megatron:(...)'"`,
   单层会被 bash/hydra 各吃一层报 `mismatched input '('`。
3. **环境变量**:`AREAL_SPMD_MODE=1` 和 `AREAL_LLM_SERVER_ADDRS=`(空值)必须设,
   否则 `scheduler.type=null` 报 `Unknown scheduler type: None`(slurm launcher 会自动注入,
   手写 torchrun 时要自己加)。
4. **mbridge HF save 限制**:PP 切分若让某个 stage 全是 dense 层(80 层模型 PP≥20 时
   stage0 的 5 层 < first_k_dense_replace=4 才安全),save 时报
   `assert all(x >= 1 for x in pp_stage_expert_shards)`。生产 PP=8/16 不受影响。
5. **不要动**:EP<16、CP<4 必 OOM(0606 实证);`use_torch_optimizer_for_cpu_offload`
   保持 false。

## 7. 工件索引

| 工件 | 路径 |
|---|---|
| 烟雾测试脚本 | `AReaL/examples/swe/smoke_sft_dp2_offload.sbatch` + `_inner.sh` |
| 缩小版模型 | `/storage/openpsi/users/chucai.dzq/models/ring-max25-mini` |
| 烟雾测试日志 | `/storage/openpsi/experiments/logs/admin/chucai-smoke-dp2-offload/` |
| 0608 基线日志分析脚本 | `/storage/openpsi/users/chucai.dzq/tmp/profile_ring_max_log.py` |
| 数据集长度分布脚本 | `/storage/openpsi/users/chucai.dzq/tmp/scan_seq_lens.py` |
| AutoParallel 默认值修复 | github.com/dingzhiqiang/AutoParallel commit f70f2a5 |
| areal-friend `--optimizer-offload` | AReaL-friend commit d197f1e(chucai.dzq/dev) |
| 过程笔记 | `easy-code/sessions/2026-06/2026-06-11-ring-max-sft-profile-dp2-feasibility.md` |

## 8. 遗留项

- d1p4 + fraction=0.3 的对照点未采到(节点被收回),不影响结论(fraction 扫描已在
  dp2 上完成,dp1/dp2 对照已在 off 配置下完成)。
- 五个 Megatron fusion 开关的收益验证(建议 dp2 稳定后,一次一个变量)。
- mbridge 纯 dense PP stage 的 save 断言,可给 AReaL 提 issue。
