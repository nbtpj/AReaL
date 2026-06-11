# ring-max SFT dp2 + optimizer offload

## 配置

在 0608 命令基础上改一行、加两行,其余不动:

```bash
"allocation_mode='megatron:(attn:d2p8t4c4|ffn:d2p8e16)'"      # d1p16 -> d2p8
+actor.megatron.optimizer_cpu_offload=true
+actor.megatron.optimizer_offload_fraction=0.3
```

注意:
- `+` 前缀必须有(base yaml 没这些 key,少了直接 ConfigCompositionException)。
- allocation_mode 双层引号,单层会报 `mismatched input '('`。
- 手写 torchrun 时要设 `AREAL_SPMD_MODE=1`、`AREAL_LLM_SERVER_ADDRS=`(空),
  否则 `scheduler.type=null` 报 Unknown scheduler。slurm launcher 会自动注入。
- EP、CP 不要动:e8 / c2 在 0606 都 OOM 过(专家权重翻倍 / 长序列激活翻倍,
  数据 p99.9=101k max=129k)。

## 显存

0606 OOM 的根因是 dp2 后差 ~10GB,正好是 optimizer states 量级。
AutoParallel 修正默认值后(commit f70f2a5)估算和实测能对上:

| allocation | 不开 offload | f=0.3 | f=1.0 | 实测 |
|---|---|---|---|---|
| d1p16t4c4\|e16(0608) | 120G | | | 跑通 |
| d2p8t4c4\|e16 | 136G(贴 140 红线) | ~121G | 85G | 0606 OOM @ ~137G |

f=0.3 余量 ~19G;不够就升 0.5 / 1.0,CPU 侧最多 467G/node(机器 1.5T)。

## 实验

单机 8 卡,0.92B 同构缩小模型(8 层/H1024/64 专家,MoE+MLA+group_norm+dense_replace
全保留),12 步,同 seed 同数据。脚本 `examples/swe/smoke_sft_dp2_offload.sbatch`:

```bash
sbatch --reservation=swe-rl \
  --export=ALL,OFFLOAD=on,FRACTION=0.3,ALLOC='attn:d2p2t2|ffn:d2p2e2' \
  examples/swe/smoke_sft_dp2_offload.sbatch
```

fraction 扫描(dp2):

| | job | train_step 稳态 | vs off |
|---|---|---|---|
| off | 931227 | 4.59s | - |
| f=0.3 | 931238 | 4.63s | +0.9% |
| f=0.5 | 931239 | 4.72s | +2.8% |
| f=1.0 | 931213 | 4.94s | +7.6% |
| f=1.0+overlap | 931240 | 4.94s | overlap 没用 |

dp1 vs dp2(都不开 offload,按生产 d1p16→d2p8 比例缩成 d1p4→d2p2):

| | train_step 稳态 |
|---|---|
| d1p4t2\|e2 | 5.99s |
| d2p2t2\|e2 | 4.59s(快 23%) |

全部 12/12 步跑通,loss 正常,逐步 token 数各组完全一致。
fraction 链路确认有效(引擎日志可见 `optimizer_offload_fraction=0.3` 传入
HybridDeviceOptimizer,Megatron 官方实现,DistributedOptimizer 有适配)。
f=1.0 此前在 5/21 test_glm5_async_on/off 跑过完整训练,f<1 是这次补的验证。

## 预期收益

- 基线 0608:289s/step,53.2k tok/s(528 步实测,bubble ~10% 非瓶颈)。
- dp2 后每 rank microbatch 128→64,p8 bubble 持平,DP 翻倍。
- 预期 289s → 150-170s,约 1.8×;offload f=0.3 开销 <1%(小模型实测 0.9%,
  真实 step 150s+ 摊薄后更低)。
- 后续还有 5 个没开的 fusion(overlap_param_gather / moe_shared_expert_overlap /
  moe_permute_fusion / moe_router_fusion / cross_entropy_loss_fusion),
  预估再 +10-20%,建议 dp2 稳定后一次开一个。

## 起跑检查

前 20 步看 train_step 和显存峰值再放量。OOM 就升 fraction,不要动 EP/CP。

## 坑

- mbridge save:PP 切到某 stage 全 dense 层(80 层 PP≥20)会断言
  `pp_stage_expert_shards >= 1`。PP=8/16 没事。
- `use_torch_optimizer_for_cpu_offload` 保持 false(torch 路径慢)。

## 相关文件

- 烟雾测试日志:`/storage/openpsi/experiments/logs/admin/chucai-smoke-dp2-offload/`
- 缩小模型:`/storage/openpsi/users/chucai.dzq/models/ring-max25-mini`
- 0608 日志分析:`/storage/openpsi/users/chucai.dzq/tmp/profile_ring_max_log.py`
- 过程笔记:`easy-code/sessions/2026-06/2026-06-11-ring-max-sft-profile-dp2-feasibility.md`
