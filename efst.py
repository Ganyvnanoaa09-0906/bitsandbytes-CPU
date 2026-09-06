"""EFST - Expert-Specific Fine-Tuning for CPU MoE training.

EFST = 专家专项微调：在 MoE 模型上只微调被选中的专家（以及可选的 router），
冻结其余参数，从而大幅减少优化器状态、梯度内存和反向传播开销。
配合 LoRA 注入时，可以进一步把每个被选中专家的可训练参数量压缩到 adapter 级别。

设计目标：
- 不依赖特定模型实现，自动识别常见 MoE 结构：
  ``experts`` / ``moe`` / ``block_sparse_moe`` / ``Experts`` 等容器；
- 支持人工指定专家，也支持先用一小批校准数据统计路由使用频率，
  自动选出最热的 top-k 专家；
- 支持“全专家微调”和“专家内 LoRA”两种模式；
- 纯 CPU / AVX2 友好：冻结的参数不产生 grad，也就不占用优化器状态，
  对 12~16GB 内存机器更友好。

典型用法::

    from efst import EFSTConfig, apply_efst

    config = EFSTConfig(
        top_k=2,
        calibration_dataloader=train_loader,
        calibration_forward_fn=lambda batch: model(**batch),
        tune_router=True,
        lora=True,
        lora_r=8,
        lora_alpha=16,
    )
    info = apply_efst(model, config)
    print(info)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "EFSTConfig",
    "EFSTResult",
    "ExpertGroup",
    "apply_efst",
    "find_expert_groups",
    "freeze_all",
    "unfreeze_all",
    "unfreeze_experts",
    "collect_expert_usage",
    "select_top_experts",
    "add_lora_to_experts",
    "report_trainable",
]

# 常见 MoE 专家容器名字片段
_EXPERT_NAME_HINTS = ("expert", "moe")
# 常见 router/gate 名字片段
_ROUTER_NAME_HINTS = ("router", "gate")


@dataclass
class ExpertGroup:
    """一个 MoE 专家容器，例如 ``model.layers.0.mlp.experts``。

    两种形态：
    - ``children`` 非空：经典 ModuleList/ModuleDict 专家（每个子模块一个专家）；
    - ``tensor_param_names`` 非空：现代 3D 张量专家（transformers 5.15+
      Qwen3Next/Qwen3.5 的 ``Qwen3NextExperts``：权重是 ``[num_experts, ...]``
      的大 Parameter，没有子模块）。``num_experts`` 记录专家数。
    """

    path: str
    module: nn.Module
    children: List[Tuple[str, nn.Module]] = field(default_factory=list)
    tensor_param_names: Tuple[str, ...] = ()
    num_experts: int = 0

    @property
    def is_tensor(self) -> bool:
        return bool(self.tensor_param_names)

    def __len__(self) -> int:
        return self.num_experts if self.is_tensor else len(self.children)

    def keys(self) -> List[str]:
        if self.is_tensor:
            return [str(i) for i in range(self.num_experts)]
        return [k for k, _ in self.children]

    def get_expert(self, key: Union[int, str]) -> Union[nn.Module, int]:
        if self.is_tensor:
            idx = int(key)
            if 0 <= idx < self.num_experts:
                return idx  # 返回专家索引（tensor 形态没有独立子模块）
            raise KeyError(f"{self.path} has no expert {key!r}; available: 0..{self.num_experts - 1}")
        for k, m in self.children:
            if k == str(key) or k == key:
                return m
        raise KeyError(f"{self.path} has no expert {key!r}; available: {self.keys()}")


@dataclass
class EFSTConfig:
    """EFST 配置。

    - 如果给了 ``expert_indices``，则直接使用指定专家；
    - 如果给了 ``top_k``，则先用 ``calibration_dataloader`` 统计路由频率，
      自动选择最热的 top-k 个专家；
    - ``expert_indices`` 和 ``top_k`` 都没给时，默认微调所有专家。
    """

    expert_indices: Optional[Union[Sequence[int], Dict[str, Sequence[Union[int, str]]]]] = None
    top_k: Optional[int] = None
    calibration_dataloader: Optional[Iterable] = None
    calibration_forward_fn: Optional[Callable[[Any], Any]] = None
    num_calibration_batches: Optional[int] = None
    tune_router: bool = False
    lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


@dataclass
class EFSTResult:
    """apply_efst 的返回信息。"""

    selected_experts: Dict[str, List[Union[int, str]]] = field(default_factory=dict)
    trainable_before: int = 0
    trainable_after: int = 0
    total_params: int = 0
    frozen_params: int = 0
    lora_injected: bool = False
    groups: List[ExpertGroup] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "EFST applied:",
            f"  expert groups : {len(self.groups)}",
            f"  total params  : {self.total_params:,}",
            f"  trainable     : {self.trainable_before:,} -> {self.trainable_after:,}",
            f"  frozen        : {self.frozen_params:,}",
            f"  lora          : {self.lora_injected}",
        ]
        for path, keys in self.selected_experts.items():
            lines.append(f"  {path}: {keys}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# 专家容器识别
# ---------------------------------------------------------------------------
def _name_hits(name: str, hints: Tuple[str, ...]) -> bool:
    low = name.lower()
    return any(h in low for h in hints)


# 常见的 3D 张量专家权重参数名（transformers 5.15+ Qwen3Next/Qwen3.5 风格）
_TENSOR_EXPERT_PARAM_HINTS = ("gate_up_proj", "down_proj", "up_proj", "w1", "w2", "w3")


def _tensor_expert_info(module: nn.Module) -> Tuple[str, int]:
    """若 module 是 3D 张量专家（权重 [E, ...] 且没有子模块），返回
    (参数名, 专家数)；否则返回 ("", 0)。"""
    for name, p in module.named_parameters(recurse=False):
        if p.dim() == 3 and any(h in name for h in _TENSOR_EXPERT_PARAM_HINTS):
            return name, p.shape[0]
    return "", 0


def find_expert_groups(model: nn.Module) -> List[ExpertGroup]:
    """自动找出模型里的 MoE 专家容器。

    返回的 ``ExpertGroup`` 是包含若干子模块的容器，例如 ModuleList/ModuleDict，
    也可能是自定义的 ``Experts`` 模块。每个子模块视为一个专家。
    也识别 3D 张量专家（``Qwen3NextExperts`` 等：权重 [E, ...]，无子模块）。
    """
    groups: List[ExpertGroup] = []
    for name, module in model.named_modules():
        if not name:
            continue
        # 已经在某个已识别容器内部，跳过，避免嵌套重复
        if any(name == g.path or name.startswith(g.path + ".") for g in groups):
            continue

        type_name = type(module).__name__.lower()
        is_container = isinstance(module, (nn.ModuleList, nn.ModuleDict, nn.Sequential))
        has_children = len(list(module.children())) > 0
        looks_by_name = _name_hits(name, _EXPERT_NAME_HINTS)
        looks_by_type = _name_hits(type_name, _EXPERT_NAME_HINTS)

        # 3D 张量专家：类型名含 expert/moe 且持有 [E, ...] 参数
        # （先于容器判断：这类模块可能带着 act_fn 等非专家子模块）
        tparam, ne = _tensor_expert_info(module)
        if tparam and ne > 0:
            groups.append(ExpertGroup(
                path=name, module=module,
                tensor_param_names=(tparam,), num_experts=ne))
            continue

        if not has_children:
            continue
        if not (looks_by_name or looks_by_type):
            continue
        # 避免把普通 MLP 里恰好叫 "gate_proj" 的 Linear 当专家容器
        if not is_container and not looks_by_type:
            continue

        children = list(module.named_children())
        if not children:
            continue
        # 若容器内部直接持有 3D 张量专家（如 ``mlp.experts``），则该容器
        # 只是 MoE 块（gate/shared_expert 混杂），不作为专家容器——真正的
        # 专家容器是内部那个 tensor 模块，它会在后续遍历中单独成组。
        if any(_tensor_expert_info(m)[0] for _, m in children):
            continue
        groups.append(ExpertGroup(path=name, module=module, children=children))

    if not groups:
        logger.warning(
            "No MoE expert containers found. If this is a MoE model, "
            "please check the module names; EFST will freeze everything."
        )
    return groups


def _normalize_expert_indices(
    groups: List[ExpertGroup],
    expert_indices: Optional[Union[Sequence[int], Dict[str, Sequence[Union[int, str]]]]],
) -> Dict[str, List[Union[int, str]]]:
    """把用户传入的专家选择归一化成 {group_path: [keys]}。"""
    selected: Dict[str, List[Union[int, str]]] = {}
    if expert_indices is None:
        for g in groups:
            selected[g.path] = list(range(len(g)))
        return selected

    if isinstance(expert_indices, dict):
        # key 可以是完整路径，也可以是末尾片段（如 "mlp.experts"）
        for g in groups:
            keys: List[Union[int, str]] = []
            for pat, vals in expert_indices.items():
                if g.path == pat or g.path.endswith("." + pat) or pat in g.path:
                    for v in vals:
                        keys.append(v)
            if keys:
                # 去重但保持顺序
                seen = set()
                uniq = []
                for v in keys:
                    s = str(v)
                    if s not in seen:
                        seen.add(s)
                        uniq.append(v)
                selected[g.path] = uniq
        return selected

    # 一个列表应用到所有专家组
    for g in groups:
        selected[g.path] = list(expert_indices)
    return selected


# ---------------------------------------------------------------------------
# 冻结 / 解冻
# ---------------------------------------------------------------------------
def freeze_all(model: nn.Module) -> None:
    """冻结全部参数。"""
    if model is None:
        raise ValueError("freeze_all: model 不能为 None")
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_all(model: nn.Module) -> None:
    """解冻全部参数。"""
    if model is None:
        raise ValueError("unfreeze_all: model 不能为 None")
    for p in model.parameters():
        p.requires_grad_(True)


def _unfreeze_module(module: nn.Module) -> int:
    n = 0
    for p in module.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)
        n += p.numel()
    return n


def _unfreeze_tensor_experts(group: ExpertGroup, keys: List[Union[int, str]]) -> int:
    """按专家索引解冻 3D 张量专家（先拆分，再只解冻选中专家的行）。"""
    split_3d_expert_params(group.module)
    n = 0
    for key in keys:
        idx = int(key)
        for name in group.tensor_param_names:
            plist = getattr(group.module, name, None)
            if isinstance(plist, nn.ParameterList) and 0 <= idx < len(plist):
                p = plist[idx]
                if not p.requires_grad:
                    p.requires_grad_(True)
                n += p.numel()
    return n


def _find_router_modules(model: nn.Module) -> List[nn.Module]:
    mods = []
    for name, module in model.named_modules():
        if not name:
            continue
        # gate_proj/up_proj/down_proj 是专家内部线性层，不是 router
        if any(x in name for x in ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3")):
            continue
        if _name_hits(name, _ROUTER_NAME_HINTS) or _name_hits(type(module).__name__, _ROUTER_NAME_HINTS):
            mods.append(module)
    return mods


def _tensor_expert_forward(module: nn.Module):
    """3D 张量专家（已拆成 ParameterList）的逐专家 forward。

    语义与 transformers 的 ``Qwen3NextExperts.forward`` 一致：
    按路由 one-hot 掩码 gather 每专家的 token，gate/up/down 计算后
    加权 index_add 回结果。与 ``grouped_mm_experts_forward`` 的差别只是
    权重从单一大 3D 张量换成 ``ParameterList``（索引语义相同）。
    """

    def forward(hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=module.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        act = module.act_fn
        for e in range(module.num_experts):
            m = expert_mask[e]
            if not bool(m.any()):
                continue
            top_k_pos, token_idx = torch.where(m)
            current_state = hidden_states[token_idx]
            gate, up = torch.nn.functional.linear(
                current_state, module.gate_up_proj[e]).chunk(2, dim=-1)
            current_hidden_states = act(gate) * up
            current_hidden_states = torch.nn.functional.linear(
                current_hidden_states, module.down_proj[e])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states)
        return final_hidden_states

    return forward


def split_3d_expert_params(module: nn.Module) -> int:
    """把 3D 张量专家权重（``[E, ...]`` 的单一大 Parameter）拆成
    ``nn.ParameterList`` of 2D Parameter（每个专家一个，初始全冻结）。

    拆分后按专家设置 ``requires_grad`` 才成为可能（PyTorch 的
    requires_grad 是参数级的，原始 3D 张量无法按行冻结）。拆分同时把
    ``forward`` 换成逐专家循环版（``ParameterList`` 索引语义与原来的
    ``tensor[i]`` 一致，但 transformers 的 grouped-matmul 实现需要单个
    3D 张量，不能直接用）。

    返回拆分掉的参数个数（0 表示该模块不是 3D 专家）。
    """
    split = 0
    for name in _TENSOR_EXPERT_PARAM_HINTS:
        p = getattr(module, name, None)
        if isinstance(p, nn.Parameter) and p.dim() == 3:
            E = p.shape[0]
            plist = nn.ParameterList(
                nn.Parameter(p[i].detach().clone(), requires_grad=False) for i in range(E)
            )
            # 先把旧 Parameter 从 _parameters 里摘掉，否则 __setattr__ 会因
            # 类型不符（ParameterList 不是 Parameter）抛 TypeError
            delattr(module, name)
            setattr(module, name, plist)
            split += 1
    if split:
        if not getattr(module, "num_experts", None):
            # 兜底：从第一个拆分参数的长度取专家数
            for name in _TENSOR_EXPERT_PARAM_HINTS:
                pl = getattr(module, name, None)
                if isinstance(pl, nn.ParameterList):
                    module.num_experts = len(pl)
                    break
        module.forward = _tensor_expert_forward(module)
    return split


def unfreeze_experts(
    model: nn.Module,
    expert_indices: Optional[Union[Sequence[int], Dict[str, Sequence[Union[int, str]]]]] = None,
    tune_router: bool = False,
) -> Dict[str, List[Union[int, str]]]:
    """冻结全模型后，只解冻指定专家（以及可选的 router）。

    返回实际解冻的 {group_path: [expert_keys]}。
    """
    groups = find_expert_groups(model)
    selected = _normalize_expert_indices(groups, expert_indices)

    for g in groups:
        keys = selected.get(g.path, [])
        if g.is_tensor:
            _unfreeze_tensor_experts(g, keys)
            continue
        for key in keys:
            try:
                expert = g.get_expert(key)
            except KeyError:
                logger.warning("Skip missing expert %s[%s]", g.path, key)
                continue
            _unfreeze_module(expert)

    if tune_router:
        for router in _find_router_modules(model):
            _unfreeze_module(router)

    return selected


# ---------------------------------------------------------------------------
# 路由统计：自动选 top-k 专家
# ---------------------------------------------------------------------------
def collect_expert_usage(
    model: nn.Module,
    dataloader: Iterable,
    forward_fn: Optional[Callable[[Any], Any]] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Dict[Union[int, str], int]]:
    """用校准数据统计每个专家被路由调用的次数。

    返回 ``{group_path: {expert_key: call_count}}``。
    统计完会自动移除 hook，不会影响后续训练。

    经典专家容器：在子模块上挂 forward hook 计数。
    3D 张量专家（无子模块）：forward 的第二个位置参数是 ``top_k_index``，
    直接统计其取值。
    """
    groups = find_expert_groups(model)
    counters: Dict[str, Dict[Union[int, str], int]] = {
        g.path: {k: 0 for k in g.keys()} for g in groups
    }
    handles = []

    def make_hook(path: str, key: Union[int, str]):
        def hook(module, args, output):
            counters[path][key] += 1
        return hook

    def make_tensor_hook(path: str):
        def hook(module, args, output):
            # 3D 专家 forward(hidden_states, top_k_index, top_k_weights, ...)
            idx = args[1] if len(args) > 1 else None
            if idx is None:
                return
            for e in idx.detach().cpu().flatten().tolist():
                e = int(e)
                if e in counters[path]:
                    counters[path][e] += 1
        return hook

    for g in groups:
        if g.is_tensor:
            handles.append(g.module.register_forward_hook(make_tensor_hook(g.path)))
        else:
            for key, expert in g.children:
                handles.append(expert.register_forward_hook(make_hook(g.path, key)))

    was_training = model.training   # 记录原来的 train/eval 模式，finally 恢复
    try:
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= max_batches:
                    break
                if forward_fn is not None:
                    forward_fn(batch)
                else:
                    model(batch)
    finally:
        for h in handles:
            h.remove()
        if was_training:   # 用户校准前若本来就是 train 模式，恢复（避免静默改掉 dropout 状态）
            model.train()
        model.train()

    return counters


def select_top_experts(
    model: nn.Module,
    dataloader: Iterable,
    top_k: int,
    forward_fn: Optional[Callable[[Any], Any]] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, List[Union[int, str]]]:
    """根据校准数据选出每个专家容器里最热的 top_k 个专家。"""
    if top_k < 1:
        raise ValueError(f"top_k 必须是 >=1 的整数，得到 {top_k}（0/负数会导致选不到 / 选反）")
    usage = collect_expert_usage(model, dataloader, forward_fn=forward_fn, max_batches=max_batches)
    selected: Dict[str, List[Union[int, str]]] = {}
    for path, counter in usage.items():
        ranked = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
        selected[path] = [k for k, _ in ranked[:top_k]]
        logger.info("EFST top-%d for %s: %s (usage=%s)", top_k, path, selected[path], ranked)
    return selected


# ---------------------------------------------------------------------------
# 专家内 LoRA（轻量实现，不依赖 peft）
# ---------------------------------------------------------------------------
class _LoraLinear(nn.Module):
    """给 nn.Linear 做最简 LoRA 注入。"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("lora_r must be > 0")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.scaling = alpha / r

        # 冻结原权重
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + lora_out * self.scaling


def _replace_linears_with_lora(module: nn.Module, r: int, alpha: float, dropout: float) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            lora_linear = _LoraLinear(child, r=r, alpha=alpha, dropout=dropout)
            setattr(module, name, lora_linear)
            replaced += 1
        else:
            replaced += _replace_linears_with_lora(child, r=r, alpha=alpha, dropout=dropout)
    return replaced


def add_lora_to_experts(
    model: nn.Module,
    expert_indices: Optional[Union[Sequence[int], Dict[str, Sequence[Union[int, str]]]]] = None,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> Dict[str, int]:
    """把选中专家内部的 ``nn.Linear`` 替换成带 LoRA 的版本。

    返回 ``{group_path: replaced_linear_count}``。
    注意：调用前通常应先 ``freeze_all(model)``，替换后 LoRA 参数默认可训练。
    3D 张量专家没有 ``nn.Linear`` 子模块，无法注入 LoRA（返回 0 并告警），
    请配合 ``unfreeze_experts`` 直接按专家行解冻。
    """
    groups = find_expert_groups(model)
    selected = _normalize_expert_indices(groups, expert_indices)
    replaced_map: Dict[str, int] = {}
    for g in groups:
        count = 0
        if g.is_tensor:
            logger.warning(
                "%s is a 3D tensor expert (no nn.Linear children); LoRA injection "
                "skipped - use unfreeze_experts() to tune expert rows directly",
                g.path,
            )
            replaced_map[g.path] = 0
            continue
        for key in selected.get(g.path, []):
            try:
                expert = g.get_expert(key)
            except KeyError:
                logger.warning("Skip missing expert %s[%s]", g.path, key)
                continue
            count += _replace_linears_with_lora(expert, r=r, alpha=alpha, dropout=dropout)
        replaced_map[g.path] = count
    return replaced_map


def _freeze_all_except_lora_and_router(model: nn.Module, tune_router: bool) -> int:
    trainable = 0
    router_param_ids = set()
    if tune_router:
        for m in _find_router_modules(model):
            router_param_ids.update(id(p) for p in m.parameters())
    for name, p in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        is_router = id(p) in router_param_ids
        if is_lora or is_router:
            p.requires_grad_(True)
            trainable += p.numel()
        else:
            p.requires_grad_(False)
    return trainable


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def apply_efst(model: nn.Module, config: Optional[EFSTConfig] = None) -> EFSTResult:
    """对 MoE 模型应用 EFST。

    流程：
    1. 如果配置了 top_k，先跑一小段校准数据统计路由，选出最热专家；
    2. 冻结全模型；
    3. 解冻指定专家（或给指定专家注入 LoRA）；
    4. 可选解冻 router。

    返回 ``EFSTResult``。
    """
    if config is None:
        config = EFSTConfig()
    if model is None:
        raise ValueError("apply_efst: model 不能为 None")

    groups = find_expert_groups(model)
    total = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 1. 确定要微调的专家
    selected: Dict[str, List[Union[int, str]]] = {}
    if config.expert_indices is not None:
        selected = _normalize_expert_indices(groups, config.expert_indices)
    elif config.top_k is not None:
        if config.calibration_dataloader is None:
            raise ValueError("EFSTConfig.top_k is set but calibration_dataloader is None")
        selected = select_top_experts(
            model,
            config.calibration_dataloader,
            top_k=config.top_k,
            forward_fn=config.calibration_forward_fn,
            max_batches=config.num_calibration_batches,
        )
    else:
        selected = _normalize_expert_indices(groups, None)

    # 2. 冻结全部
    freeze_all(model)

    # 3. 注入 LoRA 或直接解冻专家
    lora_injected = False
    if config.lora:
        replaced = add_lora_to_experts(
            model,
            expert_indices=selected,
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        lora_injected = sum(replaced.values()) > 0
        trainable_after = _freeze_all_except_lora_and_router(model, config.tune_router)
        # 3D 张量专家不支持 LoRA：直接按专家行解冻（lora 模式下的合理回退）
        for g in groups:
            if g.is_tensor:
                keys = selected.get(g.path, [])
                trainable_after += _unfreeze_tensor_experts(g, keys)
        if config.tune_router:  # _freeze_all_except_lora_and_router 已处理 router
            pass
    else:
        # 直接解冻专家本体
        unfreeze_experts(model, expert_indices=selected, tune_router=config.tune_router)
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    result = EFSTResult(
        selected_experts=selected,
        trainable_before=trainable_before,
        trainable_after=trainable_after,
        total_params=total,
        frozen_params=total - trainable_after,
        lora_injected=lora_injected,
        groups=groups,
    )
    logger.info("\n" + result.summary())
    return result


def report_trainable(model: nn.Module) -> str:
    """打印当前可训练参数分布。"""
    total = 0
    trainable = 0
    rows = []
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            rows.append(f"  {name}: {p.numel():,}")
    lines = [
        f"total params  : {total:,}",
        f"trainable     : {trainable:,} ({100.0 * trainable / max(total, 1):.3f}%)",
    ]
    if rows:
        lines.append("trainable params:")
        lines.extend(rows[:200])
        if len(rows) > 200:
            lines.append(f"  ... and {len(rows) - 200} more")
    return "\n".join(lines)
