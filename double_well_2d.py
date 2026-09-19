
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import copy
import random
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm, TwoSlopeNorm

torch.set_default_dtype(torch.float64)
plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class TeeStream:
    """同时把标准输出写到终端和 txt 文件。"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DoubleWellBimodal2D:
    def __init__(self):
        self.alpha = 2.0
        self.beta = 2.0
        self.gamma = 2.0
        self.sigma = 0.5

        self.mu1 = np.array([-2.0, 1.5])
        self.mu2 = np.array([2.0, 1.5])
        self.sig1 = 0.2
        self.sig2 = 0.5

    @property
    def D(self):
        return 0.5 * self.sigma**2

    def drift_torch(self, xy):
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        bx = self.alpha * x - self.beta * x**3
        by = -self.gamma * y
        return torch.cat([bx, by], dim=1)

    def simulate_stationary_samples(
        self,
        n_paths=25000,
        burn_in=8.0,
        sample_duration=2.0,
        sample_interval=0.1,
        dt=0.001,
        initial_left_fraction=0.5,
        seed=42,
    ):
        rng = np.random.default_rng(seed)

        if not (0.0 < initial_left_fraction < 1.0):
            raise ValueError("initial_left_fraction 必须在 (0, 1) 内。")

        n1 = int(round(n_paths * initial_left_fraction))
        n2 = n_paths - n1

        X = np.concatenate([
            rng.normal(self.mu1[0], self.sig1, n1),
            rng.normal(self.mu2[0], self.sig2, n2),
        ])
        Y = np.concatenate([
            rng.normal(self.mu1[1], self.sig1, n1),
            rng.normal(self.mu2[1], self.sig2, n2),
        ])

        burn_steps = int(round(burn_in / dt))
        sample_steps = int(round(sample_duration / dt))
        interval_steps = max(1, int(round(sample_interval / dt)))
        total_steps = burn_steps + sample_steps

        sqrt_dt_sigma = self.sigma * np.sqrt(dt)

        samples_x = []
        samples_y = []

        expected = n_paths * (sample_steps // interval_steps)
        print(
            f"[数据] 路径数={n_paths}，总步数={total_steps}，"
            f"预计收集约 {expected} 个稳态样本。"
        )
        print(
            f"[数据] 初始左势阱比例={initial_left_fraction:.2f}，"
            f"右势阱比例={1.0 - initial_left_fraction:.2f}"
        )

        for step in range(total_steps):
            bx = self.alpha * X - self.beta * X**3
            by = -self.gamma * Y

            X += bx * dt + sqrt_dt_sigma * rng.standard_normal(n_paths)
            Y += by * dt + sqrt_dt_sigma * rng.standard_normal(n_paths)

            if step >= burn_steps and (step - burn_steps) % interval_steps == 0:
                samples_x.append(X.copy())
                samples_y.append(Y.copy())

        X_all = np.concatenate(samples_x)
        Y_all = np.concatenate(samples_y)

        print(f"[数据] 实际收集稳态样本数：{X_all.size}")
        return X_all, Y_all

    def exact_steady_solution(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if X.shape != Y.shape or X.ndim != 2:
            raise ValueError("X 和 Y 必须是相同形状的二维规则网格。")

        power = (
            (2.0 / self.sigma**2)
            * (
                (self.alpha / 2.0) * X**2
                - (self.beta / 4.0) * X**4
                - (self.gamma / 2.0) * Y**2
            )
        )

        power -= np.max(power)
        p_unnorm = np.exp(power)

        x_values = np.unique(X)
        y_values = np.unique(Y)

        if x_values.size < 2 or y_values.size < 2:
            raise ValueError("归一化至少要求每个方向两个不同网格点。")

        dx = float(np.mean(np.diff(x_values)))
        dy = float(np.mean(np.diff(y_values)))

        z = float(np.sum(p_unnorm) * dx * dy)
        if (not np.isfinite(z)) or (z <= 0.0):
            raise FloatingPointError(f"解析密度归一化失败：Z={z}")

        return p_unnorm / z


def sobol_rectangle(n, bounds, seed=0):
    x_min, x_max, y_min, y_max = bounds
    engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
    unit = engine.draw(n).to(torch.float64)

    pts = torch.empty_like(unit)
    pts[:, 0] = x_min + (x_max - x_min) * unit[:, 0]
    pts[:, 1] = y_min + (y_max - y_min) * unit[:, 1]
    return pts



def symmetric_sobol_rectangle(n, bounds, seed=0):
    """
    旧版兼容函数：统一版正式训练不调用它。

    旧用途是两个纯 PINN 的方差缩减采样。

    在原均匀 Sobol 点基础上加入 (±x, ±y) 反射。
    对当前关于 x=0、y=0 对称的矩形与漂移，这不会改变连续积分目标，
    只降低有限批次导致的左右势阱不平衡。
    """
    if n <= 0:
        raise ValueError("n 必须为正整数。")

    base_n = max(1, (n + 3) // 4)
    base = sobol_rectangle(base_n, bounds, seed=seed)

    x = base[:, 0:1]
    y = base[:, 1:2]

    pts = torch.cat(
        [
            torch.cat([ x,  y], dim=1),
            torch.cat([-x,  y], dim=1),
            torch.cat([ x, -y], dim=1),
            torch.cat([-x, -y], dim=1),
        ],
        dim=0,
    )
    return pts[:n].contiguous()

def sobol_boundary(n_per_side, bounds, seed=0):
    """
    矩形区域边界采样。

    对 Ω = [x_min, x_max] × [y_min, y_max]，
    返回边界点、外法向量、边界边长权重。
    """
    x_min, x_max, y_min, y_max = bounds
    width = x_max - x_min
    height = y_max - y_min

    engine = torch.quasirandom.SobolEngine(dimension=1, scramble=True, seed=seed)
    s = engine.draw(n_per_side).to(torch.float64).reshape(-1, 1)

    y_vals = y_min + height * s
    x_vals = x_min + width * s

    left = torch.cat([
        torch.full_like(y_vals, x_min),
        y_vals,
    ], dim=1)
    right = torch.cat([
        torch.full_like(y_vals, x_max),
        y_vals,
    ], dim=1)
    bottom = torch.cat([
        x_vals,
        torch.full_like(x_vals, y_min),
    ], dim=1)
    top = torch.cat([
        x_vals,
        torch.full_like(x_vals, y_max),
    ], dim=1)

    pts = torch.cat([left, right, bottom, top], dim=0)

    n_left = torch.tensor([-1.0, 0.0], dtype=torch.float64).repeat(n_per_side, 1)
    n_right = torch.tensor([1.0, 0.0], dtype=torch.float64).repeat(n_per_side, 1)
    n_bottom = torch.tensor([0.0, -1.0], dtype=torch.float64).repeat(n_per_side, 1)
    n_top = torch.tensor([0.0, 1.0], dtype=torch.float64).repeat(n_per_side, 1)

    normals = torch.cat([n_left, n_right, n_bottom, n_top], dim=0)

    w_left = torch.full((n_per_side, 1), height, dtype=torch.float64)
    w_right = torch.full((n_per_side, 1), height, dtype=torch.float64)
    w_bottom = torch.full((n_per_side, 1), width, dtype=torch.float64)
    w_top = torch.full((n_per_side, 1), width, dtype=torch.float64)

    weights = torch.cat([w_left, w_right, w_bottom, w_top], dim=0)

    return pts, normals, weights


def build_histogram_reference(
    x_samples,
    y_samples,
    bounds=(-2.0, 2.0, -1.2, 1.2),
    bin_width=0.04,
):
    """
    保持原始直方图逻辑不变。
    """
    x_min, x_max, y_min, y_max = bounds

    nx = int(round((x_max - x_min) / bin_width))
    ny = int(round((y_max - y_min) / bin_width))

    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)

    counts, _, _ = np.histogram2d(
        x_samples,
        y_samples,
        bins=[x_edges, y_edges],
    )

    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    density = counts / (x_samples.size * dx * dy)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="ij")
    xy_data = np.column_stack([Xc.ravel(), Yc.ravel()])
    v_data = density.ravel()

    print(f"[直方图] 网格数={nx}×{ny}={nx*ny}，网格宽度=({dx:.4f}, {dy:.4f})")
    print(f"[直方图] 截断区域内 Monte Carlo 质量：{density.sum() * dx * dy:.8f}")

    return (
        torch.tensor(xy_data, dtype=torch.float64),
        torch.tensor(v_data, dtype=torch.float64).unsqueeze(1),
    )


def exact_density_on_grid(process, X, Y):
    return process.exact_steady_solution(X, Y)


def validate_reference_error(process, y_data, v_data):
    pts = y_data.numpy()
    x_unique = np.unique(pts[:, 0])
    y_unique = np.unique(pts[:, 1])

    dx = x_unique[1] - x_unique[0]
    dy = y_unique[1] - y_unique[0]

    Xg, Yg = np.meshgrid(x_unique, y_unique, indexing="ij")
    exact = exact_density_on_grid(process, Xg, Yg)

    exact_flat = exact.ravel()
    ref_flat = v_data.squeeze(1).numpy()

    mask = exact_flat > 0.01 * exact_flat.max()

    rel_err = np.mean(np.abs(ref_flat[mask] - exact_flat[mask]) / (exact_flat[mask] + 1e-14))
    rel_l2 = np.sqrt(
        np.sum((ref_flat - exact_flat)**2) * dx * dy
        / (np.sum(exact_flat**2) * dx * dy + 1e-30)
    )

    print("\n" + "=" * 76)
    print("Monte Carlo 直方图参考密度质量检查")
    print("-" * 76)
    print(f"有效密度区域平均相对误差：{100.0 * rel_err:.4f}%")
    print(f"相对 L2 误差：{100.0 * rel_l2:.4f}%")
    print("=" * 76 + "\n")


def inverse_softplus(value):
    value = torch.as_tensor(value, dtype=torch.float64)
    return torch.log(torch.expm1(value))


class GeneralFPNN(nn.Module):
    """
    正密度网络。

    与 8D 严格版本中的含数据网络保持同一幅值语义：
    不含 trainable gain，也不做训练后 output scaling；
    数据方法的绝对幅值只能通过普通 MC MSE 在训练中学习。
    """
    def __init__(self, x_scale=2.0, y_scale=1.2, width=128, depth=4):
        super().__init__()

        self.register_buffer(
            "input_scale",
            torch.tensor([x_scale, y_scale], dtype=torch.float64).view(1, 2),
        )

        layers = []
        in_dim = 2
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.Tanh())
            in_dim = width
        layers.append(nn.Linear(width, 1))
        self.shape_net = nn.Sequential(*layers)

        for m in self.shape_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        last = self.shape_net[-1]
        nn.init.normal_(last.weight, mean=0.0, std=2e-2)
        nn.init.constant_(last.bias, -1.0)

    def base_density(self, xy):
        z = xy / self.input_scale
        raw = self.shape_net(z)
        return nn.functional.softplus(raw) + 1e-12

    def forward(self, xy):
        return self.base_density(xy)


class MassNormalizedDensity(nn.Module):
    """
    旧缓存兼容用 hard-mass 包装器。

    新训练的纯 PINN 不再调用本类；单位积分改为显式 L_normal 损失。
    """
    def __init__(self, base_net, norm_points, domain_area, target_mass):
        super().__init__()
        self.base_net = base_net

        self.register_buffer("norm_points", norm_points.detach().clone())
        self.register_buffer(
            "target_mass_tensor",
            torch.tensor(float(target_mass), dtype=torch.float64),
        )
        self.domain_area = float(domain_area)

    def base_density(self, xy):
        return self.base_net.base_density(xy)

    def normalizer(self):
        q = self.base_net.base_density(self.norm_points)
        z = self.domain_area * q.mean()
        return z + 1e-30


    def forward(self, xy):
        q = self.base_net.base_density(xy)
        return self.target_mass_tensor * q / self.normalizer()



def normalized_data_loss(net, x, target, data_scale):
    mse = torch.mean((net(x) - target)**2)
    return mse / data_scale, mse


def energy_identity_loss(
    net,
    pts,
    process,
    domain_area,
    u_floor=1e-12,
    create_graph=True,
    return_details=False,
):
    """
    原始 Weak loss。纯 Weak PINN 与 Weak + data 共用本函数。
    """
    x = pts.detach().clone().requires_grad_(True)
    u = net(x)

    grad_u = grad(
        outputs=u.sum(),
        inputs=x,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]

    b = process.drift_torch(x)
    flux_residual = process.D * grad_u - b * u

    u_safe = torch.clamp(u, min=u_floor)
    pointwise_energy = torch.sum(flux_residual**2, dim=1, keepdim=True) / u_safe

    raw_energy = domain_area * pointwise_energy.mean()
    mass_like = domain_area * u.mean()
    relative_energy = raw_energy / (mass_like + 1e-14)

    if return_details:
        return relative_energy, raw_energy, mass_like
    return relative_energy


def strong_fp_residual_loss(net, pts, process, domain_area):
    """
    原始稳态 Fokker-Planck 强残差。
    纯 Strong PINN 与 Strong + data 共用本函数。
    """
    x = pts.detach().clone().requires_grad_(True)
    u = net(x)

    b = process.drift_torch(x)
    bu = b * u

    grad_bu_x = grad(bu[:, 0:1].sum(), x, create_graph=True, retain_graph=True)[0][:, 0:1]
    grad_bu_y = grad(bu[:, 1:2].sum(), x, create_graph=True, retain_graph=True)[0][:, 1:2]
    div_bu = grad_bu_x + grad_bu_y

    grad_u = grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    u_xx = grad(grad_u[:, 0:1].sum(), x, create_graph=True, retain_graph=True)[0][:, 0:1]
    u_yy = grad(grad_u[:, 1:2].sum(), x, create_graph=True, retain_graph=True)[0][:, 1:2]
    lap_u = u_xx + u_yy

    residual = -div_bu + process.D * lap_u
    return domain_area * torch.mean(residual**2)



def noflux_boundary_loss(
    net,
    boundary_pts,
    boundary_normals,
    boundary_weights,
    process,
):
    """
    原零流边界损失，保留兼容。
    """
    x = boundary_pts.detach().clone().requires_grad_(True)
    u = net(x)

    grad_u = grad(
        outputs=u.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True,
    )[0]

    flux_residual = process.D * grad_u - process.drift_torch(x) * u
    normal_flux = torch.sum(flux_residual * boundary_normals, dim=1, keepdim=True)

    return torch.mean(boundary_weights * normal_flux**2) / (
        torch.mean(boundary_weights) + 1e-30
    )



def estimate_reference_mass_from_grid(y_data, v_data):
    pts = y_data.detach().cpu().numpy()
    values = v_data.detach().cpu().numpy().reshape(-1)

    x_unique = np.unique(pts[:, 0])
    y_unique = np.unique(pts[:, 1])

    dx = float(np.mean(np.diff(x_unique)))
    dy = float(np.mean(np.diff(y_unique)))
    return float(np.sum(values) * dx * dy)


def make_reference_subset(y_ref, v_ref, n_ref=1024, seed=0, density_fraction=0.7):
    n_total = y_ref.shape[0]
    if n_ref is None or n_ref >= n_total:
        return y_ref.clone(), v_ref.clone()

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    y_cpu = y_ref.detach().cpu()
    v_cpu = v_ref.detach().cpu()

    n_density = int(round(n_ref * density_fraction))
    n_uniform = n_ref - n_density

    weights = v_cpu.reshape(-1).clone()
    weights = weights + 1e-14
    weights = weights / weights.sum()

    idx_density = torch.multinomial(
        weights,
        num_samples=min(n_density, n_total),
        replacement=False,
        generator=g,
    )
    idx_uniform = torch.randperm(n_total, generator=g)[:min(n_uniform, n_total)]

    idx = torch.unique(torch.cat([idx_density, idx_uniform], dim=0))
    while idx.numel() < n_ref:
        extra = torch.randint(0, n_total, (n_ref - idx.numel(),), generator=g)
        idx = torch.unique(torch.cat([idx, extra], dim=0))

    idx = idx[:n_ref]
    return y_cpu[idx].clone(), v_cpu[idx].clone()


def compute_physics_loss(physics_type, net, pts, process, domain_area):
    if physics_type == "weak":
        return energy_identity_loss(
            net=net,
            pts=pts,
            process=process,
            domain_area=domain_area,
            create_graph=True,
        )
    if physics_type == "strong":
        return strong_fp_residual_loss(
            net=net,
            pts=pts,
            process=process,
            domain_area=domain_area,
        )
    raise ValueError(f"未知 physics_type: {physics_type}")


def compute_mass_loss(net, pts, domain_area, target_mass=1.0):
    """Monte-Carlo/Sobol quadrature for L_normal=(int_Omega u dx-target)^2."""
    u = net(pts.detach())
    mass_like = domain_area * u.mean()
    target = torch.as_tensor(target_mass, dtype=mass_like.dtype, device=mass_like.device)
    mass_loss = (mass_like - target) ** 2
    return mass_loss, mass_like




METHODS = {
    "Data-only NN": dict(
        use_data=True,
        physics=None,
        use_lbfgs=False,
    ),
    "Strong PINN": dict(
        use_data=False,
        physics="strong",
        use_lbfgs=False,
        boundary="noflux",
        hard_mass=False,
        physics_lr_multiplier=4.0,
        physics_grad_clip=5.0,
    ),
    "Weak PINN": dict(
        use_data=False,
        physics="weak",
        use_lbfgs=False,
        boundary="noflux",
        hard_mass=False,
        physics_lr_multiplier=4.0,
        physics_grad_clip=5.0,
    ),

    # 与 8D 命名和实验语义统一：
    # Adam-only 严格停在 Stage B；+L-BFGS 只是在同一 Stage-B checkpoint 上追加 Stage C。
    "Strong + MC Adam-only": dict(
        use_data=True,
        physics="strong",
        use_lbfgs=False,
    ),
    "Weak + MC Adam-only": dict(
        use_data=True,
        physics="weak",
        use_lbfgs=False,
    ),
    "Strong + MC + L-BFGS": dict(
        use_data=True,
        physics="strong",
        use_lbfgs=True,
    ),
    "Weak + MC + L-BFGS": dict(
        use_data=True,
        physics="weak",
        use_lbfgs=True,
    ),
}

for _cfg in METHODS.values():
    _cfg.setdefault("boundary", None)
    _cfg.setdefault("boundary_weight", 0.0)
    _cfg.setdefault("hard_mass", False)
    _cfg.setdefault("physics_lr_multiplier", 1.0)
    _cfg.setdefault("physics_grad_clip", 1.0)
    _cfg.setdefault("data_steps_per_cycle", 1)
    _cfg.setdefault("checkpoint_data_weight", 1.0)
    _cfg.setdefault("checkpoint_physics_weight", 1.0)


def method_coordinate_budget(method_name, args, effective_n_ref):
    """
    与 8D 一样统计真正进入训练或 checkpoint/model selection 的唯一坐标。

    不计：
    - Euler-Maruyama 原始 MC 快照（它们只用于形成直方图）；
    - 最终解析评价网格；
    - 仅打印质量用的诊断点。

    Stage C 复用 Stage B 坐标，所以不会增加唯一坐标。
    """
    cfg = METHODS[method_name]
    details = {
        "mc_bins": int(effective_n_ref) if cfg["use_data"] else 0,
        "physics_pool": int(args.physics_pool_size) if cfg["physics"] else 0,
        "physics_monitor": int(args.fixed_check_size) if cfg["physics"] else 0,
        "hard_mass": int(args.hard_mass_quad_size) if cfg["hard_mass"] else 0,
        "boundary": (
            int(4 * args.boundary_n_per_side)
            if cfg["boundary"] == "noflux"
            else 0
        ),
    }
    details["total"] = int(sum(details.values()))
    return details


def _score_improved(new_value, old_value, min_delta=0.0):
    if not np.isfinite(new_value):
        return False
    if not np.isfinite(old_value):
        return True
    return new_value < old_value - float(min_delta)


def _new_base_net(args, device):
    return GeneralFPNN(
        x_scale=2.0,
        y_scale=1.2,
        width=args.width,
        depth=args.depth,
    ).to(device)


def _clone_general_net(net, args, device):
    """
    混合方法 Stage B 后复制网络，保证 +L-BFGS 与 Adam-only
    真正来自同一个 checkpoint，而不是重新训练一遍。
    """
    if not isinstance(net, GeneralFPNN):
        raise TypeError("混合方法应使用未包装的 GeneralFPNN。")
    cloned = _new_base_net(args, device)
    cloned.load_state_dict(copy.deepcopy(net.state_dict()))
    return cloned


def train_stage_ab(
    method_name,
    net,
    y_data,
    v_data,
    process,
    device,
    physics_bounds,
    target_mass,
    args,
    seed,
):
    """
    只执行 Stage A + Stage B。

    这是统一的 Adam 阶段：
    - Data-only: Stage A + Stage B(data Adam)
    - pure PINN: Stage B(physics Adam)
    - MC+physics: Stage A + Stage B(data Adam + physics Adam)

    重要：
    - Stage-B 起点先登记为候选 checkpoint；
    - 返回 Stage-B 最优网络和原 physics pool / monitor，
      供配对的 +L-BFGS 方法直接继续 Stage C。
    """
    cfg = METHODS[method_name]
    net = net.to(device)

    x_min, x_max, y_min, y_max = physics_bounds
    domain_area = (x_max - x_min) * (y_max - y_min)

    if cfg["use_data"]:
        if y_data is None or v_data is None:
            raise ValueError(f"{method_name} 需要 MC 数据。")
        y_data = y_data.to(device)
        v_data = v_data.to(device)
    else:
        # 仅为统一接口；不会进入数据 loss。
        y_data = torch.empty((0, 2), dtype=torch.float64, device=device)
        v_data = torch.empty((0, 1), dtype=torch.float64, device=device)

    # Pure PINNs use an explicit normalization penalty; no hard rescaling is applied.
    is_pure_pinn = (not cfg["use_data"]) and (cfg["physics"] is not None)

    print("\n" + "=" * 96)
    print(f"开始训练：{method_name}")
    print("=" * 96)
    if cfg["use_data"]:
        print("[数据] Stage A/B 均为 MC 直方图箱中心普通点值 MSE")
        print("[幅值] 无 gain、无解析幅值校准、无训练后质量缩放")
    if cfg["physics"] is not None:
        print("[物理] Stage B 使用固定 Sobol physics pool + 独立固定 monitor")
        print("[Stage B] data Adam 与 physics Adam 分开更新；起点先登记为 checkpoint")
    if is_pure_pinn:
        print(
            "[纯 PINN objective] "
            "lambda_phy*L_phy + lambda_boundary*L_boundary + lambda_normal*L_normal"
        )
        print(
            f"[纯 PINN 权重] phy={args.pure_physics_weight:g}, "
            f"boundary={args.pure_boundary_weight:g}, "
            f"normal={args.pure_normal_weight:g}; "
            "L_normal=(int_Omega u dx - 1)^2"
        )

    data_scale = (
        torch.mean(v_data**2).detach() + 1e-14
        if cfg["use_data"]
        else torch.tensor(1.0, dtype=torch.float64, device=device)
    )

    data_loader = None
    if cfg["use_data"]:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 12345)
        data_loader = DataLoader(
            TensorDataset(y_data, v_data),
            batch_size=min(args.data_batch_size, y_data.shape[0]),
            shuffle=True,
            drop_last=False,
            generator=gen,
        )

    history = {
        "method": method_name,
        "stage_a_best_epoch": None,
        "stage_b_best_cycle": None,
        "stage_b_best_score": None,
        "stage_c_best_iter": None,
        "stage_c_best_monitor": None,
        # Diagnostic histories for paper figures only.
        "stage_a_epoch": [],
        "stage_a_data_mse": [],
        "stage_b_cycle": [],
        "stage_b_monitor_score": [],
        "stage_b_data_mse": [],
        "stage_b_physics_loss": [],
        "stage_c_iter": [],
        "stage_c_monitor_score": [],
        "stage_c_data_mse": [],
        "stage_c_physics_loss": [],
    }

    # ---------------------------------------------------------
    # Stage A: ordinary MC pointwise MSE pretraining
    # ---------------------------------------------------------
    if cfg["use_data"] and args.pretrain_epochs > 0:
        print("\n[Stage A] MC 箱中心点值数据预训练")
        opt_pre = optim.AdamW(
            net.parameters(),
            lr=1.0e-3,
            weight_decay=1e-9,
        )
        sch_pre = optim.lr_scheduler.CosineAnnealingLR(
            opt_pre,
            T_max=args.pretrain_epochs,
            eta_min=5e-5,
        )

        best_pre_state = copy.deepcopy(net.state_dict())
        best_pre_mse = float("inf")
        best_pre_epoch = 0

        for epoch in range(args.pretrain_epochs):
            net.train()
            total_mse = 0.0
            total_count = 0

            for xb, vb in data_loader:
                opt_pre.zero_grad(set_to_none=True)
                rel, mse = normalized_data_loss(net, xb, vb, data_scale)
                rel.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt_pre.step()
                total_mse += float(mse.detach()) * xb.shape[0]
                total_count += xb.shape[0]

            sch_pre.step()
            epoch_mse = total_mse / max(total_count, 1)
            history["stage_a_epoch"].append(epoch + 1)
            history["stage_a_data_mse"].append(float(epoch_mse))

            if epoch_mse < best_pre_mse:
                best_pre_mse = epoch_mse
                best_pre_epoch = epoch + 1
                best_pre_state = copy.deepcopy(net.state_dict())

            if (
                (epoch + 1) % max(100, args.pretrain_epochs // 4) == 0
                or epoch == args.pretrain_epochs - 1
            ):
                print(
                    f"预训练轮次 {epoch + 1:5d} | "
                    f"数据 MSE={epoch_mse:.8e} | "
                    f"best_epoch={best_pre_epoch}"
                )

        net.load_state_dict(best_pre_state)
        history["stage_a_best_epoch"] = best_pre_epoch

    # ---------------------------------------------------------
    # Fixed Stage-B coordinates.
    # No symmetric reflection augmentation: use ordinary Sobol for all methods.
    # ---------------------------------------------------------
    if cfg["physics"] is not None:
        physics_pool = sobol_rectangle(
            args.physics_pool_size,
            physics_bounds,
            seed=888 + seed,
        ).to(device)
        fixed_check_points = sobol_rectangle(
            args.fixed_check_size,
            physics_bounds,
            seed=2027 + seed,
        ).to(device)
    else:
        physics_pool = None
        # Diagnostic-only points; they do not enter checkpoint score.
        fixed_check_points = sobol_rectangle(
            args.fixed_check_size,
            physics_bounds,
            seed=2027 + seed,
        ).to(device)

    if cfg["boundary"] == "noflux":
        boundary_pts, boundary_normals, boundary_weights = sobol_boundary(
            n_per_side=args.boundary_n_per_side,
            bounds=physics_bounds,
            seed=3030 + seed,
        )
        boundary_pts = boundary_pts.to(device)
        boundary_normals = boundary_normals.to(device)
        boundary_weights = boundary_weights.to(device)
    else:
        boundary_pts = boundary_normals = boundary_weights = None

    def physics_loss(points):
        if cfg["physics"] is None:
            return torch.tensor(0.0, dtype=torch.float64, device=device)
        return compute_physics_loss(
            cfg["physics"], net, points, process, domain_area
        )

    def boundary_loss():
        if boundary_pts is None:
            return torch.tensor(0.0, dtype=torch.float64, device=device)
        return noflux_boundary_loss(
            net,
            boundary_pts,
            boundary_normals,
            boundary_weights,
            process,
        )

    if cfg["use_data"]:
        with torch.no_grad():
            data_start, data_mse_start = normalized_data_loss(
                net, y_data, v_data, data_scale
            )
        data_loss_scale = torch.clamp(data_start.detach(), min=1e-10)
    else:
        data_start = data_mse_start = torch.tensor(
            0.0, dtype=torch.float64, device=device
        )
        data_loss_scale = torch.tensor(
            1.0, dtype=torch.float64, device=device
        )

    if cfg["physics"] is not None:
        with torch.enable_grad():
            physics_start = physics_loss(fixed_check_points).detach()
        physics_loss_scale = torch.clamp(
            physics_start.detach(), min=1e-10
        )
    else:
        physics_start = torch.tensor(
            0.0, dtype=torch.float64, device=device
        )
        physics_loss_scale = torch.tensor(
            1.0, dtype=torch.float64, device=device
        )

    if boundary_pts is not None:
        with torch.enable_grad():
            boundary_start = boundary_loss().detach()
        boundary_loss_scale = torch.clamp(
            boundary_start.detach(), min=1e-10
        )
    else:
        boundary_start = torch.tensor(
            0.0, dtype=torch.float64, device=device
        )
        boundary_loss_scale = torch.tensor(
            1.0, dtype=torch.float64, device=device
        )

    if is_pure_pinn:
        with torch.no_grad():
            normal_start, mass_start = compute_mass_loss(
                net, fixed_check_points, domain_area, target_mass=1.0
            )
    else:
        normal_start = torch.tensor(
            0.0, dtype=torch.float64, device=device
        )
        with torch.no_grad():
            mass_start = domain_area * net(fixed_check_points).mean()

    data_w = float(cfg["checkpoint_data_weight"])
    physics_w = float(cfg["checkpoint_physics_weight"])
    boundary_w = float(cfg["boundary_weight"])
    pure_physics_w = float(args.pure_physics_weight)
    pure_boundary_w = float(args.pure_boundary_weight)
    pure_normal_w = float(args.pure_normal_weight)

    if is_pure_pinn:
        best_score = float(
            pure_physics_w * physics_start
            + pure_boundary_w * boundary_start
            + pure_normal_w * normal_start
        )
    else:
        best_score = 0.0
        if cfg["use_data"]:
            best_score += data_w
        if cfg["physics"] is not None:
            best_score += physics_w
        if boundary_pts is not None:
            best_score += boundary_w

    best_state = copy.deepcopy(net.state_dict())
    best_cycle = 0

    print("\n[Stage B 起点 = 合法 checkpoint]")
    print(f"数据相对损失 = {data_start.item():.8e}")
    print(f"数据 MSE      = {data_mse_start.item():.8e}")
    print(f"物理损失      = {physics_start.item():.8e}")
    print(f"边界损失      = {boundary_start.item():.8e}")
    print(f"归一化损失    = {normal_start.item():.8e}")
    print(f"诊断质量      = {mass_start.item():.8f}")
    print(f"起点 score    = {best_score:.8e}")

    optimizer_data = (
        optim.Adam(net.parameters(), lr=args.data_lr)
        if cfg["use_data"]
        else None
    )
    effective_physics_lr = args.physics_lr * float(
        cfg["physics_lr_multiplier"]
    )
    optimizer_physics = (
        optim.Adam(
            [p for p in net.parameters() if p.requires_grad],
            lr=effective_physics_lr,
        )
        if cfg["physics"] is not None
        else None
    )

    scheduler_data = (
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer_data,
            T_max=max(1, args.joint_cycles),
            eta_min=1e-5,
        )
        if optimizer_data is not None
        else None
    )
    scheduler_physics = (
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer_physics,
            T_max=max(1, args.joint_cycles),
            eta_min=min(5e-6, 0.1 * effective_physics_lr),
        )
        if optimizer_physics is not None
        else None
    )

    data_iterator = iter(data_loader) if data_loader is not None else None
    n_pool = physics_pool.shape[0] if physics_pool is not None else 0
    log_every = max(100, args.joint_cycles // 8)

    print("\n[Stage B] 分离 Adam：physics update + data update")

    for cycle in range(args.joint_cycles):
        net.train()

        if optimizer_physics is not None:
            start_idx = (cycle * args.physics_batch_size) % n_pool
            end_idx = start_idx + args.physics_batch_size
            if end_idx <= n_pool:
                physics_pts = physics_pool[start_idx:end_idx]
            else:
                physics_pts = torch.cat(
                    [
                        physics_pool[start_idx:],
                        physics_pool[:end_idx - n_pool],
                    ],
                    dim=0,
                )

            optimizer_physics.zero_grad(set_to_none=True)
            p_loss = physics_loss(physics_pts)

            if is_pure_pinn:
                b_loss = boundary_loss()
                n_loss, _ = compute_mass_loss(
                    net, physics_pts, domain_area, target_mass=1.0
                )
                total_phys = (
                    pure_physics_w * p_loss
                    + pure_boundary_w * b_loss
                    + pure_normal_w * n_loss
                )
            else:
                total_phys = p_loss / physics_loss_scale
                if boundary_pts is not None:
                    b_loss = boundary_loss()
                    total_phys = total_phys + boundary_w * (
                        b_loss / boundary_loss_scale
                    )

            total_phys.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.requires_grad],
                float(cfg["physics_grad_clip"]),
            )
            optimizer_physics.step()
            scheduler_physics.step()

        if optimizer_data is not None:
            for _ in range(int(cfg["data_steps_per_cycle"])):
                try:
                    xb, vb = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(data_loader)
                    xb, vb = next(data_iterator)

                optimizer_data.zero_grad(set_to_none=True)
                d_loss, _ = normalized_data_loss(
                    net, xb, vb, data_scale
                )
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer_data.step()
            scheduler_data.step()

        if (
            (cycle + 1) % log_every == 0
            or cycle == args.joint_cycles - 1
        ):
            net.eval()

            if cfg["use_data"]:
                with torch.no_grad():
                    check_data, check_data_mse = normalized_data_loss(
                        net, y_data, v_data, data_scale
                    )
                data_ratio = float(
                    (check_data / data_loss_scale).detach()
                )
            else:
                check_data = check_data_mse = torch.tensor(
                    0.0, dtype=torch.float64, device=device
                )
                data_ratio = 0.0

            if cfg["physics"] is not None:
                with torch.enable_grad():
                    check_phys = physics_loss(
                        fixed_check_points
                    ).detach()
                physics_ratio = float(
                    (check_phys / physics_loss_scale).detach()
                )
            else:
                check_phys = torch.tensor(
                    0.0, dtype=torch.float64, device=device
                )
                physics_ratio = 0.0

            if boundary_pts is not None:
                with torch.enable_grad():
                    check_boundary = boundary_loss().detach()
                boundary_ratio = float(
                    (check_boundary / boundary_loss_scale).detach()
                )
            else:
                check_boundary = torch.tensor(
                    0.0, dtype=torch.float64, device=device
                )
                boundary_ratio = 0.0

            if is_pure_pinn:
                with torch.no_grad():
                    check_normal, check_mass = compute_mass_loss(
                        net,
                        fixed_check_points,
                        domain_area,
                        target_mass=1.0,
                    )
                score = float(
                    pure_physics_w * check_phys
                    + pure_boundary_w * check_boundary
                    + pure_normal_w * check_normal
                )
            else:
                check_normal = torch.tensor(
                    0.0, dtype=torch.float64, device=device
                )
                with torch.no_grad():
                    check_mass = (
                        domain_area * net(fixed_check_points).mean()
                    )

                score = 0.0
                if cfg["use_data"]:
                    score += data_w * data_ratio
                if cfg["physics"] is not None:
                    score += physics_w * physics_ratio
                if boundary_pts is not None:
                    score += boundary_w * boundary_ratio

            history["stage_b_cycle"].append(cycle + 1)
            history["stage_b_monitor_score"].append(float(score))
            history["stage_b_data_mse"].append(float(check_data_mse.detach()))
            history["stage_b_physics_loss"].append(float(check_phys.detach()))

            if _score_improved(score, best_score):
                best_score = score
                best_cycle = cycle + 1
                best_state = copy.deepcopy(net.state_dict())

            print(
                f"循环 {cycle + 1:5d} | score={score:.8e} | "
                f"data ratio={data_ratio:.6e} | "
                f"physics ratio={physics_ratio:.6e} | "
                f"data MSE={check_data_mse.item():.8e} | "
                f"physics={check_phys.item():.8e} | "
                f"boundary={check_boundary.item():.8e} | "
                f"normal={check_normal.item():.8e} | "
                f"mass={check_mass.item():.8f} | "
                f"best_cycle={best_cycle}"
            )

    net.load_state_dict(best_state)
    history["stage_b_best_cycle"] = best_cycle
    history["stage_b_best_score"] = float(best_score)

    with torch.no_grad():
        checksum = 0.0
        for p in net.parameters():
            checksum += float(p.detach().double().sum().cpu())
    history["stage_b_parameter_checksum"] = checksum

    print(
        f"[Stage B checkpoint] best_cycle={best_cycle}, "
        f"score={best_score:.8e}, checksum={checksum:.12e}"
    )

    context = {
        "physics_pool": physics_pool,
        "fixed_check_points": fixed_check_points,
        "data_scale": data_scale,
        "y_data": y_data,
        "v_data": v_data,
        "domain_area": domain_area,
        "physics_bounds": physics_bounds,
        "seed": seed,
    }
    return net, history, context


def run_stage_c_lbfgs(
    method_name,
    net,
    process,
    device,
    context,
    args,
    history=None,
):
    """
    从已经选定的 Stage-B checkpoint 继续 Stage C。

    与 8D strict 结构一致：
    - 不重新做 Stage A/B；
    - 不新增 physics 坐标；
    - train 点取自 Stage-B physics pool；
    - monitor 点取自 Stage-B fixed check；
    - 分块执行 L-BFGS 并按 monitor 回滚到最优状态。
    """
    cfg = METHODS[method_name]
    if not (
        cfg["use_lbfgs"]
        and cfg["use_data"]
        and cfg["physics"] is not None
    ):
        raise ValueError(f"{method_name} 不是 Stage-C 方法。")
    if not isinstance(net, GeneralFPNN):
        raise TypeError("含 MC 的 Stage-C 网络必须是原始 GeneralFPNN。")

    pool = context["physics_pool"]
    check = context["fixed_check_points"]
    y_data = context["y_data"]
    v_data = context["v_data"]
    data_scale = context["data_scale"]
    domain_area = context["domain_area"]

    if pool is None:
        raise RuntimeError("Stage C 缺少 Stage-B physics pool。")
    if args.lbfgs_point_size > pool.shape[0]:
        raise ValueError(
            "lbfgs_point_size 必须 <= physics_pool_size；"
            "Stage C 不允许新建物理点。"
        )
    if args.lbfgs_check_size > check.shape[0]:
        raise ValueError(
            "lbfgs_check_size 必须 <= fixed_check_size；"
            "Stage C 不允许新建 monitor 点。"
        )

    lp = pool[: args.lbfgs_point_size]
    lc = check[: args.lbfgs_check_size]

    def physics_loss(points):
        return compute_physics_loss(
            cfg["physics"], net, points, process, domain_area
        )

    print(
        "\n[Stage C] 从共享的 Stage-B checkpoint 出发；"
        "复用同一批 physics pool / monitor 做 L-BFGS"
    )
    print(
        f"[Stage C physics] reuse train={lp.shape[0]} / "
        f"pool={pool.shape[0]}, monitor={lc.shape[0]} / "
        f"check={check.shape[0]}"
    )
    print(
        f"[Stage C data] reuse MC bins={y_data.shape[0]}，不新增数据坐标"
    )

    with torch.enable_grad():
        p_start = physics_loss(lp).detach()
        p_monitor_start = physics_loss(lc).detach()
    with torch.no_grad():
        d_start, _ = normalized_data_loss(
            net, y_data, v_data, data_scale
        )
        d_monitor_start, _ = normalized_data_loss(
            net, y_data, v_data, data_scale
        )

    p_scale = torch.clamp(p_start, min=1e-10)
    d_scale = torch.clamp(d_start.detach(), min=1e-10)
    p_monitor_scale = torch.clamp(
        p_monitor_start, min=1e-10
    )
    d_monitor_scale = torch.clamp(
        d_monitor_start.detach(), min=1e-10
    )

    best_state = copy.deepcopy(net.state_dict())
    best_monitor = (
        float(args.lbfgs_data_weight)
        + float(args.lbfgs_physics_weight)
    )
    best_iter = 0
    no_improve = 0
    total_iter = 0
    closure_calls = {"n": 0}

    chunk = min(
        max(1, args.lbfgs_check_every),
        max(1, args.lbfgs_max_iter),
    )
    lb = optim.LBFGS(
        net.parameters(),
        lr=args.lbfgs_lr,
        max_iter=chunk,
        max_eval=max(50, int(1.5 * chunk)),
        tolerance_grad=1e-11,
        tolerance_change=1e-13,
        history_size=args.lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lb.zero_grad(set_to_none=True)
        d_loss, _ = normalized_data_loss(
            net, y_data, v_data, data_scale
        )
        p_loss = physics_loss(lp)
        loss = (
            args.lbfgs_data_weight * d_loss / d_scale
            + args.lbfgs_physics_weight * p_loss / p_scale
        )
        loss.backward()

        closure_calls["n"] += 1
        if closure_calls["n"] % 50 == 0:
            print(
                f"闭包调用 {closure_calls['n']:4d} | "
                f"objective={loss.detach().item():.8e} | "
                f"data={d_loss.detach().item():.8e} | "
                f"physics={p_loss.detach().item():.8e}"
            )
        return loss

    while total_iter < args.lbfgs_max_iter:
        current = min(
            chunk, args.lbfgs_max_iter - total_iter
        )
        lb.param_groups[0]["max_iter"] = current
        lb.defaults["max_iter"] = current
        lb.step(closure)
        total_iter += current

        net.eval()
        with torch.no_grad():
            check_data, check_data_mse = normalized_data_loss(
                net, y_data, v_data, data_scale
            )
            check_mass = domain_area * net(lc).mean()
        with torch.enable_grad():
            check_phys = physics_loss(lc).detach()

        data_ratio = float(
            (check_data / d_monitor_scale).detach()
        )
        physics_ratio = float(
            (check_phys / p_monitor_scale).detach()
        )
        monitor = (
            args.lbfgs_data_weight * data_ratio
            + args.lbfgs_physics_weight * physics_ratio
        )

        data_guard_ok = (
            args.lbfgs_max_data_ratio <= 0
            or data_ratio <= args.lbfgs_max_data_ratio
        )
        history["stage_c_iter"].append(int(total_iter))
        history["stage_c_monitor_score"].append(float(monitor))
        history["stage_c_data_mse"].append(float(check_data_mse.detach()))
        history["stage_c_physics_loss"].append(float(check_phys.detach()))

        if data_guard_ok and _score_improved(
            monitor, best_monitor, args.lbfgs_min_delta
        ):
            best_monitor = monitor
            best_iter = total_iter
            best_state = copy.deepcopy(net.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"L-BFGS iter {total_iter:4d} | "
            f"monitor={monitor:.8e} | "
            f"data ratio={data_ratio:.6e} | "
            f"physics ratio={physics_ratio:.6e} | "
            f"data MSE={check_data_mse.item():.8e} | "
            f"check physics={check_phys.item():.8e} | "
            f"mass={check_mass.item():.8f} | "
            f"data_guard={data_guard_ok} | "
            f"best_iter={best_iter}"
        )

        if (
            total_iter >= args.lbfgs_min_iter
            and args.lbfgs_patience_checks > 0
            and no_improve >= args.lbfgs_patience_checks
        ):
            print(
                f"[Stage C 早停] iter={total_iter}, "
                f"best_iter={best_iter}, "
                f"best_monitor={best_monitor:.8e}"
            )
            break

    net.load_state_dict(best_state)
    if history is None:
        history = {}
    history = copy.deepcopy(history)
    history["method"] = method_name
    history["stage_c_best_iter"] = best_iter
    history["stage_c_best_monitor"] = float(best_monitor)

    print(
        f"[Stage C checkpoint] best_iter={best_iter}, "
        f"monitor={best_monitor:.8e}"
    )
    print(
        "[Stage C 结束] 直接评价原始网络输出；"
        "未执行 gain、质量或幅值校准。"
    )
    return net, history



def evaluate_array_metrics(pred, exact, x_g, y_g, density_threshold=0.01):
    dx = float(np.mean(np.diff(x_g)))
    dy = float(np.mean(np.diff(y_g)))

    Xg, Yg = np.meshgrid(x_g, y_g, indexing="ij")
    mask = exact > density_threshold * exact.max()

    abs_diff = np.abs(pred - exact)
    sq_diff = (pred - exact) ** 2

    mape = np.mean(
        abs_diff[mask] / (exact[mask] + 1e-14)
    )

    # Absolute/global L2 norm on the FULL evaluation domain.
    global_l2 = np.sqrt(
        np.sum(sq_diff) * dx * dy
    )

    # Keep relative L2 internally for backward compatibility only.
    # It is no longer used in the main printed tables/plots.
    rel_l2 = np.sqrt(
        np.sum(sq_diff) * dx * dy
        / (np.sum(exact**2) * dx * dy + 1e-30)
    )

    global_l1 = np.sum(abs_diff) * dx * dy
    global_rmse = np.sqrt(np.mean(sq_diff))

    left_mask = mask & (Xg < 0.0)
    right_mask = mask & (Xg >= 0.0)

    left_mape = np.mean(
        abs_diff[left_mask] / (exact[left_mask] + 1e-14)
    )
    right_mape = np.mean(
        abs_diff[right_mask] / (exact[right_mask] + 1e-14)
    )

    y0_idx = int(np.argmin(np.abs(y_g)))
    center_mask = mask[:, y0_idx]
    center_mape = np.mean(
        abs_diff[center_mask, y0_idx]
        / (exact[center_mask, y0_idx] + 1e-14)
    )

    total_mass = np.sum(pred) * dx * dy
    left_mass = np.sum(pred[Xg < 0.0]) * dx * dy
    right_mass = np.sum(pred[Xg >= 0.0]) * dx * dy

    pred_x_reflect = pred[::-1, :]
    pred_y_reflect = pred[:, ::-1]

    x_symmetry_rel_l2 = np.sqrt(
        np.sum((pred - pred_x_reflect) ** 2) * dx * dy
        / (np.sum(pred**2) * dx * dy + 1e-30)
    )
    y_symmetry_rel_l2 = np.sqrt(
        np.sum((pred - pred_y_reflect) ** 2) * dx * dy
        / (np.sum(pred**2) * dx * dy + 1e-30)
    )

    return {
        "有效区域平均相对误差": float(mape),
        "全局L2误差": float(global_l2),
        "全局L1误差": float(global_l1),
        "全局RMSE": float(global_rmse),

        # retained only so older cached/reporting code does not break
        "相对L2误差": float(rel_l2),

        "左势阱误差": float(left_mape),
        "右势阱误差": float(right_mape),
        "中心线误差": float(center_mape),
        "诊断总质量": float(total_mass),
        "左半平面质量": float(left_mass),
        "右半平面质量": float(right_mass),
        "x反射相对L2不对称度": float(x_symmetry_rel_l2),
        "y反射相对L2不对称度": float(y_symmetry_rel_l2),
    }


def evaluate_model_on_bounds(
    net,
    process,
    device,
    eval_bounds=(-2.0, 2.0, -1.2, 1.2),
    dx=0.025,
    dy=0.025,
):
    x_min, x_max, y_min, y_max = eval_bounds
    x_g = np.arange(x_min, x_max + 0.5 * dx, dx)
    y_g = np.arange(y_min, y_max + 0.5 * dy, dy)

    Xg, Yg = np.meshgrid(x_g, y_g, indexing="ij")
    exact = exact_density_on_grid(process, Xg, Yg)

    xy_flat = np.column_stack([Xg.ravel(), Yg.ravel()])
    inp = torch.tensor(xy_flat, dtype=torch.float64, device=device)

    net.eval()
    pred_chunks = []
    with torch.no_grad():
        for chunk in torch.split(inp, 20000, dim=0):
            pred_chunks.append(net(chunk).cpu().numpy())

    pred = np.vstack(pred_chunks).reshape(Xg.shape)
    metrics = evaluate_array_metrics(pred=pred, exact=exact, x_g=x_g, y_g=y_g)
    return metrics, x_g, y_g, exact, pred


def evaluate_histogram_reference(process, y_ref, v_ref):
    pts = y_ref.detach().cpu().numpy()
    values = v_ref.detach().cpu().numpy().reshape(-1)

    x_unique = np.unique(pts[:, 0])
    y_unique = np.unique(pts[:, 1])

    pred = values.reshape(x_unique.size, y_unique.size)
    Xg, Yg = np.meshgrid(x_unique, y_unique, indexing="ij")
    exact = exact_density_on_grid(process, Xg, Yg)

    return evaluate_array_metrics(pred=pred, exact=exact, x_g=x_unique, y_g=y_unique)



def print_table(results):
    print("\n" + "=" * 126)
    print("结果汇总")
    print("=" * 126)
    print(
        f"{'方法':<30s} | "
        f"{'MAPE(%)':>12s} | "
        f"{'Global L2':>14s} | "
        f"{'Global L1':>14s} | "
        f"{'RMSE':>12s} | "
        f"{'Mass':>12s} | "
        f"{'Left Mass':>12s} | "
        f"{'Right Mass':>12s}"
    )
    print("-" * 126)

    for name, m in results.items():
        global_l2 = m.get("全局L2误差", np.nan)
        global_l1 = m.get("全局L1误差", np.nan)
        global_rmse = m.get("全局RMSE", np.nan)

        print(
            f"{name:<30s} | "
            f"{100.0 * m['有效区域平均相对误差']:12.4f} | "
            f"{global_l2:14.6e} | "
            f"{global_l1:14.6e} | "
            f"{global_rmse:12.6e} | "
            f"{m['诊断总质量']:12.6f} | "
            f"{m['左半平面质量']:12.6f} | "
            f"{m['右半平面质量']:12.6f}"
        )

    print("=" * 126 + "\n")


def print_symmetry_diagnostics(results):
    """
    补充输出左右质量差和反射不对称度。
    这对诊断双稳态问题中的单侧塌缩/对称性破缺很重要。
    """
    print("\n" + "=" * 124)
    print("对称性与双势阱诊断")
    print("=" * 124)
    print(
        f"{'方法':<30s} | "
        f"{'Left Mass':>12s} | "
        f"{'Right Mass':>12s} | "
        f"{'Mass Gap':>12s} | "
        f"{'x-Asym(%)':>12s} | "
        f"{'y-Asym(%)':>12s} | "
        f"{'Left Err(%)':>12s} | "
        f"{'Right Err(%)':>12s}"
    )
    print("-" * 124)

    for name, m in results.items():
        left_mass = m["左半平面质量"]
        right_mass = m["右半平面质量"]
        mass_gap = abs(left_mass - right_mass)

        print(
            f"{name:<30s} | "
            f"{left_mass:12.6f} | "
            f"{right_mass:12.6f} | "
            f"{mass_gap:12.6f} | "
            f"{100.0 * m['x反射相对L2不对称度']:12.4f} | "
            f"{100.0 * m['y反射相对L2不对称度']:12.4f} | "
            f"{100.0 * m['左势阱误差']:12.4f} | "
            f"{100.0 * m['右势阱误差']:12.4f}"
        )

    print("=" * 124 + "\n")



def _paper_figures_dir(out_dir):
    figures_dir = Path(out_dir) / "paper_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _method_file_tag(method_name):
    return (
        str(method_name).lower()
        .replace("+", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _display_method(method_name):
    mapping = {
        "Data-only NN": "Data-only",
        "Strong PINN": "Strong PINN",
        "Weak PINN": "Weak PINN",
        "Strong + MC Adam-only": "Strong+MC Adam",
        "Weak + MC Adam-only": "Weak+MC Adam",
        "Strong + MC + L-BFGS": "Strong+MC L-BFGS",
        "Weak + MC + L-BFGS": "Weak+MC L-BFGS",
    }
    return mapping.get(method_name, method_name)






def plot_all_method_relative_error_comparison(
    evaluation_arrays,
    figures_dir,
    tau_fraction=0.01,
):
    """
    Figure 3: full-domain relative error maps.

    Show Reference + all methods in a uniform 2x4 layout.
    The relative error is stabilized on the full domain by

        100 * |u_theta - u_ref| / max(u_ref, tau)

    with tau = tau_fraction * max(reference).
    """
    names, x_g, y_g, exact = _common_evaluation_grid(evaluation_arrays)

    extent = [
        float(x_g.min()), float(x_g.max()),
        float(y_g.min()), float(y_g.max()),
    ]
    ref_peak = float(np.nanmax(exact))
    if not np.isfinite(ref_peak) or ref_peak <= 0.0:
        ref_peak = 1.0

    tau = float(tau_fraction * ref_peak)
    denom = np.maximum(exact, tau)

    panels = [("Reference", np.zeros_like(exact))]
    for name in names:
        pred = np.asarray(
            evaluation_arrays[name]["pred"],
            dtype=np.float64,
        )
        rel_err = 100.0 * np.abs(pred - exact) / denom
        panels.append((_display_method(name), rel_err))

    competitive = [
        panel[1].ravel()
        for panel in panels
        if panel[0] in {
            "Data-only",
            "Strong+MC Adam",
            "Weak+MC Adam",
            "Strong+MC L-BFGS",
            "Weak+MC L-BFGS",
        }
    ]
    if competitive:
        pooled = np.concatenate(competitive)
    else:
        pooled = np.concatenate([panel[1].ravel() for panel in panels[1:]])

    vmax = float(np.percentile(pooled, 99.5))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(pooled))
    vmax = min(max(vmax, 1.0), 500.0)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under(cmap(0.0))

    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.1 * ncols, 3.65 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    image = None

    for ax, (label, err) in zip(axes_flat, panels):
        image = ax.imshow(
            err.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="bicubic",
            resample=True,
        )
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.grid(False)

    if image is not None:
        cbar = fig.colorbar(
            image,
            ax=[ax for ax in axes_flat[:len(panels)]],
            shrink=0.94,
            pad=0.02,
            extend="max",
        )
        cbar.set_label(
            r"Relative error  $100\,|u_\theta-u_{\rm ref}|/\max(u_{\rm ref},\tau)$  (%)"
        )

    fig.suptitle(
        "2D double well: full-domain relative error comparison",
        fontsize=13,
    )

    path = (
        Path(figures_dir)
        / "double_well_2d_global_relative_error_comparison.png"
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_bars(results, figures_dir, prefix):
    """
    Figure 4: pure MAPE bar chart comparison.
    """
    names = [name for name in METHODS if name in results]
    if not names:
        return None

    # Keep the canonical METHODS order; do not sort by MAPE.
    labels = [_display_method(n) for n in names]
    mape_values = np.asarray(
        [100.0 * results[name]["有效区域平均相对误差"] for name in names],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(
        figsize=(12.8, 5.8),
        constrained_layout=True,
    )

    x = np.arange(len(names), dtype=np.float64)
    ax.bar(x, mape_values)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Method-wise MAPE comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(True, axis="y", alpha=0.25)

    path = Path(figures_dir) / f"{prefix}_mape_bar_chart.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_reference_solution(process, physics_bounds, figures_dir, dx=0.025, dy=0.025):
    x_min, x_max, y_min, y_max = physics_bounds
    x_g = np.arange(x_min, x_max + 0.5 * dx, dx)
    y_g = np.arange(y_min, y_max + 0.5 * dy, dy)
    Xg, Yg = np.meshgrid(x_g, y_g, indexing="ij")
    exact = exact_density_on_grid(process, Xg, Yg)
    extent = [x_min, x_max, y_min, y_max]

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    image = ax.imshow(
        exact.T, origin="lower", extent=extent, aspect="auto",
        cmap="viridis", vmin=0.0, vmax=float(exact.max()),
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Reference stationary density")
    fig.colorbar(image, ax=ax, label="Stationary density")
    fig.tight_layout()
    path = Path(figures_dir) / "double_well_2d_reference_solution.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_solution_and_errors_from_arrays(
    method_name, x_g, y_g, exact, pred, figures_dir,
):
    extent = [float(x_g.min()), float(x_g.max()), float(y_g.min()), float(y_g.max())]
    tag = _method_file_tag(method_name)
    density_vmax = float(exact.max())
    abs_error = np.abs(pred - exact)
    valid = exact > 0.01 * exact.max()
    rel_error = np.full_like(exact, np.nan)
    rel_error[valid] = 100.0 * abs_error[valid] / (exact[valid] + 1e-14)
    paths = []

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    image = ax.imshow(
        pred.T, origin="lower", extent=extent, aspect="auto",
        cmap="viridis", vmin=0.0, vmax=density_vmax,
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(f"{method_name}: predicted density")
    fig.colorbar(image, ax=ax, label="Stationary density")
    fig.tight_layout()
    path = Path(figures_dir) / f"double_well_2d_{tag}_solution.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    image = ax.imshow(
        abs_error.T, origin="lower", extent=extent, aspect="auto",
        cmap="YlOrRd", vmin=0.0, vmax=density_vmax,
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(f"{method_name}: absolute error")
    fig.colorbar(image, ax=ax, label=r"$|u_\theta-u_{\mathrm{ref}}|$")
    fig.tight_layout()
    path = Path(figures_dir) / f"double_well_2d_{tag}_absolute_error.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    image = ax.imshow(
        rel_error.T, origin="lower", extent=extent, aspect="auto",
        cmap="viridis", vmin=0.0, vmax=100.0,
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(
        f"{method_name}: pointwise relative error\n"
        "Effective region; display clipped at 100%"
    )
    fig.colorbar(image, ax=ax, label="Pointwise relative error (%)")
    fig.tight_layout()
    path = Path(figures_dir) / f"double_well_2d_{tag}_relative_error.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)
    return paths


def _plot_history_family(histories, figures_dir, x_key, y_key, title, xlabel, ylabel, filename):
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    plotted = False
    all_y = []
    for method_name, history in histories.items():
        x = np.asarray(history.get(x_key, []), dtype=np.float64)
        y = np.asarray(history.get(y_key, []), dtype=np.float64)
        n = min(x.size, y.size)
        if n == 0:
            continue
        x, y = x[:n], y[:n]
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        x, y = x[finite], y[finite]
        ax.plot(x, y, linewidth=1.6, label=_display_method(method_name))
        all_y.extend(y.tolist())
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    if all(v > 0 for v in all_y):
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(figures_dir) / filename
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_training_histories(histories, figures_dir):
    specs = [
        ("stage_a_epoch", "stage_a_data_mse", "Stage A Adam training loss",
         "Epoch", "Data MSE", "double_well_2d_stage_a_adam_loss.png"),
        ("stage_b_cycle", "stage_b_monitor_score", "Stage B monitor score",
         "Cycle", "Monitor score", "double_well_2d_stage_b_monitor.png"),
        ("stage_c_iter", "stage_c_monitor_score", "Stage C L-BFGS monitor score",
         "L-BFGS iteration", "Monitor score", "double_well_2d_stage_c_lbfgs_monitor.png"),
    ]
    return [
        p for p in (
            _plot_history_family(histories, figures_dir, *spec)
            for spec in specs
        ) if p is not None
    ]




def plot_optimizer_ablation(results, figures_dir):
    """
    Adam-only vs +L-BFGS using MAPE and absolute/global L2.
    Relative L2 is intentionally not used.
    """
    pairs = [
        ("Strong + MC Adam-only", "Strong + MC + L-BFGS", "Strong"),
        ("Weak + MC Adam-only", "Weak + MC + L-BFGS", "Weak"),
    ]
    available = [
        (a, b, label)
        for a, b, label in pairs
        if a in results and b in results
    ]
    if not available:
        return []

    labels = []
    method_names = []
    for adam, lbfgs, family in available:
        labels.extend([f"{family} Adam", f"{family} + L-BFGS"])
        method_names.extend([adam, lbfgs])

    mape = np.asarray(
        [
            100.0 * results[name]["有效区域平均相对误差"]
            for name in method_names
        ],
        dtype=np.float64,
    )
    global_l2 = np.asarray(
        [
            results[name].get("全局L2误差", np.nan)
            for name in method_names
        ],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11.6, 5.3),
        constrained_layout=True,
    )
    x = np.arange(len(method_names), dtype=np.float64)

    axes[0].bar(x, mape)
    axes[0].set_ylabel("MAPE (%)")
    axes[0].set_title("Effect of Stage C L-BFGS: MAPE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, global_l2)
    axes[1].set_ylabel(r"Global $L^2$ error")
    axes[1].set_title("Effect of Stage C L-BFGS: global L2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(-2, 2),
    )

    path = (
        Path(figures_dir)
        / "double_well_2d_lbfgs_ablation_global_errors.png"
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return [path]


def _ordered_evaluation_names(evaluation_arrays):
    return [name for name in METHODS if name in evaluation_arrays]


def _common_evaluation_grid(evaluation_arrays):
    names = _ordered_evaluation_names(evaluation_arrays)
    if not names:
        raise ValueError("No evaluated methods available for comparison plots.")

    first = evaluation_arrays[names[0]]
    x_g = np.asarray(first["x_g"], dtype=np.float64)
    y_g = np.asarray(first["y_g"], dtype=np.float64)
    exact = np.asarray(first["exact"], dtype=np.float64)

    for name in names[1:]:
        item = evaluation_arrays[name]
        if (
            not np.array_equal(x_g, np.asarray(item["x_g"]))
            or not np.array_equal(y_g, np.asarray(item["y_g"]))
            or np.asarray(item["exact"]).shape != exact.shape
        ):
            raise ValueError(
                "All methods must use the same evaluation grid for fair plotting."
            )
    return names, x_g, y_g, exact





def plot_all_method_density_comparison(evaluation_arrays, figures_dir):
    """
    Full-domain density overview.

    No contour overlays, no cropping, no masks.
    Each method is shown on the same x/y domain and same reference-based
    density scale.  Global absolute L2 error is printed directly in each title
    so methods that look visually similar can still be distinguished.
    """
    names, x_g, y_g, exact = _common_evaluation_grid(evaluation_arrays)
    panels = [("Reference", exact, 0.0)] + [
        (
            _display_method(name),
            np.asarray(evaluation_arrays[name]["pred"], dtype=np.float64),
            np.sqrt(
                np.sum(
                    (
                        np.asarray(
                            evaluation_arrays[name]["pred"],
                            dtype=np.float64,
                        )
                        - exact
                    ) ** 2
                )
                * float(np.mean(np.diff(x_g)))
                * float(np.mean(np.diff(y_g)))
            ),
        )
        for name in names
    ]

    ref_peak = float(np.nanmax(exact))
    if not np.isfinite(ref_peak) or ref_peak <= 0.0:
        ref_peak = 1.0

    extent = [
        float(x_g.min()), float(x_g.max()),
        float(y_g.min()), float(y_g.max()),
    ]
    density_vmax = 1.05 * ref_peak

    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.15 * ncols, 3.70 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    image = None

    for ax, (label, values, global_l2) in zip(axes_flat, panels):
        image = ax.imshow(
            values.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=density_vmax,
            interpolation="bicubic",
            resample=True,
        )

        if label == "Reference":
            title = "Reference"
        else:
            title = f"{label}\nGlobal L2={global_l2:.3e}"

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.grid(False)

    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)

    if image is not None:
        cbar = fig.colorbar(
            image,
            ax=[ax for ax in axes_flat[:len(panels)]],
            shrink=0.94,
            pad=0.02,
            extend="max",
        )
        cbar.set_label("Stationary density")

    fig.suptitle(
        "2D double well: full-domain reference and predictions "
        "(shared density scale)",
        fontsize=13,
    )

    path = (
        Path(figures_dir)
        / "double_well_2d_all_methods_density_comparison.png"
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path




def plot_all_method_error_comparison(evaluation_arrays, figures_dir):
    """
    Figure 2: full-domain absolute error maps.

    Show Reference + all methods in a uniform 2x4 layout.
    The Reference panel is identically zero, so the whole figure layout is
    visually complete and directly comparable.
    """
    names, x_g, y_g, exact = _common_evaluation_grid(evaluation_arrays)

    extent = [
        float(x_g.min()), float(x_g.max()),
        float(y_g.min()), float(y_g.max()),
    ]

    dx = float(np.mean(np.diff(x_g)))
    dy = float(np.mean(np.diff(y_g)))

    panels = [("Reference", np.zeros_like(exact), 0.0)]
    for name in names:
        pred = np.asarray(
            evaluation_arrays[name]["pred"],
            dtype=np.float64,
        )
        err = np.abs(pred - exact)
        global_l2 = np.sqrt(np.sum(err**2) * dx * dy)
        panels.append((_display_method(name), err, global_l2))

    competitive = [
        panel[1].ravel()
        for panel in panels
        if panel[0] in {
            "Data-only",
            "Strong+MC Adam",
            "Weak+MC Adam",
            "Strong+MC L-BFGS",
            "Weak+MC L-BFGS",
        }
    ]
    if competitive:
        pooled = np.concatenate(competitive)
    else:
        pooled = np.concatenate([panel[1].ravel() for panel in panels[1:]])

    vmax = float(np.percentile(pooled, 99.7))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(pooled))
    vmax = max(vmax, 1e-12)

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_under("white")

    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.1 * ncols, 3.65 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    image = None

    for ax, (label, err, global_l2) in zip(axes_flat, panels):
        image = ax.imshow(
            err.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="bicubic",
            resample=True,
        )
        if label == "Reference":
            title = "Reference"
        else:
            title = f"{label}\nGlobal L2={global_l2:.3e}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.grid(False)

    if image is not None:
        cbar = fig.colorbar(
            image,
            ax=[ax for ax in axes_flat[:len(panels)]],
            shrink=0.94,
            pad=0.02,
            extend="max",
        )
        cbar.set_label(r"Absolute error $|u_\theta-u_{\mathrm{ref}}|$")

    fig.suptitle(
        "2D double well: full-domain absolute error comparison",
        fontsize=13,
    )

    path = (
        Path(figures_dir)
        / "double_well_2d_global_absolute_error_comparison.png"
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_all_method_error_log_comparison(evaluation_arrays, figures_dir):
    """
    Order-of-magnitude view for distinguishing the best-performing methods.

    The scale is still shared by all methods.  We plot absolute error divided
    by max(reference density), in percent, with one logarithmic colorbar.
    """
    names, x_g, y_g, exact = _common_evaluation_grid(evaluation_arrays)
    extent = [
        float(x_g.min()), float(x_g.max()),
        float(y_g.min()), float(y_g.max()),
    ]

    ref_peak = float(np.nanmax(exact))
    if not np.isfinite(ref_peak) or ref_peak <= 0.0:
        ref_peak = 1.0

    error_percent = []
    positive_values = []
    for name in names:
        pred = np.asarray(evaluation_arrays[name]["pred"], dtype=np.float64)
        err = 100.0 * np.abs(pred - exact) / ref_peak
        error_percent.append(err)
        pos = err[np.isfinite(err) & (err > 0.0)]
        if pos.size:
            positive_values.append(pos)

    if positive_values:
        all_positive = np.concatenate(positive_values)
        # Robust floor: ignore numerical near-zero noise, but do not choose a
        # separate floor for each method.
        vmin = float(np.percentile(all_positive, 2.0))
        vmax = float(np.max(all_positive))
    else:
        vmin, vmax = 1e-6, 1.0

    vmin = max(vmin, max(vmax * 1e-6, 1e-8))
    if vmin >= vmax:
        vmin = max(vmax * 1e-3, 1e-8)

    norm = LogNorm(vmin=vmin, vmax=vmax)
    ncols = len(names)

    fig, axes = plt.subplots(
        1, ncols,
        figsize=(3.05 * ncols, 3.75),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[0]
    image = None

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_under("white")

    for j, name in enumerate(names):
        image = axes[j].imshow(
            error_percent[j].T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            norm=norm,
        )
        axes[j].set_title(_display_method(name), fontsize=9)
        axes[j].set_xlabel("$x$")
        axes[j].set_ylabel("$y$" if j == 0 else "")

    if image is not None:
        cbar = fig.colorbar(
            image,
            ax=axes.tolist(),
            shrink=0.90,
            pad=0.01,
        )
        cbar.set_label(
            r"Absolute error / max(reference) (%)"
        )

    fig.suptitle(
        "2D double well: absolute-error magnitude "
        "(shared logarithmic color scale)",
        fontsize=13,
    )
    path = (
        Path(figures_dir)
        / "double_well_2d_all_methods_error_log_comparison.png"
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_final_solution_diagnostics(evaluation_arrays, figures_dir):
    """
    Focused paper figure for the actually competitive/final solutions.

    Columns:
        Strong+MC Adam
        Weak+MC Adam
        Strong+MC L-BFGS
        Weak+MC L-BFGS

    Top row:
        predicted density, one common reference-based density scale.

    Bottom row:
        signed deviation 100*(prediction-reference)/max(reference),
        one common symmetric scale. Positive means over-prediction and
        negative means under-prediction.

    This makes small differences between already-good solutions visible
    without giving every method its own arbitrary scale.
    """
    names, x_g, y_g, exact = _common_evaluation_grid(evaluation_arrays)

    preferred = [
        "Strong + MC Adam-only",
        "Weak + MC Adam-only",
        "Strong + MC + L-BFGS",
        "Weak + MC + L-BFGS",
    ]
    focus_names = [name for name in preferred if name in evaluation_arrays]
    if not focus_names:
        return None

    ref_peak = float(np.nanmax(exact))
    if not np.isfinite(ref_peak) or ref_peak <= 0.0:
        ref_peak = 1.0

    support = exact >= 0.0025 * ref_peak
    ii, jj = np.where(support)
    if ii.size and jj.size:
        x0 = float(x_g[max(0, int(ii.min()) - 5)])
        x1 = float(x_g[min(len(x_g) - 1, int(ii.max()) + 5)])
        y0 = float(y_g[max(0, int(jj.min()) - 5)])
        y1 = float(y_g[min(len(y_g) - 1, int(jj.max()) + 5)])
    else:
        x0, x1 = float(x_g.min()), float(x_g.max())
        y0, y1 = float(y_g.min()), float(y_g.max())

    extent = [
        float(x_g.min()), float(x_g.max()),
        float(y_g.min()), float(y_g.max()),
    ]

    predictions = [
        np.asarray(evaluation_arrays[name]["pred"], dtype=np.float64)
        for name in focus_names
    ]
    signed_errors = [
        100.0 * (pred - exact) / ref_peak
        for pred in predictions
    ]

    # One ROBUST shared error scale for all four competitive methods.
    # Using a common percentile prevents a few isolated pixels from flattening
    # the entire error row while still keeping cross-method comparability.
    pooled_abs = np.concatenate([
        np.abs(err[np.isfinite(err)]).ravel()
        for err in signed_errors
    ])
    err_cap = float(np.percentile(pooled_abs, 99.5))
    err_cap = max(err_cap, 0.25)
    err_norm = TwoSlopeNorm(
        vmin=-err_cap,
        vcenter=0.0,
        vmax=err_cap,
    )

    density_vmax = 1.05 * ref_peak

    ncols = len(focus_names)
    fig, axes = plt.subplots(
        2, ncols,
        figsize=(4.0 * ncols, 7.2),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    density_image = None
    error_image = None

    for j, (name, pred, signed_err) in enumerate(
        zip(focus_names, predictions, signed_errors)
    ):
        label = _display_method(name)
        if "L-BFGS" in name:
            label = f"{label}  [final]"

        density_image = axes[0, j].imshow(
            pred.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=density_vmax,
            interpolation="bicubic",
            resample=True,
        )
        axes[0, j].set_title(
            f"{label}\npeak/ref={np.nanmax(pred)/ref_peak:.3f}",
            fontsize=9,
        )

        error_image = axes[1, j].imshow(
            signed_err.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="coolwarm",
            norm=err_norm,
            interpolation="bicubic",
            resample=True,
        )

        for ax in (axes[0, j], axes[1, j]):
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$" if j == 0 else "")
            ax.grid(False)

    axes[0, 0].text(
        -0.24, 0.5, "Predicted density",
        transform=axes[0, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 0].text(
        -0.24, 0.5, "Signed deviation",
        transform=axes[1, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    if density_image is not None:
        cbar1 = fig.colorbar(
            density_image,
            ax=axes[0, :].tolist(),
            shrink=0.90,
            pad=0.012,
            extend="max",
        )
        cbar1.set_label("Stationary density")

    if error_image is not None:
        cbar2 = fig.colorbar(
            error_image,
            ax=axes[1, :].tolist(),
            shrink=0.90,
            pad=0.012,
            extend="both",
        )
        cbar2.set_label(
            "100 × (prediction - reference) / max(reference)  (%)"
        )

    fig.suptitle(
        "Final-solution comparison: density and signed deviation from reference",
        fontsize=13,
    )

    path = (
        Path(figures_dir)
        / "double_well_2d_final_solution_diagnostics.png"
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path



def plot_competitive_centerline_comparison(evaluation_arrays, figures_dir):
    """
    One-dimensional profile comparison at y ~= 0.

    Top:
        reference and four hybrid predictions on the same density axis.

    Bottom:
        signed difference from reference, normalized by the reference peak.
        This is the part that exposes differences when the top curves overlap.
    """
    names, x_g, y_g, exact = _common_evaluation_grid(evaluation_arrays)
    y0_idx = int(np.argmin(np.abs(y_g)))

    preferred = [
        "Strong + MC Adam-only",
        "Weak + MC Adam-only",
        "Strong + MC + L-BFGS",
        "Weak + MC + L-BFGS",
    ]
    focus_names = [name for name in preferred if name in evaluation_arrays]
    if not focus_names:
        return None

    ref_line = np.asarray(exact[:, y0_idx], dtype=np.float64)
    ref_peak = float(np.max(ref_line))
    if not np.isfinite(ref_peak) or ref_peak <= 0.0:
        ref_peak = 1.0

    fig, axes = plt.subplots(
        2, 1,
        figsize=(11.2, 7.8),
        sharex=True,
        constrained_layout=True,
    )

    # Top: actual solution profiles.
    axes[0].plot(
        x_g,
        ref_line,
        linewidth=3.0,
        label="Reference",
    )

    line_styles = ["-", "--", "-.", ":"]
    for i, name in enumerate(focus_names):
        pred = np.asarray(
            evaluation_arrays[name]["pred"],
            dtype=np.float64,
        )
        axes[0].plot(
            x_g,
            pred[:, y0_idx],
            linewidth=1.9,
            linestyle=line_styles[i % len(line_styles)],
            label=_display_method(name),
        )

    axes[0].set_ylabel("Stationary density")
    axes[0].set_title(
        f"Centerline density at y={y_g[y0_idx]:.3f}"
    )
    axes[0].set_ylim(0.0, 1.10 * ref_peak)
    axes[0].grid(True, alpha=0.20)
    axes[0].legend(fontsize=9, ncol=3)

    # Bottom: signed difference; much easier to distinguish close methods.
    for i, name in enumerate(focus_names):
        pred = np.asarray(
            evaluation_arrays[name]["pred"],
            dtype=np.float64,
        )
        diff_percent = (
            100.0 * (pred[:, y0_idx] - ref_line) / ref_peak
        )
        axes[1].plot(
            x_g,
            diff_percent,
            linewidth=1.9,
            linestyle=line_styles[i % len(line_styles)],
            label=_display_method(name),
        )

    axes[1].axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
        alpha=0.7,
    )
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel(
        "Signed deviation / reference peak (%)"
    )
    axes[1].set_title(
        "Difference from reference: positive = over-prediction, "
        "negative = under-prediction"
    )
    axes[1].grid(True, alpha=0.20)
    axes[1].legend(fontsize=9, ncol=2)

    # Focus on the two-well region.
    axes[1].set_xlim(-1.6, 1.6)

    path = (
        Path(figures_dir)
        / "double_well_2d_final_centerline_difference.png"
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_centerline_comparison(evaluation_arrays, figures_dir):
    """
    Overlay the y=0 centerline of the reference and every method in one axes.
    """
    names, x_g, y_g, exact = _common_evaluation_grid(evaluation_arrays)
    y0_idx = int(np.argmin(np.abs(y_g)))

    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    ax.plot(
        x_g,
        exact[:, y0_idx],
        linewidth=2.8,
        linestyle="-",
        label="Reference",
    )
    for name in names:
        pred = np.asarray(evaluation_arrays[name]["pred"], dtype=np.float64)
        ax.plot(
            x_g,
            pred[:, y0_idx],
            linewidth=1.6,
            label=_display_method(name),
        )

    ax.set_xlabel("$x$")
    ax.set_ylabel("Stationary density")
    ax.set_title(f"Centerline comparison at y={y_g[y0_idx]:.3f}")
    ax.set_xlim(float(x_g.min()), float(x_g.max()))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = Path(figures_dir) / "double_well_2d_all_methods_centerline_comparison.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def print_runtime_table(runtimes):
    print("\n" + "=" * 80)
    print("训练时间")
    print("=" * 80)
    for name, seconds in runtimes.items():
        print(
            f"{name:<32s} | "
            f"{seconds:12.2f} s | {seconds/60.0:10.2f} min"
        )
    print("=" * 80)



def evaluate_and_store(
    method_name,
    net,
    process,
    device,
    physics_bounds,
    results,
    states,
    figures_dir=None,
    evaluation_arrays=None,
):
    """
    Evaluate a method and cache its full-grid arrays.

    Plotting is deliberately deferred until all requested methods have been
    evaluated. This prevents one-method-per-figure output and allows all
    spatial comparisons to share exactly the same axes and color scales.
    """
    metrics, x_g, y_g, exact, pred = evaluate_model_on_bounds(
        net=net,
        process=process,
        device=device,
        eval_bounds=physics_bounds,
        dx=0.025,
        dy=0.025,
    )

    if evaluation_arrays is not None:
        evaluation_arrays[method_name] = {
            "x_g": np.array(x_g, copy=True),
            "y_g": np.array(y_g, copy=True),
            "exact": np.array(exact, copy=True),
            "pred": np.array(pred, copy=True),
        }

    results[method_name] = metrics
    states[method_name] = {
        k: v.detach().cpu().clone()
        for k, v in net.state_dict().items()
    }
    print(
        f"[完成] {method_name} | "
        f"MAPE={100.0*metrics['有效区域平均相对误差']:.4f}% | "
        f"Global-L2={metrics['全局L2误差']:.6e} | "
        f"Mass={metrics['诊断总质量']:.6f}"
    )
    return metrics


def run_unified_experiment(
    process,
    y_ref,
    v_ref,
    y_train,
    v_train,
    device,
    out_dir,
    physics_bounds,
    target_mass,
    args,
    methods,
    coordinate_budgets,
    training_point_budget,
):
    """
    一次性完成所有方法。

    Strong/Weak 两个混合 pair 都只做一次 Stage A+B；
    Adam-only 直接保存该 checkpoint，+L-BFGS 从同 checkpoint 分叉。
    """
    figures_dir = _paper_figures_dir(out_dir)
    results = {
        "MC histogram reference (diag)":
            evaluate_histogram_reference(process, y_ref, v_ref)
    }
    states = {}
    histories = {}
    runtimes = {}
    evaluation_arrays = {}

    def run_single_ab(method_name):
        print("\n" + "#" * 108)
        print(f"方法：{method_name}")
        print("#" * 108)
        set_seed(args.seed)
        base = _new_base_net(args, device)
        t0 = time.time()
        trained, history, context = train_stage_ab(
            method_name=method_name,
            net=base,
            y_data=y_train if METHODS[method_name]["use_data"] else None,
            v_data=v_train if METHODS[method_name]["use_data"] else None,
            process=process,
            device=device,
            physics_bounds=physics_bounds,
            target_mass=target_mass,
            args=args,
            seed=args.seed,
        )
        runtimes[method_name] = time.time() - t0
        histories[method_name] = history
        evaluate_and_store(
            method_name,
            trained,
            process,
            device,
            physics_bounds,
            results,
            states,
            figures_dir=figures_dir,
            evaluation_arrays=evaluation_arrays,
        )
        return trained, history, context

    # Data-only / pure PINNs.
    for method_name in (
        "Data-only NN",
        "Strong PINN",
        "Weak PINN",
    ):
        if method_name in methods:
            run_single_ab(method_name)

    # Hybrid pair helper: one shared Stage A+B, then fork.
    def run_pair(adam_name, lbfgs_name):
        need_adam = adam_name in methods
        need_lbfgs = lbfgs_name in methods
        if not (need_adam or need_lbfgs):
            return

        # Always train the pair's common A+B exactly once.
        print("\n" + "#" * 108)
        print(
            f"共享 Stage A+B：{adam_name} / {lbfgs_name}"
        )
        print("#" * 108)

        set_seed(args.seed)
        base = _new_base_net(args, device)
        t0 = time.time()
        stage_b_net, stage_b_history, context = train_stage_ab(
            method_name=adam_name,
            net=base,
            y_data=y_train,
            v_data=v_train,
            process=process,
            device=device,
            physics_bounds=physics_bounds,
            target_mass=target_mass,
            args=args,
            seed=args.seed,
        )
        shared_ab_seconds = time.time() - t0

        # Adam-only = selected Stage-B checkpoint.
        if need_adam:
            histories[adam_name] = copy.deepcopy(stage_b_history)
            runtimes[adam_name] = shared_ab_seconds
            evaluate_and_store(
                adam_name,
                stage_b_net,
                process,
                device,
                physics_bounds,
                results,
                states,
                figures_dir=figures_dir,
                evaluation_arrays=evaluation_arrays,
            )

        # +L-BFGS = exact same Stage-B checkpoint + Stage C only.
        if need_lbfgs:
            lb_net = _clone_general_net(
                stage_b_net, args, device
            )
            t1 = time.time()
            lb_net, lb_history = run_stage_c_lbfgs(
                method_name=lbfgs_name,
                net=lb_net,
                process=process,
                device=device,
                context=context,
                args=args,
                history=stage_b_history,
            )
            stage_c_seconds = time.time() - t1
            histories[lbfgs_name] = lb_history
            # Report total path cost A+B+C for the best-version method.
            runtimes[lbfgs_name] = (
                shared_ab_seconds + stage_c_seconds
            )
            evaluate_and_store(
                lbfgs_name,
                lb_net,
                process,
                device,
                physics_bounds,
                results,
                states,
                figures_dir=figures_dir,
                evaluation_arrays=evaluation_arrays,
            )

    run_pair(
        "Strong + MC Adam-only",
        "Strong + MC + L-BFGS",
    )
    run_pair(
        "Weak + MC Adam-only",
        "Weak + MC + L-BFGS",
    )

    # One and only one final reporting block.
    print_table(results)
    print_symmetry_diagnostics(results)
    print_runtime_table(runtimes)

    comparison_paths = []

    comparison_paths.append(
        plot_all_method_density_comparison(
            evaluation_arrays, figures_dir
        )
    )

    method_results = {
        k: v
        for k, v in results.items()
        if not k.startswith("MC histogram")
    }
    if method_results:
        comparison_paths.append(
            plot_bars(
                method_results,
                figures_dir=figures_dir,
                prefix="double_well_2d_unified_all7",
            )
        )

    comparison_paths.append(
        plot_all_method_error_comparison(
            evaluation_arrays, figures_dir
        )
    )
    comparison_paths.append(
        plot_all_method_relative_error_comparison(
            evaluation_arrays, figures_dir
        )
    )

    for path in comparison_paths:
        if path is not None:
            print(f"[统一对比图] {path.resolve()}")

    print(f"[论文图像文件夹] {figures_dir.resolve()}")

    save_path = _resolve_cache_path(args, out_dir)
    torch.save(
        {
            "results": results,
            "runtimes": runtimes,
            "histories": histories,
            "model_states": states,
            "evaluation_arrays": evaluation_arrays,
            "cache_format_version": 2,
            "figure_files": tuple(
                str(p.resolve()) for p in sorted(figures_dir.glob("*.png"))
            ),
            "args": vars(args),
            "physics_bounds": physics_bounds,
            "dimension": 2,
            "methods_run": methods,
            "mc_target_mass": target_mass,
            "training_contract": {
                "trainable_gain_for_data_methods": False,
                "posthoc_gain_calibration": False,
                "posthoc_mass_scaling_for_data_methods": False,
                "pure_pinn_hard_mass_allowed": False,
                "pure_pinn_explicit_normalization_loss": True,
                "pure_pinn_boundary_loss_for_both": True,
                "single_unified_experiment": True,
                "shared_mc_training_bins": True,
                "same_base_initialization_seed": True,
                "paired_hybrid_stage_ab_trained_once": True,
                "adam_only_is_selected_stage_b_checkpoint": True,
                "lbfgs_is_stage_c_append_only": True,
                "stage_b_start_is_checkpoint_candidate": True,
                "stage_c_reuses_stage_b_physics_sets": True,
                "stage_c_monitored_rollback": True,
                "symmetric_physics_point_augmentation": False,
                "unique_training_point_budget": int(
                    training_point_budget
                ),
                "max_training_points": int(args.max_training_points),
                "mc_training_bins": int(y_train.shape[0]),
                "physics_pool_size": int(args.physics_pool_size),
                "physics_monitor_size": int(args.fixed_check_size),
                "target_budget_semantics": "2000 MC + 1000 physics/monitor",
                "coordinate_budgets": coordinate_budgets,
                "exact_density_in_training": False,
            },
        },
        save_path,
    )
    print(f"[结果保存] {save_path}")
    return results



def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Unified 2D double-well seven-method ablation "
            "with 8D-style Stage A/B/C semantics."
        )
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "不做 MC 模拟、不训练、不重新评价模型；"
            "直接从缓存中的 evaluation_arrays/results/histories 重新出图。"
        ),
    )
    mode_group.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "不做 MC 模拟、不训练；从缓存 model_states 重建网络，"
            "重新计算评价网格并刷新 evaluation_arrays，然后重新出图。"
        ),
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=None,
        help=(
            "缓存 .pt 文件路径。默认使用 out-dir 下的 "
            "double_well_2d_unified_8d_structure_all7_results.pt。"
        ),
    )
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--only-pure",
        action="store_true",
        help="只运行纯 Strong PINN 和纯 Weak PINN。",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="double_well_2d_paper_outputs",
    )
    parser.add_argument(
        "--results-txt",
        type=str,
        default=(
            "double_well_2d_unified_8d_structure_"
            "2000plus1000_results.txt"
        ),
    )

    # MC / histogram
    parser.add_argument("--n-paths", type=int, default=25000)
    parser.add_argument("--burn-in", type=float, default=8.0)
    parser.add_argument("--sample-duration", type=float, default=2.0)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument(
        "--initial-left-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument("--bin-width", type=float, default=0.04)
    parser.add_argument("--n-ref", type=int, default=2000)
    parser.add_argument(
        "--max-training-points",
        type=int,
        default=3000,
        help=(
            "每个方法允许的最大唯一训练/模型选择坐标数。默认严格限制为 "
            "2000 个 MC 引导点 + 1000 个 physics/monitor 点。"
        ),
    )

    # Network / Stage A/B
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--pretrain-epochs", type=int, default=1200
    )
    parser.add_argument(
        "--joint-cycles", type=int, default=6000
    )
    parser.add_argument(
        "--physics-pool-size", type=int, default=800
    )
    parser.add_argument(
        "--fixed-check-size", type=int, default=200
    )
    parser.add_argument(
        "--physics-batch-size", type=int, default=256
    )
    parser.add_argument(
        "--data-batch-size", type=int, default=256
    )
    parser.add_argument("--data-lr", type=float, default=1.2e-4)
    parser.add_argument(
        "--physics-lr", type=float, default=6.0e-5
    )

    # Pure PINN objective weights / coordinates.
    parser.add_argument("--pure-physics-weight", type=float, default=1.0)
    parser.add_argument("--pure-boundary-weight", type=float, default=0.10)
    parser.add_argument("--pure-normal-weight", type=float, default=1.0)
    parser.add_argument(
        "--boundary-n-per-side", type=int, default=50
    )
    # Legacy argument retained so old command lines do not break; new training
    # does not allocate a separate hard-mass quadrature set.
    parser.add_argument(
        "--hard-mass-quad-size",
        type=int,
        default=1000,
        help=argparse.SUPPRESS,
    )

    # Stage C: all coordinates must be reused from Stage B.
    parser.add_argument(
        "--lbfgs-point-size", type=int, default=800
    )
    parser.add_argument(
        "--lbfgs-check-size", type=int, default=200
    )
    parser.add_argument(
        "--lbfgs-max-iter", type=int, default=800
    )
    parser.add_argument(
        "--lbfgs-check-every", type=int, default=50
    )
    parser.add_argument(
        "--lbfgs-min-iter", type=int, default=100
    )
    parser.add_argument(
        "--lbfgs-patience-checks", type=int, default=8
    )
    parser.add_argument(
        "--lbfgs-min-delta", type=float, default=0.0
    )
    parser.add_argument("--lbfgs-lr", type=float, default=0.5)
    parser.add_argument(
        "--lbfgs-history-size", type=int, default=100
    )
    parser.add_argument(
        "--lbfgs-data-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--lbfgs-physics-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--lbfgs-max-data-ratio",
        type=float,
        default=0.0,
        help="<=0 表示关闭 Stage-C data guard。",
    )
    return parser.parse_args()


def apply_fast_mode(args):
    if not args.fast:
        return args

    print("\n[快速模式] 已启用，仅用于检查流程。")
    args.n_paths = min(args.n_paths, 4000)
    args.burn_in = min(args.burn_in, 2.0)
    args.sample_duration = min(args.sample_duration, 0.5)
    args.n_ref = min(args.n_ref, 256)

    args.pretrain_epochs = min(args.pretrain_epochs, 30)
    args.joint_cycles = min(args.joint_cycles, 80)
    args.physics_pool_size = min(
        args.physics_pool_size, 1024
    )
    args.fixed_check_size = min(
        args.fixed_check_size, 256
    )
    args.physics_batch_size = min(
        args.physics_batch_size, 128
    )
    args.data_batch_size = min(
        args.data_batch_size, 128
    )

    args.lbfgs_point_size = min(
        args.lbfgs_point_size, args.physics_pool_size
    )
    args.lbfgs_check_size = min(
        args.lbfgs_check_size, args.fixed_check_size
    )
    args.lbfgs_max_iter = min(args.lbfgs_max_iter, 20)
    args.lbfgs_check_every = min(
        args.lbfgs_check_every, 10
    )
    args.lbfgs_min_iter = min(args.lbfgs_min_iter, 10)
    args.lbfgs_patience_checks = min(
        args.lbfgs_patience_checks, 3
    )

    args.boundary_n_per_side = min(
        args.boundary_n_per_side, 64
    )
    args.hard_mass_quad_size = min(
        args.hard_mass_quad_size, 512
    )
    args.width = min(args.width, 48)
    args.depth = min(args.depth, 2)
    return args


def selected_methods(args):
    if args.only_pure:
        return ["Strong PINN", "Weak PINN"]
    return [
        "Data-only NN",
        "Strong PINN",
        "Weak PINN",
        "Strong + MC Adam-only",
        "Weak + MC Adam-only",
        "Strong + MC + L-BFGS",
        "Weak + MC + L-BFGS",
    ]


def expected_mc_size(args):
    burn_steps = int(round(args.burn_in / args.dt))
    sample_steps = int(
        round(args.sample_duration / args.dt)
    )
    interval_steps = max(
        1, int(round(args.sample_interval / args.dt))
    )
    snapshots = (
        0
        if sample_steps <= 0
        else 1 + (sample_steps - 1) // interval_steps
    )
    return {
        "burn_steps": burn_steps,
        "sample_steps": sample_steps,
        "interval_steps": interval_steps,
        "snapshots": snapshots,
        "raw_samples": int(args.n_paths * snapshots),
    }


def planned_histogram_shape(bounds, bin_width):
    x_min, x_max, y_min, y_max = bounds
    nx = int(round((x_max - x_min) / bin_width))
    ny = int(round((y_max - y_min) / bin_width))
    return nx, ny


def print_startup_budget(
    args,
    physics_bounds,
    methods,
    coordinate_budgets,
    training_point_budget,
    effective_n_ref,
):
    mc = expected_mc_size(args)
    nx, ny = planned_histogram_shape(
        physics_bounds, args.bin_width
    )
    eval_nx = int(
        round(
            (physics_bounds[1] - physics_bounds[0])
            / 0.025
        )
    ) + 1
    eval_ny = int(
        round(
            (physics_bounds[3] - physics_bounds[2])
            / 0.025
        )
    ) + 1

    print("\n" + "=" * 116)
    print(
        "2D double-well：统一七方法 / 8D-style Stage A-B-C 结构"
    )
    print("=" * 116)
    print(
        "Adam-only = Stage A + Stage B；"
        "只有 +L-BFGS 方法从同一个 Stage-B checkpoint 进入 Stage C"
    )
    print(
        "Strong/Weak 每个 MC pair 的 Stage A+B 只训练一次，"
        "不再为 +L-BFGS 重复训练"
    )
    print(
        "Stage C 复用 Stage-B physics pool / monitor；"
        "不增加唯一物理训练坐标"
    )
    print(
        "含 MC 数据方法共享同一组训练箱；"
        "无 gain / 无解析幅值校准 / 无最终质量缩放"
    )
    print(f"methods={methods}")

    print("\n[MC 模拟规模：只用于构造直方图，不计入 NN 训练坐标预算]")
    print(
        f"  paths={args.n_paths:,}, burn_steps={mc['burn_steps']:,}, "
        f"sample_steps={mc['sample_steps']:,}, "
        f"snapshot_interval_steps={mc['interval_steps']:,}"
    )
    print(
        f"  expected snapshots={mc['snapshots']:,}, "
        f"expected raw MC samples={mc['raw_samples']:,}"
    )
    print(
        f"  histogram={nx} x {ny} = {nx*ny:,} bins, "
        f"bin_width={args.bin_width}"
    )
    print("\n[NN 训练数据规模]")
    print(
        f"  MC training bins={effective_n_ref:,} "
        f"(requested n_ref={args.n_ref:,})"
    )
    print(
        f"  physics pool={args.physics_pool_size:,}, "
        f"independent monitor={args.fixed_check_size:,}, "
        f"physics+monitor={args.physics_pool_size + args.fixed_check_size:,}"
    )
    print(
        "  hybrid target budget = "
        f"{effective_n_ref:,} MC + "
        f"{args.physics_pool_size + args.fixed_check_size:,} physics/monitor "
        f"= {effective_n_ref + args.physics_pool_size + args.fixed_check_size:,}"
    )
    print(
        f"  final evaluation grid={eval_nx} x {eval_ny} "
        f"= {eval_nx*eval_ny:,} points "
        "(evaluation only, not training)"
    )

    print(
        f"\n[最大实际唯一训练坐标预算] "
        f"{training_point_budget:,} / {args.max_training_points:,}"
    )
    print("[各方法实际坐标预算]")
    for method_name in methods:
        d = coordinate_budgets[method_name]
        print(
            f"  {method_name:30s} | total={d['total']:6d} | "
            f"MC={d['mc_bins']:5d}, "
            f"physics={d['physics_pool']:5d}, "
            f"monitor={d['physics_monitor']:4d}, "
            f"hard-mass={d['hard_mass']:5d}, "
            f"boundary={d['boundary']:4d}"
        )

    print("\n[训练超参数]")
    for key in (
        "max_training_points",
        "pretrain_epochs",
        "joint_cycles",
        "data_batch_size",
        "physics_batch_size",
        "data_lr",
        "physics_lr",
        "physics_pool_size",
        "fixed_check_size",
        "lbfgs_point_size",
        "lbfgs_check_size",
        "lbfgs_max_iter",
        "width",
        "depth",
    ):
        print(f"  {key:<24s}= {getattr(args, key)}")
    print("=" * 116 + "\n")


def run_smoke_test(seed=42):
    """No MC data: verify the explicit three-term loss for both pure PINNs."""
    set_seed(seed)
    process = DoubleWellBimodal2D()
    bounds = (-2.0, 2.0, -1.2, 1.2)
    area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
    pts = sobol_rectangle(128, bounds, seed=seed + 20)
    bp, bn, bw = sobol_boundary(32, bounds, seed=seed + 30)

    for offset, physics_type in enumerate(("strong", "weak"), start=1):
        set_seed(seed + offset)
        net = GeneralFPNN(
            x_scale=2.0, y_scale=1.2, width=32, depth=2
        )
        p_loss = compute_physics_loss(
            physics_type, net, pts, process, area
        )
        b_loss = noflux_boundary_loss(
            net, bp, bn, bw, process
        )
        n_loss, mass = compute_mass_loss(
            net, pts, area, target_mass=1.0
        )
        total = p_loss + 0.10 * b_loss + n_loss
        total.backward()
        print(
            f"[{physics_type}] physics={p_loss.item():.8e}, "
            f"boundary={b_loss.item():.8e}, "
            f"normal={n_loss.item():.8e}, mass={mass.item():.8f}, "
            f"total={total.item():.8e}"
        )

    print("2D pure-PINN three-term forward/backward: PASS")


def _resolve_cache_path(args, out_dir):
    if getattr(args, "cache_file", None):
        path = Path(args.cache_file).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path
    return (
        Path(out_dir)
        / "double_well_2d_unified_8d_structure_all7_results.pt"
    )


def _load_cache(cache_path):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"找不到缓存文件：{cache_path.resolve()}\n"
            "请先正常完整运行一次，或者用 --cache-file 指向已有缓存。"
        )
    # PyTorch >= 2.6 defaults to weights_only=True.  This cache contains
    # numpy arrays and metadata generated by this script, so explicitly
    # request a full load.  Fall back for older PyTorch versions.
    try:
        saved = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        saved = torch.load(cache_path, map_location="cpu")
    if not isinstance(saved, dict):
        raise TypeError(f"缓存格式不正确：{cache_path.resolve()}")
    return saved


def _save_cache(saved, cache_path):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, cache_path)
    print(f"[缓存保存] {cache_path.resolve()}")


def _render_cached_outputs(saved, out_dir):
    """
    Recreate all paper figures from already evaluated arrays.

    This function performs no simulation, no optimization and no model
    evaluation.  It is therefore the fast path used by --plot-only.
    """
    evaluation_arrays = saved.get("evaluation_arrays")
    if not evaluation_arrays:
        raise RuntimeError(
            "这个缓存里没有 evaluation_arrays，不能直接 --plot-only。\n"
            "如果它是旧版缓存，可以先尝试 --eval-only；"
            "否则请用当前脚本正常完整运行一次生成新版缓存。"
        )

    results = dict(saved.get("results", {}))
    histories = saved.get("histories", {})

    # Refresh all scalar metrics from cached evaluation arrays so new metrics
    # (especially Global L2) are available without retraining or eval-only.
    for method_name, item in evaluation_arrays.items():
        if method_name not in METHODS:
            continue
        results[method_name] = evaluate_array_metrics(
            pred=np.asarray(item["pred"], dtype=np.float64),
            exact=np.asarray(item["exact"], dtype=np.float64),
            x_g=np.asarray(item["x_g"], dtype=np.float64),
            y_g=np.asarray(item["y_g"], dtype=np.float64),
        )
    saved["results"] = results
    runtimes = saved.get("runtimes", {})
    figures_dir = _paper_figures_dir(out_dir)

    print("\n" + "=" * 96)
    print("缓存绘图模式：跳过 MC 模拟 / Stage A / Stage B / Stage C / 模型评价")
    print("=" * 96)

    if results:
        print_table(results)
        print_symmetry_diagnostics(results)
    if runtimes:
        print_runtime_table(runtimes)

    comparison_paths = []

    comparison_paths.append(
        plot_all_method_density_comparison(
            evaluation_arrays, figures_dir
        )
    )

    method_results = {
        k: v
        for k, v in results.items()
        if not k.startswith("MC histogram")
    }
    if method_results:
        comparison_paths.append(
            plot_bars(
                method_results,
                figures_dir=figures_dir,
                prefix="double_well_2d_unified_all7",
            )
        )

    comparison_paths.append(
        plot_all_method_error_comparison(
            evaluation_arrays, figures_dir
        )
    )
    comparison_paths.append(
        plot_all_method_relative_error_comparison(
            evaluation_arrays, figures_dir
        )
    )

    print("\n[缓存重新生成图像]")
    for path in comparison_paths:
        if path is not None:
            print(f"  {Path(path).resolve()}")

    saved["figure_files"] = tuple(
        str(p.resolve())
        for p in sorted(figures_dir.glob("*.png"))
    )
    saved["cache_format_version"] = max(
        int(saved.get("cache_format_version", 1)), 2
    )
    return saved


def _rebuild_model_from_saved_state(
    method_name,
    state,
    saved_args,
    physics_bounds,
    target_mass,
    device,
):
    """
    Rebuild a model for evaluation. New checkpoints use GeneralFPNN directly;
    legacy pure-PINN caches containing norm_points still load through the old
    MassNormalizedDensity wrapper.
    """
    width = int(saved_args.get("width", 128))
    depth = int(saved_args.get("depth", 4))

    base = GeneralFPNN(
        x_scale=2.0,
        y_scale=1.2,
        width=width,
        depth=depth,
    ).to(device)

    # New pure-PINN checkpoints are plain networks. If an old cache contains
    # norm_points, reconstruct the legacy hard-mass wrapper for compatibility.
    if "norm_points" in state:
        norm_points = state["norm_points"].detach().clone().to(
            device=device, dtype=torch.float64
        )
        x_min, x_max, y_min, y_max = physics_bounds
        domain_area = (x_max - x_min) * (y_max - y_min)
        net = MassNormalizedDensity(
            base_net=base,
            norm_points=norm_points,
            domain_area=domain_area,
            target_mass=float(target_mass),
        ).to(device)
    else:
        net = base

    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def run_plot_only(args, out_dir):
    cache_path = _resolve_cache_path(args, out_dir)
    saved = _load_cache(cache_path)
    saved = _render_cached_outputs(saved, out_dir)
    _save_cache(saved, cache_path)
    print("[plot-only 完成] 没有重新训练，也没有重新计算模型预测。")


def run_eval_only(args, out_dir):
    """
    Re-evaluate saved model checkpoints on the fixed evaluation grid without
    running Monte Carlo or any optimizer.  Useful when evaluation/plot logic
    changes but the trained checkpoints should remain untouched.
    """
    cache_path = _resolve_cache_path(args, out_dir)
    saved = _load_cache(cache_path)

    states = saved.get("model_states")
    if not states:
        raise RuntimeError(
            "缓存里没有 model_states，无法 --eval-only。"
        )

    saved_args = dict(saved.get("args", {}))
    physics_bounds = tuple(
        saved.get(
            "physics_bounds",
            (-2.0, 2.0, -1.2, 1.2),
        )
    )
    target_mass = float(saved.get("mc_target_mass", 1.0))
    device = torch.device("cpu")

    methods = [
        name
        for name in saved.get("methods_run", list(states.keys()))
        if name in states and name in METHODS
    ]
    if not methods:
        methods = [
            name for name in states
            if name in METHODS
        ]

    print("\n" + "=" * 96)
    print("缓存评价模式：跳过 MC 模拟和全部训练，只重建 checkpoint 并重新评价")
    print("=" * 96)
    print(f"[缓存] {cache_path.resolve()}")
    print(f"[方法] {methods}")

    process = DoubleWellBimodal2D()
    evaluation_arrays = {}
    results = dict(saved.get("results", {}))
    rebuilt_states = {}

    for method_name in methods:
        print(f"[重新评价] {method_name}")
        net = _rebuild_model_from_saved_state(
            method_name=method_name,
            state=states[method_name],
            saved_args=saved_args,
            physics_bounds=physics_bounds,
            target_mass=target_mass,
            device=device,
        )
        evaluate_and_store(
            method_name=method_name,
            net=net,
            process=process,
            device=device,
            physics_bounds=physics_bounds,
            results=results,
            states=rebuilt_states,
            figures_dir=None,
            evaluation_arrays=evaluation_arrays,
        )

    saved["results"] = results
    saved["model_states"] = rebuilt_states
    saved["evaluation_arrays"] = evaluation_arrays
    saved["cache_format_version"] = 2

    saved = _render_cached_outputs(saved, out_dir)
    _save_cache(saved, cache_path)
    print("[eval-only 完成] checkpoint 未训练，只重新计算了评价网格和图像。")


def execute(args, out_dir):
    if getattr(args, "plot_only", False):
        run_plot_only(args, out_dir)
        return

    if getattr(args, "eval_only", False):
        run_eval_only(args, out_dir)
        return

    if args.smoke_test:
        run_smoke_test(args.seed)
        return

    args = apply_fast_mode(args)
    set_seed(args.seed)
    start_all = time.time()

    process = DoubleWellBimodal2D()
    physics_bounds = (-2.0, 2.0, -1.2, 1.2)
    device = torch.device("cpu")
    methods = selected_methods(args)

    nx, ny = planned_histogram_shape(
        physics_bounds, args.bin_width
    )
    effective_n_ref = min(args.n_ref, nx * ny)

    coordinate_budgets = {
        m: method_coordinate_budget(
            m, args, effective_n_ref
        )
        for m in methods
    }
    training_point_budget = max(
        d["total"] for d in coordinate_budgets.values()
    )

    if args.lbfgs_point_size > args.physics_pool_size:
        raise ValueError(
            "lbfgs_point_size 必须 <= physics_pool_size。"
        )
    if args.lbfgs_check_size > args.fixed_check_size:
        raise ValueError(
            "lbfgs_check_size 必须 <= fixed_check_size。"
        )

    physics_monitor_budget = (
        int(args.physics_pool_size)
        + int(args.fixed_check_size)
    )
    if physics_monitor_budget > 1000:
        raise ValueError(
            "physics_pool_size + fixed_check_size 超过 1000。"
            f" 当前={physics_monitor_budget}。"
            "本版本默认按 2000 MC + 1000 physics/monitor 控制规模。"
        )

    if (
        args.max_training_points > 0
        and training_point_budget > args.max_training_points
    ):
        raise ValueError(
            f"唯一训练坐标预算超限：{training_point_budget:,} > "
            f"{args.max_training_points:,}。"
            "默认预算为 2000 个 MC 引导点 + "
            "1000 个 physics/monitor 坐标。"
        )

    # 用户要求：数据规模和坐标预算在真正模拟/训练前先展示。
    print_startup_budget(
        args,
        physics_bounds,
        methods,
        coordinate_budgets,
        training_point_budget,
        effective_n_ref,
    )

    x_samples, y_samples = process.simulate_stationary_samples(
        n_paths=args.n_paths,
        burn_in=args.burn_in,
        sample_duration=args.sample_duration,
        sample_interval=args.sample_interval,
        dt=args.dt,
        initial_left_fraction=args.initial_left_fraction,
        seed=args.seed,
    )

    y_ref, v_ref = build_histogram_reference(
        x_samples=x_samples,
        y_samples=y_samples,
        bounds=physics_bounds,
        bin_width=args.bin_width,
    )

    # 解析解只做 MC 参考质量检查，不进入训练/模型选择。
    validate_reference_error(process, y_ref, v_ref)

    target_mass = estimate_reference_mass_from_grid(
        y_ref, v_ref
    )
    print(f"[MC 诊断目标质量] {target_mass:.8f}")

    # 与 8D 一样：MC 引导坐标只生成一次，所有含数据方法共享。
    y_train, v_train = make_reference_subset(
        y_ref=y_ref,
        v_ref=v_ref,
        n_ref=effective_n_ref,
        seed=args.seed + 1111,
        density_fraction=0.7,
    )
    print(
        f"[共享 MC 训练箱] {y_train.shape[0]:,} / "
        f"{y_ref.shape[0]:,} bins；"
        "所有含数据方法完全复用这一份"
    )

    run_unified_experiment(
        process=process,
        y_ref=y_ref,
        v_ref=v_ref,
        y_train=y_train,
        v_train=v_train,
        device=device,
        out_dir=out_dir,
        physics_bounds=physics_bounds,
        target_mass=target_mass,
        args=args,
        methods=methods,
        coordinate_budgets=coordinate_budgets,
        training_point_budget=training_point_budget,
    )

    print(
        f"\n总用时={(time.time()-start_all)/60.0:.2f} 分钟"
    )


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "plot_only", False):
        result_path = out_dir / "plot_only_results.txt"
    elif getattr(args, "eval_only", False):
        result_path = out_dir / "eval_only_results.txt"
    else:
        result_path = out_dir / args.results_txt

    original_stdout = sys.stdout
    with open(
        result_path, "w", encoding="utf-8", buffering=1
    ) as f:
        sys.stdout = TeeStream(original_stdout, f)
        try:
            print(
                f"[文本结果文件] {result_path.resolve()}"
            )
            execute(args, out_dir)
            print(
                f"\n[文本结果保存完成] "
                f"{result_path.resolve()}"
            )
        finally:
            sys.stdout = original_stdout

    print(f"全部输出已保存到：{result_path.resolve()}")


if __name__ == "__main__":
    main()
