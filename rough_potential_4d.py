
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import copy
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset


torch.set_default_dtype(torch.float64)
plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def volume(bounds):
    b = np.asarray(bounds, dtype=np.float64)
    return float(np.prod(b[:, 1] - b[:, 0]))


def sobol_box(n, bounds, seed=0):
    if n <= 0:
        raise ValueError("n 必须为正整数")
    b = np.asarray(bounds, dtype=np.float64)
    dim = b.shape[0]
    eng = torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed)
    u = eng.draw(n).to(torch.float64)
    lo = torch.tensor(b[:, 0], dtype=torch.float64).view(1, dim)
    hi = torch.tensor(b[:, 1], dtype=torch.float64).view(1, dim)
    return lo + (hi - lo) * u


def sobol_boundary(n_per_face, bounds, seed=0):
    b = np.asarray(bounds, dtype=np.float64)
    dim = b.shape[0]
    lengths = b[:, 1] - b[:, 0]
    points, normals, weights = [], [], []

    for axis in range(dim):
        others = [j for j in range(dim) if j != axis]
        face_area = float(np.prod(lengths[others]))
        for side, value in enumerate((b[axis, 0], b[axis, 1])):
            eng = torch.quasirandom.SobolEngine(
                dim - 1, scramble=True, seed=seed + 1009 * axis + 97 * side
            )
            u = eng.draw(n_per_face).to(torch.float64)
            p = torch.empty((n_per_face, dim), dtype=torch.float64)
            lo = torch.tensor(b[others, 0], dtype=torch.float64)
            hi = torch.tensor(b[others, 1], dtype=torch.float64)
            p[:, others] = lo + (hi - lo) * u
            p[:, axis] = float(value)

            n = torch.zeros((n_per_face, dim), dtype=torch.float64)
            n[:, axis] = -1.0 if side == 0 else 1.0
            w = torch.full((n_per_face, 1), face_area, dtype=torch.float64)
            points.append(p); normals.append(n); weights.append(w)

    return torch.cat(points), torch.cat(normals), torch.cat(weights)


class RoughPotential4D:
    dimension = 4

    def __init__(self, epsilon=0.10):
        if epsilon <= 0:
            raise ValueError("epsilon 必须为正数")
        self.epsilon = float(epsilon)
        self.sigma = 0.5
        self.Z = None

    @property
    def D(self):
        return 0.5 * self.sigma**2

    def potential_numpy(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != 4:
            raise ValueError("输入最后一维必须为 4")
        eps = self.epsilon
        rough = eps**4 * np.prod(np.sin(2.0 * np.pi * x / eps), axis=-1)
        return (
            0.25 * (x[..., 0] ** 2 - 1.0) ** 2
            + 0.25 * (x[..., 1] ** 2 - 1.0) ** 2
            + 0.5 * x[..., 2] ** 2
            + 0.5 * x[..., 3] ** 2
            + rough
        )

    def drift_torch(self, x):
        if x.ndim != 2 or x.shape[1] != 4:
            raise ValueError("drift 输入必须为 (N,4)")
        eps = self.epsilon
        ph = 2.0 * np.pi * x / eps
        s, c = torch.sin(ph), torch.cos(ph)
        a = 2.0 * np.pi * eps**3
        b1 = x[:, 0:1] - x[:, 0:1]**3 - a*c[:,0:1]*s[:,1:2]*s[:,2:3]*s[:,3:4]
        b2 = x[:, 1:2] - x[:, 1:2]**3 - a*s[:,0:1]*c[:,1:2]*s[:,2:3]*s[:,3:4]
        b3 = -x[:, 2:3] - a*s[:,0:1]*s[:,1:2]*c[:,2:3]*s[:,3:4]
        b4 = -x[:, 3:4] - a*s[:,0:1]*s[:,1:2]*s[:,2:3]*c[:,3:4]
        return torch.cat([b1, b2, b3, b4], dim=1)

    def prepare_normalizer(self, bounds, n_points=262144, seed=0):
        pts = sobol_box(n_points, bounds, seed).numpy()
        self.Z = volume(bounds) * float(np.mean(np.exp(-self.potential_numpy(pts) / self.D)))
        if not np.isfinite(self.Z) or self.Z <= 0:
            raise FloatingPointError(f"归一化失败 Z={self.Z}")
        print(f"[解析参考] Sobol 归一化点数={n_points}, Z={self.Z:.12e}")

    def exact_density_numpy(self, x):
        if self.Z is None:
            raise RuntimeError("请先调用 prepare_normalizer")
        return np.exp(-self.potential_numpy(x) / self.D) / self.Z

    def simulate_stationary_samples(
        self, n_paths=25000, burn_in=8.0, sample_duration=2.0,
        sample_interval=0.1, dt=0.001, seed=42
    ):
        rng = np.random.default_rng(seed)
        signs = np.array([[-1,-1],[-1,1],[1,-1],[1,1]], dtype=np.float64)
        ids = np.arange(n_paths) % 4
        rng.shuffle(ids)
        x = np.zeros((n_paths, 4), dtype=np.float64)
        x[:, :2] = signs[ids] + 0.20 * rng.standard_normal((n_paths, 2))
        x[:, 2:] = 0.20 * rng.standard_normal((n_paths, 2))

        burn_steps = int(round(burn_in / dt))
        sample_steps = int(round(sample_duration / dt))
        interval = max(1, int(round(sample_interval / dt)))
        total_steps = burn_steps + sample_steps
        noise = self.sigma * np.sqrt(dt)
        eps = self.epsilon
        a = 2.0 * np.pi * eps**3
        samples = []

        print(f"[数据] 路径数={n_paths}, 总步数={total_steps}, 四个势阱均匀初始化")
        for step in range(total_steps):
            ph = 2.0 * np.pi * x / eps
            s, c = np.sin(ph), np.cos(ph)
            b = np.empty_like(x)
            b[:,0] = x[:,0]-x[:,0]**3-a*c[:,0]*s[:,1]*s[:,2]*s[:,3]
            b[:,1] = x[:,1]-x[:,1]**3-a*s[:,0]*c[:,1]*s[:,2]*s[:,3]
            b[:,2] = -x[:,2]-a*s[:,0]*s[:,1]*c[:,2]*s[:,3]
            b[:,3] = -x[:,3]-a*s[:,0]*s[:,1]*s[:,2]*c[:,3]
            x += b * dt + noise * rng.standard_normal(x.shape)
            if step >= burn_steps and (step - burn_steps) % interval == 0:
                samples.append(x.copy())

        out = np.concatenate(samples, axis=0)
        print(f"[数据] 实际收集 4D 样本数={out.shape[0]}")
        return out


def build_histogram_reference(samples, bounds, bin_widths=(0.2,0.2,0.2,0.2)):
    b = np.asarray(bounds, dtype=np.float64)
    widths = np.asarray(bin_widths, dtype=np.float64)
    edges, centers, shape = [], [], []
    for j in range(4):
        n = int(round((b[j,1] - b[j,0]) / widths[j]))
        e = np.linspace(b[j,0], b[j,1], n+1)
        edges.append(e)
        centers.append(0.5 * (e[:-1] + e[1:]))
        shape.append(n)

    counts, _ = np.histogramdd(samples, bins=edges)
    actual = np.array([e[1]-e[0] for e in edges])
    bin_volume = float(np.prod(actual))
    density = counts / (samples.shape[0] * bin_volume)
    raw_mass = float(density.sum() * bin_volume)
    if raw_mass <= 0:
        raise FloatingPointError("区域内无样本")
    density /= raw_mass

    mesh = np.meshgrid(*centers, indexing="ij")
    pts = np.column_stack([m.ravel() for m in mesh])
    vals = density.ravel()
    print(f"[4D直方图] shape={tuple(shape)}, bins={vals.size}, raw mass={raw_mass:.8f}, normalized mass={vals.sum()*bin_volume:.8f}")
    return (
        torch.tensor(pts, dtype=torch.float64),
        torch.tensor(vals, dtype=torch.float64).unsqueeze(1),
        bin_volume,
    )


def validate_reference(process, y, v, bin_volume):
    pts = y.numpy()
    ref = v.numpy().reshape(-1)
    exact = process.exact_density_numpy(pts)
    mask = exact > 0.01 * exact.max()
    mape = np.mean(np.abs(ref[mask]-exact[mask])/(exact[mask]+1e-14))
    rel_l2 = np.sqrt(np.sum((ref-exact)**2)*bin_volume/(np.sum(exact**2)*bin_volume+1e-30))
    print("\n" + "="*76)
    print("4D Monte Carlo 直方图质量检查")
    print("-"*76)
    print(f"有效区域 MAPE={100*mape:.4f}%")
    print(f"相对 L2={100*rel_l2:.4f}%")
    print("="*76 + "\n")


class GeneralFPNN(nn.Module):
    """正密度网络；没有显式全局 gain 参数。"""
    def __init__(self, input_scales=(2,2,1.2,1.2), width=128, depth=4):
        super().__init__()
        dim = len(input_scales)
        self.register_buffer(
            "input_scale",
            torch.tensor(input_scales, dtype=torch.float64).view(1, dim),
        )
        layers, in_dim = [], dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.Tanh()]
            in_dim = width
        layers.append(nn.Linear(width, 1))
        self.shape_net = nn.Sequential(*layers)
        for module in self.shape_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.shape_net[-1].weight, mean=0.0, std=2e-2)
        nn.init.constant_(self.shape_net[-1].bias, -1.0)

    def base_density(self, x):
        return nn.functional.softplus(self.shape_net(x / self.input_scale)) + 1e-12

    def forward(self, x):
        return self.base_density(x)


class MassNormalizedDensity(nn.Module):
    """旧缓存兼容用 hard-mass 包装器；新纯 PINN 训练不再调用。"""
    def __init__(self, base_net, norm_points, domain_area, target_mass=1.0):
        super().__init__()
        self.base_net = base_net
        self.register_buffer("norm_points", norm_points.detach().clone())
        self.register_buffer(
            "target_mass_tensor",
            torch.tensor(float(target_mass), dtype=torch.float64),
        )
        self.domain_area = float(domain_area)

    def base_density(self, x):
        return self.base_net.base_density(x)

    def normalizer(self):
        return self.domain_area * self.base_net.base_density(self.norm_points).mean() + 1e-30

    def forward(self, x):
        return self.target_mass_tensor * self.base_net.base_density(x) / self.normalizer()


def normalized_data_loss(net, x, target, scale):
    mse = torch.mean((net(x)-target)**2)
    return mse/scale, mse


def weak_loss(net, points, process, domain_area):
    x = points.detach().clone().requires_grad_(True)
    u = net(x)
    gu = grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    flux = process.D * gu - process.drift_torch(x) * u
    raw = domain_area * torch.mean(torch.sum(flux**2, dim=1, keepdim=True)/torch.clamp(u,min=1e-12))
    return raw / (domain_area*u.mean() + 1e-14)


def strong_loss(net, points, process, domain_area):
    x = points.detach().clone().requires_grad_(True)
    u = net(x)
    bu = process.drift_torch(x) * u
    div = 0.0
    for j in range(4):
        div = div + grad(bu[:,j:j+1].sum(), x, create_graph=True, retain_graph=True)[0][:,j:j+1]
    gu = grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    lap = 0.0
    for j in range(4):
        lap = lap + grad(gu[:,j:j+1].sum(), x, create_graph=True, retain_graph=True)[0][:,j:j+1]
    r = -div + process.D * lap
    return domain_area * torch.mean(r**2)


def noflux_loss(net, points, normals, weights, process):
    x = points.detach().clone().requires_grad_(True)
    u = net(x)
    gu = grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    flux = process.D * gu - process.drift_torch(x) * u
    nf = torch.sum(flux * normals, dim=1, keepdim=True)
    return torch.mean(weights * nf**2) / (torch.mean(weights)+1e-30)


def normalization_loss(net, points, domain_volume, target_mass=1.0):
    """L_normal=(int_Omega u_theta dx-target_mass)^2 via Sobol quadrature."""
    mass = domain_volume * net(points.detach()).mean()
    target = torch.as_tensor(
        target_mass, dtype=mass.dtype, device=mass.device
    )
    return (mass - target) ** 2, mass


def make_reference_subset(y, v, n_ref=6000, seed=0, density_fraction=0.7):
    n = y.shape[0]
    if n_ref >= n:
        return y.clone(), v.clone()
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    nd = int(round(n_ref*density_fraction))
    w = v.reshape(-1).clone() + 1e-14
    w /= w.sum()
    a = torch.multinomial(w, min(nd,n), replacement=False, generator=g)
    b = torch.randperm(n, generator=g)[:min(n_ref-nd,n)]
    idx = torch.unique(torch.cat([a,b]))
    while idx.numel() < n_ref:
        idx = torch.unique(torch.cat([idx, torch.randint(0,n,(n_ref-idx.numel(),),generator=g)]))
    idx = idx[:n_ref]
    return y[idx].clone(), v[idx].clone()


METHODS = {
    "Data-only NN": dict(use_data=True, physics=None, use_lbfgs=False),
    "Strong PINN": dict(use_data=False, physics="strong", use_lbfgs=False, boundary="noflux", hard_mass=False, physics_lr_multiplier=4.0, physics_grad_clip=5.0),
    "Weak PINN": dict(use_data=False, physics="weak", use_lbfgs=False, boundary="noflux", hard_mass=False, physics_lr_multiplier=4.0, physics_grad_clip=5.0),
    "Strong + data Adam-only": dict(use_data=True, physics="strong", use_lbfgs=False),
    "Weak + data Adam-only": dict(use_data=True, physics="weak", use_lbfgs=False),
    "Strong + data + LBFGS": dict(use_data=True, physics="strong", use_lbfgs=True),
    "Weak + data + LBFGS": dict(use_data=True, physics="weak", use_lbfgs=True),
}

for c in METHODS.values():
    c.setdefault("boundary", None)
    c.setdefault("boundary_weight", 0.0)
    c.setdefault("hard_mass", False)
    c.setdefault("physics_lr_multiplier", 1.0)
    c.setdefault("physics_grad_clip", 1.0)


def _relative_improvement(value, best, min_delta):
    if not np.isfinite(best):
        return True
    return value < best - min_delta * max(abs(best), 1e-12)


def _cyclic_batch(pool, start, batch_size):
    n = pool.shape[0]
    end = start + batch_size
    if end <= n:
        return pool[start:end]
    return torch.cat([pool[start:], pool[:end-n]], dim=0)


def train_model(
    method_name, net, process, device, bounds, args, target_mass=1.0,
    y_data=None, v_data=None, ):
    cfg = METHODS[method_name]
    net = net.to(device)
    if hasattr(net, "gain_raw"):
        raise RuntimeError("带数据网络中不允许存在 gain_raw")

    if cfg["use_data"]:
        if y_data is None or v_data is None:
            raise ValueError(f"{method_name} 需要 MC 数据")
        y_data, v_data = y_data.to(device), v_data.to(device)
    elif y_data is not None or v_data is not None:
        raise ValueError(f"{method_name} 是纯 PINN，禁止传入 MC 数据")

    vol = volume(bounds)
    is_pure_pinn = (not cfg["use_data"]) and (cfg["physics"] is not None)

    print("\n" + "="*92)
    print(f"开始训练：{method_name}")
    print("="*92)
    print("[采样] 普通 4D Sobol；无反射、无对称扩充、无对称约束")
    if cfg["use_data"]:
        print("[带数据规则] MC 点值损失 + 指定物理损失；无 gain、无质量投影、无质量惩罚、无边界损失")
    else:
        print("[纯 PINN] 无 MC 数据；使用显式物理/边界/单位积分三项损失")
        print(
            f"[纯 PINN 权重] phy={args.pure_physics_weight:g}, "
            f"boundary={args.pure_boundary_weight:g}, "
            f"normal={args.pure_normal_weight:g}"
        )
    print(f"[早停] {'开启' if args.early_stop else '关闭'}")

    if cfg["use_data"]:
        data_scale = torch.mean(v_data**2).detach() + 1e-14
        loader_generator = torch.Generator(device="cpu")
        loader_generator.manual_seed(args.seed + 12345)
        loader = DataLoader(
            TensorDataset(y_data, v_data),
            batch_size=min(args.data_batch_size, y_data.shape[0]),
            shuffle=True,
            drop_last=False,
            generator=loader_generator,
        )
    else:
        data_scale = torch.tensor(1.0, dtype=torch.float64, device=device)
        loader = None

    history = {
        "method": method_name,
        "stage_a_best_epoch": None,
        "stage_a_epochs_run": 0,
        "stage_b_best_cycle": None,
        "stage_b_cycles_run": 0,
        "stage_c_best_iter": None,
        "stage_c_iters_run": 0,
        "early_stop": bool(args.early_stop),
        # Diagnostic histories used only for paper figures.
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

    # ------------------------------------------------------------------
    # Stage A: MC 数据预训练。
    # 默认关闭早停时，保持旧版“每轮训练均值选最优”的行为。
    # 开启早停时，使用完整 MC 引导子集 MSE 检查并回滚。
    # ------------------------------------------------------------------
    if cfg["use_data"] and args.pretrain_epochs > 0:
        print("\n[Stage A] 数据预训练（无 gain、无质量投影）")
        opt = optim.AdamW(
            net.parameters(), lr=args.pretrain_lr, weight_decay=1e-9
        )
        sch = optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=args.pretrain_epochs,
            eta_min=args.pretrain_eta_min,
        )
        best_state = copy.deepcopy(net.state_dict())
        best_mse = float("inf")
        best_epoch = 0
        patience_count = 0

        for epoch in range(args.pretrain_epochs):
            net.train()
            total, count = 0.0, 0
            for xb, vb in loader:
                opt.zero_grad(set_to_none=True)
                rel_loss, mse_loss = normalized_data_loss(
                    net, xb, vb, data_scale
                )
                rel_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), args.data_grad_clip_pretrain
                )
                opt.step()
                total += float(mse_loss.detach()) * xb.shape[0]
                count += xb.shape[0]
            sch.step()
            epoch_mse = total / max(count, 1)
            history["stage_a_epochs_run"] = epoch + 1
            history["stage_a_epoch"].append(epoch + 1)
            history["stage_a_data_mse"].append(float(epoch_mse))

            if not args.early_stop and epoch_mse < best_mse:
                best_mse = epoch_mse
                best_epoch = epoch + 1
                best_state = copy.deepcopy(net.state_dict())

            should_check = (
                (epoch + 1) % args.pretrain_check_every == 0
                or epoch == args.pretrain_epochs - 1
            )
            full_mse = None
            if should_check:
                net.eval()
                with torch.no_grad():
                    _, full_mse_t = normalized_data_loss(
                        net, y_data, v_data, data_scale
                    )
                full_mse = float(full_mse_t)

                if args.early_stop:
                    if _relative_improvement(
                        full_mse, best_mse, args.pretrain_min_delta
                    ):
                        best_mse = full_mse
                        best_epoch = epoch + 1
                        best_state = copy.deepcopy(net.state_dict())
                        patience_count = 0
                    else:
                        patience_count += 1

                if (
                    (epoch + 1) % args.log_every == 0
                    or epoch == args.pretrain_epochs - 1
                ):
                    print(
                        f"预训练轮次 {epoch+1:5d} | "
                        f"数据 MSE={full_mse:.8e} | best_epoch={best_epoch}"
                    )

                if (
                    args.early_stop
                    and epoch + 1 >= args.pretrain_min_epochs
                    and patience_count >= args.pretrain_patience_checks
                ):
                    print(
                        f"[Stage A 早停] epoch={epoch+1}, "
                        f"best_epoch={best_epoch}, best_MSE={best_mse:.8e}"
                    )
                    break

        net.load_state_dict(best_state)
        history["stage_a_best_epoch"] = best_epoch

    # ------------------------------------------------------------------
    # Stage B preparation.
    # ------------------------------------------------------------------
    pool = sobol_box(
        args.physics_pool_size, bounds, 888 + args.seed
    ).to(device)
    check = sobol_box(
        args.fixed_check_size, bounds, 2027 + args.seed
    ).to(device)
    n_pool = pool.shape[0]

    def phys(points):
        if cfg["physics"] == "weak":
            return weak_loss(net, points, process, vol)
        if cfg["physics"] == "strong":
            return strong_loss(net, points, process, vol)
        return torch.tensor(0.0, dtype=torch.float64, device=device)

    if cfg["boundary"] == "noflux":
        bp, bn, bw = sobol_boundary(
            args.boundary_n_per_face, bounds, 3030 + args.seed
        )
        bp, bn, bw = bp.to(device), bn.to(device), bw.to(device)
    else:
        bp = bn = bw = None

    def bnd():
        if bp is None:
            return torch.tensor(0.0, dtype=torch.float64, device=device)
        return noflux_loss(net, bp, bn, bw, process)

    if cfg["use_data"]:
        with torch.no_grad():
            d0, dm0 = normalized_data_loss(
                net, y_data, v_data, data_scale
            )
        dscale = torch.clamp(d0.detach(), min=1e-10)
    else:
        d0 = dm0 = torch.tensor(0.0, dtype=torch.float64, device=device)
        dscale = torch.tensor(1.0, dtype=torch.float64, device=device)

    if cfg["physics"] is not None:
        with torch.enable_grad():
            p0 = phys(check)
        pscale = torch.clamp(p0.detach(), min=1e-10)
    else:
        p0 = torch.tensor(0.0, dtype=torch.float64, device=device)
        pscale = torch.tensor(1.0, dtype=torch.float64, device=device)

    if bp is not None:
        with torch.enable_grad():
            b0 = bnd()
        bscale = torch.clamp(b0.detach(), min=1e-10)
    else:
        b0 = torch.tensor(0.0, dtype=torch.float64, device=device)
        bscale = torch.tensor(1.0, dtype=torch.float64, device=device)

    if is_pure_pinn:
        with torch.no_grad():
            n0, m0 = normalization_loss(
                net, check, vol, target_mass=1.0
            )
    else:
        n0 = torch.tensor(0.0, dtype=torch.float64, device=device)
        with torch.no_grad():
            m0 = vol * net(check).mean()

    print("\n[Stage B 起点]")
    print(f"数据相对损失={d0.item():.8e}")
    print(f"数据 MSE={dm0.item():.8e}")
    print(f"物理损失={p0.item():.8e}")
    print(f"边界损失={b0.item():.8e}")
    print(f"归一化损失={n0.item():.8e}")
    print(f"诊断质量={m0.item():.8f}（带数据方法中只诊断，不优化）")

    opt_d = (
        optim.Adam(net.parameters(), lr=args.data_lr)
        if cfg["use_data"] else None
    )
    eff_lr = args.physics_lr * float(cfg["physics_lr_multiplier"])
    opt_p = (
        optim.Adam(
            [p for p in net.parameters() if p.requires_grad], lr=eff_lr
        )
        if cfg["physics"] else None
    )
    sch_d = (
        optim.lr_scheduler.CosineAnnealingLR(
            opt_d,
            T_max=max(1, args.joint_cycles * args.data_steps_per_cycle),
            eta_min=args.data_eta_min,
        ) if opt_d else None
    )
    sch_p = (
        optim.lr_scheduler.CosineAnnealingLR(
            opt_p,
            T_max=max(1, args.joint_cycles * args.physics_steps_per_cycle),
            eta_min=min(args.physics_eta_min, 0.1 * eff_lr),
        ) if opt_p else None
    )
    iterator = iter(loader) if loader else None

    # Strict Stage-B checkpoint semantics:
    # 起点永远是合法候选，而不是只有 --early-stop 时才进入候选集。
    # 这样如果后续 Adam 更新破坏了已经较好的 Stage-A 状态，
    # Adam-only 和 Stage-C 都会回滚到真正的全程最优点。
    if is_pure_pinn:
        start_score = float(
            args.pure_physics_weight * p0
            + args.pure_boundary_weight * b0
            + args.pure_normal_weight * n0
        )
    else:
        start_score = 0.0
        if cfg["use_data"]:
            start_score += args.checkpoint_data_weight * float((d0 / dscale).detach())
        if cfg["physics"]:
            start_score += args.checkpoint_physics_weight * float((p0 / pscale).detach())
        if bp is not None:
            start_score += float((
                cfg["boundary_weight"] * b0 / bscale
            ).detach())

    best_score = float(start_score)
    best_state = copy.deepcopy(net.state_dict())
    best_cycle = 0
    patience_count = 0

    print(
        f"[Stage B checkpoint] 起点登记为合法候选："
        f"cycle=0, score={best_score:.8e}"
    )

    print("\n[Stage B] 联合训练")
    for cycle in range(args.joint_cycles):
        net.train()

        if cfg["physics"]:
            for substep in range(args.physics_steps_per_cycle):
                update_index = cycle * args.physics_steps_per_cycle + substep
                start_index = (
                    update_index * args.physics_batch_size
                ) % n_pool
                pts = _cyclic_batch(
                    pool, start_index, args.physics_batch_size
                )
                opt_p.zero_grad(set_to_none=True)
                pl = phys(pts)
                bl = bnd()
                if is_pure_pinn:
                    nl, _ = normalization_loss(
                        net, pts, vol, target_mass=1.0
                    )
                    total_p = (
                        args.pure_physics_weight * pl
                        + args.pure_boundary_weight * bl
                        + args.pure_normal_weight * nl
                    )
                else:
                    total_p = pl / pscale
                    if bp is not None:
                        total_p = total_p + (
                            cfg["boundary_weight"] * bl / bscale
                        )
                total_p.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in net.parameters() if p.requires_grad],
                    float(cfg["physics_grad_clip"]),
                )
                opt_p.step()
                sch_p.step()

        if cfg["use_data"]:
            for _ in range(args.data_steps_per_cycle):
                try:
                    xb, vb = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    xb, vb = next(iterator)
                opt_d.zero_grad(set_to_none=True)
                dl, _ = normalized_data_loss(
                    net, xb, vb, data_scale
                )
                dl.backward()
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), args.data_grad_clip
                )
                opt_d.step()
                sch_d.step()

        history["stage_b_cycles_run"] = cycle + 1
        should_check = (
            (cycle + 1) % args.checkpoint_every == 0
            or cycle == args.joint_cycles - 1
        )
        if not should_check:
            continue

        net.eval()
        if cfg["use_data"]:
            with torch.no_grad():
                cd, cdm = normalized_data_loss(
                    net, y_data, v_data, data_scale
                )
        else:
            cd = cdm = torch.tensor(
                0.0, dtype=torch.float64, device=device
            )
        if cfg["physics"]:
            with torch.enable_grad():
                cp = phys(check)
        else:
            cp = torch.tensor(0.0, dtype=torch.float64, device=device)
        if bp is not None:
            with torch.enable_grad():
                cb = bnd()
        else:
            cb = torch.tensor(0.0, dtype=torch.float64, device=device)
        if is_pure_pinn:
            with torch.no_grad():
                cn, cm = normalization_loss(
                    net, check, vol, target_mass=1.0
                )
            score = float(
                args.pure_physics_weight * cp
                + args.pure_boundary_weight * cb
                + args.pure_normal_weight * cn
            )
        else:
            cn = torch.tensor(0.0, dtype=torch.float64, device=device)
            with torch.no_grad():
                cm = vol * net(check).mean()

            score = 0.0
            if cfg["use_data"]:
                score += args.checkpoint_data_weight * float((cd / dscale).detach())
            if cfg["physics"]:
                score += args.checkpoint_physics_weight * float((cp / pscale).detach())
            if bp is not None:
                score += float((cfg["boundary_weight"] * cb / bscale).detach())

        history["stage_b_cycle"].append(cycle + 1)
        history["stage_b_monitor_score"].append(float(score))
        history["stage_b_data_mse"].append(float(cdm.detach()))
        history["stage_b_physics_loss"].append(float(cp.detach()))

        if _relative_improvement(
            score, best_score,
            args.joint_min_delta if args.early_stop else 0.0,
        ):
            best_score = score
            best_cycle = cycle + 1
            best_state = copy.deepcopy(net.state_dict())
            patience_count = 0
        elif args.early_stop:
            patience_count += 1

        if (
            (cycle + 1) % args.log_every == 0
            or cycle == args.joint_cycles - 1
        ):
            print(
                f"循环 {cycle+1:5d} | score={score:.6e} | "
                f"data MSE={cdm.item():.8e} | physics={cp.item():.8e} | "
                f"boundary={cb.item():.8e} | normal={cn.item():.8e} | "
                f"diagnostic mass={cm.item():.8f} | "
                f"best_cycle={best_cycle}"
            )

        if (
            args.early_stop
            and cycle + 1 >= args.joint_min_cycles
            and patience_count >= args.joint_patience_checks
        ):
            print(
                f"[Stage B 早停] cycle={cycle+1}, "
                f"best_cycle={best_cycle}, best_score={best_score:.8e}"
            )
            break

    net.load_state_dict(best_state)
    history["stage_b_best_cycle"] = best_cycle
    print(
        f"[Stage B rollback] 已回滚到全程最优 checkpoint："
        f"best_cycle={best_cycle}, best_score={best_score:.8e}"
    )

    # ------------------------------------------------------------------
    # Stage C: strict shared-point L-BFGS.
    # 1) 从 Stage-B 最优 checkpoint 进入；
    # 2) 不新增 physics / monitor 坐标；
    # 3) train 点复用 Stage-B physics pool；
    # 4) monitor 点复用 Stage-B fixed check；
    # 5) 无论 --early-stop 是否开启，最终都回滚到 Stage-C monitor 最优状态。
    # ------------------------------------------------------------------
    if cfg["use_lbfgs"] and cfg["physics"] and cfg["use_data"]:
        print(
            "\n[Stage C] 从 Stage-B 最优 checkpoint 出发；"
            "复用 Stage-B physics pool / monitor 做 L-BFGS"
        )

        if args.lbfgs_point_size > pool.shape[0]:
            raise ValueError(
                "lbfgs_point_size 必须 <= physics_pool_size；"
                "strict Stage C 不允许新建物理点。"
            )

        # 训练物理点只允许来自 Stage-B pool。
        # 默认 lbfgs_point_size == physics_pool_size，因此使用完整 pool；
        # 若用户显式调小，则确定性使用 pool 的前缀，仍不产生新坐标。
        lp = pool[:args.lbfgs_point_size]
        lc = check

        print(
            f"[Stage C shared coordinates] "
            f"physics train={lp.shape[0]} / pool={pool.shape[0]}, "
            f"monitor={lc.shape[0]} / check={check.shape[0]}"
        )
        print(
            f"[Stage C data] reuse MC reference bins={y_data.shape[0]}，"
            "不新增数据坐标"
        )
        print(
            f"[Stage C weights] "
            f"data={args.lbfgs_data_weight:.3f}, "
            f"physics={args.lbfgs_physics_weight:.3f}"
        )

        # Stage-C train scales：只由进入 L-BFGS 的 Stage-B 最优 checkpoint 决定。
        with torch.enable_grad():
            pstart = phys(lp).detach()
            pmon_start = phys(lc).detach()
        with torch.no_grad():
            dstart, _ = normalized_data_loss(
                net, y_data, v_data, data_scale
            )
            dmon_start, _ = normalized_data_loss(
                net, y_data, v_data, data_scale
            )

        ps = torch.clamp(pstart, min=1e-10)
        ds = torch.clamp(dstart.detach(), min=1e-10)
        pmon_scale = torch.clamp(pmon_start, min=1e-10)
        dmon_scale = torch.clamp(dmon_start.detach(), min=1e-10)

        # Stage-C 起点本身也是合法候选。
        best_lb_state = copy.deepcopy(net.state_dict())
        best_lb_score = (
            float(args.lbfgs_data_weight)
            + float(args.lbfgs_physics_weight)
        )
        best_lb_iter = 0
        patience_count = 0
        total_iter = 0
        calls = {"n": 0}

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
            dl, _ = normalized_data_loss(
                net, y_data, v_data, data_scale
            )
            pl = phys(lp)
            loss = (
                args.lbfgs_data_weight * dl / ds
                + args.lbfgs_physics_weight * pl / ps
            )
            loss.backward()
            calls["n"] += 1
            if calls["n"] % 50 == 0:
                print(
                    f"闭包调用 {calls['n']:4d} | "
                    f"objective={loss.detach().item():.8e} | "
                    f"data={dl.detach().item():.8e} | "
                    f"physics={pl.detach().item():.8e}"
                )
            return loss

        while total_iter < args.lbfgs_max_iter:
            current_chunk = min(
                chunk, args.lbfgs_max_iter - total_iter
            )
            lb.param_groups[0]["max_iter"] = current_chunk
            lb.defaults["max_iter"] = current_chunk
            lb.step(closure)
            total_iter += current_chunk
            history["stage_c_iters_run"] = total_iter

            net.eval()
            with torch.no_grad():
                cd, cdm = normalized_data_loss(
                    net, y_data, v_data, data_scale
                )
                cm = vol * net(lc).mean()
            with torch.enable_grad():
                cp = phys(lc).detach()

            data_ratio = float((cd / dmon_scale).detach())
            physics_ratio = float((cp / pmon_scale).detach())
            monitor = (
                args.lbfgs_data_weight * data_ratio
                + args.lbfgs_physics_weight * physics_ratio
            )
            history["stage_c_iter"].append(int(total_iter))
            history["stage_c_monitor_score"].append(float(monitor))
            history["stage_c_data_mse"].append(float(cdm.detach()))
            history["stage_c_physics_loss"].append(float(cp.detach()))

            if _relative_improvement(
                monitor, best_lb_score, args.lbfgs_min_delta
            ):
                best_lb_score = float(monitor)
                best_lb_iter = total_iter
                best_lb_state = copy.deepcopy(net.state_dict())
                patience_count = 0
            else:
                patience_count += 1

            print(
                f"L-BFGS iter {total_iter:4d} | "
                f"monitor={monitor:.8e} | "
                f"data ratio={data_ratio:.6e} | "
                f"physics ratio={physics_ratio:.6e} | "
                f"data MSE={cdm.item():.8e} | "
                f"check physics={cp.item():.8e} | "
                f"diagnostic mass={cm.item():.8f} | "
                f"best_iter={best_lb_iter}"
            )

            # --early-stop 只决定是否提前结束；
            # 即使关闭，也会跑满后回滚到 monitor 最优 checkpoint。
            if (
                args.early_stop
                and total_iter >= args.lbfgs_min_iter
                and args.lbfgs_patience_checks > 0
                and patience_count >= args.lbfgs_patience_checks
            ):
                print(
                    f"[Stage C 早停] iter={total_iter}, "
                    f"best_iter={best_lb_iter}, "
                    f"best_monitor={best_lb_score:.8e}"
                )
                break

        net.load_state_dict(best_lb_state)
        history["stage_c_best_iter"] = best_lb_iter
        history["stage_c_best_monitor"] = float(best_lb_score)
        print(
            f"[Stage C rollback] 已回滚到 L-BFGS monitor 最优状态："
            f"best_iter={best_lb_iter}, "
            f"best_monitor={best_lb_score:.8e}"
        )
    print(
        f"\n[{method_name}] 训练完成。"
        f"best_score={best_score:.8e}, best_cycle={best_cycle}"
    )
    return net, history


def evaluate_points(pred, exact, pts, vol, threshold=0.01):
    pred = np.asarray(pred).reshape(-1)
    exact = np.asarray(exact).reshape(-1)
    pts = np.asarray(pts)
    mask = exact > threshold * exact.max()
    mape = np.mean(np.abs(pred[mask]-exact[mask])/(exact[mask]+1e-14))
    rel_l2 = np.sqrt(np.mean((pred-exact)**2)/(np.mean(exact**2)+1e-30))
    mass = vol * float(np.mean(pred))

    wells = {}
    for s1, s2, name in [(-1,-1,"--"),(-1,1,"-+"),(1,-1,"+-"),(1,1,"++")]:
        m1 = pts[:,0] < 0 if s1 < 0 else pts[:,0] >= 0
        m2 = pts[:,1] < 0 if s2 < 0 else pts[:,1] >= 0
        indicator = (m1 & m2).astype(np.float64)
        wells[name] = vol * float(np.mean(pred * indicator))
    gap = max(wells.values()) - min(wells.values())
    return {
        "有效区域平均相对误差": float(mape),
        "相对L2误差": float(rel_l2),
        "诊断总质量": float(mass),
        "势阱--质量": wells["--"],
        "势阱-+质量": wells["-+"],
        "势阱+-质量": wells["+-"],
        "势阱++质量": wells["++"],
        "四势阱质量极差": float(gap),
    }


def line_error(net, process, device, bounds, n=321):
    x1 = np.linspace(bounds[0][0], bounds[0][1], n)
    pts = np.column_stack([x1, np.ones_like(x1), np.zeros_like(x1), np.zeros_like(x1)])
    exact = process.exact_density_numpy(pts)
    with torch.no_grad():
        pred = net(torch.tensor(pts, dtype=torch.float64, device=device)).cpu().numpy().reshape(-1)
    mask = exact > 0.01 * exact.max()
    return float(np.mean(np.abs(pred[mask]-exact[mask])/(exact[mask]+1e-14)))


def evaluate_model(net, process, device, bounds, eval_points):
    pts = eval_points.numpy()
    with torch.no_grad():
        chunks = [net(c.to(device)).cpu().numpy() for c in torch.split(eval_points, 20000)]
    pred = np.vstack(chunks).reshape(-1)
    exact = process.exact_density_numpy(pts)
    out = evaluate_points(pred, exact, pts, volume(bounds))
    out["截线误差"] = line_error(net, process, device, bounds)
    return out


def evaluate_histogram(process, y, v, bounds):
    pts = y.numpy()
    pred = v.numpy().reshape(-1)
    exact = process.exact_density_numpy(pts)
    out = evaluate_points(pred, exact, pts, volume(bounds))
    out["截线误差"] = float("nan")
    return out


def print_table(results):
    print("\n" + "="*126)
    print("4D 结果汇总")
    print("="*126)
    print(f"{'方法':<30s} | {'MAPE(%)':>11s} | {'Rel-L2(%)':>11s} | {'Line(%)':>11s} | {'Mass':>10s} | {'Well gap':>10s} | {'--':>9s} | {'-+':>9s} | {'+-':>9s} | {'++':>9s}")
    print("-"*126)
    for name, m in results.items():
        line = f"{100*m['截线误差']:11.4f}" if np.isfinite(m["截线误差"]) else f"{'N/A':>11s}"
        print(
            f"{name:<30s} | {100*m['有效区域平均相对误差']:11.4f} | {100*m['相对L2误差']:11.4f} | {line} | "
            f"{m['诊断总质量']:10.6f} | {m['四势阱质量极差']:10.6f} | {m['势阱--质量']:9.6f} | "
            f"{m['势阱-+质量']:9.6f} | {m['势阱+-质量']:9.6f} | {m['势阱++质量']:9.6f}"
        )
    print("="*126 + "\n")


def print_runtime(runtimes):
    print("\n" + "="*72)
    print("各方法训练时间")
    print("="*72)
    print(f"{'方法':<36s} | {'秒':>12s} | {'分钟':>12s}")
    print("-"*72)
    for name, sec in runtimes.items():
        print(f"{name:<36s} | {sec:12.2f} | {sec/60:12.2f}")
    print("="*72 + "\n")



def _paper_figures_dir(out_dir):
    figures_dir = Path(out_dir) / "paper_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _display_method(method_name):
    mapping = {
        "Data-only NN": "Data-only",
        "Strong PINN": "Strong PINN",
        "Weak PINN": "Weak PINN",
        "Strong + data Adam-only": "Strong+data Adam",
        "Weak + data Adam-only": "Weak+data Adam",
        "Strong + data + LBFGS": "Strong+data L-BFGS",
        "Weak + data + LBFGS": "Weak+data L-BFGS",
    }
    return mapping.get(method_name, method_name)


def _method_file_tag(method_name):
    return (
        str(method_name).lower()
        .replace("+", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def plot_bars(results, figures_dir):
    names = list(results)
    labels = [_display_method(n) for n in names]
    mape = [100.0 * results[n]["有效区域平均相对误差"] for n in names]
    l2 = [100.0 * results[n]["相对L2误差"] for n in names]

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.bar(labels, mape)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("4D rough potential: MAPE comparison")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(Path(figures_dir) / "rough_potential_4d_mape_bar.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.bar(labels, l2)
    ax.set_ylabel("Relative L2 error (%)")
    ax.set_title("4D rough potential: relative L2 comparison")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(Path(figures_dir) / "rough_potential_4d_rel_l2_bar.png", dpi=300)
    plt.close(fig)


def _slice_grid_4d(bounds, resolution=221):
    b = np.asarray(bounds, dtype=np.float64)
    x1 = np.linspace(b[0, 0], b[0, 1], resolution)
    x2 = np.linspace(b[1, 0], b[1, 1], resolution)
    xx, yy = np.meshgrid(x1, x2, indexing="xy")
    pts = np.zeros((resolution * resolution, 4), dtype=np.float64)
    pts[:, 0] = xx.ravel()
    pts[:, 1] = yy.ravel()
    return x1, x2, pts


def plot_reference_slice(process, bounds, figures_dir, resolution=221):
    x1, x2, pts = _slice_grid_4d(bounds, resolution)
    exact = process.exact_density_numpy(pts).reshape(resolution, resolution)
    extent = [float(x1.min()), float(x1.max()), float(x2.min()), float(x2.max())]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(
        exact, origin="lower", extent=extent, aspect="equal",
        cmap="viridis", vmin=0.0, vmax=float(exact.max()),
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Reference density slice ($x_3=x_4=0$)")
    fig.colorbar(image, ax=ax, label="Stationary density")
    fig.tight_layout()
    path = Path(figures_dir) / "rough_potential_4d_reference_slice_x1_x2.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_method_slice_and_errors(
    net, method_name, process, device, bounds, figures_dir, resolution=221,
):
    x1, x2, pts = _slice_grid_4d(bounds, resolution)
    tensor_pts = torch.tensor(pts, dtype=torch.float64)
    net.eval()
    with torch.no_grad():
        pred = np.concatenate([
            net(chunk.to(device)).cpu().numpy().reshape(-1)
            for chunk in torch.split(tensor_pts, 20000)
        ]).reshape(resolution, resolution)
    exact = process.exact_density_numpy(pts).reshape(resolution, resolution)
    extent = [float(x1.min()), float(x1.max()), float(x2.min()), float(x2.max())]
    density_vmax = float(exact.max())
    abs_error = np.abs(pred - exact)
    valid = exact > 0.01 * exact.max()
    rel_error = np.full_like(exact, np.nan)
    rel_error[valid] = 100.0 * abs_error[valid] / (exact[valid] + 1e-14)
    tag = _method_file_tag(method_name)
    paths = []

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(
        pred, origin="lower", extent=extent, aspect="equal",
        cmap="viridis", vmin=0.0, vmax=density_vmax,
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"{method_name}: density slice ($x_3=x_4=0$)")
    fig.colorbar(image, ax=ax, label="Stationary density")
    fig.tight_layout()
    path = Path(figures_dir) / f"rough_potential_4d_{tag}_solution_slice.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(
        abs_error, origin="lower", extent=extent, aspect="equal",
        cmap="YlOrRd", vmin=0.0, vmax=density_vmax,
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"{method_name}: absolute-error slice")
    fig.colorbar(image, ax=ax, label=r"$|u_\theta-u_{\mathrm{ref}}|$")
    fig.tight_layout()
    path = Path(figures_dir) / f"rough_potential_4d_{tag}_absolute_error_slice.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(
        rel_error, origin="lower", extent=extent, aspect="equal",
        cmap="viridis", vmin=0.0, vmax=100.0,
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(
        f"{method_name}: pointwise relative-error slice\n"
        "Effective region; display clipped at 100%"
    )
    fig.colorbar(image, ax=ax, label="Pointwise relative error (%)")
    fig.tight_layout()
    path = Path(figures_dir) / f"rough_potential_4d_{tag}_relative_error_slice.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)
    return paths


def plot_method_profile(
    net, method_name, process, device, bounds, figures_dir, n=401,
):
    x1 = np.linspace(bounds[0][0], bounds[0][1], n)
    pts = np.column_stack([
        x1,
        np.ones_like(x1),
        np.zeros_like(x1),
        np.zeros_like(x1),
    ])
    exact = process.exact_density_numpy(pts)
    with torch.no_grad():
        pred = net(
            torch.tensor(pts, dtype=torch.float64, device=device)
        ).cpu().numpy().reshape(-1)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(x1, exact, linewidth=2.0, label="Reference")
    ax.plot(x1, pred, linewidth=1.8, label=method_name)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("Stationary density")
    ax.set_title(r"1D profile: $x_2=1,\ x_3=x_4=0$")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = Path(figures_dir) / f"rough_potential_4d_{_method_file_tag(method_name)}_profile_x1.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


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
         "Epoch", "Data MSE", "rough_potential_4d_stage_a_adam_loss.png"),
        ("stage_b_cycle", "stage_b_monitor_score", "Stage B monitor score",
         "Cycle", "Monitor score", "rough_potential_4d_stage_b_monitor.png"),
        ("stage_c_iter", "stage_c_monitor_score", "Stage C L-BFGS monitor score",
         "L-BFGS iteration", "Monitor score", "rough_potential_4d_stage_c_lbfgs_monitor.png"),
    ]
    return [
        p for p in (
            _plot_history_family(histories, figures_dir, *spec)
            for spec in specs
        ) if p is not None
    ]


def plot_optimizer_ablation(results, figures_dir):
    pairs = [
        ("Strong + data Adam-only", "Strong + data + LBFGS", "Strong"),
        ("Weak + data Adam-only", "Weak + data + LBFGS", "Weak"),
    ]
    available = [(a, b, label) for a, b, label in pairs if a in results and b in results]
    if not available:
        return []
    labels = []
    for adam, lbfgs, family in available:
        labels.extend([f"{family} Adam", f"{family} + L-BFGS"])
    paths = []
    for metric_key, ylabel, suffix in (
        ("有效区域平均相对误差", "MAPE (%)", "mape"),
        ("相对L2误差", "Relative L2 error (%)", "rel_l2"),
    ):
        values = []
        for adam, lbfgs, _ in available:
            values.extend([
                100.0 * results[adam][metric_key],
                100.0 * results[lbfgs][metric_key],
            ])
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        ax.bar(labels, values)
        ax.set_ylabel(ylabel)
        ax.set_title("Training ablation: effect of Stage C L-BFGS")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path = Path(figures_dir) / f"rough_potential_4d_lbfgs_ablation_{suffix}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)
    return paths


def _ordered_evaluation_names(evaluation_arrays):
    """Keep the canonical method order exactly as METHODS; never sort by metric."""
    return [name for name in METHODS if name in evaluation_arrays]


def _clear_existing_pngs(figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for path in figures_dir.glob("*.png"):
        path.unlink()


def evaluate_slice_and_store_arrays(
    net, process, device, bounds, resolution=221,
):
    x_g, y_g, pts = _slice_grid_4d(bounds, resolution)
    tensor_pts = torch.tensor(pts, dtype=torch.float64)
    net.eval()
    with torch.no_grad():
        pred = np.concatenate([
            net(chunk.to(device)).cpu().numpy().reshape(-1)
            for chunk in torch.split(tensor_pts, 20000)
        ]).reshape(resolution, resolution)
    exact = process.exact_density_numpy(pts).reshape(resolution, resolution)
    return {
        "x_g": np.asarray(x_g, dtype=np.float64),
        "y_g": np.asarray(y_g, dtype=np.float64),
        "exact": np.asarray(exact, dtype=np.float64),
        "pred": np.asarray(pred, dtype=np.float64),
        "slice_dims": (0, 1),
        "fixed_coordinates": (0.0, 0.0),
    }


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
            raise ValueError("All methods must use the same 2D slice grid.")
    return names, x_g, y_g, exact


def _comparison_layout(panels, x_g, y_g, values_key, cmap, vmin, vmax,
                       colorbar_label, title, path, title_formatter=None):
    extent = [
        float(x_g.min()), float(x_g.max()),
        float(y_g.min()), float(y_g.max()),
    ]
    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.15*ncols, 3.70*nrows),
        sharex=True, sharey=True, squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    image = None
    for ax, panel in zip(axes_flat, panels):
        values = panel[values_key]
        image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic",
            resample=True,
        )
        label = panel["label"]
        ax.set_title(
            title_formatter(panel) if title_formatter is not None else label,
            fontsize=9,
        )
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.grid(False)
    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)
    if image is not None:
        cbar = fig.colorbar(
            image,
            ax=[ax for ax in axes_flat[:len(panels)]],
            shrink=0.94, pad=0.02, extend="max",
        )
        cbar.set_label(colorbar_label)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return Path(path)


def plot_all_method_density_comparison(evaluation_arrays, figures_dir):
    names, x_g, y_g, exact = _common_evaluation_grid(evaluation_arrays)
    dx = float(np.mean(np.diff(x_g)))
    dy = float(np.mean(np.diff(y_g)))
    panels = [{"label": "Reference", "values": exact, "global_l2": 0.0}]
    for name in names:
        pred = np.asarray(evaluation_arrays[name]["pred"], dtype=np.float64)
        global_l2 = float(np.sqrt(np.sum((pred-exact)**2)*dx*dy))
        panels.append({
            "label": _display_method(name),
            "values": pred,
            "global_l2": global_l2,
        })
    ref_peak = float(np.nanmax(exact))
    if not np.isfinite(ref_peak) or ref_peak <= 0.0:
        ref_peak = 1.0
    path = Path(figures_dir) / "rough_potential_4d_all_methods_density_comparison.png"
    return _comparison_layout(
        panels, x_g, y_g, "values", "viridis", 0.0, 1.05*ref_peak,
        "Stationary density",
        "4D rough potential: reference and predictions on $x_3=x_4=0$",
        path,
        title_formatter=lambda p: (
            "Reference" if p["label"] == "Reference"
            else f'{p["label"]}\nSlice L2={p["global_l2"]:.3e}'
        ),
    )


def plot_mape_bar_comparison(results, figures_dir):
    names = [name for name in METHODS if name in results]
    if not names:
        return None
    labels = [_display_method(name) for name in names]
    values = np.asarray([
        100.0 * results[name]["有效区域平均相对误差"]
        for name in names
    ], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12.8, 5.8), constrained_layout=True)
    x = np.arange(len(names), dtype=np.float64)
    ax.bar(x, values)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Method-wise MAPE comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    path = Path(figures_dir) / "rough_potential_4d_mape_bar_chart.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_all_method_error_comparison(evaluation_arrays, figures_dir):
    names, x_g, y_g, exact = _common_evaluation_grid(evaluation_arrays)
    dx = float(np.mean(np.diff(x_g)))
    dy = float(np.mean(np.diff(y_g)))
    panels = [{"label": "Reference", "error": np.zeros_like(exact), "global_l2": 0.0}]
    pooled = []
    for name in names:
        pred = np.asarray(evaluation_arrays[name]["pred"], dtype=np.float64)
        err = np.abs(pred-exact)
        pooled.append(err.ravel())
        panels.append({
            "label": _display_method(name),
            "error": err,
            "global_l2": float(np.sqrt(np.sum(err**2)*dx*dy)),
        })
    all_err = np.concatenate(pooled)
    vmax = float(np.percentile(all_err, 99.7))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = max(float(np.nanmax(all_err)), 1e-12)
    path = Path(figures_dir) / "rough_potential_4d_global_absolute_error_comparison.png"
    return _comparison_layout(
        panels, x_g, y_g, "error", "YlOrRd", 0.0, vmax,
        r"Absolute error $|u_\theta-u_{\mathrm{ref}}|$",
        "4D rough potential: absolute-error comparison on $x_3=x_4=0$",
        path,
        title_formatter=lambda p: (
            "Reference" if p["label"] == "Reference"
            else f'{p["label"]}\nSlice L2={p["global_l2"]:.3e}'
        ),
    )


def plot_all_method_relative_error_comparison(
    evaluation_arrays, figures_dir, tau_fraction=0.01,
):
    names, x_g, y_g, exact = _common_evaluation_grid(evaluation_arrays)
    ref_peak = float(np.nanmax(exact))
    if not np.isfinite(ref_peak) or ref_peak <= 0.0:
        ref_peak = 1.0
    tau = float(tau_fraction * ref_peak)
    denom = np.maximum(exact, tau)
    panels = [{"label": "Reference", "error": np.zeros_like(exact)}]
    pooled = []
    for name in names:
        pred = np.asarray(evaluation_arrays[name]["pred"], dtype=np.float64)
        err = 100.0*np.abs(pred-exact)/denom
        pooled.append(err.ravel())
        panels.append({"label": _display_method(name), "error": err})
    all_err = np.concatenate(pooled)
    vmax = float(np.percentile(all_err, 99.5))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(all_err))
    vmax = min(max(vmax, 1.0), 500.0)
    path = Path(figures_dir) / "rough_potential_4d_global_relative_error_comparison.png"
    return _comparison_layout(
        panels, x_g, y_g, "error", "viridis", 0.0, vmax,
        r"Relative error $100|u_\theta-u_{\rm ref}|/\max(u_{\rm ref},\tau)$ (%)",
        "4D rough potential: relative-error comparison on $x_3=x_4=0$",
        path,
    )


def render_four_comparison_figures(results, evaluation_arrays, figures_dir):
    _clear_existing_pngs(figures_dir)
    method_results = {
        k: v for k, v in results.items()
        if not k.startswith("MC histogram")
    }
    paths = [
        plot_all_method_density_comparison(evaluation_arrays, figures_dir),
        plot_mape_bar_comparison(method_results, figures_dir),
        plot_all_method_error_comparison(evaluation_arrays, figures_dir),
        plot_all_method_relative_error_comparison(evaluation_arrays, figures_dir),
    ]
    return tuple(str(Path(p).resolve()) for p in paths if p is not None)



def run_comparison(process, y_ref, v_ref, device, out_dir, bounds, args):
    if args.only_weak_hybrid:
        methods = ["Weak + data Adam-only", "Weak + data + LBFGS"]
    elif args.only_pure:
        methods = ["Strong PINN", "Weak PINN"]
    elif args.only_hybrids:
        methods = [
            "Data-only NN",
            "Strong + data Adam-only",
            "Weak + data Adam-only",
            "Strong + data + LBFGS",
            "Weak + data + LBFGS",
        ]
    else:
        methods = list(METHODS)

    eval_points = sobol_box(args.eval_size, bounds, args.seed + 9001)
    figures_dir = _paper_figures_dir(out_dir)
    results = {
        "MC histogram (diag)": evaluate_histogram(process, y_ref, v_ref, bounds)
    }
    runtimes, histories, model_states = {}, {}, {}
    evaluation_arrays = {}
    y_shared, v_shared = make_reference_subset(
        y_ref, v_ref, args.n_ref, args.seed + 8001, args.density_fraction,
    )

    print("\n" + "#"*108)
    print("开始 4D 完整消融比较")
    print("#"*108)
    print("绘图策略：训练期间不出单方法图；全部方法完成后只输出四张统一对比图。")
    print("缓存策略：保存 scalar results / histories / model_states / evaluation_arrays，后续可 --plot-only。")

    for i, name in enumerate(methods):
        print("\n" + "#"*108)
        print(f"最终方法 {i+1}/{len(methods)}：{name}")
        print("#"*108)
        set_seed(args.seed)
        cfg = METHODS[name]
        yd, vd = (
            (y_shared.clone(), v_shared.clone())
            if cfg["use_data"] else (None, None)
        )
        net = GeneralFPNN(width=args.width, depth=args.depth).to(device)
        start_time = time.time()
        net, history = train_model(
            name, net, process, device, bounds, args,
            target_mass=1.0, y_data=yd, v_data=vd,
        )
        elapsed = time.time() - start_time
        metrics = evaluate_model(net, process, device, bounds, eval_points)
        evaluation_arrays[name] = evaluate_slice_and_store_arrays(
            net, process, device, bounds, resolution=221,
        )
        results[name] = metrics
        runtimes[name] = elapsed
        histories[name] = history
        model_states[name] = {
            key: value.detach().cpu().clone()
            for key, value in net.state_dict().items()
        }
        print(
            f"[完成] {name} | "
            f"MAPE={100*metrics['有效区域平均相对误差']:.4f}% | "
            f"Rel-L2={100*metrics['相对L2误差']:.4f}% | "
            f"well gap={metrics['四势阱质量极差']:.6f} | "
            f"time={elapsed/60:.3f} min"
        )
        del net

    print_table(results)
    print_runtime(runtimes)
    figure_files = render_four_comparison_figures(
        results, evaluation_arrays, figures_dir,
    )
    print("\n[四张统一对比图]")
    for path in figure_files:
        print(f"  {path}")

    save_path = _resolve_cache_path(args, out_dir)
    saved = {
        "results": results,
        "runtimes": runtimes,
        "histories": histories,
        "model_states": model_states,
        "evaluation_arrays": evaluation_arrays,
        "cache_format_version": 2,
        "figure_files": figure_files,
        "args": vars(args),
        "bounds": bounds,
        "epsilon": process.epsilon,
        "dimension": 4,
        "methods_run": methods,
        "reference_points": y_ref.detach().cpu(),
        "reference_values": v_ref.detach().cpu(),
        "evaluation_slice": {"dims": (0, 1), "fixed": (0.0, 0.0), "resolution": 221},
        "rotation_removed": True,
        "symmetric_sampling_removed": True,
        "pure_pinn_uses_mc_data": False,
        "hybrid_has_gain": False,
        "hybrid_uses_mass_projection": False,
        "hybrid_uses_mass_penalty": False,
        "hybrid_uses_boundary_loss": False,
        "pure_pinn_hard_mass": False,
                "pure_pinn_explicit_normalization_loss": True,
                "pure_pinn_boundary_loss_for_both": True,
        "potential": (
            "0.25*(x1^2-1)^2+0.25*(x2^2-1)^2+"
            "0.5*x3^2+0.5*x4^2+eps^4*prod(sin(2*pi*xi/eps))"
        ),
    }
    _save_cache(saved, save_path)
    return results, runtimes


def _resolve_cache_path(args, out_dir):
    if getattr(args, "cache_file", None):
        path = Path(args.cache_file).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path
    return Path(out_dir) / "rough_potential_4d_full_ablation_results.pt"


def _load_cache(cache_path):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"找不到缓存文件：{cache_path.resolve()}\n"
            "请先完整运行一次，或用 --cache-file 指向已有 .pt。"
        )
    try:
        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
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


def _rebuild_model_from_saved_state(method_name, state, saved_args, bounds, device):
    width = int(saved_args.get("width", 128))
    depth = int(saved_args.get("depth", 4))
    base = GeneralFPNN(width=width, depth=depth).to(device)
    # New pure-PINN checkpoints are plain networks. Old checkpoints that
    # contain norm_points are still reconstructed through the legacy wrapper.
    if "norm_points" in state:
        target_mass = float(state.get(
            "target_mass_tensor", torch.tensor(1.0, dtype=torch.float64)
        ).item())
        net = MassNormalizedDensity(
            base,
            state["norm_points"].detach().clone(),
            volume(bounds),
            target_mass,
        ).to(device)
    else:
        net = base
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def _render_cached_outputs(saved, out_dir):
    evaluation_arrays = saved.get("evaluation_arrays")
    if not evaluation_arrays:
        raise RuntimeError(
            "这个缓存没有 evaluation_arrays，不能直接 --plot-only。\n"
            "旧版 .pt 请先运行 --eval-only；它会读取 model_states，"
            "重新计算一次评价网格并把缓存升级为可直接绘图的版本。"
        )
    results = dict(saved.get("results", {}))
    figures_dir = _paper_figures_dir(out_dir)
    figure_files = render_four_comparison_figures(
        results, evaluation_arrays, figures_dir,
    )
    saved["figure_files"] = figure_files
    saved["cache_format_version"] = max(int(saved.get("cache_format_version", 1)), 2)
    print("[plot-only] 仅从缓存数组重画四张图，没有训练，也没有重新计算网络预测。")
    return saved


def run_plot_only(args, out_dir):
    cache_path = _resolve_cache_path(args, out_dir)
    saved = _load_cache(cache_path)
    saved = _render_cached_outputs(saved, out_dir)
    _save_cache(saved, cache_path)


def run_eval_only(args, out_dir):
    cache_path = _resolve_cache_path(args, out_dir)
    saved = _load_cache(cache_path)
    states = saved.get("model_states")
    if not states:
        raise RuntimeError("缓存里没有 model_states，无法 --eval-only。")

    saved_args = dict(saved.get("args", {}))
    bounds = tuple(tuple(v) for v in saved.get(
        "bounds", ((-2.0,2.0),(-2.0,2.0),(-1.2,1.2),(-1.2,1.2))
    ))
    epsilon = float(saved.get("epsilon", saved_args.get("epsilon", 0.10)))
    seed = int(saved_args.get("seed", args.seed))
    eval_size = int(saved_args.get("eval_size", 65536))
    normalizer_size = int(saved_args.get("normalizer_size", 262144))
    device = torch.device("cpu")

    process = RoughPotential4D(epsilon)
    process.prepare_normalizer(bounds, normalizer_size, seed + 5001)
    eval_points = sobol_box(eval_size, bounds, seed + 9001)

    methods = [
        name for name in saved.get("methods_run", list(states.keys()))
        if name in states and name in METHODS
    ]
    if not methods:
        methods = [name for name in states if name in METHODS]

    results = dict(saved.get("results", {}))
    evaluation_arrays = {}
    rebuilt_states = {}
    for name in methods:
        print(f"[重新评价] {name}")
        net = _rebuild_model_from_saved_state(
            name, states[name], saved_args, bounds, device,
        )
        results[name] = evaluate_model(net, process, device, bounds, eval_points)
        evaluation_arrays[name] = evaluate_slice_and_store_arrays(
            net, process, device, bounds, resolution=221,
        )
        rebuilt_states[name] = {
            k: v.detach().cpu().clone() for k, v in net.state_dict().items()
        }

    saved["results"] = results
    saved["model_states"] = rebuilt_states
    saved["evaluation_arrays"] = evaluation_arrays
    saved["methods_run"] = methods
    saved["cache_format_version"] = 2
    saved = _render_cached_outputs(saved, out_dir)
    _save_cache(saved, cache_path)
    print("[eval-only] 没有重新训练；旧 checkpoint 已升级为可 --plot-only 的缓存。")


def parse_args():
    p = argparse.ArgumentParser(
        description="4D reversible rough-potential full ablation (no-gain hybrids)"
    )
    p.add_argument("--fast", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument("--plot-only", action="store_true", help="直接从新版 .pt 的 evaluation_arrays 只重画四张对比图。")
    mode_group.add_argument("--eval-only", action="store_true", help="从旧/新 .pt 的 model_states 重建网络并刷新 evaluation_arrays，不重新训练。")
    p.add_argument("--cache-file", type=str, default=None, help="缓存 .pt 路径；默认使用 out-dir/rough_potential_4d_full_ablation_results.pt。")
    p.add_argument("--only-pure", action="store_true")
    p.add_argument("--only-hybrids", action="store_true")
    p.add_argument("--only-weak-hybrid", action="store_true")
    p.add_argument("--early-stop", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        default="rough_potential_4d_paper_outputs",
    )
    p.add_argument(
        "--results-txt",
        default="rough_potential_4d_full_ablation_results.txt",
    )
    p.add_argument("--epsilon", type=float, default=0.10)

    # 本 strict 版本测试较小坐标预算：8000 MC 参考箱 + 4000 physics train；monitor 单独用于模型选择。
    p.add_argument("--n-paths", type=int, default=210000)
    p.add_argument("--burn-in", type=float, default=8.0)
    p.add_argument("--sample-duration", type=float, default=2.0)
    p.add_argument("--sample-interval", type=float, default=0.1)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--bin-width-double", type=float, default=0.20)
    p.add_argument("--bin-width-harmonic", type=float, default=0.20)

    p.add_argument("--n-ref", type=int, default=8000)
    p.add_argument("--density-fraction", type=float, default=0.70)
    p.add_argument("--eval-size", type=int, default=65536)
    p.add_argument("--normalizer-size", type=int, default=262144)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)

    p.add_argument("--pretrain-epochs", type=int, default=1200)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--pretrain-eta-min", type=float, default=5e-5)
    p.add_argument("--data-grad-clip-pretrain", type=float, default=2.0)

    p.add_argument("--joint-cycles", type=int, default=3000)
    p.add_argument("--physics-pool-size", type=int, default=4000)
    p.add_argument("--fixed-check-size", type=int, default=1000)
    p.add_argument("--data-batch-size", type=int, default=512)
    p.add_argument("--physics-batch-size", type=int, default=1000)
    p.add_argument("--data-steps-per-cycle", type=int, default=1)
    p.add_argument("--physics-steps-per-cycle", type=int, default=2)
    p.add_argument("--data-lr", type=float, default=1.2e-4)
    p.add_argument("--physics-lr", type=float, default=6e-5)
    p.add_argument("--data-eta-min", type=float, default=1e-5)
    p.add_argument("--physics-eta-min", type=float, default=5e-6)
    p.add_argument("--data-grad-clip", type=float, default=1.0)
    p.add_argument("--checkpoint-data-weight", type=float, default=1.0)
    p.add_argument("--checkpoint-physics-weight", type=float, default=1.0)
    p.add_argument("--checkpoint-every", type=int, default=375)
    p.add_argument("--log-every", type=int, default=375)

    p.add_argument("--lbfgs-point-size", type=int, default=4000)
    p.add_argument("--lbfgs-max-iter", type=int, default=500)
    p.add_argument("--lbfgs-lr", type=float, default=0.5)
    p.add_argument("--lbfgs-history-size", type=int, default=100)
    p.add_argument("--lbfgs-data-weight", type=float, default=1.0)
    p.add_argument("--lbfgs-physics-weight", type=float, default=1.0)

    p.add_argument("--pure-physics-weight", type=float, default=1.0)
    p.add_argument("--pure-boundary-weight", type=float, default=0.10)
    p.add_argument("--pure-normal-weight", type=float, default=1.0)
    p.add_argument("--boundary-n-per-face", type=int, default=128)
    # Kept only for backward-compatible command lines / old caches.
    p.add_argument(
        "--hard-mass-quad-size",
        type=int,
        default=8192,
        help=argparse.SUPPRESS,
    )

    # 可选提前停止：Stage B/C 最优状态回滚始终开启；该开关只控制是否提前结束训练。
    p.add_argument("--pretrain-check-every", type=int, default=50)
    p.add_argument("--pretrain-patience-checks", type=int, default=8)
    p.add_argument("--pretrain-min-epochs", type=int, default=400)
    p.add_argument("--pretrain-min-delta", type=float, default=1e-4)
    p.add_argument("--joint-patience-checks", type=int, default=4)
    p.add_argument("--joint-min-cycles", type=int, default=1000)
    p.add_argument("--joint-min-delta", type=float, default=1e-4)
    p.add_argument("--lbfgs-check-every", type=int, default=25)
    p.add_argument("--lbfgs-patience-checks", type=int, default=4)
    p.add_argument("--lbfgs-min-iter", type=int, default=100)
    p.add_argument("--lbfgs-min-delta", type=float, default=1e-4)
    return p.parse_args()


def apply_fast(args):
    if not args.fast:
        return args
    print("[快速模式] 仅检查完整消融流程")
    args.n_paths = min(args.n_paths, 1000)
    args.burn_in = min(args.burn_in, 0.2)
    args.sample_duration = min(args.sample_duration, 0.1)
    args.sample_interval = min(args.sample_interval, 0.05)
    args.dt = max(args.dt, 0.005)
    args.n_ref = min(args.n_ref, 128)
    args.eval_size = min(args.eval_size, 512)
    args.normalizer_size = min(args.normalizer_size, 2048)
    args.pretrain_epochs = min(args.pretrain_epochs, 4)
    args.pretrain_min_epochs = min(args.pretrain_min_epochs, 2)
    args.pretrain_check_every = 1
    args.joint_cycles = min(args.joint_cycles, 4)
    args.joint_min_cycles = min(args.joint_min_cycles, 2)
    args.checkpoint_every = 1
    args.log_every = 1
    args.physics_pool_size = min(args.physics_pool_size, 32)
    args.fixed_check_size = min(args.fixed_check_size, 16)
    args.physics_batch_size = min(args.physics_batch_size, 8)
    args.data_batch_size = min(args.data_batch_size, 32)
    args.lbfgs_point_size = min(args.lbfgs_point_size, 16)
    args.lbfgs_max_iter = min(args.lbfgs_max_iter, 2)
    args.lbfgs_check_every = 1
    args.physics_steps_per_cycle = 1
    args.data_steps_per_cycle = 1
    args.boundary_n_per_face = min(args.boundary_n_per_face, 2)
    args.hard_mass_quad_size = min(args.hard_mass_quad_size, 64)
    args.width = min(args.width, 12)
    args.depth = min(args.depth, 1)
    args.bin_width_double = max(args.bin_width_double, 1.0)
    args.bin_width_harmonic = max(args.bin_width_harmonic, 0.8)
    return args


def smoke_test(seed=42, epsilon=0.1):
    set_seed(seed)
    process = RoughPotential4D(epsilon)
    bounds = ((-2,2),(-2,2),(-1.2,1.2),(-1.2,1.2))
    vol = volume(bounds)
    pts = sobol_box(24, bounds, seed + 20)
    bp, bn, bw = sobol_boundary(2, bounds, seed + 40)

    direct = GeneralFPNN(width=16, depth=1)
    target = torch.full((24, 1), 0.01, dtype=torch.float64)
    dscale = torch.mean(target**2) + 1e-14
    dl, _ = normalized_data_loss(direct, pts, target, dscale)
    wl = weak_loss(direct, pts, process, vol)
    (dl + wl).backward()

    for offset, physics_type in enumerate(("strong", "weak"), start=1):
        set_seed(seed + offset)
        net = GeneralFPNN(width=12, depth=1)
        pl = (
            strong_loss(net, pts, process, vol)
            if physics_type == "strong"
            else weak_loss(net, pts, process, vol)
        )
        bl = noflux_loss(net, bp, bn, bw, process)
        nl, mass = normalization_loss(net, pts, vol, target_mass=1.0)
        total = pl + 0.10 * bl + nl
        total.backward()
        print(
            f"[{physics_type}] physics={pl.item():.8e}, "
            f"boundary={bl.item():.8e}, normal={nl.item():.8e}, "
            f"mass={mass.item():.8f}, total={total.item():.8e}"
        )

    print("4D pure-PINN three-term forward/backward: PASS")




class TeeStream:
    def __init__(self,*streams): self.streams=streams
    def write(self,data):
        for s in self.streams: s.write(data); s.flush()
        return len(data)
    def flush(self):
        for s in self.streams: s.flush()
    def isatty(self): return any(getattr(s,"isatty",lambda:False)() for s in self.streams)



def execute(args, out_dir):
    if getattr(args, "plot_only", False):
        run_plot_only(args, out_dir)
        return
    if getattr(args, "eval_only", False):
        run_eval_only(args, out_dir)
        return
    if args.smoke_test:
        smoke_test(args.seed, args.epsilon)
        return
    args = apply_fast(args)
    if args.lbfgs_point_size > args.physics_pool_size:
        raise ValueError(
            "lbfgs_point_size 不能超过 physics_pool_size；"
            "strict Stage C 只能复用 Stage-B physics pool。"
        )
    set_seed(args.seed)
    start = time.time()
    process = RoughPotential4D(args.epsilon)
    bounds = ((-2.0,2.0),(-2.0,2.0),(-1.2,1.2),(-1.2,1.2))
    device = torch.device("cpu")

    print("\n" + "="*108)
    print("4D 完整消融实验配置")
    print("="*108)
    for key in (
        "epsilon", "n_paths", "bin_width_double",
        "bin_width_harmonic", "n_ref", "density_fraction",
        "physics_pool_size", "fixed_check_size",
        "physics_batch_size", "physics_steps_per_cycle",
        "pretrain_epochs", "joint_cycles", "lbfgs_point_size",
        "lbfgs_max_iter", "eval_size", "normalizer_size",
        "early_stop",
    ):
        print(f"{key:<28s}= {getattr(args, key)}")
    print("最终图像输出固定为 4 张：density / MAPE / absolute error / relative error。")
    print("MAPE 柱状图严格按 METHODS 中的方法顺序，不按数值排序。")
    print("="*108 + "\n")

    process.prepare_normalizer(bounds, args.normalizer_size, args.seed + 5001)
    samples = process.simulate_stationary_samples(
        args.n_paths, args.burn_in, args.sample_duration,
        args.sample_interval, args.dt, args.seed,
    )
    y, v, bin_volume = build_histogram_reference(
        samples, bounds,
        (args.bin_width_double, args.bin_width_double,
         args.bin_width_harmonic, args.bin_width_harmonic),
    )
    validate_reference(process, y, v, bin_volume)
    results, runtimes = run_comparison(
        process, y, v, device, out_dir, bounds, args,
    )
    elapsed = time.time() - start
    print(f"\n4D 完整消融结束，总用时={elapsed/60:.2f} 分钟")
    return {"results": results, "runtimes": runtimes, "elapsed": elapsed}


def main():
    args=parse_args(); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    path=out_dir/args.results_txt; original=sys.stdout
    with open(path,"w",encoding="utf-8",buffering=1) as f:
        sys.stdout=TeeStream(original,f)
        try:
            print(f"[文本结果文件] {path.resolve()}")
            execute(args,out_dir)
            print(f"\n[文本结果保存完成] {path.resolve()}")
        finally:
            sys.stdout=original
    print(f"全部输出已保存到：{path.resolve()}")


if __name__ == "__main__":
    main()