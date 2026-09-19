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


def make_importance_physics_set(
    n,
    bounds,
    process,
    gaussian_fraction=0.85,
    std_scale=1.25,
    seed=0,
):
    """
    生成固定的 importance-corrected 物理积分点。

    proposal q = a*q_G + (1-a)*Uniform(Omega)，其中 q_G 是截断在
   计算盒内的独立高斯。高斯标准差由原点线性化 OU 尺度
        sqrt(D/lambda_i)
    给出，再乘 std_scale。该 proposal 只使用漂移/扩散参数，不使用
    MC 标签或解析稳态密度。

    返回 dimensionless 权重
        w(x) = 1 / (|Omega| q(x)).
    因此
        int_Omega f(x) dx = |Omega| E_q[w(x) f(x)].
    """
    if n <= 0:
        raise ValueError("importance physics 点数必须为正")
    if not (0.0 <= gaussian_fraction < 1.0):
        raise ValueError("gaussian_fraction 必须位于 [0,1)")
    if std_scale <= 0:
        raise ValueError("std_scale 必须为正")

    b = np.asarray(bounds, dtype=np.float64)
    dim = b.shape[0]
    if dim != process.dimension:
        raise ValueError("proposal 维数与过程不一致")
    vol = volume(bounds)
    lo = torch.tensor(b[:, 0], dtype=torch.float64).view(1, dim)
    hi = torch.tensor(b[:, 1], dtype=torch.float64).view(1, dim)
    std = torch.tensor(
        std_scale * np.sqrt(process.D / process.lambdas),
        dtype=torch.float64,
    ).view(1, dim)

    n_g = int(round(n * gaussian_fraction))
    n_u = n - n_g
    pieces = []

    if n_g > 0:
        eng_g = torch.quasirandom.SobolEngine(
            dim, scramble=True, seed=seed + 17
        )
        u = eng_g.draw(n_g).to(torch.float64)
        sqrt2 = np.sqrt(2.0)
        cdf_lo = 0.5 * (1.0 + torch.erf(lo / (std * sqrt2)))
        cdf_hi = 0.5 * (1.0 + torch.erf(hi / (std * sqrt2)))
        p = cdf_lo + u * (cdf_hi - cdf_lo)
        p = torch.clamp(p, 1e-14, 1.0 - 1e-14)
        z = sqrt2 * torch.erfinv(2.0 * p - 1.0)
        pieces.append(std * z)

    if n_u > 0:
        pieces.append(sobol_box(n_u, bounds, seed + 29))

    points = torch.cat(pieces, dim=0)

    # Exact mixture density at every generated point.
    z = points / std
    sqrt2 = np.sqrt(2.0)
    cdf_lo = 0.5 * (1.0 + torch.erf(lo / (std * sqrt2)))
    cdf_hi = 0.5 * (1.0 + torch.erf(hi / (std * sqrt2)))
    trunc_mass = torch.clamp(cdf_hi - cdf_lo, min=1e-300)
    log_qg = torch.sum(
        -0.5 * z ** 2
        - torch.log(std)
        - 0.5 * np.log(2.0 * np.pi)
        - torch.log(trunc_mass),
        dim=1,
        keepdim=True,
    )
    qg = torch.exp(log_qg)
    qu = 1.0 / vol
    q = gaussian_fraction * qg + (1.0 - gaussian_fraction) * qu
    importance = 1.0 / (vol * torch.clamp(q, min=1e-300))

    # Deterministic shuffle prevents component blocks in cyclic mini-batches.
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 41)
    perm = torch.randperm(points.shape[0], generator=g)
    points = points[perm].contiguous()
    importance = importance[perm].contiguous()

    ess = float((importance.sum() ** 2 / (importance.square().sum() + 1e-30)))
    print(
        f"[importance physics] n={n}, gaussian={n_g}, uniform={n_u}, "
        f"std_scale={std_scale:.3f}, weight ESS={ess:.1f}"
    )
    return points, importance


class RoughPotential8D:
    dimension = 8

    def __init__(
        self,
        epsilon=0.05,
        rough_strength=1.0,
        lambdas=(2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7),
    ):
        if epsilon <= 0:
            raise ValueError("epsilon 必须为正数")
        if rough_strength < 0:
            raise ValueError("rough_strength 不能为负数")
        if len(lambdas) != self.dimension:
            raise ValueError("lambdas 必须有 8 个分量")

        self.epsilon = float(epsilon)
        self.rough_strength = float(rough_strength)
        self.lambdas = np.asarray(lambdas, dtype=np.float64)
        if np.any(self.lambdas <= 0):
            raise ValueError("所有 lambda_i 必须为正数")

        self.sigma = 0.5
        self.Z = None

        # 8 个循环四元耦合项：
        # (0,1,2,3), (1,2,3,4), ..., (7,0,1,2)
        self.terms = np.asarray(
            [[(j + m) % self.dimension for m in range(4)]
             for j in range(self.dimension)],
            dtype=np.int64,
        )

        # 每个坐标属于 4 个四元项；每个项对对应 Hessian 行最多贡献 4*C。
        # 所以粗糙项的 Hessian 行绝对值和保守上界为 16*C。
        c = (
            self.rough_strength
            * (2.0 * np.pi) ** 2
            * self.epsilon ** 2
        )
        self.convexity_lower_bound = float(self.lambdas.min() - 16.0 * c)
        if self.convexity_lower_bound <= 0:
            raise ValueError(
                "当前参数不能由保守界保证单峰。"
                f" Hessian 下界={self.convexity_lower_bound:.6e}。"
                "请减小 epsilon/rough_strength，或增大 lambdas。"
            )

    @property
    def D(self):
        return 0.5 * self.sigma ** 2

    def _rough_products_numpy(self, s):
        """
        s: (..., 8)
        返回每个循环四元项的正弦乘积，shape (..., 8)。
        """
        st = s[..., self.terms]  # (..., 8 terms, 4 local coordinates)
        return np.prod(st, axis=-1)

    def potential_numpy(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != self.dimension:
            raise ValueError("输入最后一维必须为 8")

        base = 0.5 * np.sum(self.lambdas * x ** 2, axis=-1)
        ph = 2.0 * np.pi * x / self.epsilon
        s = np.sin(ph)
        rough = (
            self.rough_strength
            * self.epsilon ** 4
            * np.sum(self._rough_products_numpy(s), axis=-1)
        )
        return base + rough

    def _rough_grad_numpy(self, x):
        """
        计算 rough 部分对 x 的梯度，shape (N, 8)。
        """
        ph = 2.0 * np.pi * x / self.epsilon
        s = np.sin(ph)
        c = np.cos(ph)

        st = s[:, self.terms]  # (N, 8 terms, 4)
        ct = c[:, self.terms]

        prod_ex = np.empty_like(st)
        prod_ex[:, :, 0] = st[:, :, 1] * st[:, :, 2] * st[:, :, 3]
        prod_ex[:, :, 1] = st[:, :, 0] * st[:, :, 2] * st[:, :, 3]
        prod_ex[:, :, 2] = st[:, :, 0] * st[:, :, 1] * st[:, :, 3]
        prod_ex[:, :, 3] = st[:, :, 0] * st[:, :, 1] * st[:, :, 2]

        local = ct * prod_ex
        out = np.zeros_like(x)
        coef = (
            self.rough_strength
            * 2.0 * np.pi
            * self.epsilon ** 3
        )

        # 对固定 local position，terms[:, k] 是 0,...,7 的一个排列，
        # 因此可直接按列累加，不需要利用任何概率结构。
        for k in range(4):
            out[:, self.terms[:, k]] += coef * local[:, :, k]
        return out

    def drift_numpy(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError("drift 输入必须为 (N,8)")
        return -(x * self.lambdas.reshape(1, -1) + self._rough_grad_numpy(x))

    def drift_torch(self, x):
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError("drift 输入必须为 (N,8)")

        lam = torch.as_tensor(
            self.lambdas, dtype=x.dtype, device=x.device
        ).view(1, -1)
        terms = torch.as_tensor(
            self.terms, dtype=torch.long, device=x.device
        )

        ph = 2.0 * np.pi * x / self.epsilon
        s = torch.sin(ph)
        c = torch.cos(ph)

        st = s[:, terms]  # (N, 8 terms, 4)
        ct = c[:, terms]

        prod_ex = torch.empty_like(st)
        prod_ex[:, :, 0] = st[:, :, 1] * st[:, :, 2] * st[:, :, 3]
        prod_ex[:, :, 1] = st[:, :, 0] * st[:, :, 2] * st[:, :, 3]
        prod_ex[:, :, 2] = st[:, :, 0] * st[:, :, 1] * st[:, :, 3]
        prod_ex[:, :, 3] = st[:, :, 0] * st[:, :, 1] * st[:, :, 2]

        local = ct * prod_ex
        rough_grad = torch.zeros_like(x)
        coef = (
            self.rough_strength
            * 2.0 * np.pi
            * self.epsilon ** 3
        )
        for k in range(4):
            rough_grad[:, terms[:, k]] = (
                rough_grad[:, terms[:, k]] + coef * local[:, :, k]
            )

        return -(x * lam + rough_grad)

    def prepare_normalizer(
        self,
        bounds,
        n_points=1048576,
        seed=0,
        chunk_size=65536,
    ):
        """
        仅为解析参考计算 Z。使用普通 8D Sobol 积分，不参与训练。
        分块生成，避免一次性保存过大的中间数组。
        """
        b = np.asarray(bounds, dtype=np.float64)
        dim = b.shape[0]
        eng = torch.quasirandom.SobolEngine(
            dim, scramble=True, seed=seed
        )
        lo = torch.tensor(b[:, 0], dtype=torch.float64).view(1, dim)
        hi = torch.tensor(b[:, 1], dtype=torch.float64).view(1, dim)

        total = 0.0
        done = 0
        while done < n_points:
            m = min(chunk_size, n_points - done)
            u = eng.draw(m).to(torch.float64)
            pts = (lo + (hi - lo) * u).numpy()
            vals = np.exp(-self.potential_numpy(pts) / self.D)
            total += float(vals.sum())
            done += m

        self.Z = volume(bounds) * total / float(n_points)
        if not np.isfinite(self.Z) or self.Z <= 0:
            raise FloatingPointError(f"归一化失败 Z={self.Z}")

        print(
            f"[解析参考] 8D Sobol 归一化点数={n_points}, "
            f"Z={self.Z:.12e}"
        )
        print(
            f"[单峰检查] 保守 Hessian 下界="
            f"{self.convexity_lower_bound:.8e} > 0"
        )

    def exact_density_numpy(self, x):
        if self.Z is None:
            raise RuntimeError("请先调用 prepare_normalizer")
        return np.exp(-self.potential_numpy(x) / self.D) / self.Z

    def simulate_histogram_path_batch(
        self,
        grid,
        counts,
        n_paths,
        burn_in,
        sample_duration,
        sample_interval,
        dt,
        seed,
        batch_number,
    ):
        """
        对一批独立路径做普通 Euler-Maruyama，并把采样时刻直接累计到
        8D 直方图。无反射、无解析稳态初始化、不保存全部快照。
        """
        if n_paths <= 0:
            raise ValueError("n_paths 必须为正数")
        if dt <= 0 or burn_in < 0 or sample_duration <= 0:
            raise ValueError("时间参数非法")
        if sample_interval <= 0:
            raise ValueError("sample_interval 必须为正数")

        rng = np.random.default_rng(seed)
        x = 0.25 * rng.standard_normal(
            (n_paths, self.dimension)
        ).astype(np.float64)

        burn_steps = int(round(burn_in / dt))
        sample_steps = int(round(sample_duration / dt))
        interval = max(1, int(round(sample_interval / dt)))
        total_steps = burn_steps + sample_steps
        noise = self.sigma * np.sqrt(dt)

        raw_samples = 0
        inside_samples = 0
        snapshots = 0

        print(
            f"[MC batch {batch_number}] paths={n_paths}, "
            f"steps={total_steps}, burn={burn_steps}, "
            f"interval_steps={interval}"
        )

        for step in range(total_steps):
            x += (
                self.drift_numpy(x) * dt
                + noise * rng.standard_normal(x.shape)
            )

            completed = step + 1
            should_sample = (
                completed > burn_steps
                and (completed - burn_steps) % interval == 0
            )
            if should_sample:
                raw, inside = grid.accumulate(counts, x)
                raw_samples += raw
                inside_samples += inside
                snapshots += 1

            if (
                completed % max(1, total_steps // 5) == 0
                or completed == total_steps
            ):
                print(
                    f"[MC batch {batch_number}] "
                    f"Euler {completed}/{total_steps}"
                )

        print(
            f"[MC batch {batch_number}] snapshots={snapshots}, "
            f"raw={raw_samples}, inside={inside_samples}, "
            f"inside fraction="
            f"{inside_samples/max(raw_samples,1):.8f}"
        )
        return {
            "raw_samples": int(raw_samples),
            "inside_samples": int(inside_samples),
            "snapshots": int(snapshots),
        }


class HistogramGrid8D:
    """规则 8D 直方图网格；计数数组使用扁平索引保存。"""

    def __init__(self, bounds, requested_width=0.2):
        b = np.asarray(bounds, dtype=np.float64)
        if b.shape != (8, 2):
            raise ValueError("bounds 必须为 (8,2)")
        if requested_width <= 0:
            raise ValueError("requested_width 必须为正数")

        self.bounds = b
        self.lo = b[:, 0].copy()
        self.hi = b[:, 1].copy()
        lengths = self.hi - self.lo

        n_bins = np.rint(lengths / requested_width).astype(np.int64)
        if np.any(n_bins < 2):
            raise ValueError("每一维至少需要两个直方图箱子")

        self.shape = tuple(int(v) for v in n_bins)
        self.widths = lengths / n_bins
        self.total_bins = int(np.prod(n_bins, dtype=np.int64))
        self.bin_volume = float(np.prod(self.widths))

        print(
            f"[8D histogram] shape={self.shape}, "
            f"total bins={self.total_bins:,}, "
            f"actual widths={tuple(float(v) for v in self.widths)}, "
            f"bin volume={self.bin_volume:.12e}"
        )

    def flat_indices(self, points):
        x = np.asarray(points, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 8:
            raise ValueError("points 必须为 (N,8)")

        inside = np.all(
            (x >= self.lo.reshape(1, -1))
            & (x <= self.hi.reshape(1, -1)),
            axis=1,
        )
        xin = x[inside]
        if xin.shape[0] == 0:
            return np.empty(0, dtype=np.int64), inside

        idx = np.floor(
            (xin - self.lo.reshape(1, -1))
            / self.widths.reshape(1, -1)
        ).astype(np.int64)
        for j, n in enumerate(self.shape):
            idx[:, j] = np.clip(idx[:, j], 0, n - 1)

        flat = idx[:, 0].copy()
        for j in range(1, 8):
            flat = flat * self.shape[j] + idx[:, j]
        return flat, inside

    def accumulate(self, counts, points):
        flat, inside = self.flat_indices(points)
        if flat.size:
            unique, freq = np.unique(flat, return_counts=True)
            counts[unique] += freq.astype(counts.dtype, copy=False)
        return int(points.shape[0]), int(inside.sum())

    def centers_from_flat(self, flat_indices):
        flat = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
        work = flat.copy()
        multi = np.empty((flat.size, 8), dtype=np.int64)

        for j in range(7, -1, -1):
            n = self.shape[j]
            multi[:, j] = work % n
            work //= n

        return (
            self.lo.reshape(1, -1)
            + (multi.astype(np.float64) + 0.5)
            * self.widths.reshape(1, -1)
        )

    def sobol_unique_flat_indices(
        self,
        n,
        seed,
        excluded=None,
    ):
        if n <= 0:
            return np.empty(0, dtype=np.int64)

        excluded_set = (
            set() if excluded is None
            else set(np.asarray(excluded, dtype=np.int64).tolist())
        )
        chosen = []
        chosen_set = set()
        eng = torch.quasirandom.SobolEngine(
            8, scramble=True, seed=seed
        )

        while len(chosen) < n:
            need = n - len(chosen)
            draw_n = max(1024, 2 * need)
            u = eng.draw(draw_n).to(torch.float64).numpy()
            idx = np.floor(
                u * np.asarray(self.shape, dtype=np.float64)
            ).astype(np.int64)
            idx = np.minimum(
                idx,
                np.asarray(self.shape, dtype=np.int64) - 1,
            )

            flat = idx[:, 0].copy()
            for j in range(1, 8):
                flat = flat * self.shape[j] + idx[:, j]

            for value in flat:
                ivalue = int(value)
                if (
                    ivalue not in excluded_set
                    and ivalue not in chosen_set
                ):
                    chosen.append(ivalue)
                    chosen_set.add(ivalue)
                    if len(chosen) >= n:
                        break

        return np.asarray(chosen, dtype=np.int64)


def precompute_exact_histogram_centers(
    process,
    grid,
    chunk_size=262144,
):
    """
    只用于 MC 质量验收和最终诊断。
    不进入数据标签构造之外的任何训练目标。
    """
    exact = np.empty(grid.total_bins, dtype=np.float64)
    print(
        f"[MC 验收] 预计算 {grid.total_bins:,} 个箱中心的解析密度"
    )

    for start in range(0, grid.total_bins, chunk_size):
        end = min(start + chunk_size, grid.total_bins)
        centers = grid.centers_from_flat(
            np.arange(start, end, dtype=np.int64)
        )
        exact[start:end] = process.exact_density_numpy(centers)

        if (
            end == grid.total_bins
            or end % max(chunk_size, grid.total_bins // 10) < chunk_size
        ):
            print(
                f"[MC 验收] exact centers {end:,}/"
                f"{grid.total_bins:,}"
            )

    if not np.all(np.isfinite(exact)):
        raise FloatingPointError("解析箱中心密度出现非有限值")
    return exact


def evaluate_histogram_counts(
    counts,
    inside_count,
    grid,
    exact_center_density,
    threshold=0.01,
    chunk_size=262144,
):
    if inside_count <= 0:
        raise FloatingPointError("盒内 MC 样本数为零")

    exact = np.asarray(exact_center_density, dtype=np.float64)
    if exact.shape != (grid.total_bins,):
        raise ValueError("exact_center_density shape 不匹配")

    denom = float(inside_count) * grid.bin_volume
    exact_max = float(exact.max())
    cutoff = threshold * exact_max

    relative_sum = 0.0
    relative_count = 0
    error_sq_sum = 0.0
    exact_sq_sum = 0.0
    pred_moment = np.zeros(8, dtype=np.float64)
    exact_moment = np.zeros(8, dtype=np.float64)

    for start in range(0, grid.total_bins, chunk_size):
        end = min(start + chunk_size, grid.total_bins)
        pred = counts[start:end].astype(np.float64) / denom
        ex = exact[start:end]
        mask = ex > cutoff

        if np.any(mask):
            relative_sum += float(
                np.sum(
                    np.abs(pred[mask] - ex[mask])
                    / (ex[mask] + 1e-14)
                )
            )
            relative_count += int(mask.sum())

        error_sq_sum += float(np.sum((pred - ex) ** 2))
        exact_sq_sum += float(np.sum(ex ** 2))

        centers = grid.centers_from_flat(
            np.arange(start, end, dtype=np.int64)
        )
        pred_moment += (
            grid.bin_volume
            * np.sum(pred[:, None] * centers, axis=0)
        )
        exact_moment += (
            grid.bin_volume
            * np.sum(ex[:, None] * centers, axis=0)
        )

    mass = (
        float(counts.sum())
        / float(inside_count)
    )
    if not np.isclose(mass, 1.0, atol=1e-12, rtol=1e-12):
        raise FloatingPointError(
            f"直方图质量归一化失败：mass={mass:.16e}"
        )

    peak_index = int(np.argmax(exact))
    peak_pred = (
        float(counts[peak_index]) / denom
    )
    peak_exact = float(exact[peak_index])

    return {
        "有效区域平均相对误差": (
            relative_sum / max(relative_count, 1)
        ),
        "相对L2误差": np.sqrt(
            error_sq_sum / (exact_sq_sum + 1e-30)
        ),
        "诊断总质量": mass,
        "一阶矩L2误差": float(
            np.linalg.norm(pred_moment - exact_moment)
        ),
        "斜截线误差": float("nan"),
        "中心峰值相对误差": abs(
            peak_pred - peak_exact
        ) / (peak_exact + 1e-14),
        "有效箱数": int(relative_count),
        "非零箱数": int(np.count_nonzero(counts)),
        "箱中心解析质量": (
            grid.bin_volume * float(exact.sum())
        ),
        "盒内MC样本数": int(inside_count),
    }




def make_histogram_training_subset(
    counts,
    inside_count,
    grid,
    n_ref,
    density_fraction,
    seed,
):
    """Faithful 4D-style subset: density-weighted + uniform histogram bins."""
    if not (0.0 <= density_fraction <= 1.0):
        raise ValueError("density_fraction 必须位于 [0,1]")
    if n_ref <= 0:
        raise ValueError("n_ref 必须为正数")
    if n_ref > grid.total_bins:
        raise ValueError("n_ref 不能超过直方图箱数")

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    values = torch.as_tensor(counts, dtype=torch.float64)
    weights = values + 1e-14
    weights /= weights.sum()

    n_density = int(round(n_ref * density_fraction))
    n_uniform = n_ref - n_density
    idx_density = torch.multinomial(
        weights,
        num_samples=n_density,
        replacement=False,
        generator=g,
    )
    idx_uniform = torch.randperm(
        grid.total_bins, generator=g
    )[:n_uniform]
    idx = torch.unique(torch.cat([idx_density, idx_uniform]))
    while idx.numel() < n_ref:
        extra = torch.randint(
            0,
            grid.total_bins,
            (n_ref - idx.numel(),),
            generator=g,
        )
        idx = torch.unique(torch.cat([idx, extra]))
    idx = idx[:n_ref].numpy().astype(np.int64, copy=False)

    points = grid.centers_from_flat(idx)
    density = counts[idx].astype(np.float64) / (
        float(inside_count) * grid.bin_volume
    )
    print(
        f"[训练参考子集] n_ref={n_ref}, "
        f"density-weighted={n_density}, uniform={n_uniform}, "
        f"zero-target fraction={np.mean(density == 0.0):.6f}"
    )
    return (
        torch.tensor(points, dtype=torch.float64),
        torch.tensor(density, dtype=torch.float64).unsqueeze(1),
        idx,
    )

def load_histogram_reference_from_counts(
    process,
    bounds,
    args,
    out_dir,
):
    """
    优先读取上一轮保存并通过验收的 8D 直方图计数。
    如果计数文件缺失，则使用完全相同的 Euler-Maruyama 配置从零生成，
    每批保存一次；训练、损失、网络、验收阈值和后续流程均保持不变。
    """
    counts_path = Path(args.hist_counts_file)
    grid = HistogramGrid8D(
        bounds, args.hist_bin_width
    )
    generated_from_scratch = not counts_path.exists()
    if generated_from_scratch and not args.allow_mc_rebuild_if_missing:
        raise FileNotFoundError(
            "未找到上一轮要复用的 MC 直方图计数文件："
            f"{counts_path.resolve()}\n"
            "本版默认不重新生成、不追加 MC，以确保七个实验都使用完全相同的 "
            "7,337,767 个盒内样本。请检查文件路径，或通过 "
            "--hist-counts-file 指定实际位置。只有确实需要重新生成时，"
            "才显式添加 --allow-mc-rebuild-if-missing。"
        )
    if generated_from_scratch:
        print(
            "[MC cache missing] Explicit rebuild was enabled; "
            "a new histogram will be generated and saved."
        )
    else:
        print(f"[MC cache] Reusing cached histogram counts: {counts_path.resolve()}")

    augmentation_convergence = []
    new_raw = 0
    new_inside = 0
    batches_run = 0
    metrics = None

    if generated_from_scratch:
        print(
            "[MC 文件缺失] 未找到上一轮直方图计数："
            f"{counts_path.resolve()}"
        )
        print(
            "[MC 从零重建] 使用当前脚本原有的普通 Euler-Maruyama "
            "配置重新生成；不改变后续训练设置。"
        )
        counts_path.parent.mkdir(parents=True, exist_ok=True)
        counts = np.zeros(grid.total_bins, dtype=np.int64)
        inside_count = 0
    else:
        counts = np.load(
            counts_path,
            mmap_mode=None,
            allow_pickle=False,
        )

        if counts.ndim != 1:
            raise ValueError(
                f"计数数组必须是一维，实际 shape={counts.shape}"
            )
        if counts.size != grid.total_bins:
            raise ValueError(
                "计数文件与当前网格不匹配："
                f"counts.size={counts.size:,}, "
                f"expected={grid.total_bins:,}"
            )
        if not np.issubdtype(counts.dtype, np.integer):
            raise TypeError(
                f"计数数组必须为整数类型，实际 dtype={counts.dtype}"
            )
        if np.any(counts < 0):
            raise ValueError("计数数组中存在负数")

        counts = np.asarray(counts, dtype=np.int64)
        inside_count = int(counts.sum())
        if inside_count <= 0:
            raise FloatingPointError("直方图计数总和为零")

        expected_inside = int(args.expected_mc_inside_samples)
        if (
            expected_inside > 0
            and not args.augment_existing_mc
            and inside_count != expected_inside
        ):
            raise ValueError(
                "MC 规模检查失败：当前计数文件包含 "
                f"{inside_count:,} 个盒内样本，但本版要求直接使用 "
                f"{expected_inside:,} 个。请确认 --hist-counts-file 指向上一轮保存的 "
                "rough_potential_8d_histogram_counts_augmented_to_15pct.npy。"
                "如有意使用其他规模，请显式设置 "
                "--expected-mc-inside-samples 0。"
            )

        print(f"[复用 MC] 读取计数文件：{counts_path.resolve()}")
        print(
            f"[复用 MC] inside samples={inside_count:,}, "
            f"nonzero bins={np.count_nonzero(counts):,}"
        )

    exact_grid = precompute_exact_histogram_centers(
        process,
        grid,
        chunk_size=args.hist_eval_chunk_size,
    )

    def mc_target_met(current_metrics):
        # 用户指定的 MC 目标是有效区域 MAPE；Rel-L2 仅保留为诊断。
        # 对宽度 0.2 的 8D 直方图，箱平均被放在箱中心会造成不可由
        # 增加 MC 样本消除的峰部系统偏差，因此不再把 Rel-L2 作为硬门槛。
        return (
            current_metrics["有效区域平均相对误差"]
            <= args.mc_target_mape
        )

    if generated_from_scratch:
        for batch_id in range(args.mc_max_batches):
            batch_number = batch_id + 1
            batch_info = process.simulate_histogram_path_batch(
                grid=grid,
                counts=counts,
                n_paths=args.mc_path_batch_size,
                burn_in=args.burn_in,
                sample_duration=args.sample_duration,
                sample_interval=args.sample_interval,
                dt=args.dt,
                seed=args.seed + 100003 * batch_number,
                batch_number=batch_number,
            )
            new_raw += int(batch_info["raw_samples"])
            new_inside += int(batch_info["inside_samples"])
            batches_run = batch_number
            inside_count = int(counts.sum())

            metrics = evaluate_histogram_counts(
                counts=counts,
                inside_count=inside_count,
                grid=grid,
                exact_center_density=exact_grid,
                threshold=args.eval_threshold,
                chunk_size=args.hist_eval_chunk_size,
            )
            augmentation_convergence.append(
                {
                    "batch": batch_number,
                    "inside_samples": inside_count,
                    "mape": float(metrics["有效区域平均相对误差"]),
                    "rel_l2": float(metrics["相对L2误差"]),
                    "source": "generated_from_scratch",
                }
            )

            # 保存到原参数指定的位置。即使后面未达标或程序被重新启动，
            # 已完成批次也不会丢失；下一次运行会按“已有计数”继续追加。
            np.save(counts_path, counts)
            print(
                f"[MC 重建 {batch_number}/{args.mc_max_batches}] "
                f"inside={inside_count:,}, "
                f"MAPE={100*metrics['有效区域平均相对误差']:.4f}%, "
                f"Rel-L2={100*metrics['相对L2误差']:.4f}%"
            )
            print(f"[MC 重建计数已保存] {counts_path.resolve()}")

            if mc_target_met(metrics):
                print(
                    "[MC 重建] 已达到设定的 MAPE 目标；"
                    "Rel-L2 仍只作诊断。"
                )
                break

        if metrics is None or inside_count <= 0:
            raise RuntimeError("MC 从零重建未产生有效盒内样本。")
    else:
        metrics = evaluate_histogram_counts(
            counts=counts,
            inside_count=inside_count,
            grid=grid,
            exact_center_density=exact_grid,
            threshold=args.eval_threshold,
            chunk_size=args.hist_eval_chunk_size,
        )

        augmented_path = (
            out_dir
            / "rough_potential_8d_histogram_counts_augmented_to_15pct.npy"
        )
        if args.augment_existing_mc and not mc_target_met(metrics):
            print(
                "[MC 追加] 现有计数未达到 15% 目标；"
                "只追加普通 Euler-Maruyama 样本并与原计数逐箱相加。"
            )
            for batch_id in range(args.mc_max_batches):
                batch_number = batch_id + 1
                batch_info = process.simulate_histogram_path_batch(
                    grid=grid,
                    counts=counts,
                    n_paths=args.mc_path_batch_size,
                    burn_in=args.burn_in,
                    sample_duration=args.sample_duration,
                    sample_interval=args.sample_interval,
                    dt=args.dt,
                    seed=args.seed + 7000003 + 100003 * batch_number,
                    batch_number=batch_number,
                )
                new_raw += int(batch_info["raw_samples"])
                new_inside += int(batch_info["inside_samples"])
                batches_run = batch_number
                inside_count = int(counts.sum())
                metrics = evaluate_histogram_counts(
                    counts=counts,
                    inside_count=inside_count,
                    grid=grid,
                    exact_center_density=exact_grid,
                    threshold=args.eval_threshold,
                    chunk_size=args.hist_eval_chunk_size,
                )
                augmentation_convergence.append(
                    {
                        "batch": batch_number,
                        "inside_samples": inside_count,
                        "mape": float(
                            metrics["有效区域平均相对误差"]
                        ),
                        "rel_l2": float(metrics["相对L2误差"]),
                        "source": "augmented_existing_counts",
                    }
                )
                np.save(augmented_path, counts)
                print(
                    f"[MC 追加 {batch_number}/{args.mc_max_batches}] "
                    f"inside={inside_count:,}, "
                    f"MAPE="
                    f"{100*metrics['有效区域平均相对误差']:.4f}%, "
                    f"Rel-L2={100*metrics['相对L2误差']:.4f}%"
                )
                if mc_target_met(metrics):
                    print(
                        "[MC 追加] 已达到设定的 MAPE 目标；"
                        "Rel-L2 仅作诊断。"
                    )
                    break
            if batches_run > 0:
                counts_path = augmented_path
                print(f"[MC 追加计数保存] {counts_path.resolve()}")

    print("\n" + "=" * 80)
    print(
        "新建 8D MC 直方图验收"
        if generated_from_scratch
        else "复用 8D MC 直方图验收"
    )
    print("-" * 80)
    print(
        f"MAPE="
        f"{100*metrics['有效区域平均相对误差']:.4f}% "
        f"(target <= {100*args.mc_target_mape:.2f}%)"
    )
    print(
        f"Rel-L2="
        f"{100*metrics['相对L2误差']:.4f}% "
        f"(diagnostic reference "
        f"{100*args.mc_target_rel_l2:.2f}%; not a hard gate)"
    )
    print(
        f"Mass={metrics['诊断总质量']:.12f}, "
        f"inside samples={inside_count:,}, "
        f"effective bins={metrics['有效箱数']:,}"
    )
    print(
        f"Peak-center error="
        f"{100*metrics['中心峰值相对误差']:.4f}%"
    )
    print("=" * 80 + "\n")

    accepted = (
        metrics["有效区域平均相对误差"]
        <= args.mc_target_mape
    )
    if not accepted and not args.allow_unqualified_mc:
        source_text = "新生成的" if generated_from_scratch else "读取的"
        raise RuntimeError(
            f"{source_text} MC 直方图未通过 MAPE 硬验收，停止训练。"
            f" MAPE={100*metrics['有效区域平均相对误差']:.4f}%,"
            f" Rel-L2={100*metrics['相对L2误差']:.4f}%（仅诊断）。"
            f" 当前计数已保存到：{counts_path.resolve()}。"
            "再次运行会读取已保存计数并继续追加，不会从零开始。"
        )

    y_data, v_data, selected_indices = (
        make_histogram_training_subset(
            counts=counts,
            inside_count=inside_count,
            grid=grid,
            n_ref=args.n_ref,
            density_fraction=args.density_fraction,
            seed=args.seed + 7001,
        )
    )

    return {
        "grid": grid,
        "counts": counts,
        "exact_grid": exact_grid,
        "metrics": metrics,
        "convergence": augmentation_convergence,
        "accepted": bool(accepted),
        "batches_run": int(batches_run),
        "raw_samples": int(new_raw) if batches_run > 0 else None,
        "inside_samples": inside_count,
        "augmented_inside_samples": int(new_inside),
        "y_data": y_data,
        "v_data": v_data,
        "selected_indices": selected_indices,
        "counts_path": str(counts_path.resolve()),
        "reused_counts": not generated_from_scratch,
    }


def build_adaptive_histogram_reference(
    process,
    bounds,
    args,
    out_dir,
):
    """
    流式累计 MC 直方图，并使用解析密度做“仅评价”的硬验收。
    正常模式下，只有同时满足 MAPE 和 Rel-L2 阈值才允许进入训练。
    """
    grid = HistogramGrid8D(
        bounds, args.hist_bin_width
    )
    counts = np.zeros(
        grid.total_bins, dtype=np.int64
    )
    exact_grid = precompute_exact_histogram_centers(
        process,
        grid,
        chunk_size=args.hist_eval_chunk_size,
    )

    total_raw = 0
    total_inside = 0
    convergence = []
    accepted = False
    final_metrics = None
    batches_run = 0

    for batch_id in range(args.mc_max_batches):
        batch_number = batch_id + 1
        batch_info = process.simulate_histogram_path_batch(
            grid=grid,
            counts=counts,
            n_paths=args.mc_path_batch_size,
            burn_in=args.burn_in,
            sample_duration=args.sample_duration,
            sample_interval=args.sample_interval,
            dt=args.dt,
            seed=args.seed + 100003 * batch_number,
            batch_number=batch_number,
        )
        total_raw += batch_info["raw_samples"]
        total_inside += batch_info["inside_samples"]
        batches_run = batch_number

        should_check = (
            batch_number >= args.mc_min_batches
            and (
                batch_number % args.mc_check_every == 0
                or batch_number == args.mc_max_batches
            )
        )
        if not should_check:
            continue

        final_metrics = evaluate_histogram_counts(
            counts=counts,
            inside_count=total_inside,
            grid=grid,
            exact_center_density=exact_grid,
            threshold=args.eval_threshold,
            chunk_size=args.hist_eval_chunk_size,
        )
        convergence.append(
            {
                "batch": batch_number,
                "raw_samples": int(total_raw),
                "inside_samples": int(total_inside),
                "mape": float(
                    final_metrics[
                        "有效区域平均相对误差"
                    ]
                ),
                "rel_l2": float(
                    final_metrics["相对L2误差"]
                ),
                "mass": float(
                    final_metrics["诊断总质量"]
                ),
            }
        )

        print("\n" + "=" * 80)
        print("8D MC 直方图验收")
        print("-" * 80)
        print(
            f"batches={batch_number}, "
            f"raw samples={total_raw:,}, "
            f"inside samples={total_inside:,}"
        )
        print(
            f"MAPE="
            f"{100*final_metrics['有效区域平均相对误差']:.4f}% "
            f"(target <= {100*args.mc_target_mape:.2f}%)"
        )
        print(
            f"Rel-L2="
            f"{100*final_metrics['相对L2误差']:.4f}% "
            f"(target <= {100*args.mc_target_rel_l2:.2f}%)"
        )
        print(
            f"Mass={final_metrics['诊断总质量']:.12f}, "
            f"nonzero bins={final_metrics['非零箱数']:,}, "
            f"effective bins={final_metrics['有效箱数']:,}"
        )
        print(
            f"Peak-center error="
            f"{100*final_metrics['中心峰值相对误差']:.4f}% "
            f"(仅诊断，不作为 20% 验收条件)"
        )
        print("=" * 80 + "\n")

        accepted = (
            final_metrics["有效区域平均相对误差"]
            <= args.mc_target_mape
            and final_metrics["相对L2误差"]
            <= args.mc_target_rel_l2
        )
        if accepted:
            print(
                "[MC 验收通过] 直方图 MAPE 与 Rel-L2 "
                "均达到设定阈值。"
            )
            break

    if final_metrics is None:
        final_metrics = evaluate_histogram_counts(
            counts=counts,
            inside_count=total_inside,
            grid=grid,
            exact_center_density=exact_grid,
            threshold=args.eval_threshold,
            chunk_size=args.hist_eval_chunk_size,
        )

    if not accepted and not args.allow_unqualified_mc:
        raise RuntimeError(
            "MC 直方图未通过硬验收，因此停止训练。"
            f" 当前 MAPE="
            f"{100*final_metrics['有效区域平均相对误差']:.4f}%,"
            f" Rel-L2="
            f"{100*final_metrics['相对L2误差']:.4f}%。"
            "请提高 --mc-max-batches、"
            "--mc-path-batch-size，或检查 dt/burn-in。"
        )

    counts_path = out_dir / "rough_potential_8d_histogram_counts.npy"
    np.save(counts_path, counts)
    print(f"[直方图计数保存] {counts_path}")

    y_data, v_data, selected_indices = (
        make_histogram_training_subset(
            counts=counts,
            inside_count=total_inside,
            grid=grid,
            n_ref=args.n_ref,
            density_fraction=args.density_fraction,
            seed=args.seed + 7001,
        )
    )

    return {
        "grid": grid,
        "counts": counts,
        "exact_grid": exact_grid,
        "metrics": final_metrics,
        "convergence": convergence,
        "accepted": bool(accepted),
        "batches_run": int(batches_run),
        "raw_samples": int(total_raw),
        "inside_samples": int(total_inside),
        "y_data": y_data,
        "v_data": v_data,
        "selected_indices": selected_indices,
        "counts_path": str(counts_path),
    }





class GeneralFPNN(nn.Module):
    """Positive density network with direct-density or log-density output."""
    def __init__(
        self,
        input_scales=(0.7,) * 8,
        width=256,
        depth=4,
        first_width=512,
        density_parameterization="exp",
        log_density_min=-30.0,
        log_density_max=12.0,
        initial_log_density=-2.7,
    ):
        super().__init__()
        dim = len(input_scales)
        if dim != 8:
            raise ValueError("8D 网络必须有 8 个输入尺度")
        if depth < 1 or width <= 0 or first_width <= 0:
            raise ValueError("网络宽度和深度必须为正")
        if density_parameterization not in {"softplus", "exp"}:
            raise ValueError("density_parameterization 必须是 softplus 或 exp")
        if log_density_min >= log_density_max:
            raise ValueError("log_density_min 必须小于 log_density_max")
        self.density_parameterization = density_parameterization
        self.log_density_min = float(log_density_min)
        self.log_density_max = float(log_density_max)
        self.initial_log_density = float(initial_log_density)
        self.register_buffer(
            "input_scale",
            torch.tensor(input_scales, dtype=torch.float64).view(1, dim),
        )
        # Non-trainable scale kept for state-dict compatibility with v6.3.
        # In this ablation it remains exactly 1.0 for the entire run.
        self.register_buffer(
            "output_scale", torch.tensor(1.0, dtype=torch.float64)
        )
        self.register_buffer(
            "mass_calibration_count", torch.tensor(0, dtype=torch.int64)
        )
        layers = [nn.Linear(dim, first_width), nn.Tanh()]
        in_dim = first_width
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, width), nn.Tanh()]
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.shape_net = nn.Sequential(*layers)
        for module in self.shape_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        if self.density_parameterization == "exp":
            nn.init.normal_(self.shape_net[-1].weight, mean=0.0, std=1e-2)
            nn.init.constant_(self.shape_net[-1].bias, self.initial_log_density)
        else:
            nn.init.normal_(self.shape_net[-1].weight, mean=0.0, std=2e-2)
            nn.init.constant_(self.shape_net[-1].bias, -1.0)

    def base_density(self, x):
        raw = self.shape_net(x / self.input_scale)
        if self.density_parameterization == "softplus":
            return nn.functional.softplus(raw) + 1e-12
        log_u = torch.clamp(
            raw, min=self.log_density_min, max=self.log_density_max
        )
        return torch.exp(log_u) + 1e-12

    def forward(self, x):
        return self.output_scale * self.base_density(x)


class MassNormalizedDensity(nn.Module):
    """Legacy hard-mass wrapper kept only for loading old cached checkpoints."""
    def __init__(self, base_net, norm_points, domain_volume, target_mass=1.0):
        super().__init__()
        self.base_net = base_net
        self.register_buffer("norm_points", norm_points.detach().clone())
        self.register_buffer(
            "target_mass_tensor",
            torch.tensor(float(target_mass), dtype=torch.float64),
        )
        self.domain_volume = float(domain_volume)

    def base_density(self, x):
        return self.base_net.base_density(x)

    def normalizer(self):
        return (
            self.domain_volume
            * self.base_net.base_density(self.norm_points).mean()
            + 1e-30
        )

    def forward(self, x):
        return (
            self.target_mass_tensor
            * self.base_net.base_density(x)
            / self.normalizer()
        )


def histogram_observable(
    net,
    x,
    cell_widths=None,
    fd_fraction=0.5,
):
    """
    Return point values, or a second-order approximation of the model's
    histogram-cell average:
        avg_cell u = u(c) + sum_j h_j^2/24 * d_jj u(c) + O(h^4).
    The diagonal Hessian is evaluated by centered finite differences, so the
    data loss still needs only ordinary parameter backpropagation and never
    uses the analytic density.
    """
    if cell_widths is None:
        return net(x)
    if not (0.0 < fd_fraction <= 0.5):
        raise ValueError("fd_fraction 必须位于 (0, 0.5]")
    widths = torch.as_tensor(
        cell_widths, dtype=x.dtype, device=x.device
    ).reshape(-1)
    if widths.numel() != x.shape[1]:
        raise ValueError("cell_widths 维数与输入不一致")
    steps = fd_fraction * widths
    dim = x.shape[1]
    shifts = torch.zeros(
        (1 + 2 * dim, dim), dtype=x.dtype, device=x.device
    )
    for j in range(dim):
        shifts[1 + 2 * j, j] = steps[j]
        shifts[2 + 2 * j, j] = -steps[j]
    expanded = x[:, None, :] + shifts[None, :, :]
    values = net(expanded.reshape(-1, dim)).reshape(
        x.shape[0], 1 + 2 * dim, 1
    )
    center = values[:, 0, :]
    average = center.clone()
    for j in range(dim):
        plus = values[:, 1 + 2 * j, :]
        minus = values[:, 2 + 2 * j, :]
        second = (plus - 2.0 * center + minus) / (steps[j] ** 2)
        average = average + (widths[j] ** 2 / 24.0) * second
    return average


def normalized_data_loss(
    net, x, target, scale, cell_widths=None, fd_fraction=0.5
):
    pred = histogram_observable(
        net, x, cell_widths=cell_widths, fd_fraction=fd_fraction
    )
    mse = torch.mean((pred - target) ** 2)
    return mse / scale, mse


def make_density_weighted_subset(y, v, n_points, density_fraction, seed):
    """Deterministic density-weighted + uniform subset for Stage C data."""
    n = y.shape[0]
    if n_points <= 0 or n_points >= n:
        return y, v
    if not (0.0 <= density_fraction <= 1.0):
        raise ValueError("density_fraction 必须位于 [0,1]")
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    nd = int(round(n_points * density_fraction))
    nu = n_points - nd
    weights = v.detach().cpu().reshape(-1).clone() + 1e-14
    weights /= weights.sum()
    a = torch.multinomial(weights, nd, replacement=False, generator=g)
    b = torch.randperm(n, generator=g)[:nu]
    idx = torch.unique(torch.cat([a, b]))
    while idx.numel() < n_points:
        extra = torch.randint(0, n, (n_points - idx.numel(),), generator=g)
        idx = torch.unique(torch.cat([idx, extra]))
    idx = idx[:n_points].to(y.device)
    return y[idx], v[idx]


def weak_loss(
    net,
    points,
    process,
    domain_volume,
    importance_weights=None,
):
    """原 weak flux 比值；importance 权重只用于无偏体积积分。"""
    x = points.detach().clone().requires_grad_(True)
    u = net(x)
    gu = grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    flux = process.D * gu - process.drift_torch(x) * u
    pointwise = torch.sum(flux ** 2, dim=1, keepdim=True) / torch.clamp(
        u, min=1e-12
    )
    if importance_weights is None:
        w = torch.ones_like(u)
    else:
        w = importance_weights.detach().to(dtype=u.dtype, device=u.device)
    numerator = torch.mean(w * pointwise)
    denominator = torch.mean(w * u)
    return numerator / (denominator + 1e-14)


def strong_loss(
    net,
    points,
    process,
    domain_volume,
    importance_weights=None,
):
    x = points.detach().clone().requires_grad_(True)
    u = net(x)
    bu = process.drift_torch(x) * u
    div_bu = torch.zeros_like(u)
    for j in range(x.shape[1]):
        div_bu = div_bu + grad(
            bu[:, j:j+1].sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0][:, j:j+1]
    gu = grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    lap_u = torch.zeros_like(u)
    for j in range(x.shape[1]):
        lap_u = lap_u + grad(
            gu[:, j:j+1].sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0][:, j:j+1]
    residual = -div_bu + process.D * lap_u
    if importance_weights is None:
        w = torch.ones_like(residual)
    else:
        w = importance_weights.detach().to(
            dtype=residual.dtype, device=residual.device
        )
    return domain_volume * torch.mean(w * residual ** 2)


def sobol_boundary(n_per_face, bounds, seed=0):
    if n_per_face <= 0:
        raise ValueError("n_per_face 必须为正")
    b = np.asarray(bounds, dtype=np.float64)
    dim = b.shape[0]
    pts_all, normals_all, weights_all = [], [], []
    for axis in range(dim):
        face_area = float(np.prod(np.delete(b[:, 1] - b[:, 0], axis)))
        for side, value in [(-1.0, b[axis, 0]), (1.0, b[axis, 1])]:
            eng = torch.quasirandom.SobolEngine(
                dim, scramble=True, seed=seed + 1009 * axis + int(side > 0)
            )
            u = eng.draw(n_per_face).to(torch.float64)
            lo = torch.tensor(b[:, 0], dtype=torch.float64).view(1, dim)
            hi = torch.tensor(b[:, 1], dtype=torch.float64).view(1, dim)
            pts = lo + (hi - lo) * u
            pts[:, axis] = float(value)
            normals = torch.zeros((n_per_face, dim), dtype=torch.float64)
            normals[:, axis] = side
            weights = torch.full(
                (n_per_face, 1), face_area, dtype=torch.float64
            )
            pts_all.append(pts)
            normals_all.append(normals)
            weights_all.append(weights)
    return (
        torch.cat(pts_all, dim=0),
        torch.cat(normals_all, dim=0),
        torch.cat(weights_all, dim=0),
    )


def noflux_loss(net, points, normals, weights, process):
    x = points.detach().clone().requires_grad_(True)
    u = net(x)
    gu = grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    flux = process.D * gu - process.drift_torch(x) * u
    normal_flux = torch.sum(flux * normals, dim=1, keepdim=True)
    return torch.mean(weights * normal_flux ** 2) / (
        torch.mean(weights) + 1e-30
    )


def normalization_loss(
    net,
    points,
    domain_volume,
    importance_weights=None,
    target_mass=1.0,
):
    """Importance-corrected L_normal=(int_Omega u_theta dx-target)^2."""
    u = net(points.detach())
    if importance_weights is None:
        w = torch.ones_like(u)
    else:
        w = importance_weights.detach().to(dtype=u.dtype, device=u.device)
    mass = domain_volume * torch.mean(w * u)
    target = torch.as_tensor(
        target_mass, dtype=mass.dtype, device=mass.device
    )
    return (mass - target) ** 2, mass


def _cyclic_batch(pool, start, batch_size):
    n = pool.shape[0]
    end = start + batch_size
    if end <= n:
        return pool[start:end]
    return torch.cat([pool[start:], pool[:end - n]], dim=0)


def _relative_improvement(value, best, min_delta):
    if not np.isfinite(best):
        return True
    return value < best - min_delta * max(abs(best), 1e-12)


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

    # The following four hybrid methods have exactly the same Stage A/B
    # configuration within each strong/weak pair. Adam-only stops after
    # Stage B; only the +L-BFGS member continues to Stage C.
    "Strong + MC Adam-only": dict(
        use_data=True,
        physics="strong",
        use_lbfgs=False,
        physics_lr_multiplier=4.0,
        physics_grad_clip=5.0,
        data_steps_per_cycle=2,
        checkpoint_data_weight=0.20,
        checkpoint_physics_weight=1.0,
    ),
    "Weak + MC Adam-only": dict(
        use_data=True,
        physics="weak",
        use_lbfgs=False,
        physics_lr_multiplier=4.0,
        physics_grad_clip=5.0,
        data_steps_per_cycle=2,
        checkpoint_data_weight=0.20,
        checkpoint_physics_weight=1.0,
    ),
    "Strong + MC + L-BFGS": dict(
        use_data=True,
        physics="strong",
        use_lbfgs=True,
        physics_lr_multiplier=4.0,
        physics_grad_clip=5.0,
        data_steps_per_cycle=2,
        checkpoint_data_weight=0.20,
        checkpoint_physics_weight=1.0,
    ),
    "Weak + MC + L-BFGS": dict(
        use_data=True,
        physics="weak",
        use_lbfgs=True,
        physics_lr_multiplier=4.0,
        physics_grad_clip=5.0,
        data_steps_per_cycle=2,
        checkpoint_data_weight=0.20,
        checkpoint_physics_weight=1.0,
    ),
}
for cfg in METHODS.values():
    cfg.setdefault("boundary", None)
    cfg.setdefault("boundary_weight", 0.0)
    cfg.setdefault("hard_mass", False)
    cfg.setdefault("physics_lr_multiplier", 1.0)
    cfg.setdefault("physics_grad_clip", 1.0)
    cfg.setdefault("data_steps_per_cycle", None)
    cfg.setdefault("checkpoint_data_weight", None)
    cfg.setdefault("checkpoint_physics_weight", None)



def estimate_model_mass(net, points, domain_volume, device, chunk_size=32768):
    total = 0.0
    count = 0
    net.eval()
    with torch.no_grad():
        for chunk in torch.split(points, chunk_size):
            values = net(chunk.to(device))
            total += float(values.sum().cpu())
            count += values.numel()
    return domain_volume * total / max(count, 1)


def calibrate_once_to_mc_mass(
    net, bounds, target_mass, n_points, seed, device, chunk_size=32768
):
    """Exactly one post-training global multiplication; no exact density."""
    if not isinstance(net, GeneralFPNN):
        raise TypeError("一次质量校准只允许用于未包装的含数据网络")
    if int(net.mass_calibration_count.item()) != 0:
        raise RuntimeError("全局质量校准已执行过，禁止第二次执行")
    qpts = sobol_box(n_points, bounds, seed)
    current_mass = estimate_model_mass(
        net, qpts, volume(bounds), device, chunk_size
    )
    if not np.isfinite(current_mass) or current_mass <= 0.0:
        raise FloatingPointError(f"校准前质量非法: {current_mass}")
    factor = float(target_mass) / current_mass
    with torch.no_grad():
        net.output_scale.mul_(factor)
        net.mass_calibration_count.add_(1)
    check_mass = estimate_model_mass(
        net, qpts, volume(bounds), device, chunk_size
    )
    print("\n" + "=" * 80)
    print("Weak + MC + L-BFGS：唯一一次 MC 总质量全局校准")
    print("-" * 80)
    print(f"MC target mass={target_mass:.12f}")
    print(f"model mass before={current_mass:.12f}")
    print(f"global factor={factor:.12f}")
    print(f"model mass after={check_mass:.12f}")
    print("解析密度未参与；不改变网络形状；后续禁止再次校准")
    print("=" * 80)
    return {
        "target_mass": float(target_mass),
        "mass_before": float(current_mass),
        "factor": float(factor),
        "mass_after": float(check_mass),
        "n_points": int(n_points),
    }


def train_model(
    method_name, net, process, device, bounds, args,
    y_data=None, v_data=None, target_mass=1.0,
):
    cfg = METHODS[method_name]
    net = net.to(device)
    vol = volume(bounds)

    if cfg["use_data"]:
        if y_data is None or v_data is None:
            raise ValueError(f"{method_name} 需要 MC 数据")
        y_data = y_data.to(device)
        v_data = v_data.to(device)
    is_pure_pinn = (not cfg["use_data"]) and (cfg["physics"] is not None)

    print("\n" + "=" * 96)
    print(f"开始训练：{method_name}")
    print("=" * 96)
    print("[数据] Stage A/B 为 MC 箱中心普通点值 MSE；Stage C 可选箱平均 MSE")
    print("[物理] 原 weak/strong 体积积分；局部高斯+uniform 仅做严格重要性采样")
    print("[Stage B] data Adam 与 physics Adam 分开更新；起点先登记为候选 checkpoint")
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
    if method_name == "Weak + MC + L-BFGS":
        print("[质量] 禁用最终 MC 总质量缩放；直接保存和评价原始网络输出")

    stage_c_cell_widths = None
    if cfg["use_data"] and args.histogram_data_mode == "cell-average":
        b = np.asarray(bounds, dtype=np.float64)
        lengths = b[:, 1] - b[:, 0]
        n_bins = np.rint(lengths / args.hist_bin_width).astype(np.int64)
        stage_c_cell_widths = torch.tensor(
            lengths / n_bins, dtype=torch.float64, device=device
        )
        print(
            "[数据算子] Stage A/B 保持原箱中心点值 MSE；"
            "Stage C 用二阶中心差分比较模型箱平均与 MC 箱平均"
        )
    elif cfg["use_data"]:
        print("[数据算子] 全阶段兼容模式：MC 箱平均直接当箱中心点值")

    data_scale = (
        torch.mean(v_data ** 2).detach() + 1e-14
        if cfg["use_data"]
        else torch.tensor(1.0, dtype=torch.float64, device=device)
    )
    loader = None
    if cfg["use_data"]:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(args.seed + 12345)
        loader = DataLoader(
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
        "mass_calibration": None,
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

    # Stage A: same ordinary pointwise MC pretraining, shortened for sparse 8D data.
    if cfg["use_data"] and args.pretrain_epochs > 0:
        print("\n[Stage A] MC 箱中心点值数据预训练")
        opt = optim.AdamW(
            net.parameters(), lr=args.pretrain_lr, weight_decay=1e-9
        )
        sch = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.pretrain_epochs, eta_min=args.pretrain_eta_min
        )
        best_state = copy.deepcopy(net.state_dict())
        best_mse = float("inf")
        best_epoch = 0
        for epoch in range(args.pretrain_epochs):
            net.train()
            total, count = 0.0, 0
            for xb, vb in loader:
                opt.zero_grad(set_to_none=True)
                rel, mse = normalized_data_loss(net, xb, vb, data_scale)
                rel.backward()
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), args.data_grad_clip_pretrain
                )
                opt.step()
                total += float(mse.detach()) * xb.shape[0]
                count += xb.shape[0]
            sch.step()
            epoch_mse = total / max(count, 1)
            history["stage_a_epoch"].append(epoch + 1)
            history["stage_a_data_mse"].append(float(epoch_mse))
            if epoch_mse < best_mse:
                best_mse = epoch_mse
                best_epoch = epoch + 1
                best_state = copy.deepcopy(net.state_dict())
            if ((epoch + 1) % args.pretrain_log_every == 0
                    or epoch == args.pretrain_epochs - 1):
                print(
                    f"预训练轮次 {epoch+1:5d} | "
                    f"数据 MSE={epoch_mse:.8e} | best_epoch={best_epoch}"
                )
        net.load_state_dict(best_state)
        history["stage_a_best_epoch"] = best_epoch

    # Importance-corrected physics sets. The continuous objective is unchanged.
    pool, pool_w = make_importance_physics_set(
        args.physics_pool_size,
        bounds,
        process,
        gaussian_fraction=args.physics_gaussian_fraction,
        std_scale=args.physics_proposal_std_scale,
        seed=args.seed + 888,
    )
    check, check_w = make_importance_physics_set(
        args.fixed_check_size,
        bounds,
        process,
        gaussian_fraction=args.physics_gaussian_fraction,
        std_scale=args.physics_proposal_std_scale,
        seed=args.seed + 2027,
    )
    pool, pool_w = pool.to(device), pool_w.to(device)
    check, check_w = check.to(device), check_w.to(device)

    def phys(points, weights):
        if cfg["physics"] == "weak":
            return weak_loss(net, points, process, vol, weights)
        if cfg["physics"] == "strong":
            return strong_loss(net, points, process, vol, weights)
        return torch.tensor(0.0, dtype=torch.float64, device=device)

    if cfg["boundary"] == "noflux":
        bp, bn, bw = sobol_boundary(
            args.boundary_n_per_face, bounds, args.seed + 3030
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
            d0, dm0 = normalized_data_loss(net, y_data, v_data, data_scale)
        dscale = torch.clamp(d0.detach(), min=1e-10)
    else:
        d0 = dm0 = torch.tensor(0.0, dtype=torch.float64, device=device)
        dscale = torch.tensor(1.0, dtype=torch.float64, device=device)
    if cfg["physics"] is not None:
        with torch.enable_grad():
            p0 = phys(check, check_w)
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
                net,
                check,
                vol,
                importance_weights=check_w,
                target_mass=1.0,
            )
    else:
        n0 = torch.tensor(0.0, dtype=torch.float64, device=device)
        with torch.no_grad():
            m0 = vol * torch.mean(check_w * net(check))

    method_checkpoint_data_weight = (
        args.checkpoint_data_weight
        if cfg["checkpoint_data_weight"] is None
        else float(cfg["checkpoint_data_weight"])
    )
    method_checkpoint_physics_weight = (
        args.checkpoint_physics_weight
        if cfg["checkpoint_physics_weight"] is None
        else float(cfg["checkpoint_physics_weight"])
    )
    method_data_steps = (
        args.data_steps_per_cycle
        if cfg["data_steps_per_cycle"] is None
        else int(cfg["data_steps_per_cycle"])
    )
    if method_data_steps <= 0:
        raise ValueError("method_data_steps 必须为正")

    if is_pure_pinn:
        initial_score = float(
            args.pure_physics_weight * p0
            + args.pure_boundary_weight * b0
            + args.pure_normal_weight * n0
        )
    else:
        initial_score = 0.0
        if cfg["use_data"]:
            initial_score += method_checkpoint_data_weight
        if cfg["physics"]:
            initial_score += method_checkpoint_physics_weight
        if bp is not None:
            initial_score += cfg["boundary_weight"]

    print("\n[Stage B 起点]")
    print(f"数据 MSE={dm0.item():.8e}")
    print(f"物理损失={p0.item():.8e}")
    print(f"边界损失={b0.item():.8e}")
    print(f"归一化损失={n0.item():.8e}")
    print(f"importance 诊断质量={m0.item():.8f}")
    print(f"起点 checkpoint score={initial_score:.8e}（已登记）")
    print(
        "[Stage B 配置] "
        f"physics_lr={args.physics_lr * float(cfg['physics_lr_multiplier']):.3e}, "
        f"data_steps/cycle={method_data_steps}, "
        f"checkpoint weights="
        f"({method_checkpoint_data_weight:.3f}, "
        f"{method_checkpoint_physics_weight:.3f})"
    )

    opt_d = optim.Adam(net.parameters(), lr=args.data_lr) if cfg["use_data"] else None
    effective_physics_lr = args.physics_lr * float(cfg["physics_lr_multiplier"])
    opt_p = (
        optim.Adam(
            [p for p in net.parameters() if p.requires_grad],
            lr=effective_physics_lr,
        ) if cfg["physics"] else None
    )
    n_data_updates = max(
        1,
        int(np.ceil(args.joint_cycles / max(args.hybrid_data_update_every, 1)))
        * max(method_data_steps, 1),
    )
    sch_d = (
        optim.lr_scheduler.CosineAnnealingLR(
            opt_d, T_max=n_data_updates, eta_min=args.data_eta_min
        ) if opt_d else None
    )
    sch_p = (
        optim.lr_scheduler.CosineAnnealingLR(
            opt_p,
            T_max=max(1, args.joint_cycles * args.physics_steps_per_cycle),
            eta_min=min(args.physics_eta_min, 0.1 * effective_physics_lr),
        ) if opt_p else None
    )
    iterator = iter(loader) if loader else None

    # Critical bug fix: Stage B start is a real candidate, not +infinity.
    best_score = float(initial_score)
    best_state = copy.deepcopy(net.state_dict())
    best_cycle = 0
    n_pool = pool.shape[0]
    no_improve_checks = 0

    print("\n[Stage B] 分离 Adam：固定小规模 physics pool 与 MC 锚定交替更新")
    for cycle in range(args.joint_cycles):
        net.train()
        if cfg["physics"]:
            for substep in range(args.physics_steps_per_cycle):
                update_index = cycle * args.physics_steps_per_cycle + substep
                start = (update_index * args.physics_batch_size) % n_pool
                pts = _cyclic_batch(pool, start, args.physics_batch_size)
                iw = _cyclic_batch(pool_w, start, args.physics_batch_size)
                opt_p.zero_grad(set_to_none=True)
                pl = phys(pts, iw)
                bl = bnd()
                if is_pure_pinn:
                    nl, _ = normalization_loss(
                        net,
                        pts,
                        vol,
                        importance_weights=iw,
                        target_mass=1.0,
                    )
                    total_p = (
                        args.pure_physics_weight * pl
                        + args.pure_boundary_weight * bl
                        + args.pure_normal_weight * nl
                    )
                else:
                    total_p = pl / pscale
                    if bp is not None:
                        total_p = total_p + cfg["boundary_weight"] * bl / bscale
                total_p.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in net.parameters() if p.requires_grad],
                    float(cfg["physics_grad_clip"]),
                )
                opt_p.step()
                sch_p.step()

        do_data_update = cfg["use_data"] and (
            cfg["physics"] is None
            or cycle % max(args.hybrid_data_update_every, 1) == 0
        )
        if do_data_update:
            for _ in range(method_data_steps):
                try:
                    xb, vb = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    xb, vb = next(iterator)
                opt_d.zero_grad(set_to_none=True)
                dl, _ = normalized_data_loss(net, xb, vb, data_scale)
                dl.backward()
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), args.data_grad_clip
                )
                opt_d.step()
                sch_d.step()

        should_check = (
            (cycle + 1) % args.checkpoint_every == 0
            or cycle == args.joint_cycles - 1
        )
        if not should_check:
            continue
        net.eval()
        if cfg["use_data"]:
            with torch.no_grad():
                cd, cdm = normalized_data_loss(net, y_data, v_data, data_scale)
        else:
            cd = cdm = torch.tensor(0.0, dtype=torch.float64, device=device)
        if cfg["physics"]:
            with torch.enable_grad():
                cp = phys(check, check_w)
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
                    net,
                    check,
                    vol,
                    importance_weights=check_w,
                    target_mass=1.0,
                )
        else:
            cn = torch.tensor(0.0, dtype=torch.float64, device=device)
            with torch.no_grad():
                cm = vol * torch.mean(check_w * net(check))

        data_ratio = float((cd / dscale).detach()) if cfg["use_data"] else 0.0
        physics_ratio = float((cp / pscale).detach()) if cfg["physics"] else 0.0
        if is_pure_pinn:
            score = float(
                args.pure_physics_weight * cp
                + args.pure_boundary_weight * cb
                + args.pure_normal_weight * cn
            )
        else:
            score = 0.0
            if cfg["use_data"]:
                score += method_checkpoint_data_weight * data_ratio
            if cfg["physics"]:
                score += method_checkpoint_physics_weight * physics_ratio
            if bp is not None:
                score += float(cfg["boundary_weight"] * (cb / bscale).detach())

        history["stage_b_cycle"].append(cycle + 1)
        history["stage_b_monitor_score"].append(float(score))
        history["stage_b_data_mse"].append(float(cdm.detach()))
        history["stage_b_physics_loss"].append(float(cp.detach()))

        if _relative_improvement(score, best_score, args.joint_min_delta):
            best_score = score
            best_cycle = cycle + 1
            best_state = copy.deepcopy(net.state_dict())
            no_improve_checks = 0
        else:
            no_improve_checks += 1

        if ((cycle + 1) % args.log_every == 0
                or cycle == args.joint_cycles - 1):
            print(
                f"循环 {cycle+1:5d} | score={score:.8e} | "
                f"data ratio={data_ratio:.6e} | physics ratio={physics_ratio:.6e} | "
                f"data MSE={cdm.item():.8e} | physics={cp.item():.8e} | "
                f"boundary={cb.item():.8e} | normal={cn.item():.8e} | "
                f"mass={cm.item():.8f} | best_cycle={best_cycle}"
            )

        if (
            args.joint_patience_checks > 0
            and cycle + 1 >= args.joint_min_cycles
            and no_improve_checks >= args.joint_patience_checks
        ):
            print(
                f"[Stage B 早停] cycle={cycle+1}, best_cycle={best_cycle}, "
                f"best_score={best_score:.8e}"
            )
            break

    net.load_state_dict(best_state)
    history["stage_b_best_cycle"] = best_cycle
    history["stage_b_best_score"] = best_score
    with torch.no_grad():
        stage_b_checksum = 0.0
        for parameter in net.parameters():
            stage_b_checksum += float(parameter.detach().double().sum().cpu())
    history["stage_b_parameter_checksum"] = stage_b_checksum
    print(
        f"[Stage B checkpoint] best_cycle={best_cycle}, "
        f"score={best_score:.8e}, checksum={stage_b_checksum:.12e}"
    )

    # Stage C: importance-corrected fixed train/check sets and monitored rollback.
    if cfg["use_lbfgs"] and cfg["physics"] and cfg["use_data"]:
        print(
            "\n[Stage C] 从上面已经回滚选定的 Stage-B checkpoint 出发；"
            "复用相同坐标做 L-BFGS"
        )
        if args.lbfgs_point_size > pool.shape[0]:
            raise ValueError(
                "lbfgs_point_size 不能超过 physics_pool_size；"
                "v6.3 为控制唯一训练点预算，Stage C 不再新建物理点。"
            )
        if args.lbfgs_check_size > check.shape[0]:
            raise ValueError(
                "lbfgs_check_size 不能超过 fixed_check_size；"
                "v6.3 为控制唯一训练点预算，Stage C 不再新建监控点。"
            )
        lp = pool[:args.lbfgs_point_size]
        lp_w = pool_w[:args.lbfgs_point_size]
        lc = check[:args.lbfgs_check_size]
        lc_w = check_w[:args.lbfgs_check_size]
        print(
            f"[Stage C physics] reuse train={lp.shape[0]} / "
            f"pool={pool.shape[0]}, monitor={lc.shape[0]} / "
            f"check={check.shape[0]}"
        )
        y_lb, v_lb = make_density_weighted_subset(
            y_data, v_data, args.lbfgs_data_size,
            args.lbfgs_data_density_fraction, args.seed + 31337,
        )
        y_lc, v_lc = make_density_weighted_subset(
            y_data, v_data, args.lbfgs_data_check_size,
            args.lbfgs_data_density_fraction, args.seed + 41339,
        )
        print(
            f"[Stage C data] train bins={y_lb.shape[0]}, "
            f"check bins={y_lc.shape[0]}, mode={args.histogram_data_mode}"
        )

        with torch.enable_grad():
            pstart = phys(lp, lp_w).detach()
            pmon0 = phys(lc, lc_w).detach()
        with torch.no_grad():
            dstart, _ = normalized_data_loss(
                net, y_lb, v_lb, data_scale,
                cell_widths=stage_c_cell_widths,
                fd_fraction=args.cell_average_fd_fraction,
            )
            dmon0, _ = normalized_data_loss(
                net, y_lc, v_lc, data_scale,
                cell_widths=stage_c_cell_widths,
                fd_fraction=args.cell_average_fd_fraction,
            )
        ps = torch.clamp(pstart, min=1e-10)
        ds = torch.clamp(dstart.detach(), min=1e-10)
        pmon_scale = torch.clamp(pmon0, min=1e-10)
        dmon_scale = torch.clamp(dmon0.detach(), min=1e-10)

        best_lb_state = copy.deepcopy(net.state_dict())
        best_lb_score = args.lbfgs_data_weight + args.lbfgs_physics_weight
        best_lb_iter = 0
        total_iter = 0
        no_improve = 0
        calls = {"n": 0}
        chunk = min(args.lbfgs_check_every, args.lbfgs_max_iter)
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
                net, y_lb, v_lb, data_scale,
                cell_widths=stage_c_cell_widths,
                fd_fraction=args.cell_average_fd_fraction,
            )
            pl = phys(lp, lp_w)
            loss = (
                args.lbfgs_data_weight * dl / ds
                + args.lbfgs_physics_weight * pl / ps
            )
            loss.backward()
            calls["n"] += 1
            if calls["n"] % 50 == 0:
                print(
                    f"闭包调用 {calls['n']:4d} | objective={loss.item():.8e} | "
                    f"data={dl.item():.8e} | physics={pl.item():.8e}"
                )
            return loss

        while total_iter < args.lbfgs_max_iter:
            current = min(chunk, args.lbfgs_max_iter - total_iter)
            lb.param_groups[0]["max_iter"] = current
            lb.defaults["max_iter"] = current
            lb.step(closure)
            total_iter += current

            net.eval()
            with torch.no_grad():
                cd, cdm = normalized_data_loss(
                    net, y_lc, v_lc, data_scale,
                    cell_widths=stage_c_cell_widths,
                    fd_fraction=args.cell_average_fd_fraction,
                )
                cm = vol * torch.mean(lc_w * net(lc))
            with torch.enable_grad():
                cp = phys(lc, lc_w)
            data_ratio = float((cd / dmon_scale).detach())
            physics_ratio = float((cp / pmon_scale).detach())
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
            history["stage_c_data_mse"].append(float(cdm.detach()))
            history["stage_c_physics_loss"].append(float(cp.detach()))
            if data_guard_ok and _relative_improvement(
                monitor, best_lb_score, args.lbfgs_min_delta
            ):
                best_lb_score = monitor
                best_lb_iter = total_iter
                best_lb_state = copy.deepcopy(net.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            print(
                f"L-BFGS iter {total_iter:4d} | monitor={monitor:.8e} | "
                f"data ratio={data_ratio:.6e} | physics ratio={physics_ratio:.6e} | "
                f"data MSE={cdm.item():.8e} | check physics={cp.item():.8e} | "
                f"mass={cm.item():.8f} | data_guard={data_guard_ok} | "
                f"best_iter={best_lb_iter}"
            )
            if (
                total_iter >= args.lbfgs_min_iter
                and no_improve >= args.lbfgs_patience_checks
            ):
                print(
                    f"[Stage C 早停] iter={total_iter}, best_iter={best_lb_iter}, "
                    f"best_monitor={best_lb_score:.8e}"
                )
                break

        net.load_state_dict(best_lb_state)
        history["stage_c_best_iter"] = best_lb_iter
        history["stage_c_best_monitor"] = best_lb_score

    if method_name == "Weak + MC + L-BFGS":
        if not isinstance(net, GeneralFPNN):
            raise TypeError("未校准消融要求使用未包装的含数据网络")
        if int(net.mass_calibration_count.item()) != 0:
            raise RuntimeError("未校准消融中 mass_calibration_count 必须保持为 0")
        if not np.isclose(
            float(net.output_scale.item()), 1.0, atol=0.0, rtol=0.0
        ):
            raise RuntimeError("未校准消融中 output_scale 必须严格保持为 1.0")
        history["mass_calibration"] = {
            "enabled": False,
            "factor": 1.0,
            "mass_calibration_count": 0,
        }
        print("\n" + "=" * 80)
        print("Weak + MC + L-BFGS：最终 MC 总质量校准已禁用")
        print("-" * 80)
        print("不生成校准积分点；不估计校准前质量；不乘任何全局因子")
        print("output_scale=1.000000000000, mass_calibration_count=0")
        print("后续评价与保存均使用 Stage C 回滚后的原始网络输出")
        print("=" * 80)

    print(
        f"\n[{method_name}] 训练完成。"
        f"stage_b_best_score={best_score:.8e}, best_cycle={best_cycle}"
    )
    return net, history

def evaluate_points(pred, exact, pts, vol, threshold=0.01):
    pred = np.asarray(pred).reshape(-1)
    exact = np.asarray(exact).reshape(-1)
    pts = np.asarray(pts, dtype=np.float64)
    mask = exact > threshold * exact.max()
    mape = np.mean(np.abs(pred[mask] - exact[mask]) / (exact[mask] + 1e-14))
    rel_l2 = np.sqrt(
        np.mean((pred - exact) ** 2) / (np.mean(exact ** 2) + 1e-30)
    )
    mass = vol * float(np.mean(pred))
    pred_mean = vol * np.mean(pred[:, None] * pts, axis=0)
    exact_mean = vol * np.mean(exact[:, None] * pts, axis=0)
    return {
        "有效区域平均相对误差": float(mape),
        "相对L2误差": float(rel_l2),
        "诊断总质量": float(mass),
        "一阶矩L2误差": float(np.linalg.norm(pred_mean - exact_mean)),
    }


def oblique_line_data(net, process, device, bound, n=401):
    direction = np.asarray(
        [1.0, -0.83, 0.67, -0.51, 0.39, -0.28, 0.19, -0.11],
        dtype=np.float64,
    )
    direction /= np.max(np.abs(direction))
    t = np.linspace(-0.95 * bound, 0.95 * bound, n)
    pts = t[:, None] * direction[None, :]
    exact = process.exact_density_numpy(pts)
    with torch.no_grad():
        pred = net(torch.tensor(pts, dtype=torch.float64, device=device))
        pred = pred.cpu().numpy().reshape(-1)
    mask = exact > 0.01 * exact.max()
    err = float(np.mean(np.abs(pred[mask] - exact[mask]) / (exact[mask] + 1e-14)))
    return t, pred, exact, err


def evaluate_model(net, process, device, bounds, eval_points):
    pts = eval_points.numpy()
    with torch.no_grad():
        chunks = [
            net(c.to(device)).cpu().numpy()
            for c in torch.split(eval_points, 20000)
        ]
    pred = np.vstack(chunks).reshape(-1)
    exact = process.exact_density_numpy(pts)
    out = evaluate_points(pred, exact, pts, volume(bounds))
    _, _, _, line_err = oblique_line_data(
        net, process, device, float(np.asarray(bounds)[0, 1])
    )
    out["斜截线误差"] = line_err
    origin = torch.zeros((1, 8), dtype=torch.float64, device=device)
    with torch.no_grad():
        pred_peak = float(net(origin).cpu().item())
    exact_peak = float(process.exact_density_numpy(np.zeros((1, 8)))[0])
    out["中心峰值相对误差"] = abs(pred_peak - exact_peak) / (exact_peak + 1e-14)
    return out


def print_table(results):
    print("\n" + "=" * 132)
    print("8D 单峰 rough-potential：严格七方法对照结果")
    print("=" * 132)
    print(
        f"{'方法':<34s} | {'MAPE(%)':>11s} | {'Rel-L2(%)':>11s} | "
        f"{'Line(%)':>11s} | {'Peak(%)':>11s} | {'Mass':>11s} | {'Moment-L2':>12s}"
    )
    print("-" * 132)
    for name, m in results.items():
        line = f"{100*m['斜截线误差']:11.4f}" if np.isfinite(m.get("斜截线误差", np.nan)) else f"{'N/A':>11s}"
        peak = f"{100*m['中心峰值相对误差']:11.4f}" if np.isfinite(m.get("中心峰值相对误差", np.nan)) else f"{'N/A':>11s}"
        print(
            f"{name:<34s} | {100*m['有效区域平均相对误差']:11.4f} | "
            f"{100*m['相对L2误差']:11.4f} | {line} | {peak} | "
            f"{m['诊断总质量']:11.6f} | {m['一阶矩L2误差']:12.6e}"
        )
    print("=" * 132 + "\n")



def _paper_figures_dir(out_dir):
    figures_dir = Path(out_dir) / "paper_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def sanitize_filename(text):
    safe = []
    for ch in str(text):
        safe.append(ch if ch.isalnum() else "_")
    compact = "".join(safe)
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact.strip("_") or "unnamed"


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


def plot_bars(results, figures_dir):
    model_results = {
        k: v for k, v in results.items() if not k.startswith("MC histogram")
    }
    names = list(model_results)
    labels = [_display_method(n) for n in names]
    mape = [100.0 * model_results[n]["有效区域平均相对误差"] for n in names]
    l2 = [100.0 * model_results[n]["相对L2误差"] for n in names]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(labels, mape)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("8D rough potential: MAPE comparison")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(Path(figures_dir) / "rough_potential_8d_ablation_mape.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(labels, l2)
    ax.set_ylabel("Relative L2 error (%)")
    ax.set_title("8D rough potential: relative L2 comparison")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(Path(figures_dir) / "rough_potential_8d_ablation_rel_l2.png", dpi=300)
    plt.close(fig)


def compute_solution_slice(
    net,
    process,
    device,
    bounds,
    resolution=201,
    slice_dims=(0, 1),
):
    if resolution < 2:
        raise ValueError("heatmap resolution must be at least 2")
    if len(slice_dims) != 2:
        raise ValueError("slice_dims must contain exactly two dimensions")
    dim_x, dim_y = int(slice_dims[0]), int(slice_dims[1])
    if not (0 <= dim_x < 8 and 0 <= dim_y < 8) or dim_x == dim_y:
        raise ValueError("slice dimensions must be two distinct indices in [0, 7]")

    b = np.asarray(bounds, dtype=np.float64)
    xs = np.linspace(float(b[dim_x, 0]), float(b[dim_x, 1]), resolution)
    ys = np.linspace(float(b[dim_y, 0]), float(b[dim_y, 1]), resolution)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((resolution * resolution, 8), dtype=np.float64)
    pts[:, dim_x] = xx.ravel()
    pts[:, dim_y] = yy.ravel()

    exact = process.exact_density_numpy(pts).reshape(resolution, resolution)
    pred = None
    if net is not None:
        tensor_pts = torch.tensor(pts, dtype=torch.float64)
        with torch.no_grad():
            pred = np.concatenate([
                net(chunk.to(device)).cpu().numpy().reshape(-1)
                for chunk in torch.split(tensor_pts, 20000)
            ]).reshape(resolution, resolution)
    return xs, ys, pred, exact


def plot_reference_slice(
    process, bounds, figures_dir, resolution=201, slice_dims=(0, 1),
):
    xs, ys, _, exact = compute_solution_slice(
        None, process, torch.device("cpu"), bounds,
        resolution=resolution, slice_dims=slice_dims,
    )
    dim_x, dim_y = int(slice_dims[0]), int(slice_dims[1])
    extent = [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(
        exact, origin="lower", extent=extent, aspect="auto",
        cmap="viridis", vmin=0.0, vmax=float(exact.max()),
    )
    ax.set_xlabel(f"$x_{dim_x+1}$")
    ax.set_ylabel(f"$x_{dim_y+1}$")
    ax.set_title(
        f"Reference density slice ($x_{dim_x+1}$, $x_{dim_y+1}$; others = 0)"
    )
    fig.colorbar(image, ax=ax, label="Stationary density")
    fig.tight_layout()
    path = Path(figures_dir) / (
        f"rough_potential_8d_reference_slice_x{dim_x+1}_x{dim_y+1}.png"
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_method_slice_and_errors(
    net,
    process,
    device,
    bounds,
    method_name,
    figures_dir,
    resolution=201,
    slice_dims=(0, 1),
):
    xs, ys, pred, exact = compute_solution_slice(
        net, process, device, bounds,
        resolution=resolution, slice_dims=slice_dims,
    )
    dim_x, dim_y = int(slice_dims[0]), int(slice_dims[1])
    extent = [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())]
    density_vmax = float(exact.max())
    abs_error = np.abs(pred - exact)
    valid = exact > 0.01 * exact.max()
    rel_error = np.full_like(exact, np.nan)
    rel_error[valid] = 100.0 * abs_error[valid] / (exact[valid] + 1e-14)
    tag = sanitize_filename(method_name)
    paths = []

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(
        pred, origin="lower", extent=extent, aspect="auto",
        cmap="viridis", vmin=0.0, vmax=density_vmax,
    )
    ax.set_xlabel(f"$x_{dim_x+1}$")
    ax.set_ylabel(f"$x_{dim_y+1}$")
    ax.set_title(f"{method_name}: predicted density slice (others = 0)")
    fig.colorbar(image, ax=ax, label="Stationary density")
    fig.tight_layout()
    path = Path(figures_dir) / f"rough_potential_8d_{tag}_solution_slice.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(
        abs_error, origin="lower", extent=extent, aspect="auto",
        cmap="YlOrRd", vmin=0.0, vmax=density_vmax,
    )
    ax.set_xlabel(f"$x_{dim_x+1}$")
    ax.set_ylabel(f"$x_{dim_y+1}$")
    ax.set_title(f"{method_name}: absolute-error slice")
    fig.colorbar(image, ax=ax, label=r"$|u_\theta-u_{\mathrm{ref}}|$")
    fig.tight_layout()
    path = Path(figures_dir) / f"rough_potential_8d_{tag}_absolute_error_slice.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(
        rel_error, origin="lower", extent=extent, aspect="auto",
        cmap="viridis", vmin=0.0, vmax=100.0,
    )
    ax.set_xlabel(f"$x_{dim_x+1}$")
    ax.set_ylabel(f"$x_{dim_y+1}$")
    ax.set_title(
        f"{method_name}: pointwise relative-error slice\n"
        "Effective region; display clipped at 100%"
    )
    fig.colorbar(image, ax=ax, label="Pointwise relative error (%)")
    fig.tight_layout()
    path = Path(figures_dir) / f"rough_potential_8d_{tag}_relative_error_slice.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)
    return paths


def plot_method_profile(
    net, process, device, bounds, method_name, figures_dir, n=401,
):
    x1 = np.linspace(bounds[0][0], bounds[0][1], n)
    pts = np.zeros((n, 8), dtype=np.float64)
    pts[:, 0] = x1
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
    ax.set_title("1D profile: $x_2=\\cdots=x_8=0$")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = Path(figures_dir) / (
        f"rough_potential_8d_{sanitize_filename(method_name)}_profile_x1.png"
    )
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
         "Epoch", "Data MSE", "rough_potential_8d_stage_a_adam_loss.png"),
        ("stage_b_cycle", "stage_b_monitor_score", "Stage B monitor score",
         "Cycle", "Monitor score", "rough_potential_8d_stage_b_monitor.png"),
        ("stage_c_iter", "stage_c_monitor_score", "Stage C L-BFGS monitor score",
         "L-BFGS iteration", "Monitor score", "rough_potential_8d_stage_c_lbfgs_monitor.png"),
    ]
    return [
        p for p in (
            _plot_history_family(histories, figures_dir, *spec)
            for spec in specs
        ) if p is not None
    ]


def plot_optimizer_ablation(results, figures_dir):
    model_results = {
        k: v for k, v in results.items() if not k.startswith("MC histogram")
    }
    pairs = [
        ("Strong + MC Adam-only", "Strong + MC + L-BFGS", "Strong"),
        ("Weak + MC Adam-only", "Weak + MC + L-BFGS", "Weak"),
    ]
    available = [(a, b, label) for a, b, label in pairs if a in model_results and b in model_results]
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
                100.0 * model_results[adam][metric_key],
                100.0 * model_results[lbfgs][metric_key],
            ])
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        ax.bar(labels, values)
        ax.set_ylabel(ylabel)
        ax.set_title("Training ablation: effect of Stage C L-BFGS")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path = Path(figures_dir) / f"rough_potential_8d_lbfgs_ablation_{suffix}.png"
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
    net, process, device, bounds, resolution=201, slice_dims=(0, 1),
):
    xs, ys, pred, exact = compute_solution_slice(
        net, process, device, bounds,
        resolution=resolution, slice_dims=slice_dims,
    )
    return {
        "x_g": np.asarray(xs, dtype=np.float64),
        "y_g": np.asarray(ys, dtype=np.float64),
        "exact": np.asarray(exact, dtype=np.float64),
        "pred": np.asarray(pred, dtype=np.float64),
        "slice_dims": tuple(int(v) for v in slice_dims),
        "fixed_other_coordinates": 0.0,
    }


def _common_evaluation_grid(evaluation_arrays):
    names = _ordered_evaluation_names(evaluation_arrays)
    if not names:
        raise ValueError("No evaluated methods available for comparison plots.")
    first = evaluation_arrays[names[0]]
    x_g = np.asarray(first["x_g"], dtype=np.float64)
    y_g = np.asarray(first["y_g"], dtype=np.float64)
    exact = np.asarray(first["exact"], dtype=np.float64)
    slice_dims = tuple(first.get("slice_dims", (0, 1)))
    for name in names[1:]:
        item = evaluation_arrays[name]
        if (
            not np.array_equal(x_g, np.asarray(item["x_g"]))
            or not np.array_equal(y_g, np.asarray(item["y_g"]))
            or np.asarray(item["exact"]).shape != exact.shape
            or tuple(item.get("slice_dims", slice_dims)) != slice_dims
        ):
            raise ValueError("All methods must use the same 8D 2D-slice grid.")
    return names, x_g, y_g, exact, slice_dims


def _comparison_layout(panels, x_g, y_g, slice_dims, values_key, cmap, vmin, vmax,
                       colorbar_label, title, path, title_formatter=None):
    extent = [
        float(x_g.min()), float(x_g.max()),
        float(y_g.min()), float(y_g.max()),
    ]
    dim_x, dim_y = (int(slice_dims[0]), int(slice_dims[1]))
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
        image = ax.imshow(
            panel[values_key],
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic",
            resample=True,
        )
        ax.set_title(
            title_formatter(panel) if title_formatter is not None else panel["label"],
            fontsize=9,
        )
        ax.set_xlabel(f"$x_{dim_x+1}$")
        ax.set_ylabel(f"$x_{dim_y+1}$")
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
    names, x_g, y_g, exact, slice_dims = _common_evaluation_grid(evaluation_arrays)
    dx = float(np.mean(np.diff(x_g)))
    dy = float(np.mean(np.diff(y_g)))
    panels = [{"label": "Reference", "values": exact, "global_l2": 0.0}]
    for name in names:
        pred = np.asarray(evaluation_arrays[name]["pred"], dtype=np.float64)
        panels.append({
            "label": _display_method(name),
            "values": pred,
            "global_l2": float(np.sqrt(np.sum((pred-exact)**2)*dx*dy)),
        })
    ref_peak = float(np.nanmax(exact))
    if not np.isfinite(ref_peak) or ref_peak <= 0.0:
        ref_peak = 1.0
    dim_x, dim_y = int(slice_dims[0]), int(slice_dims[1])
    path = Path(figures_dir) / "rough_potential_8d_all_methods_density_comparison.png"
    return _comparison_layout(
        panels, x_g, y_g, slice_dims, "values", "viridis", 0.0, 1.05*ref_peak,
        "Stationary density",
        f"8D rough potential: reference and predictions on "
        f"$(x_{dim_x+1},x_{dim_y+1})$ slice (others = 0)",
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
    path = Path(figures_dir) / "rough_potential_8d_mape_bar_chart.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_all_method_error_comparison(evaluation_arrays, figures_dir):
    names, x_g, y_g, exact, slice_dims = _common_evaluation_grid(evaluation_arrays)
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
    dim_x, dim_y = int(slice_dims[0]), int(slice_dims[1])
    path = Path(figures_dir) / "rough_potential_8d_global_absolute_error_comparison.png"
    return _comparison_layout(
        panels, x_g, y_g, slice_dims, "error", "YlOrRd", 0.0, vmax,
        r"Absolute error $|u_\theta-u_{\mathrm{ref}}|$",
        f"8D rough potential: absolute-error comparison on "
        f"$(x_{dim_x+1},x_{dim_y+1})$ slice",
        path,
        title_formatter=lambda p: (
            "Reference" if p["label"] == "Reference"
            else f'{p["label"]}\nSlice L2={p["global_l2"]:.3e}'
        ),
    )


def plot_all_method_relative_error_comparison(
    evaluation_arrays, figures_dir, tau_fraction=0.01,
):
    names, x_g, y_g, exact, slice_dims = _common_evaluation_grid(evaluation_arrays)
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
    dim_x, dim_y = int(slice_dims[0]), int(slice_dims[1])
    path = Path(figures_dir) / "rough_potential_8d_global_relative_error_comparison.png"
    return _comparison_layout(
        panels, x_g, y_g, slice_dims, "error", "viridis", 0.0, vmax,
        r"Relative error $100|u_\theta-u_{\rm ref}|/\max(u_{\rm ref},\tau)$ (%)",
        f"8D rough potential: relative-error comparison on "
        f"$(x_{dim_x+1},x_{dim_y+1})$ slice",
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


def _resolve_cache_path(args, out_dir):
    if getattr(args, "cache_file", None):
        path = Path(args.cache_file).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path
    return Path(out_dir) / "rough_potential_8d_v6_5_strict_stage_b_fair_all7_results.pt"


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
    bound = float(saved_args.get("bound", np.asarray(bounds, dtype=np.float64)[0,1]))
    width = int(saved_args.get("width", 128))
    depth = int(saved_args.get("depth", 4))
    first_width = int(saved_args.get("first_width", 256))
    density_parameterization = str(saved_args.get("density_parameterization", "exp"))
    log_density_min = float(saved_args.get("log_density_min", -30.0))
    log_density_max = float(saved_args.get("log_density_max", 12.0))
    init = saved_args.get("initial_log_density", None)
    if init is None:
        init = -np.log(volume(bounds))
    base = GeneralFPNN(
        input_scales=(bound,) * 8,
        width=width,
        depth=depth,
        first_width=first_width,
        density_parameterization=density_parameterization,
        log_density_min=log_density_min,
        log_density_max=log_density_max,
        initial_log_density=float(init),
    ).to(device)
    # New pure-PINN checkpoints are plain networks. Old caches containing
    # norm_points remain loadable through the legacy hard-mass wrapper.
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
            "重新计算一次评价点与 2D slice，并升级缓存。"
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
    bound = float(saved_args.get("bound", 0.70))
    bounds = tuple(tuple(v) for v in saved.get(
        "bounds", tuple([(-bound, bound)] * 8)
    ))
    epsilon = float(saved_args.get("epsilon", 0.05))
    rough_strength = float(saved_args.get("rough_strength", 1.0))
    seed = int(saved_args.get("seed", args.seed))
    eval_size = int(saved_args.get("eval_size", 131072))
    normalizer_size = int(saved_args.get("normalizer_size", 1048576))
    resolution = int(saved_args.get("heatmap_resolution", 201))
    slice_dims = tuple(int(v) for v in saved_args.get("heatmap_slice_dims", (0, 1)))
    device = torch.device("cpu")

    process = RoughPotential8D(
        epsilon=epsilon,
        rough_strength=rough_strength,
    )
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
            net, process, device, bounds,
            resolution=resolution, slice_dims=slice_dims,
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
        description="Faithful 2D/4D-style seven-method ablation for 8D rough potential"
    )
    p.add_argument("--fast", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument("--plot-only", action="store_true", help="直接从新版 .pt 的 evaluation_arrays 只重画四张对比图。")
    mode_group.add_argument("--eval-only", action="store_true", help="从旧/新 .pt 的 model_states 重建网络并刷新 evaluation_arrays，不重新训练。")
    p.add_argument("--cache-file", type=str, default=None, help="缓存 .pt 路径；默认使用 out-dir/rough_potential_8d_v6_5_strict_stage_b_fair_all7_results.pt。")
    p.add_argument("--only-weak-best", action="store_true")
    p.add_argument("--only-weak-hybrid", action="store_true")
    p.add_argument("--only-pure", action="store_true")
    p.add_argument("--only-hybrids", action="store_true")
    p.add_argument(
        "--only-lbfgs",
        action="store_true",
        help=(
            "只运行两个带 MC 和 L-BFGS 的方法："
            "Strong + MC + L-BFGS 与 Weak + MC + L-BFGS。"
        ),
    )
    p.add_argument(
        "--only-adam-and-weak-lbfgs",
        action="store_true",
        help=(
            "只运行三个训练方法：Strong + MC Adam-only、"
            "Weak + MC Adam-only、Weak + MC + L-BFGS。"
            "Adam-only 严格停在 Stage B；MC reference 仍进入结果表。"
        ),
    )
    p.add_argument("--allow-unqualified-mc", action="store_true")
    p.add_argument(
        "--augment-existing-mc",
        dest="augment_existing_mc",
        action="store_true",
        help="显式允许在现有 MC 计数上继续追加 Euler-Maruyama 样本。",
    )
    p.add_argument(
        "--no-augment-existing-mc",
        dest="augment_existing_mc",
        action="store_false",
        help="不追加 MC；直接使用现有计数文件（本版默认）。",
    )
    p.set_defaults(augment_existing_mc=False)
    p.add_argument(
        "--allow-mc-rebuild-if-missing",
        action="store_true",
        help=(
            "只有显式指定该参数时，MC 文件缺失才允许从零重建；"
            "默认直接报错，避免误用新生成的小规模 MC。"
        ),
    )
    p.add_argument(
        "--expected-mc-inside-samples",
        type=int,
        default=7_337_767,
        help=(
            "要求复用的 MC 计数文件包含指定数量的盒内样本。"
            "本版默认严格锁定上一轮直方图的 7,337,767 个盒内样本；"
            "设为 0 可关闭规模检查。"
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir", default="rough_potential_8d_paper_outputs"
    )
    p.add_argument(
        "--results-txt", default="rough_potential_8d_v6_5_strict_stage_b_fair_all7_results.txt"
    )
    p.add_argument(
        "--hist-counts-file",
        default=str(
            Path(__file__).resolve().with_name(
                "rough_potential_8d_histogram_counts_augmented_to_15pct.npy"
            )
        ),
        help=(
            "复用已验收的 8D MC 直方图计数。默认查找脚本同目录下的 "
            "rough_potential_8d_histogram_counts_augmented_to_15pct.npy；"
            "该文件应对应 7,337,767 个盒内样本。也可显式传入其他路径。"
        ),
    )
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--rough-strength", type=float, default=1.0)
    p.add_argument("--bound", type=float, default=0.70)
    p.add_argument("--hist-bin-width", type=float, default=0.20)
    p.add_argument("--hist-eval-chunk-size", type=int, default=262144)
    p.add_argument("--mc-target-mape", type=float, default=0.15)
    p.add_argument("--mc-target-rel-l2", type=float, default=0.15, help="仅用于日志诊断，不参与 v6.3 的 MC 硬验收。")
    p.add_argument("--eval-threshold", type=float, default=0.01)
    p.add_argument("--n-ref", type=int, default=10000)
    p.add_argument("--max-training-points", type=int, default=17500)
    p.add_argument("--density-fraction", type=float, default=0.90)
    p.add_argument("--histogram-data-mode", choices=("point", "cell-average"), default="cell-average")
    p.add_argument("--cell-average-fd-fraction", type=float, default=0.50)
    p.add_argument("--eval-size", type=int, default=131072)
    p.add_argument(
        "--heatmap-resolution",
        type=int,
        default=201,
        help="Resolution of the final 2D heatmaps for each experiment.",
    )
    p.add_argument(
        "--heatmap-slice-dims",
        type=int,
        nargs=2,
        default=(0, 1),
        metavar=("DIM_X", "DIM_Y"),
        help="Two coordinate indices used for the 2D slice heatmaps; all other coordinates are fixed at 0.",
    )
    p.add_argument("--normalizer-size", type=int, default=1048576)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--first-width", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--density-parameterization", choices=("softplus", "exp"), default="exp")
    p.add_argument("--log-density-min", type=float, default=-30.0)
    p.add_argument("--log-density-max", type=float, default=12.0)
    p.add_argument("--initial-log-density", type=float, default=None)

    # Kept from successful 4D defaults unless dimension alone requires more points.
    p.add_argument("--pretrain-epochs", type=int, default=800)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--pretrain-eta-min", type=float, default=5e-5)
    p.add_argument("--pretrain-log-every", type=int, default=200)
    p.add_argument("--data-grad-clip-pretrain", type=float, default=2.0)
    p.add_argument("--joint-cycles", type=int, default=2400)
    p.add_argument("--physics-pool-size", type=int, default=6000)
    p.add_argument("--fixed-check-size", type=int, default=1500)
    p.add_argument("--data-batch-size", type=int, default=512)
    p.add_argument("--physics-batch-size", type=int, default=1024)
    p.add_argument("--data-steps-per-cycle", type=int, default=1)
    p.add_argument("--physics-steps-per-cycle", type=int, default=2)
    p.add_argument("--hybrid-data-update-every", type=int, default=1)
    p.add_argument("--physics-gaussian-fraction", type=float, default=0.70)
    p.add_argument("--physics-proposal-std-scale", type=float, default=1.20)
    p.add_argument("--data-lr", type=float, default=8.0e-5)
    p.add_argument("--physics-lr", type=float, default=1.0e-5)
    p.add_argument("--data-eta-min", type=float, default=1e-5)
    p.add_argument("--physics-eta-min", type=float, default=5e-6)
    p.add_argument("--data-grad-clip", type=float, default=1.0)
    p.add_argument("--checkpoint-data-weight", type=float, default=1.0)
    p.add_argument("--checkpoint-physics-weight", type=float, default=1.0)
    p.add_argument("--joint-min-delta", type=float, default=1e-4)
    p.add_argument("--joint-min-cycles", type=int, default=600)
    p.add_argument("--joint-patience-checks", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--lbfgs-point-size", type=int, default=6000)
    p.add_argument("--lbfgs-data-size", type=int, default=2048)
    p.add_argument("--lbfgs-data-check-size", type=int, default=2048)
    p.add_argument("--lbfgs-data-density-fraction", type=float, default=0.95)
    p.add_argument("--lbfgs-max-data-ratio", type=float, default=1.5)
    p.add_argument("--lbfgs-max-iter", type=int, default=600)
    p.add_argument("--lbfgs-lr", type=float, default=0.20)
    p.add_argument("--lbfgs-history-size", type=int, default=50)
    p.add_argument("--lbfgs-data-weight", type=float, default=0.20)
    p.add_argument("--lbfgs-physics-weight", type=float, default=1.0)
    p.add_argument("--lbfgs-check-size", type=int, default=1500)
    p.add_argument("--lbfgs-check-every", type=int, default=20)
    p.add_argument("--lbfgs-min-iter", type=int, default=80)
    p.add_argument("--lbfgs-patience-checks", type=int, default=6)
    p.add_argument("--lbfgs-min-delta", type=float, default=1e-4)

    p.add_argument("--pure-physics-weight", type=float, default=1.0)
    p.add_argument("--pure-boundary-weight", type=float, default=0.10)
    p.add_argument("--pure-normal-weight", type=float, default=1.0)
    p.add_argument("--boundary-n-per-face", type=int, default=32)
    # Legacy CLI compatibility only; new pure-PINN training does not allocate
    # a separate hard-mass quadrature set.
    p.add_argument(
        "--hard-mass-quad-size",
        type=int,
        default=8192,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--mass-calibration-size", type=int, default=262144)
    p.add_argument("--mass-calibration-batch-size", type=int, default=32768)
    # Compatibility parameters used by retained MC-generation functions.
    p.add_argument("--mc-path-batch-size", type=int, default=50000)
    p.add_argument("--mc-min-batches", type=int, default=10)
    p.add_argument("--mc-max-batches", type=int, default=20)
    p.add_argument("--mc-check-every", type=int, default=1)
    p.add_argument("--burn-in", type=float, default=5.0)
    p.add_argument("--sample-duration", type=float, default=10.0)
    p.add_argument("--sample-interval", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.002)
    return p.parse_args()


def apply_fast(args):
    if not args.fast:
        return args
    args.normalizer_size = 2048
    args.eval_size = 512
    args.heatmap_resolution = 41
    args.n_ref = 128
    args.pretrain_epochs = 2
    args.pretrain_log_every = 1
    args.joint_cycles = 2
    args.physics_pool_size = 32
    args.fixed_check_size = 16
    args.data_batch_size = 32
    args.physics_batch_size = 8
    args.physics_steps_per_cycle = 1
    args.hybrid_data_update_every = 1
    args.physics_gaussian_fraction = 0.7
    args.histogram_data_mode = "point"
    args.lbfgs_data_size = 16
    args.lbfgs_data_check_size = 16
    args.joint_min_cycles = 1
    args.joint_patience_checks = 2
    args.checkpoint_every = 1
    args.log_every = 1
    args.lbfgs_point_size = 16
    args.lbfgs_max_iter = 2
    args.lbfgs_check_size = 16
    args.max_training_points = 1000
    args.lbfgs_check_every = 1
    args.lbfgs_min_iter = 1
    args.lbfgs_patience_checks = 2
    args.hard_mass_quad_size = 64
    args.boundary_n_per_face = 2
    args.mass_calibration_size = 128
    args.mass_calibration_batch_size = 64
    args.hist_bin_width = max(args.hist_bin_width, 0.7)
    args.mc_target_mape = 10.0
    args.mc_target_rel_l2 = 10.0
    args.augment_existing_mc = False
    args.expected_mc_inside_samples = 0
    args.allow_unqualified_mc = True
    args.width = 12
    args.first_width = 16
    args.depth = 1
    return args


def select_methods(args):
    if args.only_adam_and_weak_lbfgs:
        return [
            "Strong + MC Adam-only",
            "Weak + MC Adam-only",
            "Weak + MC + L-BFGS",
        ]
    if args.only_weak_best:
        return ["Weak + MC + L-BFGS"]
    if args.only_weak_hybrid:
        return ["Weak + MC Adam-only", "Weak + MC + L-BFGS"]
    if args.only_pure:
        return ["Strong PINN", "Weak PINN"]
    if args.only_lbfgs:
        return [
            "Strong + MC + L-BFGS",
            "Weak + MC + L-BFGS",
        ]
    if args.only_hybrids:
        return [
            "Data-only NN", "Strong + MC Adam-only", "Weak + MC Adam-only",
            "Strong + MC + L-BFGS", "Weak + MC + L-BFGS",
        ]
    return list(METHODS)


def smoke_test(seed=42):
    set_seed(seed)
    process = RoughPotential8D(epsilon=0.05)
    bounds = tuple([(-0.7, 0.7)] * 8)
    vol = volume(bounds)
    ip, iw = make_importance_physics_set(
        16, bounds, process, 0.7, 1.20, seed + 2
    )
    bp, bn, bw = sobol_boundary(2, bounds, seed + 40)

    hybrid = GeneralFPNN(
        width=12,
        depth=1,
        density_parameterization="exp",
        initial_log_density=-np.log(vol),
    )
    target = torch.full((16, 1), 0.02, dtype=torch.float64)
    scale = torch.mean(target ** 2) + 1e-14
    dl, _ = normalized_data_loss(hybrid, ip, target, scale)
    wl = weak_loss(hybrid, ip, process, vol, iw)
    (dl + wl).backward()

    for offset, physics_type in enumerate(("strong", "weak"), start=1):
        set_seed(seed + offset)
        net = GeneralFPNN(
            width=12,
            depth=1,
            density_parameterization="exp",
            initial_log_density=-np.log(vol),
        )
        subset = slice(None, 4) if physics_type == "strong" else slice(None)
        pts = ip[subset]
        weights = iw[subset]
        pl = (
            strong_loss(net, pts, process, vol, weights)
            if physics_type == "strong"
            else weak_loss(net, pts, process, vol, weights)
        )
        bl = noflux_loss(net, bp, bn, bw, process)
        nl, mass = normalization_loss(
            net, pts, vol, importance_weights=weights, target_mass=1.0
        )
        total = pl + 0.10 * bl + nl
        total.backward()
        print(
            f"[{physics_type}] physics={pl.item():.8e}, "
            f"boundary={bl.item():.8e}, normal={nl.item():.8e}, "
            f"mass={mass.item():.8f}, total={total.item():.8e}"
        )

    print("8D pure-PINN three-term forward/backward: PASS")




class TeeStream:
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
    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


def method_coordinate_budget(method_name, args):
    """
    Count unique coordinates that actually enter training/model selection.

    Reused Stage-C subsets do not add coordinates.
    MC histogram simulation samples are not counted as NN coordinates; the
    n_ref selected histogram bins are counted.
    """
    cfg = METHODS[method_name]
    details = {
        "mc_bins": int(args.n_ref) if cfg["use_data"] else 0,
        "physics_pool": int(args.physics_pool_size) if cfg["physics"] else 0,
        "physics_monitor": int(args.fixed_check_size) if cfg["physics"] else 0,
        "hard_mass": int(args.hard_mass_quad_size) if cfg["hard_mass"] else 0,
        "boundary": (
            int(2 * 8 * args.boundary_n_per_face)
            if cfg["boundary"] == "noflux"
            else 0
        ),
    }
    details["total"] = int(sum(details.values()))
    return details



def execute(args, out_dir):
    if getattr(args, "plot_only", False):
        run_plot_only(args, out_dir)
        return
    if getattr(args, "eval_only", False):
        run_eval_only(args, out_dir)
        return
    if args.smoke_test:
        smoke_test(args.seed)
        return
    args = apply_fast(args)

    methods_for_budget = select_methods(args)
    coordinate_budgets = {
        method_name: method_coordinate_budget(method_name, args)
        for method_name in methods_for_budget
    }
    training_point_budget = max(
        details["total"] for details in coordinate_budgets.values()
    )

    if args.lbfgs_point_size > args.physics_pool_size:
        raise ValueError(
            "lbfgs_point_size 必须 <= physics_pool_size，"
            "因为 Stage C 复用 Stage B physics pool。"
        )
    if args.lbfgs_check_size > args.fixed_check_size:
        raise ValueError(
            "lbfgs_check_size 必须 <= fixed_check_size，"
            "因为 Stage C 复用 Stage B monitor set。"
        )
    if (
        args.max_training_points > 0
        and training_point_budget > args.max_training_points
    ):
        raise ValueError(
            f"唯一训练点预算超限：{training_point_budget:,} > "
            f"{args.max_training_points:,}。"
        )

    set_seed(args.seed)
    start_all = time.time()
    device = torch.device("cpu")
    process = RoughPotential8D(
        epsilon=args.epsilon,
        rough_strength=args.rough_strength,
    )
    bounds = tuple([(-args.bound, args.bound)] * 8)

    print("\n" + "=" * 116)
    print("8D rough-potential strict Stage-B 公平消融配置")
    print("=" * 116)
    print("最终图像输出固定为 4 张：density / MAPE / absolute error / relative error。")
    print("MAPE 柱状图严格按 METHODS 中的方法顺序，不按数值排序。")
    print("训练、importance correction、MC counts 读取和 Stage A/B/C 语义保持原样。")
    print(f"methods={select_methods(args)}")
    print(
        f"[最大实际唯一坐标预算] {training_point_budget:,} / "
        f"{args.max_training_points:,}; Stage C 全部复用/抽取"
    )
    for key in (
        "n_ref", "max_training_points", "density_fraction",
        "pretrain_epochs", "joint_cycles",
        "physics_steps_per_cycle", "data_steps_per_cycle",
        "hybrid_data_update_every", "data_lr", "physics_lr",
        "physics_gaussian_fraction", "physics_proposal_std_scale",
        "physics_pool_size", "fixed_check_size",
        "physics_batch_size", "lbfgs_point_size", "lbfgs_max_iter",
        "first_width", "width", "depth",
        "density_parameterization", "histogram_data_mode",
        "cell_average_fd_fraction", "lbfgs_data_weight",
        "lbfgs_physics_weight", "lbfgs_max_data_ratio",
    ):
        print(f"{key:<30s}= {getattr(args, key)}")
    print(f"hist counts file             = {Path(args.hist_counts_file)}")
    print("=" * 116 + "\n")

    process.prepare_normalizer(
        bounds, args.normalizer_size, args.seed + 5001
    )
    hist = load_histogram_reference_from_counts(process, bounds, args, out_dir)
    y_data, v_data = hist["y_data"], hist["v_data"]
    target_mass = float(hist["metrics"]["诊断总质量"])
    eval_points = sobol_box(args.eval_size, bounds, args.seed + 9001)

    figures_dir = _paper_figures_dir(out_dir)
    results = {"MC histogram reference (diag)": dict(hist["metrics"])}
    runtimes, histories, states = {}, {}, {}
    evaluation_arrays = {}
    methods = select_methods(args)

    for i, method in enumerate(methods):
        print("\n" + "#" * 108)
        print(f"方法 {i+1}/{len(methods)}：{method}")
        print("#" * 108)
        set_seed(args.seed)
        init_log_density = (
            -np.log(volume(bounds))
            if args.initial_log_density is None
            else args.initial_log_density
        )
        net = GeneralFPNN(
            input_scales=(args.bound,) * 8,
            width=args.width,
            depth=args.depth,
            first_width=args.first_width,
            density_parameterization=args.density_parameterization,
            log_density_min=args.log_density_min,
            log_density_max=args.log_density_max,
            initial_log_density=init_log_density,
        ).to(device)
        t0 = time.time()
        trained, history = train_model(
            method_name=method,
            net=net,
            process=process,
            device=device,
            bounds=bounds,
            args=args,
            y_data=y_data.clone() if METHODS[method]["use_data"] else None,
            v_data=v_data.clone() if METHODS[method]["use_data"] else None,
            target_mass=target_mass,
        )
        elapsed = time.time() - t0
        metrics = evaluate_model(
            trained, process, device, bounds, eval_points
        )
        results[method] = metrics
        runtimes[method] = elapsed
        histories[method] = history
        states[method] = {
            k: v.detach().cpu().clone()
            for k, v in trained.state_dict().items()
        }
        evaluation_arrays[method] = evaluate_slice_and_store_arrays(
            trained, process, device, bounds,
            resolution=args.heatmap_resolution,
            slice_dims=tuple(args.heatmap_slice_dims),
        )
        print(
            f"[完成] {method} | MAPE={100*metrics['有效区域平均相对误差']:.4f}% | "
            f"Rel-L2={100*metrics['相对L2误差']:.4f}% | "
            f"Mass={metrics['诊断总质量']:.6f} | {elapsed/60:.2f} min"
        )
        del trained, net

    print_table(results)
    print("\n" + "=" * 76)
    print("训练时间")
    print("=" * 76)
    for name, seconds in runtimes.items():
        print(f"{name:<34s} | {seconds:12.2f} s | {seconds/60:10.2f} min")
    print("=" * 76)

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
        "model_states": states,
        "evaluation_arrays": evaluation_arrays,
        "cache_format_version": 2,
        "figure_files": figure_files,
        "args": vars(args),
        "bounds": bounds,
        "dimension": 8,
        "tuning_version": "6.5-strict-stage-b-four-figure-cache",
        "methods_run": methods,
        "histogram_counts_path": hist["counts_path"],
        "histogram_selected_indices": hist["selected_indices"],
        "reference_points": y_data.detach().cpu(),
        "reference_values": v_data.detach().cpu(),
        "mc_target_mass": target_mass,
        "evaluation_slice": {
            "dims": tuple(args.heatmap_slice_dims),
            "resolution": int(args.heatmap_resolution),
            "fixed_other_coordinates": 0.0,
        },
        "training_contract": {
            "pointwise_histogram_mse": True,
            "cell_average_loss": bool(args.histogram_data_mode == "cell-average"),
            "importance_corrected_physics_sampling": True,
            "existing_mc_augmented_only_by_new_em_samples": bool(args.augment_existing_mc),
            "compact_single_peak_network": True,
            "unique_training_point_budget": int(training_point_budget),
            "coordinate_budgets": coordinate_budgets,
            "stage_c_reuses_stage_b_physics_sets": True,
            "separate_data_physics_adam": True,
            "monitored_lbfgs_same_measure": True,
            "exact_density_in_training": False,
            "weak_best_single_mc_mass_calibration": False,
            "final_mass_calibration_disabled": True,
            "pure_pinn_hard_mass": False,
            "pure_pinn_explicit_normalization_loss": True,
            "pure_pinn_boundary_loss_for_both": True,
            "final_output_scale": 1.0,
        },
    }
    _save_cache(saved, save_path)
    print(f"总用时={(time.time()-start_all)/60:.2f} 分钟")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / args.results_txt
    original = sys.stdout
    with open(path, "w", encoding="utf-8", buffering=1) as f:
        sys.stdout = TeeStream(original, f)
        try:
            print(f"[文本结果文件] {path.resolve()}")
            execute(args, out_dir)
            print(f"\n[文本结果保存完成] {path.resolve()}")
        finally:
            sys.stdout = original
    print(f"全部输出已保存到：{path.resolve()}")


if __name__ == "__main__":
    main()
