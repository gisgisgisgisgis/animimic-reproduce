"""Differentiable implicit FEM (paper Sec. 3.2 and Appendix A).

Fixed Corotated elasticity, backward Euler via Newton with line search, and the
adjoint of Eq. 19-21. The paper uses Warp. This module implements the same math
in batched PyTorch, so it runs on CUDA with float64 and needs no extra build.
The Hessian is applied matrix-free in closed form.
"""
import torch


def lame(E, nu):
    return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def _cofactor(F):
    f1, f2, f3 = F[..., 0], F[..., 1], F[..., 2]
    return torch.stack(
        (torch.cross(f2, f3, dim=-1), torch.cross(f3, f1, dim=-1), torch.cross(f1, f2, dim=-1)), dim=-1,
    )


def _skew(w):
    o = torch.zeros_like(w[..., 0])
    return torch.stack((o, -w[..., 2], w[..., 1], w[..., 2], o, -w[..., 0], -w[..., 1], w[..., 0], o), -1).reshape(
        *w.shape[:-1], 3, 3,
    )


class TetFEM:
    def __init__(self, rest, tets, density=1000.0, nu=0.4, device="cpu"):
        self.device = torch.device(device)
        self.X = torch.as_tensor(rest, dtype=torch.float64, device=self.device)
        tets = torch.as_tensor(tets, dtype=torch.long, device=self.device).clone()
        Dm = self._edges(self.X, tets)
        flip = torch.det(Dm) < 0
        tets[flip, 2], tets[flip, 3] = tets[flip, 3].clone(), tets[flip, 2].clone()
        self.tets = tets
        Dm = self._edges(self.X, tets)
        self.Dm_inv = torch.linalg.inv(Dm)
        self.vol = torch.det(Dm).abs() / 6.0
        self.nu = nu
        self.num_nodes = self.X.shape[0]
        self.mass = self._assemble_scalar((self.vol * density / 4.0)[:, None].repeat(1, 4))

        grad_n = self.Dm_inv.transpose(1, 2)  # columns are grad N_1..3
        grad0 = -grad_n.sum(-1, keepdim=True)
        self.grad_n = torch.cat((grad0, grad_n), dim=-1)  # (M, 3, 4)

    @staticmethod
    def _edges(x, tets):
        p = x[tets]
        return (p[:, 1:] - p[:, :1]).transpose(1, 2)

    def _assemble_scalar(self, values):
        out = torch.zeros(self.X.shape[0], device=self.device, dtype=torch.float64)
        out.index_add_(0, self.tets.flatten(), values.flatten())
        return out

    def _assemble(self, elem):
        """elem: (M, 4, 3) -> nodal (N, 3)."""
        out = torch.zeros_like(self.X)
        idx = self.tets.flatten()
        for d in range(3):
            out[:, d].index_add_(0, idx, elem[..., d].flatten())
        return out

    def state(self, x):
        F = self._edges(x, self.tets) @ self.Dm_inv
        U, S, Vh = torch.linalg.svd(F)
        d = torch.sign(torch.det(U @ Vh))
        U = U.clone()
        U[:, :, 2] = U[:, :, 2] * d[:, None]
        S = S.clone()
        S[:, 2] = S[:, 2] * d
        C = _cofactor(F)
        J = (F[..., 0] * C[..., 0]).sum(-1)
        return {"F": F, "R": U @ Vh, "S": S, "V": Vh.transpose(1, 2), "C": C, "J": J}

    def energy(self, st, mu, lam):
        return (self.vol * (mu * (st["F"] - st["R"]).pow(2).sum((1, 2)) + 0.5 * lam * (st["J"] - 1.0) ** 2)).sum()

    def _element_force(self, P):
        """Elastic gradient per element: V P grad N_a, shape (M, 4, 3)."""
        return (self.vol[:, None, None] * (P @ self.grad_n)).transpose(1, 2)

    def piola(self, st, mu, lam):
        m, l = mu[:, None, None], lam[:, None, None]
        return 2.0 * m * (st["F"] - st["R"]) + l * (st["J"] - 1.0)[:, None, None] * st["C"]

    def elastic_grad(self, st, mu, lam):
        return self._assemble(self._element_force(self.piola(st, mu, lam)))

    def hess_vec(self, st, dx, mu, lam):
        F, R, S, V, C, J = st["F"], st["R"], st["S"], st["V"], st["C"], st["J"]
        dF = self._edges(dx, self.tets) @ self.Dm_inv
        RtdF = R.transpose(1, 2) @ dF
        A = RtdF - RtdF.transpose(1, 2)
        axial = torch.stack((A[:, 2, 1], A[:, 0, 2], A[:, 1, 0]), -1)
        s1, s2, s3 = S[:, 0], S[:, 1], S[:, 2]
        denom = torch.stack((s2 + s3, s1 + s3, s1 + s2), -1).clamp_min(1e-6)
        w = (V @ ((V.transpose(1, 2) @ axial[..., None])[..., 0] / denom)[..., None])[..., 0]
        dR = R @ _skew(w)

        dJ = (C * dF).sum((1, 2))
        f1, f2, f3 = F[..., 0], F[..., 1], F[..., 2]
        d1, d2, d3 = dF[..., 0], dF[..., 1], dF[..., 2]
        cr = lambda a, b: torch.cross(a, b, dim=-1)
        dC = torch.stack((
            cr(d2, f3) + cr(f2, d3), cr(d3, f1) + cr(f3, d1), cr(d1, f2) + cr(f1, d2),
        ), -1)
        m, l = mu[:, None, None], lam[:, None, None]
        dP = 2.0 * m * (dF - dR) + l * dJ[:, None, None] * C + l * (J - 1.0)[:, None, None] * dC
        return self._assemble(self._element_force(dP))

    def _diag(self, mu, lam, inertia):
        norm2 = self.grad_n.pow(2).sum(1)  # (M, 4)
        stiff = self._assemble_scalar((2.0 * mu + lam)[:, None] * self.vol[:, None] * norm2)
        return inertia + stiff[:, None]

    def _pcg(self, st, b, inertia, mu, lam, free, tol, max_iter):
        diag = self._diag(mu, lam, inertia)
        apply = lambda v: free * (inertia * v + self.hess_vec(st, free * v, mu, lam))
        x = torch.zeros_like(b)
        r = b.clone()
        z = r / diag
        p = z.clone()
        rz = (r * z).sum()
        r0 = r.norm().clamp_min(1e-30)
        for k in range(max_iter):
            Ap = apply(p)
            pAp = (p * Ap).sum()
            if pAp <= 0:
                return z if k == 0 else x
            a = rz / pAp
            x = x + a * p
            r = r - a * Ap
            if r.norm() < tol * r0:
                break
            z = r / diag
            rz_new = (r * z).sum()
            p = z + (rz_new / rz) * p
            rz = rz_new
        return x

    @torch.no_grad()
    def solve_step(self, x, x_tilde, mu, lam, free, h, max_newton=30, tol=1e-7, cg_iters=300, rel_tol=0.0):
        """One backward-Euler step: argmin 1/(2h^2)||x - x~||_M^2 + Psi(x), with pinned nodes held.

        Stops when the largest force residual is below tol x the largest inertial force, or below rel_tol x
        the residual at the starting point. The first test alone scales with 1/h^2: with long steps (few
        frames per second of the clip) it asks for far more precision than the elastic forces carry."""
        inertia = (self.mass / h ** 2)[:, None]

        def objective(q):
            s = self.state(q)
            return 0.5 * (inertia * (q - x_tilde).pow(2)).sum() + self.energy(s, mu, lam), s

        e0, st = objective(x)
        scale = (inertia * x_tilde.abs()).abs().max().clamp_min(1e-12)
        g_start = None
        for _ in range(max_newton):
            g = free * (inertia * (x - x_tilde) + self.elastic_grad(st, mu, lam))
            g_max = g.abs().max()
            g_start = g_max if g_start is None else g_start
            if g_max < tol * scale or g_max < rel_tol * g_start:
                break
            dx = self._pcg(st, -g, inertia, mu, lam, free, 1e-3, cg_iters)
            slope = (g * dx).sum()
            if slope >= 0:
                dx, slope = -g / self._diag(mu, lam, inertia), -(g * (g / self._diag(mu, lam, inertia))).sum()
            alpha, accepted = 1.0, False
            while alpha > 1e-10:
                q = x + alpha * dx
                e, s = objective(q)
                if torch.isfinite(e) and e <= e0 + 1e-4 * alpha * slope:
                    x, e0, st, accepted = q, e, s, True
                    break
                alpha *= 0.5
            if not accepted:
                break
        return x

    @torch.no_grad()
    def simulate(self, x0, v0, E, pinned_idx, pinned_traj, h, guess_traj=None, **kw):
        """Roll out len(pinned_traj)-1 steps. Returns (S+1, N, 3).

        guess_traj (S+1, N, 3), optional: where Newton starts at each step, e.g. the skinned pose. It only
        changes the starting point, not the solution. Starting from the inertial prediction with only the
        pinned nodes moved stretches the tets around the pins, and Newton then needed 30+ iterations."""
        mu, lam = lame(E, self.nu)
        free = torch.ones_like(self.X)
        free[pinned_idx] = 0.0
        xs, x, v = [x0], x0, v0
        for n in range(pinned_traj.shape[0] - 1):
            x_tilde = x + h * v
            guess = x_tilde.clone() if guess_traj is None else guess_traj[n + 1].clone()
            guess[pinned_idx] = pinned_traj[n + 1]
            x_next = self.solve_step(guess, x_tilde, mu, lam, free, h, **kw)
            v = (x_next - x) / h
            x = x_next
            xs.append(x)
        return torch.stack(xs)

    @torch.no_grad()
    def adjoint_gradient(self, xs, E, pinned_idx, h, dL_dx, cg_iters=800):
        """dL/dE per tet for L = sum_n l_n(x^n), with dL_dx = (dl_n/dx^n), shape (S+1, N, 3). Eq. 19-21."""
        mu, lam = lame(E, self.nu)
        unit_mu, unit_lam = lame(torch.ones_like(E), self.nu)
        free = torch.ones_like(self.X)
        free[pinned_idx] = 0.0
        inertia = (self.mass / h ** 2)[:, None]
        dL_dE = torch.zeros_like(E)
        lam_x, lam_v = dL_dx[-1].clone(), torch.zeros_like(self.X)
        for n in range(xs.shape[0] - 2, -1, -1):
            st = self.state(xs[n + 1])
            b = free * (lam_x + lam_v / h)
            A = self._pcg(st, b, inertia, mu, lam, free, 1e-8, cg_iters)
            unit = self._element_force(self.piola(st, unit_mu, unit_lam))
            dL_dE -= (A[self.tets] * unit).sum((1, 2))
            prev_lam_v = lam_v
            lam_v = free * (inertia * A) * h
            lam_x = dL_dx[n] + free * (inertia * A) - prev_lam_v / h
        return dL_dE
