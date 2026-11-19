"""Fixed-corotated FEM with implicit Euler and implicit-adjoint gradients.

Dense reference solver for gradient validation and small tetrahedral problems.
The Hessian is solved implicitly, rather than differentiating Newton iterations.
The production step assembles exact per-element Hessians into a sparse matrix
and factorizes it, instead of iterating Hessian-vector products.
"""
import torch
from contextvars import ContextVar
from torch.func import jacrev, vmap

BACKEND = ContextVar("fem_backend", default="torch")
NEWTON_RELATIVE_TOLERANCE = 1e-6
NEWTON_ABSOLUTE_TOLERANCE = 1e-10


class FEMConvergenceError(RuntimeError):
    """A time step could not be solved without invalid tetrahedra."""


def elastic_energy(x, rest, tets, log_e, poisson=0.45):
    if BACKEND.get() == "warp":
        from .warp_fem import WarpEnergy
        return WarpEnergy.apply(x, rest, tets, log_e, poisson)
    return torch_elastic_energy(x, rest, tets, log_e, poisson)


def volumes(x, tets):
    p = x[tets]
    return torch.linalg.det(torch.stack(
        (p[:, 1]-p[:, 0], p[:, 2]-p[:, 0], p[:, 3]-p[:, 0]), dim=-1)) / 6


def torch_elastic_energy(x, rest, tets, log_e, poisson=0.45):
    p, q = x[tets], rest[tets]
    ds = torch.stack((p[:, 1]-p[:, 0], p[:, 2]-p[:, 0], p[:, 3]-p[:, 0]), -1)
    dm = torch.stack((q[:, 1]-q[:, 0], q[:, 2]-q[:, 0], q[:, 3]-q[:, 0]), -1)
    deformation = ds @ torch.linalg.inv(dm)
    # Newton polar iteration has well-defined derivatives at repeated singular
    # values (including the identity), unlike a generic SVD backward.
    rotation = deformation
    for _ in range(12):
        rotation = 0.5 * (rotation + torch.linalg.inv(rotation).transpose(-1, -2))
    e = log_e.exp()
    mu = e / (2 * (1 + poisson))
    lam = e * poisson / ((1 + poisson) * (1 - 2 * poisson))
    density = mu * (deformation - rotation).square().sum((-1, -2))
    density = density + lam / 2 * (torch.linalg.det(deformation)-1).square()
    return (density * volumes(rest, tets).abs()).sum()


def implicit_step(rest, tets, mass, x, velocity, log_e, dt, fixed, targets,
                  iterations=30, tolerance=1e-8):
    """Differentiable reference Newton solve; prescribed nodes eliminated."""
    free = (~fixed).nonzero().flatten()
    predicted = x + dt * velocity

    def unpack(z):
        value = targets.clone()
        return value.index_copy(0, free, z.reshape(-1, 3))

    def objective(z):
        value = unpack(z)
        inertia = (mass[:, None] * (value-predicted).square()).sum() / (2*dt*dt)
        return inertia + elastic_energy(value, rest, tets, log_e)

    z = x[free].reshape(-1).detach().requires_grad_(True)
    for _ in range(iterations):
        loss = objective(z)
        gradient = torch.autograd.grad(loss, z, create_graph=True)[0]
        if gradient.detach().norm() < tolerance:
            break
        hessian = torch.autograd.functional.hessian(objective, z)
        direction = torch.linalg.solve(
            hessian + torch.eye(len(z), device=z.device, dtype=z.dtype)*1e-6,
            -gradient)
        alpha = 1.0
        for _ in range(20):
            candidate = z + alpha * direction
            if (volumes(unpack(candidate), tets) > 0).all() and objective(candidate) < loss:
                break
            alpha *= 0.5
        else:
            raise RuntimeError("FEM line search failed")
        z = candidate.detach().requires_grad_(True)
    # Implicit differentiation via one Newton correction at the converged state.
    residual = torch.autograd.grad(objective(z), z, create_graph=True)[0]
    hessian = torch.autograd.functional.hessian(objective, z).detach()
    z = z - torch.linalg.solve(hessian, residual)
    result = unpack(z)
    if not (volumes(result, tets) > 0).all():
        raise RuntimeError("Inverted tetrahedron")
    return result, (result-x)/dt


def _det3(m):
    # Explicit cofactor expansion: torch.func's second derivative of
    # torch.linalg.det is NaN at the identity (the rest state).
    return (m[0, 0]*(m[1, 1]*m[2, 2]-m[1, 2]*m[2, 1])
            - m[0, 1]*(m[1, 0]*m[2, 2]-m[1, 2]*m[2, 0])
            + m[0, 2]*(m[1, 0]*m[2, 1]-m[1, 1]*m[2, 0]))


def _element_energy(p, dm_inverse, volume, log_e, poisson):
    """One tetrahedron of torch_elastic_energy; the same formula, unbatched."""
    ds = torch.stack((p[1]-p[0], p[2]-p[0], p[3]-p[0]), -1)
    deformation = ds @ dm_inverse
    rotation = deformation
    for _ in range(12):
        rotation = 0.5 * (rotation + torch.linalg.inv(rotation).transpose(-1, -2))
    e = log_e.exp()
    mu = e / (2 * (1 + poisson))
    lam = e * poisson / ((1 + poisson) * (1 - 2 * poisson))
    density = mu * (deformation - rotation).square().sum()
    density = density + lam / 2 * (_det3(deformation)-1).square()
    return density * volume


_element_hessian = vmap(jacrev(jacrev(_element_energy)), in_dims=(0, 0, 0, 0, None))

_LEVI_CIVITA = {}


def _levi_civita(device, dtype):
    key = (str(device), dtype)
    if key not in _LEVI_CIVITA:
        eps = torch.zeros(3, 3, 3, device=device, dtype=dtype)
        for a, b, c, sign in ((0, 1, 2, 1), (1, 2, 0, 1), (2, 0, 1, 1),
                              (0, 2, 1, -1), (2, 1, 0, -1), (1, 0, 2, -1)):
            eps[a, b, c] = sign
        _LEVI_CIVITA[key] = eps
    return _LEVI_CIVITA[key]


def _deformation_hessians(deformation, u, s, v, log_e, poisson):
    """Analytic d2(psi)/dF2 (row-major vec(F)) of the fixed-corotated density.

    With F = U S V^T, the polar factor R = U V^T responds only to the three
    twist modes U (e_i e_j^T - e_j e_i^T) V^T / sqrt(2), with eigenvalue
    2/(s_i + s_j). The volume term uses d2 det(F) = eps_ikm eps_jln F_mn.
    Valid where det(F) > 0 and the 12-step Newton polar factor has converged.
    """
    n = len(deformation)
    e = log_e.exp()
    mu = e / (2 * (1 + poisson))
    lam = e * poisson / ((1 + poisson) * (1 - 2 * poisson))
    result = 2*mu[:, None, None]*torch.eye(9, device=deformation.device,
                                           dtype=deformation.dtype).expand(n, 9, 9)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        twist = (u[:, :, i, None]*v[:, None, :, j] - u[:, :, j, None]*v[:, None, :, i]) / 2**0.5
        twist = twist.reshape(n, 9)
        weight = 4*mu/(s[:, i]+s[:, j])
        result = result - weight[:, None, None]*twist[:, :, None]*twist[:, None, :]
    eps = _levi_civita(deformation.device, deformation.dtype)
    cofactor = 0.5*torch.einsum("ikm,jln,bkl,bmn->bij", eps, eps, deformation, deformation).reshape(n, 9)
    second = torch.einsum("ikm,jln,bmn->bijkl", eps, eps, deformation).reshape(n, 9, 9)
    volume_change = torch.linalg.det(deformation) - 1
    return (result + lam[:, None, None]*cofactor[:, :, None]*cofactor[:, None, :]
            + (lam*volume_change)[:, None, None]*second)


def element_hessians(x, rest, tets, log_e, poisson=0.45, project=False):
    """Exact 12x12 elastic Hessians; optionally clamped to positive semidefinite.

    Uses the analytic fixed-corotated Hessian, and per-element automatic
    differentiation of the same energy where the analytic form does not apply
    (inverted elements, or an unconverged Newton polar iteration).
    """
    q, p = rest[tets], x[tets]
    dm = torch.stack((q[:, 1]-q[:, 0], q[:, 2]-q[:, 0], q[:, 3]-q[:, 0]), -1)
    ds = torch.stack((p[:, 1]-p[:, 0], p[:, 2]-p[:, 0], p[:, 3]-p[:, 0]), -1)
    inverse = torch.linalg.inv(dm)
    volume = (torch.linalg.det(dm) / 6).abs()
    log_e = log_e.expand(len(tets)) if log_e.numel() == 1 else log_e
    deformation = ds @ inverse
    # dF/dx: F_ij depends on x_ac through delta_ic * G_aj, G = [-1^T Dm^-1; Dm^-1].
    gradient = torch.cat((-inverse.sum(1, keepdim=True), inverse), 1)
    identity = torch.eye(3, device=x.device, dtype=x.dtype)
    jacobian = torch.einsum("ic,naj->nijac", identity, gradient).reshape(len(tets), 9, 12)
    # The energy's own polar factor R gives F = R P; diagonalizing the
    # symmetric P = V S V^T yields the SVD (U = R V) far faster than a
    # batched GPU SVD.
    determinant = torch.linalg.det(deformation)
    safe = torch.where((determinant > 0)[:, None, None], deformation, identity)
    rotation = safe
    for _ in range(12):
        rotation = 0.5 * (rotation + torch.linalg.inv(rotation).transpose(-1, -2))
    stretch = rotation.transpose(-1, -2) @ safe
    asymmetry = (stretch - stretch.transpose(-1, -2)).abs().amax((1, 2))
    singular, v = torch.linalg.eigh(0.5*(stretch + stretch.transpose(-1, -2)))
    hessian_f = _deformation_hessians(deformation, rotation @ v, singular, v, log_e, poisson)
    if project:
        values, vectors = torch.linalg.eigh(0.5*(hessian_f + hessian_f.transpose(-1, -2)))
        hessian_f = (vectors * values.clamp_min(0)[:, None, :]) @ vectors.transpose(-1, -2)
    blocks = volume[:, None, None] * jacobian.transpose(1, 2) @ hessian_f @ jacobian
    orthogonality = (rotation.transpose(-1, -2) @ rotation - identity).abs().amax((1, 2))
    fallback = ((determinant <= 0) | (orthogonality > 1e-10) | (singular.min(-1).values <= 0)
                | (asymmetry > 1e-10*stretch.abs().amax((1, 2)))
                | ~torch.isfinite(blocks).all(-1).all(-1))
    if fallback.any():
        index = fallback.nonzero().flatten()
        exact = _element_hessian(p[index], inverse[index], volume[index], log_e[index], poisson)
        exact = exact.reshape(len(index), 12, 12)
        exact = 0.5 * (exact + exact.transpose(-1, -2))
        if project:
            values, vectors = torch.linalg.eigh(exact)
            exact = (vectors * values.clamp_min(0)[:, None, :]) @ vectors.transpose(-1, -2)
        blocks = blocks.index_copy(0, index, exact)
    return 0.5 * (blocks + blocks.transpose(-1, -2))


_STRUCTURES = {}


class _SparseTangent:
    """Free-DOF Hessian M/dt^2 + K assembled from element blocks, solved by LU.

    The sparsity pattern depends only on the mesh and fixed nodes, so it is
    computed once and reused; each factorization only scatters new values.
    """

    def __init__(self, tets, fixed, mass, dt):
        import numpy as np
        key = (tets.data_ptr(), tuple(tets.shape), int(tets.sum()),
               fixed.data_ptr(), int(fixed.sum()), str(tets.device))
        if key not in _STRUCTURES:
            if len(_STRUCTURES) > 8:
                _STRUCTURES.clear()
            free = ~fixed
            position = torch.full((len(fixed),), -1, dtype=torch.long, device=fixed.device)
            position[free] = torch.arange(int(free.sum()), device=fixed.device)
            node = position[tets]
            dof = (node[..., None]*3 + torch.arange(3, device=tets.device)).reshape(len(tets), 12)
            valid = (node >= 0).repeat_interleave(3, dim=1)
            mask = valid[:, :, None] & valid[:, None, :]
            size = 3*int(free.sum())
            rows = dof[:, :, None].expand(-1, 12, 12)[mask]
            cols = dof[:, None, :].expand(-1, 12, 12)[mask]
            diagonal = torch.arange(size, device=tets.device)
            rows, cols = torch.cat((rows, diagonal)), torch.cat((cols, diagonal))
            # Column-major unique entries give the CSC layout directly.
            unique, slot = torch.unique(cols*size + rows, return_inverse=True)
            unique = unique.cpu().numpy()
            column, row = unique // size, unique % size
            indptr = np.zeros(size+1, dtype=np.int64)
            np.add.at(indptr, column+1, 1)
            _STRUCTURES[key] = (mask, slot, len(unique), row.astype(np.int32),
                                np.cumsum(indptr).astype(np.int32), size, int(mask.sum()))
        (self.mask, self.slot, self.count, self.row,
         self.indptr, self.size, self.entries) = _STRUCTURES[key]
        self.diagonal = (mass[~fixed]/(dt*dt)).repeat_interleave(3)

    def factorize(self, blocks):
        import numpy as np
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import splu
        values = torch.cat((blocks[self.mask].detach(), self.diagonal.to(blocks.dtype)))
        data = torch.zeros(self.count, dtype=values.dtype, device=values.device)
        data.index_add_(0, self.slot, values)
        matrix = csc_matrix((data.cpu().numpy(), self.row, self.indptr), shape=(self.size, self.size))
        factor = splu(matrix, permc_spec="MMD_AT_PLUS_A")
        def solve(rhs):
            vector = rhs.detach().reshape(-1).cpu().numpy()
            solution = factor.solve(vector)
            residual = float(np.linalg.norm(matrix @ solution - vector))
            return torch.as_tensor(solution, dtype=rhs.dtype, device=rhs.device).reshape(rhs.shape), residual
        return solve


class _ImplicitFEM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rest, tets, mass, x, velocity, log_e, fixed, targets, dt, notify, guess=None):
        ctx.notify = notify
        free = (~fixed).nonzero().flatten()
        predicted = x + dt*velocity
        # The adjoint in backward() recomputes parameter derivatives; keep the
        # Newton graph independent of any upstream graph that produced log_e.
        log_e = log_e.detach()
        with torch.enable_grad():
            z = x[free].clone().requires_grad_()
            def objective(z):
                value = targets.index_copy(0, free, z)
                inertia = (mass[:, None]*(value-predicted).square()).sum()/(2*dt*dt)
                return inertia + elastic_energy(value, rest, tets, log_e)
            def merit(z):
                # Warp uses atomic summation. Near convergence, reduction noise
                # can exceed the expected Newton decrease. Compare candidates
                # with the same PyTorch expression used by the derivative bridge.
                value = targets.index_copy(0, free, z)
                inertia = (mass[:, None]*(value-predicted).square()).sum()/(2*dt*dt)
                return inertia + torch_elastic_energy(value, rest, tets, log_e)
            def valid(candidate):
                value = targets.index_copy(0, free, candidate)
                return bool(torch.isfinite(value).all() and (volumes(value, tets) > 0).all())
            # Start from the best valid candidate: previous positions, the
            # inertial prediction, or a caller-provided guess (e.g. the same
            # step's solution from the previous optimization iteration).
            candidates = [c.detach()[free] for c in (x, predicted, guess) if c is not None]
            candidates = [c for c in candidates if valid(c)]
            if not candidates:
                raise FEMConvergenceError("FEM prescribed boundary produces an inverted initial guess")
            # Tolerance scale: the force imbalance at the previous positions
            # when they are valid, so a warm start cannot make the
            # convergence test stricter or looser.
            first = (float(torch.autograd.grad(objective(z), z)[0].norm())
                     if valid(z.detach()) else None)
            with torch.no_grad():
                energies = [merit(c) for c in candidates]
            z = candidates[int(torch.stack(energies).argmin())].clone().requires_grad_()
            if first is None:
                first = float(torch.autograd.grad(objective(z), z)[0].norm())
            tangent = _SparseTangent(tets, fixed, mass, dt)
            converged = False
            for newton in range(50):
                if notify:
                    notify(f"Newton {newton+1}/50 | evaluating energy and gradient | dt={dt:.6g}")
                loss = objective(z)
                grad = torch.autograd.grad(loss, z)[0]
                norm = float(grad.norm())
                if notify:
                    notify(f"Newton {newton+1}/50 | gradient norm={norm:.6g}")
                # Relative to the step's initial force imbalance: an absolute
                # threshold is unreachable for large boundary motions.
                if norm <= max(NEWTON_RELATIVE_TOLERANCE*first, NEWTON_ABSOLUTE_TOLERANCE):
                    converged = True
                    break
                # The exact tangent gives quadratic convergence. Fixed-corotated
                # elasticity is nonconvex, so fall back to element Hessians
                # projected to positive semidefinite when the exact step is not
                # a usable descent direction.
                current = targets.index_copy(0, free, z.detach())
                with torch.no_grad():
                    reference = merit(z.detach())
                accepted = None
                for kind, halvings in (("exact", 8), ("projected", 25)):
                    try:
                        solve = tangent.factorize(element_hessians(
                            current, rest, tets, log_e, project=kind == "projected"))
                        direction, residual = solve(-grad)
                    except RuntimeError:
                        continue
                    slope = (grad.detach()*direction).sum()
                    if not torch.isfinite(direction).all() or slope >= 0:
                        continue
                    if notify:
                        notify(f"Newton {newton+1}/50 | {kind} sparse solve residual={residual:.6g}")
                    alpha = 1.
                    for search in range(halvings):
                        if notify:
                            notify(f"Newton {newton+1}/50 | line search {search+1}/{halvings} | alpha={alpha:.6g}")
                        trial = (z+alpha*direction).detach()
                        value = targets.index_copy(0, free, trial)
                        with torch.no_grad():
                            acceptable = ((volumes(value, tets)>0).all()
                                          and merit(trial) <= reference + 1e-4*alpha*slope)
                        if acceptable:
                            accepted = trial
                            break
                        alpha *= .5
                    if accepted is not None:
                        break
                if accepted is None:
                    raise FEMConvergenceError(
                        f"FEM line search failed (gradient norm={float(grad.norm()):.6g}, dt={dt:.6g})")
                z = accepted.requires_grad_()
            if not converged:
                raise FEMConvergenceError("FEM nonlinear solve did not converge")
            result = targets.index_copy(0, free, z).detach()
            if notify:
                notify(f"Newton converged at {newton+1}/50")
        ctx.save_for_backward(rest, tets, mass, x, velocity, log_e, fixed, targets, result)
        ctx.dt = dt
        return result

    @staticmethod
    def backward(ctx, upstream):
        rest, tets, mass, x, velocity, log_e, fixed, targets, result = ctx.saved_tensors
        dt = ctx.dt
        free = (~fixed).nonzero().flatten()
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            velocity = velocity.detach().requires_grad_()
            log_e = log_e.detach().requires_grad_()
            targets = targets.detach().requires_grad_()
            z = result[free].detach().requires_grad_()
            value = targets.index_copy(0, free, z)
            energy = (mass[:, None]*(value-x-dt*velocity).square()).sum()/(2*dt*dt)
            energy = energy+elastic_energy(value, rest, tets, log_e)
            residual = torch.autograd.grad(energy, z, create_graph=True)[0]
            notify = ctx.notify
            if notify:
                notify("backward | computing implicit adjoint")
            # The exact (unprojected) tangent gives the exact implicit derivative.
            tangent = _SparseTangent(tets, fixed, mass, dt)
            try:
                solve = tangent.factorize(element_hessians(result, rest, tets, log_e.detach()))
            except RuntimeError:
                # Singular exact tangent; the projected one is the best available.
                solve = tangent.factorize(element_hessians(result, rest, tets, log_e.detach(), project=True))
            adjoint, error = solve(upstream[free])
            if notify:
                notify(f"backward adjoint | sparse solve residual={error:.6g}")
            grads = torch.autograd.grad(residual, (x, velocity, log_e, targets),
                                        -adjoint, allow_unused=True)
        gx, gv, ge, gt = grads
        gt = torch.zeros_like(targets) if gt is None else gt
        gt[fixed] += upstream[fixed]
        return None, None, None, gx, gv, ge, None, gt, None, None, None


def matrix_free_step(rest, tets, mass, x, velocity, log_e, dt, fixed, targets, notify=None,
                     guess=None):
    """One implicit Euler step. `guess` only changes where Newton starts."""
    result = _ImplicitFEM.apply(rest, tets, mass, x, velocity, log_e, fixed, targets, dt,
                                notify, guess)
    return result, (result-x)/dt


def adaptive_step(rest, tets, mass, x, velocity, log_e, dt, fixed, targets,
                  max_halvings=8, notify=None, guess=None):
    """Retry failed steps by advancing through smaller boundary increments."""
    try:
        return matrix_free_step(rest, tets, mass, x, velocity, log_e, dt, fixed, targets, notify,
                                guess)
    except FEMConvergenceError as error:
        if max_halvings <= 0:
            raise FEMConvergenceError(f"FEM adaptive substeps exhausted: {error}") from error
        reason = str(error)
    # Leave the exception handler before retrying so failed Newton graphs are
    # released rather than retained by tracebacks throughout the recursive solve.
    if notify:
        notify(f"retrying FEM step with dt={dt/2:.6g} ({reason})")
    halfway = x.lerp(targets, .5)
    middle, mid_velocity = adaptive_step(rest, tets, mass, x, velocity, log_e,
        dt/2, fixed, halfway, max_halvings-1, notify)
    return adaptive_step(rest, tets, mass, middle, mid_velocity, log_e,
        dt/2, fixed, targets, max_halvings-1, notify)


def initial_state(rest, tets, mass, log_e, dt, fixed, targets,
                  max_halvings=8, notify=None, guess=None):
    """Move the entire volume into the initial pose before prescribing nodes.

    A rigid fit preserves all volumes. The remaining nonrigid boundary motion
    is applied by the same adaptive solver as subsequent animation frames.
    `guess` (e.g. a previous solution of this pose) only changes where the
    first Newton solve starts.
    """
    source, destination = rest[fixed], targets[fixed]
    source_center, target_center = source.mean(0), destination.mean(0)
    u, _, vh = torch.linalg.svd((source-source_center).T @ (destination-target_center))
    correction = torch.eye(3, dtype=rest.dtype, device=rest.device)
    correction[-1, -1] = torch.linalg.det(u @ vh).sign()
    rotation = u @ correction @ vh
    posed = (rest-source_center) @ rotation + target_center
    state, _ = adaptive_step(rest, tets, mass, posed, torch.zeros_like(rest),
        log_e, dt, fixed, targets, max_halvings, notify, guess)
    return state, torch.zeros_like(state)
