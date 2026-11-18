"""Fixed-corotated FEM with implicit Euler and implicit-adjoint gradients.

Dense reference solver for gradient validation and small tetrahedral problems.
The Hessian is solved implicitly, rather than differentiating Newton iterations.
"""
import torch
from contextvars import ContextVar

BACKEND = ContextVar("fem_backend", default="torch")


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


def cg(operator, rhs, tolerance=1e-8, iterations=200, notify=None):
    value = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    norm = (residual*residual).sum()
    for iteration in range(iterations):
        if notify and iteration % 10 == 0:
            notify(f"CG {iteration}/{iterations} | residual={float(norm.sqrt()):.6g}")
        if norm.sqrt() <= tolerance:
            if notify:
                notify(f"CG converged at {iteration}/{iterations} | residual={float(norm.sqrt()):.6g}")
            return value
        product = operator(direction)
        denominator = (direction*product).sum()
        if denominator <= 0:
            raise RuntimeError("FEM tangent is not positive definite")
        alpha = norm/denominator
        value = value + alpha*direction
        residual = residual-alpha*product
        next_norm = (residual*residual).sum()
        direction = residual + next_norm/norm*direction
        norm = next_norm
    raise RuntimeError("FEM linear solve did not converge")


class _ImplicitFEM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rest, tets, mass, x, velocity, log_e, fixed, targets, dt, notify):
        ctx.notify = notify
        free = (~fixed).nonzero().flatten()
        predicted = x + dt*velocity
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
            initial = targets.index_copy(0, free, z.detach())
            if not torch.isfinite(initial).all() or not (volumes(initial, tets)>0).all():
                raise FEMConvergenceError("FEM prescribed boundary produces an inverted initial guess")
            converged = False
            for newton in range(50):
                if notify:
                    notify(f"Newton {newton+1}/50 | evaluating energy and gradient | dt={dt:.6g}")
                loss = objective(z)
                grad = torch.autograd.grad(loss, z, create_graph=True)[0]
                if notify:
                    notify(f"Newton {newton+1}/50 | gradient norm={float(grad.norm()):.6g}")
                if grad.norm() < 1e-6:
                    converged = True
                    break
                damping = 1e-5
                for _ in range(12):
                    def multiply(v):
                        return torch.autograd.grad(grad, z, v, retain_graph=True)[0] + damping*v
                    try:
                        def report_cg(message):
                            notify(f"Newton {newton+1}/50 | damping={damping:.6g} | {message}")
                        direction = cg(multiply, -grad.detach(), tolerance=1e-7,
                                       notify=report_cg if notify else None)
                        break
                    except RuntimeError:
                        damping *= 10
                else:
                    direction = -grad.detach()/grad.detach().norm().clamp_min(1)
                alpha = 1.
                with torch.no_grad():
                    reference = merit(z.detach())
                    slope = (grad.detach()*direction).sum()
                for search in range(25):
                    if notify:
                        notify(f"Newton {newton+1}/50 | line search {search+1}/25 | alpha={alpha:.6g}")
                    trial = (z+alpha*direction).detach()
                    value = targets.index_copy(0, free, trial)
                    with torch.no_grad():
                        acceptable = ((volumes(value, tets)>0).all()
                                      and merit(trial) <= reference + 1e-4*alpha*slope)
                    if acceptable:
                        break
                    alpha *= .5
                else:
                    raise FEMConvergenceError(
                        f"FEM line search failed (gradient norm={float(grad.norm()):.6g}, dt={dt:.6g})")
                z = trial.requires_grad_()
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
            def multiply(v):
                return torch.autograd.grad(residual, z, v, retain_graph=True)[0]
            notify = ctx.notify
            if notify:
                notify("backward | computing implicit adjoint")
            adjoint = cg(multiply, upstream[free], tolerance=1e-7,
                         notify=(lambda message: notify(f"backward adjoint | {message}")) if notify else None)
            grads = torch.autograd.grad(residual, (x, velocity, log_e, targets),
                                        -adjoint, allow_unused=True)
        gx, gv, ge, gt = grads
        gt = torch.zeros_like(targets) if gt is None else gt
        gt[fixed] += upstream[fixed]
        return None, None, None, gx, gv, ge, None, gt, None, None


def matrix_free_step(rest, tets, mass, x, velocity, log_e, dt, fixed, targets, notify=None):
    result = _ImplicitFEM.apply(rest, tets, mass, x, velocity, log_e, fixed, targets, dt, notify)
    return result, (result-x)/dt


def adaptive_step(rest, tets, mass, x, velocity, log_e, dt, fixed, targets,
                  max_halvings=8, notify=None):
    """Retry failed steps by advancing through smaller boundary increments."""
    try:
        return matrix_free_step(rest, tets, mass, x, velocity, log_e, dt, fixed, targets, notify)
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
                  max_halvings=8, notify=None):
    """Move the entire volume into the initial pose before prescribing nodes.

    A rigid fit preserves all volumes. The remaining nonrigid boundary motion
    is applied by the same adaptive solver as subsequent animation frames.
    """
    source, destination = rest[fixed], targets[fixed]
    source_center, target_center = source.mean(0), destination.mean(0)
    u, _, vh = torch.linalg.svd((source-source_center).T @ (destination-target_center))
    correction = torch.eye(3, dtype=rest.dtype, device=rest.device)
    correction[-1, -1] = torch.linalg.det(u @ vh).sign()
    rotation = u @ correction @ vh
    posed = (rest-source_center) @ rotation + target_center
    state, _ = adaptive_step(rest, tets, mass, posed, torch.zeros_like(rest),
        log_e, dt, fixed, targets, max_halvings, notify)
    return state, torch.zeros_like(state)
