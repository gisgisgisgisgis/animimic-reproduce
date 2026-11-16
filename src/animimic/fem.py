"""Fixed-corotated FEM with implicit Euler and implicit-adjoint gradients.

Dense reference solver for gradient validation and small tetrahedral problems.
The Hessian is solved implicitly, rather than differentiating Newton iterations.
"""
import torch
from contextvars import ContextVar

BACKEND = ContextVar("fem_backend", default="torch")


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


def cg(operator, rhs, tolerance=1e-8, iterations=200):
    value = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    norm = (residual*residual).sum()
    for _ in range(iterations):
        if norm.sqrt() <= tolerance:
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
    def forward(ctx, rest, tets, mass, x, velocity, log_e, fixed, targets, dt):
        free = (~fixed).nonzero().flatten()
        predicted = x + dt*velocity
        with torch.enable_grad():
            z = x[free].clone().requires_grad_()
            def objective(z):
                value = targets.index_copy(0, free, z)
                inertia = (mass[:, None]*(value-predicted).square()).sum()/(2*dt*dt)
                return inertia + elastic_energy(value, rest, tets, log_e)
            converged = False
            for _ in range(50):
                loss = objective(z)
                grad = torch.autograd.grad(loss, z, create_graph=True)[0]
                if grad.norm() < 1e-6:
                    converged = True
                    break
                damping = 1e-5
                for _ in range(12):
                    def multiply(v):
                        return torch.autograd.grad(grad, z, v, retain_graph=True)[0] + damping*v
                    try:
                        direction = cg(multiply, -grad.detach(), tolerance=1e-7)
                        break
                    except RuntimeError:
                        damping *= 10
                else:
                    direction = -grad.detach()/grad.detach().norm().clamp_min(1)
                alpha = 1.
                for _ in range(25):
                    trial = (z+alpha*direction).detach()
                    value = targets.index_copy(0, free, trial)
                    if (volumes(value, tets)>0).all() and objective(trial) < loss:
                        break
                    alpha *= .5
                else:
                    raise RuntimeError("FEM line search failed")
                z = trial.requires_grad_()
            if not converged:
                raise RuntimeError("FEM nonlinear solve did not converge")
            result = targets.index_copy(0, free, z).detach()
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
            adjoint = cg(multiply, upstream[free], tolerance=1e-7)
            grads = torch.autograd.grad(residual, (x, velocity, log_e, targets),
                                        -adjoint, allow_unused=True)
        gx, gv, ge, gt = grads
        gt = torch.zeros_like(targets) if gt is None else gt
        gt[fixed] += upstream[fixed]
        return None, None, None, gx, gv, ge, None, gt, None


def matrix_free_step(rest, tets, mass, x, velocity, log_e, dt, fixed, targets):
    result = _ImplicitFEM.apply(rest, tets, mass, x, velocity, log_e, fixed, targets, dt)
    return result, (result-x)/dt
