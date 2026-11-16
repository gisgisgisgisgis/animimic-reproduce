"""Warp energy kernels with a PyTorch analytic tangent/implicit-adjoint bridge.

Warp evaluates the nonlinear FEM objective on the GPU; PyTorch differentiates
the identical fixed-corotated expression for Hessian-vector products. This
avoids differentiating through Newton/CG iterations or allocating dense Hessians.
"""
import torch
import warp as wp
from pathlib import Path

wp.config.kernel_cache_dir = str(Path(".cache/warp").resolve())
wp.init()


@wp.kernel
def energy_kernel(x: wp.array(dtype=wp.vec3d), rest: wp.array(dtype=wp.vec3d),
                  tets: wp.array(dtype=wp.vec4i), log_e: wp.array(dtype=wp.float64),
                  poisson: wp.float64, result: wp.array(dtype=wp.float64)):
    i = wp.tid()
    ids = tets[i]
    ds = wp.matrix_from_cols(x[ids[1]]-x[ids[0]], x[ids[2]]-x[ids[0]], x[ids[3]]-x[ids[0]])
    dm = wp.matrix_from_cols(rest[ids[1]]-rest[ids[0]], rest[ids[2]]-rest[ids[0]],
                             rest[ids[3]]-rest[ids[0]])
    deformation = ds * wp.inverse(dm)
    rotation = deformation
    for _ in range(12):
        rotation = wp.float64(0.5)*(rotation+wp.transpose(wp.inverse(rotation)))
    difference = deformation-rotation
    squared = wp.float64(0.0)
    for row in range(3):
        for col in range(3):
            squared += difference[row, col]*difference[row, col]
    modulus = wp.exp(log_e[i])
    mu = modulus/(wp.float64(2.0)*(wp.float64(1.0)+poisson))
    lam = modulus*poisson/((wp.float64(1.0)+poisson)*(wp.float64(1.0)-wp.float64(2.0)*poisson))
    determinant = wp.determinant(deformation)-wp.float64(1.0)
    energy = (mu*squared+lam/wp.float64(2.0)*determinant*determinant)*wp.abs(wp.determinant(dm))/wp.float64(6.0)
    wp.atomic_add(result, 0, energy)


class WarpEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rest, tets, log_e, poisson):
        ctx.save_for_backward(x, rest, tets, log_e)
        ctx.poisson = poisson
        # All simulation tensors are float64; vec4i indices must be int32.
        indices = tets.to(torch.int32).contiguous()
        scalar = torch.zeros(1, dtype=torch.float64, device=x.device)
        wp.launch(energy_kernel, dim=len(tets), inputs=[
            wp.from_torch(x.contiguous(), dtype=wp.vec3d, requires_grad=False),
            wp.from_torch(rest.contiguous(), dtype=wp.vec3d, requires_grad=False),
            wp.from_torch(indices, dtype=wp.vec4i, requires_grad=False),
            wp.from_torch(log_e.contiguous(), requires_grad=False), float(poisson),
            wp.from_torch(scalar, requires_grad=False)],
            device=str(x.device))
        return scalar[0]

    @staticmethod
    def backward(ctx, upstream):
        from .fem import torch_elastic_energy
        x, rest, tets, log_e = ctx.saved_tensors
        higher_order = torch.is_grad_enabled()
        with torch.enable_grad():
            position = x if x.requires_grad else x.detach().requires_grad_()
            parameter = log_e if log_e.requires_grad else log_e.detach().requires_grad_()
            energy = torch_elastic_energy(position, rest, tets, parameter, ctx.poisson)
            gx, ge = torch.autograd.grad(energy, (position, parameter),
                                         create_graph=higher_order)
        return upstream*gx, None, None, upstream*ge, None
