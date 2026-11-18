"""Fixed-camera PyTorch3D rendering and surface-point projection."""
import torch


class Renderer:
    def __init__(self, device, width=512, height=288, azimuth=0,
                 distance=2.2, elevation=10, fov=35, lighting="legacy", srgb=False,
                 use_texture=False):
        from pytorch3d.renderer import (
            FoVPerspectiveCameras, look_at_view_transform, RasterizationSettings,
            MeshRasterizer, MeshRenderer, SoftPhongShader, SoftSilhouetteShader,
            BlendParams, PointLights,
        )
        self.settings = dict(width=width, height=height, azimuth=azimuth,
                             distance=distance, elevation=elevation, fov=fov,
                             lighting=lighting, srgb=srgb, use_texture=use_texture)
        self.srgb = srgb
        self.use_texture = use_texture
        self.material = None
        r, t = look_at_view_transform(distance, elevation, azimuth, device=device)
        self.camera = FoVPerspectiveCameras(device=device, R=r, T=t,
                                           fov=fov, aspect_ratio=1.0)
        self.size = (height, width)
        self.rasterizer = MeshRasterizer(cameras=self.camera,
            raster_settings=RasterizationSettings(image_size=self.size,
                                                   blur_radius=0, faces_per_pixel=1))
        if lighting == "studio":
            lights = PointLights(device=device, location=self.camera.get_camera_center(),
                ambient_color=((.8, .8, .8),), diffuse_color=((.2, .2, .2),),
                specular_color=((0., 0., 0.),))
        elif lighting == "unlit":
            # Some game assets bake lighting into their base-color texture.
            lights = PointLights(device=device, ambient_color=((1., 1., 1.),),
                diffuse_color=((0., 0., 0.),), specular_color=((0., 0., 0.),))
        elif lighting == "legacy":
            lights = PointLights(device=device, location=[[0., 2., 3.]])
        else:
            raise ValueError(f"Unknown lighting: {lighting}")
        self.rgb = MeshRenderer(self.rasterizer, SoftPhongShader(
            device=device, cameras=self.camera, lights=lights,
            blend_params=BlendParams(background_color=(1, 1, 1))))
        soft = MeshRasterizer(cameras=self.camera,
            raster_settings=RasterizationSettings(image_size=self.size,
                blur_radius=1e-4, faces_per_pixel=10))
        self.mask = MeshRenderer(soft, SoftSilhouetteShader())

    def bind_material(self, rig):
        if self.use_texture:
            self.material = (rig["uv"], rig["texture"])

    def mesh(self, vertices, faces, colors):
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesVertex
        # A dense or deformed mesh can concentrate every face in one screen bin.
        # Reserve for the whole mesh, including near-plane clipping which can
        # split each triangle into two, for RGB, masks and surface sampling.
        capacity = max(1, 2 * len(faces))
        self.rasterizer.raster_settings.max_faces_per_bin = capacity
        self.mask.rasterizer.raster_settings.max_faces_per_bin = capacity
        if self.use_texture:
            from pytorch3d.renderer import TexturesUV
            if self.material is None:
                raise ValueError("Bind UV material before rendering a textured mesh")
            uv, texture = self.material
            appearance = TexturesUV(maps=texture[None], faces_uvs=faces[None], verts_uvs=uv[None])
        else:
            appearance = TexturesVertex(verts_features=colors[None])
        return Meshes([vertices], [faces], textures=appearance)

    def __call__(self, vertices, faces, colors):
        mesh = self.mesh(vertices, faces, colors)
        rgb = self.rgb(mesh)[0, ..., :3]
        if self.srgb:
            # glTF factors and vertex colors are linear; PNG/video displays sRGB.
            rgb = torch.where(rgb <= .0031308, 12.92*rgb,
                              1.055*rgb.clamp_min(1e-12).pow(1/2.4)-.055)
        return rgb, self.mask(mesh)[0, ..., 3]

    def project(self, points):
        return self.camera.transform_points_screen(points[None], image_size=[self.size])[0, ..., :2]

    def depth(self, points):
        return self.camera.get_world_to_view_transform().transform_points(points)[..., 2]

    def surface_samples(self, vertices, faces, colors, count=512):
        fragments = self.rasterizer(self.mesh(vertices, faces, colors))
        foreground = (fragments.pix_to_face[0, ..., 0] >= 0).float()
        eroded = -torch.nn.functional.max_pool2d(
            -foreground[None, None], 7, stride=1, padding=3)[0, 0] > 0.5
        if not eroded.any():
            eroded = foreground.bool()
        valid = eroded.nonzero()
        if not len(valid):
            raise ValueError("Mesh is outside camera")
        ids = torch.linspace(0, len(valid)-1, min(count, len(valid)),
                             device=vertices.device).long()
        y, x = valid[ids].unbind(-1)
        tri = fragments.pix_to_face[0, y, x, 0]
        beta = fragments.bary_coords[0, y, x, 0]
        return tri, beta, torch.stack((x, y), -1)
