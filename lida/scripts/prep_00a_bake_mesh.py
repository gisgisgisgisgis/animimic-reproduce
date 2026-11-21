"""Bake an input asset to world space, with no armature and no node transforms. Run with UniRig's Python.

UniRig's extract step reads world-space vertices, but its merge step can export the mesh and the
new skeleton in different frames when the mesh sits under a transformed node. The result looks
fine in a render and deforms wrongly. With every transform baked in, both steps see one frame.

Usage: python prep_00a_bake_mesh.py input.glb output.glb
"""
import os
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def load(path):
    suffix = Path(path).suffix.lower()
    if suffix in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path, import_pack_images=False)
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path, ignore_leaf_bones=False, use_image_search=False)
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise ValueError(f"Unsupported input type: {suffix}")


def main():
    source, target = sys.argv[1:3]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    load(source)
    meshes = [ob for ob in bpy.data.objects if ob.type == "MESH"]
    if not meshes:
        raise ValueError(f"{source}: no mesh found")

    # Read every world matrix first, since clearing a parent changes the children's matrices.
    world = {ob.name: ob.matrix_world.copy() for ob in meshes}
    for ob in meshes:
        ob.parent = None
        ob.modifiers.clear()
        ob.vertex_groups.clear()
        if ob.data.users > 1:
            ob.data = ob.data.copy()
        ob.data.transform(world[ob.name])
        if world[ob.name].determinant() < 0:
            ob.data.flip_normals()
        ob.matrix_world = Matrix.Identity(4)
    for ob in list(bpy.data.objects):
        if ob.type != "MESH":
            bpy.data.objects.remove(ob)

    bpy.ops.export_scene.gltf(filepath=target, export_format="GLB", export_skins=False, export_animations=False)
    print(f"Wrote {target}", flush=True)
    os._exit(0)  # bpy segfaults during normal interpreter shutdown on macOS, after the file is written


if __name__ == "__main__":
    main()
