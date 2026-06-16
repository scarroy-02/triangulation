# Converts the mesher's OBJ output into a real .blend file, headlessly:
#
#   blender --background --python obj_to_blend.py -- whitney_mesh.obj whitney_mesh.blend
#
# Works with Blender 3.x and 4.x.
import bpy
import sys

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(argv) != 2:
    print("usage: blender --background --python obj_to_blend.py -- in.obj out.blend")
    sys.exit(1)
obj_in, blend_out = argv

bpy.ops.wm.read_factory_settings(use_empty=True)
try:
    bpy.ops.wm.obj_import(filepath=obj_in)        # Blender >= 3.2
except AttributeError:
    bpy.ops.import_scene.obj(filepath=obj_in)     # legacy importer

for ob in bpy.data.objects:
    if ob.type == 'MESH':
        bpy.context.view_layer.objects.active = ob
        ob.select_set(True)
        bpy.ops.object.shade_smooth()
        # merge the duplicated vertices along shared flags (cosmetic)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=1e-9)
        bpy.ops.object.mode_set(mode='OBJECT')

bpy.ops.wm.save_as_mainfile(filepath=blend_out)
print(f"saved {blend_out}")