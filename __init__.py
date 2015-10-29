__author__ = 'H'
bl_info = {
    'name': 'Caffe-Gui-Tool',
    'author': 'Hugh Tomkins',
    'location': 'Node view - Properties panel',
    'category': 'Node View'
}

# To support reload properly, try to access a package var,
# # if it's there, reload everything
if "bpy" in locals():
    import imp

    imp.reload(CaffeGenerate)
    imp.reload(CaffeNodes)
    imp.reload(Loadproto)
    imp.reload(Arrange)
    imp.reload(graph)
    print("Reloaded multifiles")
else:
    from . import CaffeGenerate, CaffeNodes, Loadproto, Arrange, graph

    print("Imported multifiles")

import bpy
import random
from bpy.props import *


############# This class is needed for a blender init script to be simple
def initSceneProperties():
    bpy.types.Scene.traintest = bpy.props.StringProperty(
        name="Train Test Prototxt",
        default="",
        description="Get the path to the data",
        subtype='FILE_PATH'
    )

    bpy.types.Scene.solver = bpy.props.StringProperty(
        name="Solver Prototxt",
        default="",
        description="Get the path to the data",
        subtype='FILE_PATH'
    )
    bpy.types.Scene.deploy = bpy.props.StringProperty(
        name="Deploy (optional) Prototxt",
        default="",
        description="Get the path to the data",
        subtype='FILE_PATH'
    )
    return


initSceneProperties()


class DialogPanel(bpy.types.Panel):
    bl_label = "Caffe Gui Tool"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"

    def draw(self, context):
        scn = context.scene
        self.layout.operator("nodes.make_solver")
        self.layout.label("Loading Options")
        self.layout.prop(scn, "traintest")
        self.layout.prop(scn, "solver")
        self.layout.prop(scn, "deploy")
        self.layout.operator("nodes.load_solver")


def register():
    bpy.utils.register_module(__name__)
    Arrange.register()
    CaffeGenerate.register()
    CaffeNodes.register()
    Loadproto.register()


def unregister():
    Arrange.unregister()
    CaffeGenerate.unregister()
    CaffeNodes.unregister()
    bpy.utils.unregister_module(__name__)


if __name__ == "__main__":
    register()
