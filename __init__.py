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

    imp.reload(IOwriteprototxt)
    imp.reload(IOcexp)
    imp.reload(CGTNodes)
    imp.reload(IOloadprototxt)
    imp.reload(CGTArrangeHelper)
    imp.reload(CGTGraph)
    print("Reloaded multifiles")
else:
    from . import IOwriteprototxt, CGTNodes, IOloadprototxt, CGTArrangeHelper, CGTGraph, IOcexp

    print("Imported multifiles")

import bpy
import random
from bpy.props import *


def getactivefcurve():
    ncurves = 0
    for object in bpy.context.selected_objects:
        if object.animation_data:
            if object.animation_data.action:
                for curve in object.animation_data.action.fcurves.items():
                    if curve[1].select:
                        ncurves += 1
                        activeobject = object
                        activecurve = curve[1]
    if ncurves == 1:
        return activecurve, activeobject
    elif ncurves == 0:
        return None, None
    else:
        return False, False


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
    bpy.types.Scene.savetempdata = bpy.props.StringProperty(
        name="Log folder",
        default="",
        description="Folder in which to store saved graphs and log data",
        subtype='DIR_PATH'
    )
    bpy.types.Scene.loadtempdata = bpy.props.StringProperty(
        name="Log file to Load",
        default="",
        description="File to load tree and curve from",
        subtype='FILE_PATH'
    )
    bpy.types.Scene.comment = bpy.props.StringProperty(
        name="Comment",
        default="",
        description="Add a comment that helps identify the current experiment"
    )
    bpy.types.Scene.filecomment = bpy.props.StringProperty(
        name="Filename",
        default="",
        description="Add a string to beginning of filename to describe experiment"
    )
    bpy.types.Scene.loadtree = bpy.props.BoolProperty(
        name="Load Node tree",
        default=1,
        description="Load the node tree from .cexp"
    )
    bpy.types.Scene.loadloss = bpy.props.BoolProperty(
        name="Load Loss graphs",
        default=1,
        description="Load the loss data from .cexp"
    )
    bpy.types.Scene.donetraining = bpy.props.IntProperty(default=1)
    return


initSceneProperties()


class LoadDialogPanel(bpy.types.Panel):
    bl_label = "Load Prototxt"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"

    def draw(self, context):
        scn = context.scene
        self.layout.prop(scn, "traintest")
        self.layout.prop(scn, "solver")
        self.layout.prop(scn, "deploy")
        self.layout.operator("nodes.load_solver")


class RunDialogPanel(bpy.types.Panel):
    bl_label = "Run Caffe"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"

    def draw(self, context):
        scn = bpy.context.scene
        self.layout.operator("nodes.make_solver")
        self.layout.prop(scn, "savetempdata")
        self.layout.prop(scn, "filecomment")
        self.layout.prop(scn, "comment")
        self.layout.operator("nodes.run_solver")
        self.layout.operator("nodes.cancel_solver")


class GraphInfoPanel(bpy.types.Panel):
    bl_label = "Selected loss plot"
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"

    def draw(self, context):
        activecurve, activeobject = getactivefcurve()
        if activecurve == None:
            self.layout.label("No curve selected")
        elif not activecurve:
            self.layout.label("Multiple curves selected")
            self.layout.label("Select a single curve to view comments")
        else:
            try:
                self.layout.label(activeobject["comment"])
            except KeyError:
                self.layout.label("No comment")
            self.layout.operator("nodes.load_tree_from_curve")
            if activeobject["originaltree"] != '':
                self.layout.label("Original tree loaded to:")
                self.layout.label(activeobject["originaltree"])


class CexpLoadPanel(bpy.types.Panel):
    bl_label = "Load experiment"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"

    def draw(self, context):
        scn = context.scene
        self.layout.prop(scn, "loadtempdata")
        self.layout.prop(scn, "loadtree")
        self.layout.prop(scn, "loadloss")
        self.layout.operator("nodes.load_trained_solver")


def register():
    bpy.utils.register_class(RunDialogPanel)
    bpy.utils.register_class(LoadDialogPanel)
    bpy.utils.register_class(CexpLoadPanel)
    bpy.utils.register_class(GraphInfoPanel)
    # bpy.utils.register_module(__name__)
    CGTArrangeHelper.register()
    CGTGraph.register()
    IOwriteprototxt.register()
    CGTNodes.register()
    IOloadprototxt.register()
    IOcexp.register()


def unregister():
    bpy.utils.unregister_class(RunDialogPanel)
    bpy.utils.unregister_class(LoadDialogPanel)
    bpy.utils.unregister_class(CexpLoadPanel)
    bpy.utils.unregister_class(GraphInfoPanel)
    CGTArrangeHelper.unregister()
    CGTGraph.unregister()
    IOwriteprototxt.unregister()
    CGTNodes.unregister()
    IOcexp.unregister()
    # bpy.utils.unregister_module(__name__)
