__author__ = 'H'

import os
import pickle

from .IOloadprototxt import LoadFunction
import bpy
from .CGTArrangeHelper import ArrangeFunction


class anyclass(object):
    pass


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


def LoadCexpFunction(context, tempdata, datacube, loadloss, loadtree):
    datalist = pickle.load(open(tempdata, 'rb'))
    train_graph, test_graph, protodata, comment = datalist
    prototxt, Isize = protodata
    nh, nw, h, w = Isize
    datacube.name = os.path.split(tempdata)[1]
    datacube["comment"] = comment
    datacube["cexp"] = tempdata
    context.user_preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
    if loadloss:
        for loss, iteration in train_graph:
            datacube["train"] = loss
            context.user_preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
            datacube.keyframe_insert(data_path='["train"]', frame=(iteration))
        for loss, iteration in test_graph:
            datacube["test"] = loss
            context.user_preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
            datacube.keyframe_insert(data_path='["test"]', frame=(iteration))
    if loadtree:
        datacube["originaltree"] = os.path.split(tempdata)[1]
        ## Load node graph from cexp
        prevtrees = bpy.data.node_groups.items()
        LoadFunction(prototxt, 0, 0, nh, nw, h, w)
        newtrees = bpy.data.node_groups.items()
        tree = list(set(newtrees) - set(prevtrees))[0][1]
        tree.name = os.path.split(tempdata)[1]
        ArrangeFunction(context, treename=tree.name)


class LoadCexp(bpy.types.Operator):
    """Load Experiment (.cexp) file"""
    bl_idname = "nodes.load_trained_solver"
    bl_label = "Load .cexp"
    bl_options = {'REGISTER'}

    def execute(self, context):
        cscene = bpy.context.scene
        # Output path
        if 'loadtempdata' in cscene:
            tempdata = cscene['loadtempdata']
        else:
            self.report({'ERROR'}, "No .cexp path set")
            return {'FINISHED'}
        oldcubes = bpy.data.objects.items()
        bpy.ops.mesh.primitive_cube_add()
        newcubes = bpy.data.objects.items()
        datacube = list(set(newcubes) - set(oldcubes))[0][1]
        datacube.select = True
        datacube["originaltree"] = ''
        if 'loadloss' not in cscene or 'loadtree' not in cscene:
            self.report({'ERROR'}, "Blender bug: Please Double toggle the 'Load Loss graphs' and 'Load Node tree' toggles")
        LoadCexpFunction(context, tempdata, datacube, cscene["loadloss"], cscene["loadtree"])
        return {'FINISHED'}  # this lets blender know the operator finished successfully.


class LoadLossTree(bpy.types.Operator):
    """Load Experiment (.cexp) file"""
    bl_idname = "nodes.load_tree_from_curve"
    bl_label = "Load Original Node tree"
    bl_options = {'REGISTER'}

    def execute(self, context):
        activecurve, activeobject = getactivefcurve()
        if activecurve:
            tempdata = activeobject["cexp"]
            LoadCexpFunction(context, tempdata, activeobject, 0, 1)
        else:
            self.report({'ERROR'}, "Please select a single curve")
            return {'FINISHED'}
        return {'FINISHED'}  # this lets blender know the operator finished successfully.


def register():
    bpy.utils.register_class(LoadCexp)
    bpy.utils.register_class(LoadLossTree)


def unregister():
    bpy.utils.unregister_class(LoadCexp)
    bpy.utils.unregister_class(LoadLossTree)
