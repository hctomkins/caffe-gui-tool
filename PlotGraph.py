__author__ = 'Hugh'

import os
from .parse import search as findfirstraw
import datetime
import bpy
import pickle
import random
import subprocess
import string


def findfirst(search, string):
    searchresult = findfirstraw(search, string)
    if searchresult:
        toset = searchresult.fixed[0]
        return toset


def format_filename(s):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ','_') # I don't like spaces in filenames.
    return filename


def get_loss(line):
    line = line[line.index('=')+2:]
    line = line[:line.index('(')-1]
    loss = float(line)
    return loss


class TrainPlot(bpy.types.Operator):
    """Train, save .cexp, plot Cexp"""
    bl_idname = "nodes.run_solver"
    bl_label = "Train log and plot solution"
    bl_options = {'REGISTER'}

    def execute(self, context):
        train_graph = []
        test_graph = []
        bpy.ops.nodes.make_solver('EXEC_DEFAULT')

        # Get active tree
        for space in bpy.context.area.spaces:
            if space.type == 'NODE_EDITOR':
                tree = space.edit_tree

        # Get output destination, and solver file
        for node in context.selected_nodes:
            if node.bl_idname == 'SolverNodeType':
                solvername = node.solvername
                error = False
                output_dest = node.config_path
                solverfile = 'train_%s.sh'%node.solvername
                # TODO:Raise no solver node error
        if error:
            raise  OSError

        # Because blender is a 3d graphics editor, this hacky method stores all the data associated with
        # the current experiment inside a cube object, in the 3d editor.
        # When we need to draw the next point in the graph, we save the current data, load it, and plot it.
        oldcubes = bpy.data.objects.items()
        bpy.ops.mesh.primitive_cube_add()
        newcubes = bpy.data.objects.items()
        datacube = list(set(newcubes) - set(oldcubes))[0][1]
        datacube.select = True
        cscene = bpy.context.scene
        # Output path
        if 'tempdata' in cscene:
            tempdata = cscene['savetempdata']
            #os.chdir(tempdata)
        else:
            raise  OSError # Todo: proper no path error
        if 'comment' in cscene:
            comment = cscene['comment']
        else:
            comment = 'No comment for current graph'
        if 'filecomment' in cscene:
            filecomment = cscene['filecomment']
        else:
            filecomment = ''
        dumpdata = [train_graph,test_graph,tree,comment]
        currenttime = datetime.datetime.now()
        filename = format_filename(filecomment) + currenttime.strftime("_%Y.%m.%d_%H.%M_") + '.cexp'
        pickle.dump(dumpdata,open(os.path.join(tempdata,filename),'wb'),protocol=2)
        datacube.name = filename
        datacube["comment"] = comment
        # Do the training
        iteration = 0
        context.user_preferences.edit.keyframe_new_interpolation_type ='LINEAR'
        proc = subprocess.Popen(os.path.join(output_dest,solverfile), shell=False, stderr=subprocess.PIPE)
        while proc.poll() is None:
            output = proc.stderr.readline()
            output = output[:-1].decode("utf8")
            print (output)
            if 'Iteration' in output:
                iteration = findfirst('Iteration {:g},',output)
            if 'output' in output and 'loss' in output:
                loss = get_loss(output)
                if 'Test' in output:
                    datacube["test"] = loss
                    test_graph.append([loss,iteration])
                    context.user_preferences.edit.keyframe_new_interpolation_type ='LINEAR'
                    datacube.keyframe_insert(data_path='["test"]', frame=(iteration))
                elif 'Train' in output:
                    datacube["train"] = loss
                    train_graph.append([loss,iteration])
                    context.user_preferences.edit.keyframe_new_interpolation_type ='LINEAR'
                    datacube.keyframe_insert(data_path='["train"]', frame=(iteration))
                else:
                    raise OSError #TODO: WTF error
                dumpdata = [train_graph,test_graph,tree,comment]
                pickle.dump(dumpdata,open(os.path.join(tempdata,filename),'wb'),protocol=2)
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                # ^^^ This is hacky but would have to be replaced by 50 lines of code to be 'proper'
        bpy.ops.object.select_all()
        bpy.ops.object.select_all()
        return {'FINISHED'}  # this lets blender know the operator finished successfully.


class LoadPlot(bpy.types.Operator):
    """Load Experiment (.cexp) file"""
    bl_idname = "nodes.load_trained_solver"
    bl_label = "Load .cexp"
    bl_options = {'REGISTER'}

    def execute(self, context):
        oldcubes = bpy.data.objects.items()
        bpy.ops.mesh.primitive_cube_add()
        newcubes = bpy.data.objects.items()
        datacube = list(set(newcubes) - set(oldcubes))[0][1]
        datacube.select = True
        cscene = bpy.context.scene
        # Output path
        if 'loadtempdata' in cscene:
            tempdata = cscene['loadtempdata']
        else:
            raise  OSError # Todo: proper no path error
        train_graph,test_graph,tree,comment = pickle.load(open(tempdata,'rb'))
        datacube.name = os.path.split(tempdata)[1]
        datacube["comment"] = comment
        context.user_preferences.edit.keyframe_new_interpolation_type ='LINEAR'
        for loss,iteration in train_graph:
            context.user_preferences.edit.keyframe_new_interpolation_type ='LINEAR'
            datacube.keyframe_insert(data_path='["train"]', frame=(iteration))
        for loss,iteration in test_graph:
            context.user_preferences.edit.keyframe_new_interpolation_type ='LINEAR'
            datacube.keyframe_insert(data_path='["test"]', frame=(iteration))

        ## Load node graph from cexp
        prevtrees = bpy.data.node_groups.items()
        bpy.ops.node.new_node_tree()
        newtrees = bpy.data.node_groups.items()
        temptree = list(set(newtrees) - set(prevtrees))[0][1]
        temptree.name = 'temp999'
        bpy.data.node_groups[temptree.name] = tree
        return {'FINISHED'}  # this lets blender know the operator finished successfully.


def register():
    bpy.utils.register_class(TrainPlot)
    bpy.utils.register_class(LoadPlot)


def unregister():
    bpy.utils.unregister_class(TrainPlot)
    bpy.utils.unregister_class(LoadPlot)