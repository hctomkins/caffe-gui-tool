__author__ = 'Hugh'

import os
import datetime
import pickle
import string
import sys
from subprocess import PIPE, Popen
from threading import Thread
from contextlib import redirect_stdout
import io
stdout = io.StringIO()
from .IOparse import search as findfirstraw
import bpy
from queue import Queue, Empty
from .IOwriteprototxt import SolveFunction

ON_POSIX = 'posix' in sys.builtin_module_names


def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()


def findfirst(search, string):
    searchresult = findfirstraw(search, string)
    if searchresult:
        toset = searchresult.fixed[0]
        return toset


def format_filename(s):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ', '_')  # I don't like spaces in filenames.
    return filename


def get_loss(line):
    nameline = line[line.index(']'):]
    nameline = nameline[nameline.index(':')+2:]
    nameline = nameline[:nameline.index('=')-1]
    line = line[line.index('=') + 2:]
    if '(' in line:
        line = line[:line.index('(') - 1]
    loss = float(line)
    return loss,nameline


class TrainPlot(bpy.types.Operator):
    """Train, save .cexp, plont Cexp"""
    bl_idname = "nodes.run_solver"
    bl_label = "Train log and plot solution"
    bl_options = {'REGISTER'}
    _timer = None

    def execute(self, context):
        scn = context.scene
        if "donetraining" not in scn:
            scn["donetraining"] = True
        if scn["donetraining"]:
            scn["donetraining"] = False
            wm = context.window_manager
            self._timer = wm.event_timer_add(0.01, context.window)
            wm.modal_handler_add(self)
            self.train_graph = []
            self.test_graph = []
            bpy.ops.nodes.make_solver('EXEC_DEFAULT')
            # Get active tree
            for space in bpy.context.area.spaces:
                if space.type == 'NODE_EDITOR':
                    self.tree = space.edit_tree

            # Get output destination, and solver file
            for node in context.selected_nodes:
                if node.bl_idname == 'SolverNodeType':
                    error = False
            if error:
                self.report({'ERROR'}, "No Solver Node added")
                return {'FINISHED'}
            # Because blender is a 3d graphics editor, this hacky method stores all the data associated with
            # the current experiment inside a cube object, in the 3d editor.
            # When we need to draw the next point in the graph, we save the current data, load it, and plot it.
            oldcubes = bpy.data.objects.items()
            bpy.ops.mesh.primitive_cube_add()
            newcubes = bpy.data.objects.items()
            self.datacube = list(set(newcubes) - set(oldcubes))[0][1]
            self.datacube.select = True
            cscene = bpy.context.scene

            ## Get scene params
            if 'tempdata' in cscene:
                self.tempdata = cscene['savetempdata']
            else:
                self.report({'ERROR'}, "Log Folder not set")
                return {'FINISHED'}
            if 'comment' in cscene:
                self.comment = cscene['comment']
            else:
                self.comment = 'No comment for current graph'
            if 'filecomment' in cscene:
                self.filecomment = cscene['filecomment']
            else:
                self.filecomment = ''

            ## Set up operater parameters
            self.iteration = 0
            self.protodata,bashpath = SolveFunction(context)
            dumpdata = [self.train_graph, self.test_graph, self.protodata, self.comment]
            currenttime = datetime.datetime.now()
            self.filename = format_filename(self.filecomment) + currenttime.strftime("_%Y.%m.%d_%H.%M_") + '.cexp'
            pickle.dump(dumpdata, open(os.path.join(self.tempdata, self.filename), 'wb'), protocol=2)
            self.datacube.name = self.filename
            self.datacube["comment"] = self.comment
            self.datacube["cexp"] = os.path.join(self.tempdata, self.filename)
            self.datacube["originaltree"] = ''
            ## Do the training
            context.user_preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
            self.p = Popen([bashpath], stderr=PIPE, bufsize=1,
                           close_fds=ON_POSIX, shell=True)
            self.q = Queue()
            self.t = Thread(target=enqueue_output, args=(self.p.stderr, self.q))
            self.t.daemon = True  # thread dies with the program
            self.t.start()
            return {'RUNNING_MODAL'}
        else:
            self.report({'ERROR'}, "Already Training")
            return {'FINISHED'}

    def modal(self, context, event):
        if event.type == 'TIMER' and self.p.poll() is None and not context.scene["donetraining"]:
            try:
                output = self.q.get_nowait()  # or q.get(timeout=.1)
            except Empty:
                output = False
            if output:
                output = output[:-1].decode("utf8")
                print (output)
                if 'Iteration' in output:
                    self.iteration = findfirst('Iteration {:g},', output)
                if 'output' in output:
                    try:
                        loss,name = get_loss(output)
                        if len(name)>55:
                            name = name[int(55-len(name)/2):len(name)-(int(55-len(name)/2))]
                        Fail = False
                    except ValueError:
                        Fail = True
                    if 'Test' in output and not Fail:
                        self.datacube[name+"test"] = loss
                        self.test_graph.append([loss, self.iteration])
                        context.user_preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
                        self.datacube.keyframe_insert(data_path='["%stest"]'%name, frame=(self.iteration))
                    elif 'Train' in output and not Fail:
                        self.datacube[name+"train"] = loss
                        self.train_graph.append([loss, self.iteration])
                        context.user_preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
                        self.datacube.keyframe_insert(data_path='["%strain"]'%name, frame=(self.iteration))
                    elif not Fail:
                        self.report({'ERROR'}, "WTF just happened")
                        return {'FINISHED'}
                    try:
                        bpy.ops.graph.select_all_toggle()
                    except RuntimeError:
                        pass
                    dumpdata = [self.train_graph, self.test_graph, self.protodata, self.comment]
                    pickle.dump(dumpdata, open(os.path.join(self.tempdata, self.filename), 'wb'), protocol=2)
                    with redirect_stdout(stdout):
                        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                    # ^^^ This is hacky but would have to be replaced by 50 lines of code to be 'proper'

        ## Quitting
        elif self.p.poll() is not None or event.type in {'RIGHTMOUSE', 'ESC'} or context.scene["donetraining"]:
            if event.type in {'RIGHTMOUSE', 'ESC'} or context.scene["donetraining"]:
                print('Kill')
                self.p.kill()
                os.system("pkill caffe")
            if self.p.poll():
                print ('Finished')
                print('Note - If your training stopped early run the train script directly from terminal as')
                print('there is likely an error with your network')
            bpy.ops.object.select_all()
            if len(context.selected_objects) == 0:
                bpy.ops.object.select_all()
            self.cancel(context)
            return {'CANCELLED'}
        return {'PASS_THROUGH'}

    def cancel(self, context):
        context.scene["donetraining"] = True
        wm = context.window_manager
        wm.event_timer_remove(self._timer)


class CancelPlot(bpy.types.Operator):
    """Cancel training"""
    bl_idname = "nodes.cancel_solver"
    bl_label = "Cancel Training"
    bl_options = {'REGISTER'}

    def execute(self, context):
        context.scene["donetraining"] = True
        return {'FINISHED'}


def register():
    bpy.utils.register_class(TrainPlot)
    bpy.utils.register_class(CancelPlot)


def unregister():
    bpy.utils.unregister_class(CancelPlot)
    bpy.utils.unregister_class(TrainPlot)
