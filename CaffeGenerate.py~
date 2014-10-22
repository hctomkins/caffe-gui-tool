__author__ = 'hugh'
bl_info = {
    "name": "Creat Caffe solution",
    "category": "Object",
}

import bpy
import random
import numpy as np
import time
import os

def convtemplate(name, OutputLs, Padding, kernelsize, Stride, bottom, bfv, flr, blr, fdr, bdr, std, weight_filler):
    string = \
        'layers {\n\
        name: "%s"\n\
        type: CONVOLUTION\n\
        blobs_lr: %i\n\
        blobs_lr: %i\n\
        weight_decay: %i\n\
        weight_decay: %i\n\
        convolution_param {\n\
        num_output: %i\n\
        pad: %i\n\
        kernel_size: %i\n\
        stride: %i\n\
        weight_filler {\n\
        type: "%s"\n\
        std: %f\n\
        }\n\
        bias_filler {\n\
        type: "constant"\n\
        value: %f\n\
        }\n\
        }\n\
        bottom: "%s"\n\
        top: "%s"\n\
        }\n' \
        % (name, flr, blr, fdr,bdr,OutputLs, Padding, kernelsize, Stride, weight_filler, std, bfv, bottom,name)
    tb = [name, bottom]
    return string


def datatemplate(name, batchsize, trainpath, testpath,supervised, maxval=255):
    if supervised == 0:
        lstring = ''
    else:
        lstring = 'top: "label"'
    sf = 1.0 / (maxval + 1)
    string = \
        'layers {\n\
        name: "%s"\n\
        type: DATA\n\
        top: "%s"\n\
        %s\n\
        data_param {\n\
        source: "%s"\n\
        backend: LMDB\n\
        batch_size: %i\n\
        }\n\
        transform_param {\n\
        scale: %f\n\
        }\n\
        include: { phase: TRAIN }\n\
        }\n\
        layers {\n\
        name: "%s"\n\
        type: DATA\n\
        top: "%s"\n\
        %s\n\
        data_param {\n\
        source: "%s"\n\
        backend: LMDB\n\
        batch_size: %i\n\
        }\n\
        transform_param {\n\
        scale: %f\n\
        }\n\
        include: { phase: TEST }\n\
        }\n' \
        % (name, name, lstring,trainpath, batchsize, sf, name, name,lstring, testpath, batchsize, sf)
    return string


def pooltemplate(name, kernel, stride, mode, bottom):
    string = \
        'layers {\n\
        name: "%s"\n\
        type: POOLING\n\
        bottom: "%s"\n\
        top: "%s"\n\
        pooling_param {\n\
        pool: %s\n\
        kernel_size: %i\n\
        stride: %i\n\
        }\n\
        }\n' \
        % (name, bottom, name, mode, kernel, stride)
    return string


def FCtemplate(name, outputs, bottom, sparse, weight_filler, bfv,flr,blr,fdr,bdr,std,sparsity):
    if weight_filler == 'gaussian':
        if sparsity == 1 :
            sparsestring = 'sparse: %i' %sparse
        else:
            sparsestring = ''
        wfstring = 'weight_filler {\n\
        type: "gaussian"\n\
        std: %f\n\
        %s\n\
        }\n' %(std,sparsestring)
    else:
        wfstring = 'weight_filler {\n\
        type: "xavier"\n\
        }\n'
    string = \
        'layers {\n\
        name: "%s"\n\
        type: INNER_PRODUCT\n\
        blobs_lr: %i\n\
        blobs_lr: %i\n\
        weight_decay: %i\n\
        weight_decay: %i\n\
        inner_product_param {\n\
        num_output: %i\n\
        %s\n\
        bias_filler {\n\
        type: "constant"\n\
        value: %i\n\
        }\n\
        }\n\
        bottom: "%s"\n\
        top: "%s"\n\
        }\n' \
        % (name, flr, blr, fdr,bdr,outputs, wfstring, bfv, bottom, name)
    return string


def flattentemplate(name, bottom):
    string = \
        'layers {\n\
        bottom: "%s"\n\
        top: "%s"\n\
        name: "%s"\n\
        type: FLATTEN\n\
        }\n' \
        % (bottom, name, name)
    return string


def dropouttemplate(name, bottom, dropout):
    string = \
    'layers {\n\
    name: "%s"\n\
    type: DROPOUT\n\
    bottom: "%s"\n\
    top: "%s"\n\
    dropout_param {\n\
    dropout_ratio: %f\n\
    }\n\
    }\n' % (name, bottom, name, dropout)
    return string

def LRNtemplate(name, bottom, alpha, beta, size, mode):
    string = \
        'layers {\n\
        name: "%s"\n\
        type: LRN\n\
        bottom: "%s"\n\
        top: "%s"\n\
        lrn_param {\n\
        local_size: %i\n\
        alpha: %f\n\
        beta: %f\n\
        norm_region: %s\n\
        }\n\
        }\n' \
        % (name, bottom, name, size, alpha, beta, mode)
    return string


def NLtemplate(name, bottom, mode):
    string = \
        'layers {\n\
        name: "%s"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        type: %s\n\
        }\n' \
        % (name, bottom, name, mode)
    return string


def Relutemplate(bottom, name, Negativeg):
    string = \
        'layers {\n\
        name: "%s"\n\
        type: RELU\n\
        bottom: "%s"\n\
        top: "%s"\n\
        }\n' \
        % (name, bottom, name)
    return string


def SMtemplate(name, bottom, w):
    string = \
        'layers {\n\
        name: "loss"\n\
        type: SOFTMAX_LOSS\n\
        bottom: "%s"\n\
        bottom: "label"\n\
        top: "%s"\n\
        loss_weight: %f\n\
        }\n' \
        % (bottom, name, w)
    return string


def SCEtemplate(name, bottom1, bottom2, w):
    string = \
        'layers {\n\
        name: "loss"\n\
        type: SIGMOID_CROSS_ENTROPY_LOSS\n\
        bottom: "%s"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        loss_weight: %f\n\
        }\n' \
        % (bottom1, bottom2, name, w)
    return string


def EUtemplate(name, bottom1, bottom2, w):
    string = \
        'layers {\n\
        name: "loss"\n\
        type: EUCLIDEAN_LOSS\n\
        bottom: "%s"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        loss_weight: %f\n\
        }\n' \
        % (bottom1, bottom2, name, w)
    return string


def Concattemplate(name, bottom1, bottom2, dim):
    string = \
        'layers {\n\
        name: "%s"\n\
        bottom: "%s"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        type: CONCAT\n\
        concat_param {\n\
        concat_dim: %i\n\
        }\n\
        }\n' \
        % (name, bottom1, bottom2, name, dim)
    return string


def accuracytemplate(name, bottom, Testonly):
    if Testonly == 1:
        Testonly = 'include: { phase: TEST }'
    else:
        Testonly = ''
    string = \
        'layers {\n\
        name: "%s"\n\
        type: ACCURACY\n\
        bottom: "%s"\n\
        bottom: "label"\n\
        top: "%s"\n\
        %s\n\
        }\n' \
        % (name, bottom, name, Testonly)
    return string


def solvertemplate(type, learningrate, testinterval, testruns, maxiter, displayiter, snapshotiter, snapshotname,
                   snapshotpath, configpath, solvername, solver='GPU'):
    snapshotprefix = snapshotpath + snapshotname
    netpath = configpath + '%s_train_test.prototxt' %solvername
    if type == 'ADAGRAD':
        tsstring = \
            'lr_policy: "step"\n\
            gamma: 0.1\n\
            stepsize: 10000\n\
            weight_decay: 0.0005\n\
            solver_type: ADAGRAD\n'
    elif type == 'NAG':
        tsstring = \
            'lr_policy: "step"\n\
            gamma: 0.1\n\
            stepsize: 10000\n\
            weight_decay: 0.0005\n\
            momentum: 0.95\n\
            solver_type: NESTEROV\n'
    elif type == 'SGD':
        tsstring = \
            'lr_policy: "step"\n\
            gamma: 0.1\n\
            stepsize: 10000\n\
            weight_decay: 0.0005\n\
            momentum: 0.95\n'
    else:
        print ('ERROR')
        time.sleep(1000000)
    genericstring = \
        'net: "%s"\n\
        test_iter: %i\n\
        test_interval: %i\n\
        base_lr: %f\n\
        display: %i\n\
        max_iter: %i\n\
        snapshot: %i\n\
        snapshot_prefix: "%s"\n\
        solver_mode: %s\n' \
        % (netpath, testruns, testinterval, learningrate, displayiter, maxiter, snapshotiter, snapshotprefix, solver)
    solverstring = genericstring + tsstring
    return solverstring

def scripttemplate(caffepath,configpath,solvername,gpu):
    solverstring = configpath + '%s_solver.prototxt' % solvername
    caffestring = caffepath + 'caffe'
    string = '#!/usr/bin/env sh \n %s train --solver=%s --gpu=%i' %(caffestring,solverstring,gpu)
    return string

class Solve(bpy.types.Operator):
    """Generate Caffe solver"""  # blender will use this as a tooltip for menu items and buttons.
    bl_idname = "nodes.make_solver"  # unique identifier for buttons and menu items to reference.
    bl_label = "Create Solution"  # display name in the interface.
    bl_options = {'REGISTER'}  # enable undo for the operator.

    def execute(self, context):  # execute() is called by blender when running the operator.
        gtops = []
        gbottoms = []
        g2bottoms = []
        gcode = []

        # The original script
        for node in context.selected_nodes:
            bottoms = []
            nname = node.name
            string = 0
            for input in node.inputs:
                if input.is_linked == True:
                    bottomnode = input.links[0].from_node
                    while bottomnode.bl_idname == 'NodeReroute':
                        bottomnodein = bottomnode.inputs
                        bottomnodein = bottomnodein[0]
                        bottomnode = bottomnodein.links[0].from_node
                    bottom = bottomnode.name
                    print ('node %s has input %s' % (nname, bottomnode.name))
                    bottoms.extend([bottom])
            if node.bl_idname == 'DataNodeType':
                string = datatemplate(node.name, node.batchsize, node.trainpath, node.testpath, node.supervised, node.maxval)
            elif node.bl_idname == 'PoolNodeType':
                string = pooltemplate(node.name, node.kernel, node.stride, node.mode, bottoms[0])
            elif node.bl_idname == 'ConvNodeType':
                string = convtemplate(node.name, node.OutputLs, node.Padding, node.kernelsize, node.Stride, bottoms[0],0,node.filterlr,node.biaslr,node.filterdecay,node.biasdecay,node.std,node.weights)
            elif node.bl_idname == 'FCNodeType':
                string = FCtemplate(node.name, node.outputnum, bottoms[0],node.sparse,node.weights,0,node.filterlr,node.biaslr,node.filterdecay,node.biasdecay,node.std,node.sparsity)
            elif node.bl_idname == 'FlattenNodeType':
                string = flattentemplate(node.name, bottoms[0])
            elif node.bl_idname == 'LRNNodeType':
                string = LRNtemplate(node.name, bottoms[0], node.alpha, node.beta, node.size, node.mode)
            elif node.bl_idname == 'AcNodeType':
                string = NLtemplate(node.name, bottoms[0], node.mode)
            elif node.bl_idname == 'ReluNodeType':
                string = Relutemplate(bottoms[0], node.name, node.Negativeg)
            elif node.bl_idname == 'DropoutNodeType':
                string = dropouttemplate(node.name, bottoms[0], node.fac)
            elif node.bl_idname == 'SMLossNodeType':
                string = SMtemplate(node.name, bottoms[0], node.w)
            elif node.bl_idname == 'SCELossNodeType':
                string = SCEtemplate(node.name, bottoms[0], bottoms[1], node.w)
            elif node.bl_idname == 'EULossNodeType':
                string = EUtemplate(node.name, bottoms[0], bottoms[1], node.w)
            elif node.bl_idname == 'ConcatNodeType':
                string = Concattemplate(node.name, bottoms[0], bottoms[1], node.dim)
            elif node.bl_idname == 'AccuracyNodeType':
                string = accuracytemplate(node.name, bottoms[0], node.Testonly)
            elif node.bl_idname == 'NodeReroute':
                string = ''
            elif node.bl_idname == 'SolverNodeType':
                solverstring = solvertemplate(node.solver, node.learningrate, node.testinterval, node.testruns, node.maxiter,
                                        node.displayiter, node.snapshotiter, node.solvername, node.snapshotpath,
                                        node.configpath,node.solvername,solver=node.compmode)
                scriptstring = scripttemplate(node.caffexec,node.configpath,node.solvername,node.gpu)
                configpath = node.configpath
                solvername = node.solvername
            elif string == 0:
                print (node.bl_idname)
            if node.bl_idname != 'SolverNodeType':
                gcode.extend([string])
                gtops.extend([node.name])
                try:
                    gbottoms.extend([bottoms[0]])
                except IndexError:
                    gbottoms.extend([str(random.random())])
                try:
                    g2bottoms.extend([bottoms[1]])
                except IndexError:
                    g2bottoms.extend([str(random.random())])
        for juggle in range(30):
            gtops,gbottoms,g2bottoms,gcode = self.juggleorder(gtops,gbottoms, g2bottoms,gcode,0)
        #for chunk in gcode:
            #print (chunk)
            gtops,gbottoms,g2bottoms,gcode = self.juggleorder(gtops,gbottoms, g2bottoms,gcode,1)
        solution = ''
        for chunk in gcode:
            solution = solution + chunk
        #print (solution)
        os.chdir(configpath)
        ttfile = open('%s_train_test.prototxt' %solvername,mode='w')
        ttfile.write(solution)
        ttfile.close()
        solvefile = open('%s_solver.prototxt' %solvername, mode='w')
        solvefile.write(solverstring)
        solvefile.close()
        scriptfile = open('train_%s.sh' %solvername, mode='w')
        scriptfile.write(scriptstring)
        scriptfile.close()
        return {'FINISHED'}  # this lets blender know the operator finished successfully.

    def juggleorder(self, names, refs, refs2, code,prefsocket):
        goodorder = 0
        print (refs)
        print(refs2)
        checks = np.ones((len(names)))
        while np.sum(checks) > 0:

            for name in names:
                x = 0
                y = 0
                z = 0
                # Start of list is data layer
                # get location of bottom in top
                #print (name)
                #print (names)
                loc = names.index(name)
                try:
                    ref = refs.index(name)
                    x = 1
                except ValueError:
                    pass
                try:
                    float(name)
                    
                    print ('passing float')
                    print (name)
                    y = 1
                except ValueError:
                    pass
                try:
                    tmpref = refs2.index(name)
                    if x == 1 and prefsocket == 1:
                        ref = tmpref
                    elif x == 0:
                        ref = tmpref
                    z = 1
                except ValueError:
                    pass
                if x + y + z == 0:
                    # not referred to by anything, so can be as late as possible
                    ref = 10000000000000000
                    #time.sleep(10)
                #ref = 10000000
                if ref < loc:
                    names,refs,refs2,code = self.swap(loc,ref,(names,refs,refs2,code))
                    checks[loc] = 0
                else:
                    checks[loc] = 0
            print (names)
            print (refs)
            print (refs2)
            print (checks)
                # if in wrong order
        return names,refs,refs2,code

    def swap(self,orig,dest,lists):
        for list in lists:
            tmp = list[dest]
            list[dest] = list[orig]
            list[orig] = tmp
        return lists



def register():
    bpy.utils.register_class(Solve)


def unregister():
    bpy.utils.unregister_class(Solve)


# This allows you to run the script directly from blenders text editor
# to test the addon without having to install it.
if __name__ == "__main__":
    register()
