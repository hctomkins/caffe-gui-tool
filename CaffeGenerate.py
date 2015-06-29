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
        'layer {\n\
        name: "%s"\n\
        type: "Convolution"\n\
        param {\n\
        lr_mult: %i\n\
        decay_mult: %i\n\
        }\n\
        param {\n\
        lr_mult: %i\n\
        decay_mult: %i\n\
        }\n\
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
        % (name, flr, fdr, blr, bdr, OutputLs, Padding, kernelsize, Stride, weight_filler, std, bfv, bottom, name)
    tb = [name, bottom]
    return string


def deconvtemplate(name, OutputLs, Padding, kernelsize, Stride, bottom, bfv, flr, blr, fdr, bdr, std, weight_filler):
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "Deconvolution"\n\
        param {\n\
        lr_mult: %i\n\
        decay_mult: %i\n\
        }\n\
        param {\n\
        lr_mult: %i\n\
        decay_mult: %i\n\
        }\n\
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
        % (name, flr, fdr, blr, bdr, OutputLs, Padding, kernelsize, Stride, weight_filler, std, bfv, bottom, name)
    tb = [name, bottom]
    return string


def datatemplate(name, batchsize, trainpath, testpath, supervised, dbtype, meanused, imsize, maxval=255, mirror=0, meanfile=0,silout=0):
    sf = 1.0 / (maxval + 1)
    try:
        extralabel = str(int(name[-1]))
    except ValueError:
        extralabel =''
    if supervised == 0:
        lstring = ''
    else:
        lstring = 'top: "label%s"' %extralabel
    if silout and supervised:
        silencestring =\
        'layer {\n\
        bottom: "label%s"\n\
        name: "%s"\n\
        type: "Silence"\n\
        }\n'\
        %(extralabel,name+'silence')
    else:
        silencestring = ''
    if meanused != 0:
        meanstring = 'mean_file: "%s"' % meanfile
    else:
        meanstring = ''
    if dbtype == 'LMDB':
        typestring = 'Data'
        paramstring = \
            'data_param {\n\
            source: "%s"\n\
            backend: LMDB\n\
            batch_size: %i\n\
            }\n\
            transform_param {\n\
            mirror: %i\n\
            scale: %f\n\
            %s\n\
            }\n' \
            % (trainpath, batchsize, mirror, sf, meanstring)
        testparamstring = \
            'data_param {\n\
            source: "%s"\n\
            backend: LMDB\n\
            batch_size: %i\n\
            }\n\
            transform_param {\n\
            mirror: %i\n\
            scale: %f\n\
            %s \n\
            }\n' \
            % (testpath, batchsize, mirror, sf, meanstring)
    elif dbtype == 'Image files':
        typestring = 'ImageData'
        paramstring = \
            'transform_param {\n\
            mirror: %i\n\
            scale: %f\n\
            %s\n\
            }\n\
            image_data_param {\n\
            source: "%s"\n\
            batch_size: %i\n\
            new_height: %i\n\
            new_width: %i\n\
            }\n' \
            % (mirror, sf, meanstring, trainpath, batchsize,imsize,imsize)
        testparamstring = \
            'transform_param {\n\
            mirror: %i\n\
            scale: %f\n\
            %s\n\
            }\n\
            image_data_param {\n\
            source: "%s"\n\
            batch_size: %i\n\
            new_height: %i\n\
            new_width: %i\n\
            }\n' \
            % (mirror, sf, meanstring, testpath, batchsize,imsize,imsize)
    else:
        print (dbtype)
        raise EOFError
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "%s"\n\
        top: "%s"\n\
        %s\n\
        %s \n\
        include { \n\
        phase: TRAIN \n\
        }\n\
        }\n\
        layer {\n\
        name: "%s"\n\
        type: "%s"\n\
        top: "%s"\n\
        %s\n\
        %s\n\
        include {\n\
        phase: TEST \n\
        }\n\
        }\n\
        %s\n' \
        % (name, typestring, name, lstring, paramstring, name, typestring, name, lstring, testparamstring,silencestring)
    return string


def pooltemplate(name, kernel, stride, mode, bottom):
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "Pooling"\n\
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


def FCtemplate(name, outputs, bottom, sparse, weight_filler, bfv, flr, blr, fdr, bdr, std, sparsity):
    if weight_filler == 'gaussian':
        if sparsity == 1:
            sparsestring = 'sparse: %i' % sparse
        else:
            sparsestring = ''
        wfstring = 'weight_filler {\n\
        type: "gaussian"\n\
        std: %f\n\
        %s\n\
        }\n' % (std, sparsestring)
    else:
        wfstring = 'weight_filler {\n\
        type: "xavier"\n\
        }\n'
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "InnerProduct"\n\
        param {\n\
        lr_mult: %i\n\
        decay_mult: %i\n\
        }\n\
        param {\n\
        lr_mult: %i\n\
        decay_mult: %i\n\
        }\n\
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
        % (name, flr, fdr, blr, bdr, outputs, wfstring, bfv, bottom, name)
    return string


def flattentemplate(name, bottom):
    string = \
        'layer {\n\
        bottom: "%s"\n\
        top: "%s"\n\
        name: "%s"\n\
        type: "Flatten"\n\
        }\n' \
        % (bottom, name, name)
    return string


def silencetemplate(name, bottom):
    string = \
        'layer {\n\
        bottom: "%s"\n\
        name: "%s"\n\
        type: "Silence"\n\
        }\n' \
        % (bottom, name)
    return string


def dropouttemplate(name, bottom, dropout):
    string = \
        'layer {\n\
    name: "%s"\n\
    type: "Dropout"\n\
    bottom: "%s"\n\
    top: "%s"\n\
    dropout_param {\n\
    dropout_ratio: %f\n\
    }\n\
    }\n' % (name, bottom, name, dropout)
    return string


def LRNtemplate(name, bottom, alpha, beta, size, mode):
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "LRN"\n\
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
        'layer {\n\
        name: "%s"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        type: %s\n\
        }\n' \
        % (name, bottom, name, mode)
    return string


def Relutemplate(bottom, name, Negativeg):
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "ReLU"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        }\n' \
        % (name, bottom, name)
    return string


def SMtemplate(name, bottom, w):
    string = \
        'layer {\n\
        name: "loss"\n\
        type: "SoftmaxWithLoss"\n\
        bottom: "%s"\n\
        bottom: "label"\n\
        top: "%s"\n\
        loss_weight: %f\n\
        }\n' \
        % (bottom, name, w)
    return string


def SCEtemplate(name, bottom1, bottom2, w):
    string = \
        'layer {\n\
        name: "loss"\n\
        type: "SigmoidCrossEntropyLoss"\n\
        bottom: "%s"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        loss_weight: %f\n\
        }\n' \
        % (bottom1, bottom2, name, w)
    return string


def EUtemplate(name, bottom1, bottom2, w):
    string = \
        'layer {\n\
        name: "loss"\n\
        type: "EuclideanLoss"\n\
        bottom: "%s"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        loss_weight: %f\n\
        }\n' \
        % (bottom1, bottom2, name, w)
    return string


def Concattemplate(name, bottom1, bottom2, dim):
    string = \
        'layer {\n\
        name: "%s"\n\
        bottom: "%s"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        type: "Concat"\n\
        concat_param {\n\
        concat_dim: %i\n\
        }\n\
        }\n' \
        % (name, bottom1, bottom2, name, dim)
    return string


def accuracytemplate(name, bottom, Testonly):
    if Testonly == 1:
        Testonly = 'include { \n\
            phase: TEST \n\
            }'
    else:
        Testonly = ''
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "Accuracy"\n\
        bottom: "%s"\n\
        bottom: "label"\n\
        top: "%s"\n\
        %s\n\
        }\n' \
        % (name, bottom, name, Testonly)
    return string


def solvertemplate(type, learningrate, testinterval, testruns, maxiter, displayiter, snapshotiter, snapshotname,
                   snapshotpath, configpath, solvername, itersize, solver='GPU'):
    snapshotprefix = snapshotpath + snapshotname
    netpath = configpath + '%s_train_test.prototxt' % solvername
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
        iter_size: %i\n\
        snapshot: %i\n\
        snapshot_prefix: "%s"\n\
        solver_mode: %s\n' \
        % (netpath, testruns, testinterval, learningrate, displayiter, maxiter,itersize, snapshotiter, snapshotprefix, solver)
    solverstring = genericstring + tsstring
    return solverstring


def deploytemplate(batch, channels, size, datain):
    deploystring = \
        'name: "Autogen"\n\
    input: "%s"\n\
    input_dim: %i\n\
    input_dim: %i\n\
    input_dim: %i\n\
    input_dim: %i\n' % (datain, batch, channels, size, size)
    return deploystring


def scripttemplate(caffepath, configpath, solvername, gpu,solver):
    if solver == 'GPU':
        extrastring = '--gpu=%i' %gpu
    else:
        extrastring = ''
    solverstring = configpath + '%s_solver.prototxt' % solvername
    caffestring = caffepath + 'caffe'
    string = '#!/usr/bin/env sh \n %s train --solver=%s %s' % (caffestring, solverstring, extrastring)
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
        dcode = []

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
                if node.dbtype == 'LMDB':
                    string = datatemplate(node.name, node.batchsize, node.trainpath, node.testpath, node.supervised,
                                          node.dbtype, node.usemeanfile,node.imsize,node.maxval,node.mirror,node.meanfile,node.silout)
                    dstring = deploytemplate(node.batchsize, node.channels, node.imsize, node.name)
                elif node.dbtype == "Image files":
                    string = datatemplate(node.name, node.batchsize, node.trainfile, node.testfile, node.supervised,
                                          node.dbtype, node.usemeanfile,node.imsize,node.maxval,node.mirror,node.meanfile,node.silout)
                    dstring = deploytemplate(node.batchsize, node.channels, node.imsize, node.name)
            elif node.bl_idname == 'PoolNodeType':
                string = pooltemplate(node.name, node.kernel, node.stride, node.mode, bottoms[0])
                dstring = string
            elif node.bl_idname == 'ConvNodeType':
                string = convtemplate(node.name, node.OutputLs, node.Padding, node.kernelsize, node.Stride, bottoms[0],
                                      node.biasfill, node.filterlr, node.biaslr, node.filterdecay, node.biasdecay,
                                      node.std, node.weights)
                dstring = string
            elif node.bl_idname == 'DeConvNodeType':
                string = deconvtemplate(node.name, node.OutputLs, node.Padding, node.kernelsize, node.Stride, bottoms[0],
                                      node.biasfill, node.filterlr, node.biaslr, node.filterdecay, node.biasdecay,
                                      node.std, node.weights)
                dstring = string
            elif node.bl_idname == 'FCNodeType':
                string = FCtemplate(node.name, node.outputnum, bottoms[0], node.sparse, node.weights, node.biasfill,
                                    node.filterlr, node.biaslr, node.filterdecay, node.biasdecay, node.std,
                                    node.sparsity)
                dstring = string
            elif node.bl_idname == 'FlattenNodeType':
                string = flattentemplate(node.name, bottoms[0])
                dstring = string
            elif node.bl_idname == 'SilenceNodeType':
                string = silencetemplate(node.name, bottoms[0])
                dstring = string
            elif node.bl_idname == 'LRNNodeType':
                string = LRNtemplate(node.name, bottoms[0], node.alpha, node.beta, node.size, node.mode)
                dstring = string
            elif node.bl_idname == 'AcNodeType':
                string = NLtemplate(node.name, bottoms[0], node.mode)
                dstring = string
                dstring = string
            elif node.bl_idname == 'ReluNodeType':
                string = Relutemplate(bottoms[0], node.name, node.Negativeg)
                dstring = string
            elif node.bl_idname == 'DropoutNodeType':
                string = dropouttemplate(node.name, bottoms[0], node.fac)
                dstring = string
            elif node.bl_idname == 'SMLossNodeType':
                string = SMtemplate(node.name, bottoms[0], node.w)
                dstring = ''
            elif node.bl_idname == 'SCELossNodeType':
                string = SCEtemplate(node.name, bottoms[0], bottoms[1], node.w)
                dstring = ''
            elif node.bl_idname == 'EULossNodeType':
                string = EUtemplate(node.name, bottoms[0], bottoms[1], node.w)
                dstring = ''
            elif node.bl_idname == 'ConcatNodeType':
                string = Concattemplate(node.name, bottoms[0], bottoms[1], node.dim)
                dstring = string
            elif node.bl_idname == 'AccuracyNodeType':
                string = accuracytemplate(node.name, bottoms[0], node.Testonly)
                dstring = ''
            elif node.bl_idname == 'NodeReroute':
                string = ''
                dstring = ''
            elif node.bl_idname == 'SolverNodeType':
                solverstring = solvertemplate(node.solver, node.learningrate, node.testinterval, node.testruns,
                                              node.maxiter,
                                              node.displayiter, node.snapshotiter, node.solvername, node.snapshotpath,
                                              node.configpath, node.solvername, node.accumiters,solver=node.compmode)
                scriptstring = scripttemplate(node.caffexec, node.configpath, node.solvername, node.gpu,solver=node.compmode)
                configpath = node.configpath
                solvername = node.solvername
            elif string == 0:
                print (node.bl_idname)
            if node.bl_idname != 'SolverNodeType':
                gcode.extend([string])
                dcode.extend([dstring])
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
            gtops, gbottoms, g2bottoms, gcode, dcode = self.juggleorder(gtops, gbottoms, g2bottoms, gcode, 0, dcode)
            # for chunk in gcode:
            #print (chunk)
            gtops, gbottoms, g2bottoms, gcode, dcode = self.juggleorder(gtops, gbottoms, g2bottoms, gcode, 1, dcode)
        solution = ''
        for chunk in gcode:
            solution = solution + chunk
        dsolution = ''
        for chunk in dcode:
            dsolution = dsolution + chunk
        # print (solution)
        os.chdir(configpath)
        ttfile = open('%s_train_test.prototxt' % solvername, mode='w')
        ttfile.write(solution)
        ttfile.close()
        depfile = open('%s_deploy.prototxt' % solvername, mode='w')
        depfile.write(dsolution)
        depfile.close()
        solvefile = open('%s_solver.prototxt' % solvername, mode='w')
        solvefile.write(solverstring)
        solvefile.close()
        scriptfile = open('train_%s.sh' % solvername, mode='w')
        scriptfile.write(scriptstring)
        scriptfile.close()
        return {'FINISHED'}  # this lets blender know the operator finished successfully.

    def juggleorder(self, names, refs, refs2, code, prefsocket, dcode):
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
                # print (name)
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
                    names, refs, refs2, code, dcode = self.swap(loc, ref, (names, refs, refs2, code, dcode))
                    checks[loc] = 0
                else:
                    checks[loc] = 0
            print (names)
            print (refs)
            print (refs2)
            print (checks)
            # if in wrong order
        return names, refs, refs2, code, dcode

    def swap(self, orig, dest, lists):
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
