__author__ = 'hugh'
bl_info = {
    "name": "Create Caffe solution",
    "category": "Object",
}

import bpy
import random
import time
import os

def getFillerString(node, ftype):
    if ftype == 'none':
        if node.type == 'constant':
            fillerString = 'value: %f\n' % (node.value)
        elif node.type == 'xavier' or node.type == 'msra':
            fillerString = 'variance_norm: %s\n' % (node.variance_norm)
        elif node.type == 'gaussian':
            fillerString = 'mean: %f\nstd: %f\n' % (node.mean, node.std)    
            if node.sparsity:
                fillerString = fillerString + 'sparse: %i\n' % (node.sparse)    
        elif node.type == 'uniform':
            fillerString = 'min: %f\nmax: %f\n' % (node.min, node.max)
    elif ftype == 'weight':
        if node.w_type == 'constant':
            fillerString = 'value: %f\n' % (node.w_value)
        elif node.w_type == 'xavier' or node.w_type == 'msra':
            fillerString = 'variance_norm: %s\n' % (node.w_variance_norm)
        elif node.w_type == 'gaussian':
            fillerString = 'mean: %f\nstd: %f\n' % (node.w_mean, node.w_std)    
            if node.w_sparsity:
                fillerString = fillerString + 'sparse: %i\n' % (node.w_sparse)    
        elif node.w_type == 'uniform':
            fillerString = 'min: %f\nmax: %f\n' % (node.w_min, node.w_max)
    elif ftype == 'bias':
        if node.b_type == 'constant':
            fillerString = 'value: %f\n' % (node.b_value)
        elif node.b_type == 'xavier' or node.b_type == 'msra':
            fillerString = 'variance_norm: %s\n' % (node.b_variance_norm)
        elif node.b_type == 'gaussian':
            fillerString = 'mean: %f\nstd: %f\n' % (node.b_mean, node.b_std)    
            if node.b_sparsity:
                fillerString = fillerString + 'sparse: %i\n' % (node.b_sparse)    
        elif node.b_type == 'uniform':
            fillerString = 'min: %f\nmax: %f\n' % (node.b_min, node.b_max)
    
    fillerString = 'type: "%s"\n%s' % (node.type, fillerString)
    return fillerString
    
def convtemplate(node,name, OutputLs, Padding, kernelsize, Stride, bottom, bfv, flr, blr, fdr, bdr, std, weight_filler,nonsquare=0,x=0,y=0):
    w_fillerString = getFillerString(node, 'weight')
    b_fillerString = getFillerString(node, 'bias')
    
    if not nonsquare:
        kernelstring = 'kernel_size: %i'%kernelsize
    else:
        kernelstring = 'kernel_h: %i\nkernel_w: %i' %(y,x)
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
        %s\n\
        stride: %i\n\
        weight_filler {\n\
        %s\
        }\n\
        bias_filler {\n\
        %s\
        }\n\
        }\n\
        bottom: "%s"\n\
        top: "%s"\n\
        }\n' \
        % (name, flr, fdr, blr, bdr, OutputLs, Padding, kernelstring, Stride, w_fillerString,b_fillerString, bottom, top)
    tb = [name, bottom]
    return string


def deconvtemplate(node,name, OutputLs, Padding, kernelsize, Stride, bottom,top, bfv, flr, blr, fdr, bdr, std, weight_filler,nonsquare=0,x=0,y=0):
    w_fillerString = getFillerString(node, 'weight')
    b_fillerString = getFillerString(node, 'bias')
    
    if not nonsquare:
        kernelstring = 'kernel_size: %i'%kernelsize
    else:
        kernelstring = 'kernel_h: %i\nkernel_w: %i' %(y,x)
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
        %s\n\
        stride: %i\n\
        weight_filler {\n\
        %s\
        }\n\
        bias_filler {\n\
        %s\
        }\n\
        }\n\
        bottom: "%s"\n\
        top: "%s"\n\
        }\n' \
        % (name, flr, fdr, blr, bdr, OutputLs, Padding, kernelstring, Stride, w_fillerString, b_fillerString, bottom, top)
    tb = [name, bottom]
    return string


def datatemplate(name, top1, top2, batchsize, trainpath, testpath, shuffle, supervised, dbtype, meanused, imsize, maxval=255, mirror=0,
                meanfile=0, silout=0, channels=3):
    sf = 1.0 / (maxval + 1)
    if channels == 1:
        iscolour = 'is_color: 1' ### When single channel
    else:
        iscolour = ''
    try:
        extralabel = str(int(name[-1])) ####In case of more than one data layer
    except ValueError:
        extralabel = ''
    if supervised == 0:
        lstring = ''
    else:
        lstring = 'top: "%s%s"' % (top2, extralabel)
    if silout and supervised:
        silencestring = \
            'layer {\n\
        bottom: "%s%s"\n\
        name: "%s"\n\
        type: "Silence"\n\
        }\n' \
            % (top2, extralabel, name + 'silence')
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
            %s\n\
            source: "%s"\n\
            batch_size: %i\n\
            new_height: %i\n\
            new_width: %i\n\
            shuffle: %i\n\
            }\n' \
            % (mirror, sf, meanstring, iscolour, trainpath, batchsize, imsize, imsize, shuffle)
        testparamstring = \
            'transform_param {\n\
            mirror: %i\n\
            scale: %f\n\
            %s\n\
            }\n\
            image_data_param {\n\
            %s\n\
            source: "%s"\n\
            batch_size: %i\n\
            new_height: %i\n\
            new_width: %i\n\
            shuffle: %i\n\
            }\n' \
            % (mirror, sf, meanstring, iscolour, testpath, batchsize, imsize, imsize, shuffle)
    elif dbtype == 'HDF5Data':
        typestring = 'HDF5Data'
        paramstring = \
            'hdf5_data_param {\n\
            source: "%s"\n\
            batch_size: %i\n\
            shuffle: %i\n\
            }\n\
            transform_param {\n\
            mirror: %i\n\
            scale: %f\n\
            %s\n\
            }\n' \
            % (trainpath, batchsize, shuffle, mirror, sf, meanstring)
        testparamstring = \
            'hdf5_data_param {\n\
            source: "%s"\n\
            batch_size: %i\n\
            shuffle: %i\n\
            }\n\
            transform_param {\n\
            mirror: %i\n\
            scale: %f\n\
            %s \n\
            }\n' \
            % (testpath, batchsize, shuffle, mirror, sf, meanstring)
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
        % (
            name, typestring, top1, lstring, paramstring, name, typestring, top1, lstring, testparamstring,
            silencestring)
    return string


def pooltemplate(name, kernel, stride, mode, bottom, top):
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
        % (name, bottom, top, mode, kernel, stride)
    return string


def mvntemplate(name, bottom, normalize_variance, across_channels, eps):
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "MVN"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        mvn_param  {\n\
        normalize_variance: %s\n\
        across_channels: %s\n\
        eps: %f\n\
        }\n\
        }\n' \
        % (name, bottom, name, normalize_variance, across_channels, eps)
    return string
    
def FCtemplate(node, name, outputs, bottom, top, sparse, weight_filler, bfv, flr, blr, fdr, bdr, std, sparsity):
    if sparsity == 1:
        sparsestring = 'sparse: %i' % sparse
    else:
        sparsestring = ''
    if weight_filler == 'gaussian':
        wfstring = 'weight_filler {\n\
        type: "gaussian"\n\
        std: %f\n\
        %s\n\
        }\n' % (std, sparsestring)
    else:
        wfstring = 'weight_filler {\n\
        type: "xavier"\n\
        std: %f\n\
        %s\n\
        }\n'%(std,sparsestring)
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
        % (name, flr, fdr, blr, bdr, outputs, wfstring, bfv, bottom, top)
    return string


def flattentemplate(name, bottom, top):
    string = \
        'layer {\n\
        bottom: "%s"\n\
        top: "%s"\n\
        name: "%s"\n\
        type: "Flatten"\n\
        }\n' \
        % (bottom, top, name)
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


def dropouttemplate(name, bottom, top, dropout):
    string = \
        'layer {\n\
    name: "%s"\n\
    type: "Dropout"\n\
    bottom: "%s"\n\
    top: "%s"\n\
    dropout_param {\n\
    dropout_ratio: %f\n\
    }\n\
    }\n' % (name, bottom, top, dropout)
    return string


def LRNtemplate(name, bottom, top, alpha, beta, size, mode):
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
        % (name, bottom, top, size, alpha, beta, mode)
    return string


def NLtemplate(name, bottom, top, mode):
    string = \
        'layer {\n\
        name: "%s"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        type: %s\n\
        }\n' \
        % (name, bottom, top, mode)
    return string


def Relutemplate(bottom, top, name, Negativeg):
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "ReLU"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        }\n' \
        % (name, bottom, top)
    return string
    
def PRelutemplate(node, bottom):    
    fillerString = getFillerString(node,'none')    
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "PReLU"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        filler {\n\
        %s\
        }\n\
        }\n' \
        % (node.name, bottom, node.name, fillerString)
    return string

def SMtemplate(name, bottom1, bottom2, top, w):
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "SoftmaxWithLoss"\n\
        bottom: "%s"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        loss_weight: %f\n\
        }\n' \
        % (name, bottom1, bottom2, top, w)
    return string


def SCEtemplate(name, bottom1, bottom2, top, w):
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "SigmoidCrossEntropyLoss"\n\
        bottom: "%s"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        loss_weight: %f\n\
        }\n' \
        % (name, bottom1, bottom2, top, w)
    return string


def EUtemplate(name, bottom1, bottom2, top, w):
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "EuclideanLoss"\n\
        bottom: "%s"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        loss_weight: %f\n\
        }\n' \
        % (name, bottom1, bottom2, top, w)
    return string


def Concattemplate(name, bottom1, bottom2, top, axis):
    string = \
        'layer {\n\
        name: "%s"\n\
        bottom: "%s"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        type: "Concat"\n\
        concat_param {\n\
        axis: %i\n\
        }\n\
        }\n' \
        % (name, bottom1, bottom2, top, axis)
    return string


def accuracytemplate(name, bottom, top, Testonly):
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
        % (name, bottom, top, Testonly)
    return string

def argmaxtemplate(name, bottom, top, OutMaxVal, TopK):
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "ARGMAX"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        argmax_param {\n\
        out_max_val: %i\n\
        top_k: %i\n\
        }\n\
        }\n' \
        % (name, bottom, top, OutMaxVal, TopK)
    return string

def hdf5outputtemplate(name, bottom, filename):
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "HDF5Output"\n\
        bottom: "%s"\n\
        hdf5_output_param {\n\
        file_name: "%s"\n\
        }\n\
        }\n' \
        % (name, bottom, filename)
    return string


def logtemplate(name, bottom, top, scale, shift, base):
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "Log"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        log_param {\n\
        scale: %f\n\
        shift: %f\n\
        base: %f\n\
        }\n\
        }\n' \
        % (name, bottom, top, scale, shift, base)
    return string

def powertemplate(name, bottom, top, power, scale, shift):
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "Power"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        power_param {\n\
        power: %f\n\
        scale: %f\n\
        shift: %f\n\
        }\n\
        }\n' \
        % (name, bottom, top, power, scale, shift)
    return string

def reductiontemplate(name, bottom, top, operation, axis, coeff):
    string = \
        'layer {\n\
        name: "%s"\n\
        type: "Reduction"\n\
        bottom: "%s"\n\
        top: "%s"\n\
        reduction_param { \n\
        operation: %s\n\
        axis: %i\n\
        coeff: %f\n\
        }\n\
        }\n' \
        % (name, bottom, top, operation, axis, coeff)
    return string

def slicetemplate(name, bottom, tops, axis, slice_points):
    top_string = ""
    for top in tops:
        top_string += 'top: "%s"\n' % top
    slice_points_string = ""
    for slice_point in slice_points:
        slice_points_string += 'slice_point: %i\n' % slice_point

    string = \
        'layer {\n\
        name: "%s"\n\
        type: "Slice"\n\
        bottom: "%s"\n\
        %s\
        slice_param { \n\
        axis: %i\n\
        %s\n\
        }\n\
        }\n' \
        % (name, bottom, top_string, axis, slice_points_string)
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
        % (netpath, testruns, testinterval, learningrate, displayiter, maxiter, itersize, snapshotiter, snapshotprefix,
        solver)
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


def scripttemplate(caffepath, configpath, solvername, gpus, solver):
    gpustring = ''
    usedcount = 0
    for gpu, used in enumerate(gpus):
        if used:
            if usedcount != 0:
                gpustring += ',' + str(gpu)
            else:
                gpustring += str(gpu)
            usedcount += 1
    if solver == 'GPU':
        extrastring = '--gpu=%s' % gpustring
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
        gtops = []  # the top (I.E. name of) each layer
        gbottoms = []  # the first input of all nodes
        g2bottoms = []  # the second input of all nodes
        gcode = []  # the code slice of each layer
        dcode = []  # the 'deploy' code slice of each layer
        ########################################### Main loop
        for node in context.selected_nodes:
            ###################### What are all the nodes inputs?
            bottoms = []
            nname = node.name
            string = 0
            for input in node.inputs:
                if input.is_linked == True:
                    bottom = input.links[0].from_socket.output_name
                    bottoms.extend([bottom])  # Bottoms is the list of all the nodes attached behind the current node
            ###########################
            if node.bl_idname == 'DataNodeType':
                if node.dbtype == 'LMDB':
                    string = datatemplate(node.name, node.outputs[0].output_name, node.outputs[1].output_name, node.batchsize,
                                        node.trainpath, node.testpath, node.shuffle, node.supervised,
                                        node.dbtype, node.usemeanfile, node.imsize, node.maxval, node.mirror,
                                        node.meanfile, node.silout)
                    dstring = deploytemplate(node.batchsize, node.channels, node.imsize, node.name)
                elif node.dbtype == 'Image files':
                    string = datatemplate(node.name, node.outputs[0].output_name, node.outputs[1].output_name,
                                        node.batchsize, node.trainfile, node.testfile, node.shuffle, node.supervised,
                                        node.dbtype, node.usemeanfile, node.imsize, node.maxval, node.mirror,
                                        node.meanfile, node.silout, channels=node.channels)
                    dstring = deploytemplate(node.batchsize, node.channels, node.imsize, node.name)
                elif node.dbtype == 'HDF5Data':
                    string = datatemplate(node.name, node.outputs[0].output_name, node.outputs[1].output_name,
                                        node.batchsize, node.trainHDF5, node.trainHDF5, node.shuffle, node.supervised,
                                        node.dbtype, node.usemeanfile, node.imsize, node.maxval, node.mirror,
                                        node.meanfile, node.silout, channels=node.channels)
                    dstring = deploytemplate(node.batchsize, node.channels, node.imsize, node.name)
            elif node.bl_idname == 'PoolNodeType':
                string = pooltemplate(node.name, node.kernel, node.stride, node.mode, bottoms[0], node.outputs[0].output_name)
                dstring = string                
                dstring = string
            elif node.bl_idname == 'ConvNodeType':
                string = convtemplate(node,node.name, node.OutputLs, node.Padding, node.kernelsize, node.Stride, bottoms[0], node.outputs[0].output_name,
                                    node.biasfill, node.filterlr, node.biaslr, node.filterdecay, node.biasdecay,
                                    node.std, node.weights,nonsquare=node.nonsquare,x=node.kernelsizex,y=node.kernelsizey)
                dstring = string
            elif node.bl_idname == 'DeConvNodeType':
                string = deconvtemplate(node,node.name, node.OutputLs, node.Padding, node.kernelsize, node.Stride,
                                        bottoms[0], node.outputs[0].output_name,
                                        node.biasfill, node.filterlr, node.biaslr, node.filterdecay, node.biasdecay,
                                        node.std, node.weights,nonsquare=node.nonsquare,x=node.kernelsizex,y=node.kernelsizey)
                dstring = string
            elif node.bl_idname == 'FCNodeType':
                string = FCtemplate(node.name, node.outputnum, bottoms[0], node.outputs[0].output_name, node.sparse, node.weights, node.biasfill,
                                    node.filterlr, node.biaslr, node.filterdecay, node.biasdecay, node.std,
                                    node.sparsity)
                dstring = string
            elif node.bl_idname == 'FlattenNodeType':
                string = flattentemplate(node.name, bottoms[0], node.outputs[0].output_name)
                dstring = string
            elif node.bl_idname == 'SilenceNodeType':
                string = silencetemplate(node.name, bottoms[0])
                dstring = string
            elif node.bl_idname == 'LRNNodeType':
                string = LRNtemplate(node.name, bottoms[0], node.outputs[0].output_name, node.alpha, node.beta, node.size, node.mode)
                dstring = string
            elif node.bl_idname == 'AcNodeType':
                string = NLtemplate(node.name, bottoms[0], node.outputs[0].output_name, node.mode)
                dstring = string
            elif node.bl_idname == 'ReluNodeType':
                string = Relutemplate(bottoms[0], node.outputs[0].output_name, node.name, node.Negativeg)
                dstring = string
            elif node.bl_idname == 'PReluNodeType':
                string = PRelutemplate(node, in1)
                dstring = string    
            elif node.bl_idname == 'DropoutNodeType':
                string = dropouttemplate(node.name, bottoms[0], node.outputs[0].output_name, node.fac)
                dstring = string
            elif node.bl_idname == 'SMLossNodeType':
                string = SMtemplate(node.name, bottoms[0], bottoms[1], node.outputs[0].output_name, node.w)
                dstring = ''
            elif node.bl_idname == 'SCELossNodeType':
                string = SCEtemplate(node.name, bottoms[0],bottoms[1], node.outputs[0].output_name, node.w)
                dstring = ''
            elif node.bl_idname == 'EULossNodeType':
                string = EUtemplate(node.name, bottoms[0], bottoms[1], node.outputs[0].output_name, node.w)
                dstring = ''
            elif node.bl_idname == 'ConcatNodeType':
                string = Concattemplate(node.name, bottoms[0], bottoms[1], node.outputs[0].output_name, node.axis)
                dstring = string
            elif node.bl_idname == 'AccuracyNodeType':
                string = accuracytemplate(node.name, bottoms[0], node.outputs[0].output_name, node.Testonly)
                dstring = ''
            elif node.bl_idname == 'ArgMaxNodeType':
                string = argmaxtemplate(node.name, bottoms[0], node.outputs[0].output_name, node.OutMaxVal, node.TopK)
                dstring = string
            elif node.bl_idname == 'HDF5OutputNodeType':
                string = hdf5outputtemplate(node.name, bottoms[0], node.filename)
                dstring = ''
            elif node.bl_idname == 'LogNodeType':
                string = logtemplate(node.name, bottoms[0], node.outputs[0].output_name, node.scale, node.shift, node.base)
                dstring = string;
            elif node.bl_idname == 'PowerNodeType':
                string = powertemplate(node.name, bottoms[0], node.outputs[0].output_name, node.power, node.scale, node.shift)
                dstring = string;
            elif node.bl_idname == 'ReductionNodeType':
                string = reductiontemplate(node.name, bottoms[0], node.outputs[0].output_name, node.operation, node.axis, node.coeff)
                dstring = string;
            elif node.bl_idname == 'SliceNodeType':
                tops = []
                for output in node.outputs:
                    tops.append(output.output_name)
                slice_points = []
                for slice_point in node.slice_points:
                    slice_points.append(slice_point.slice_point)
                string = slicetemplate(node.name, bottoms[0], tops, node.axis, slice_points)
            elif node.bl_idname == 'NodeReroute':
                string = ''
                dstring = ''
            elif node.bl_idname == 'SolverNodeType':
                solverstring = solvertemplate(node.solver, node.learningrate, node.testinterval, node.testruns,
                                            node.maxiter,
                                            node.displayiter, node.snapshotiter, node.solvername, node.snapshotpath,
                                            node.configpath, node.solvername, node.accumiters, solver=node.compmode)
                scriptstring = scripttemplate(node.caffexec, node.configpath, node.solvername, node.gpus,
                                            solver=node.compmode)
                configpath = node.configpath
                solvername = node.solvername
            elif string == 0:
                print (node.bl_idname)
            if node.bl_idname != 'SolverNodeType':
                gcode.extend([string])
                dcode.extend([dstring])
                gtops.extend([node.name])
                try:
                    gbottoms.extend([bottoms[0]])  # first node attached to current
                except IndexError:
                    gbottoms.extend([str(random.random())])
                try:
                    g2bottoms.extend([bottoms[1]])  # Second node attached to current
                except IndexError:
                    g2bottoms.extend([str(random.random())])
        for juggle in range(30):
            gtops, gbottoms, g2bottoms, gcode, dcode = self.juggleorder(gtops, gbottoms, g2bottoms, gcode, 0, dcode)
            # for chunk in gcode:
            # print (chunk)
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
        print ('Finished solving tree')
        return {'FINISHED'}  # this lets blender know the operator finished successfully.

    def juggleorder(self, names, refs, refs2, code, prefsocket, dcode):

        '''Ever heard of a bubble sort? Meet the worlds most complicated function designed to do just that.
        It checks whether a node is dependent on the node below it, and orders all the laters in the prototxt
        by a reference number. For some reason it sort of does it twice. Best just not to touch this and hope it never
        breaks as no-one will ever EVER work out how fix it.'''
        # Names, in 1, in2, code chunk, ??, deploy code chunk
        goodorder = 0
        checks = [1] * len(names)  #make list of zeros, length names
        while sum(checks) > 0:
            for name in names:
                Referred1Socket = 0
                Bottomless = 0
                Referred2Socket = 0
                # Start of list is data layer
                # get location of bottom in top
                # print (name)
                #print (names)
                loc = names.index(name)
                try:
                    ref = refs.index(name)  # find where the current node is referred to
                    Referred1Socket = 1
                except ValueError:
                    pass
                try:
                    float(name)  #we used a float name for nodes that are bottomless
                    print ('passing float')
                    print (name)
                    Bottomless = 1
                except ValueError:
                    pass
                try:
                    tmpref = refs2.index(name)  #check a node isnt reffered to as the second socket
                    if Referred1Socket == 1 and prefsocket == 1:
                        ref = tmpref  #only put before if on second socket pass, or does not connect to a first socket
                    elif Referred1Socket == 0:  #(Will not be a bottomless node as connects to at least one socket)
                        ref = tmpref
                    Referred2Socket = 1
                except ValueError:
                    pass
                if Referred1Socket + Bottomless + Referred2Socket == 0:
                    # not referred to by anything, so can be as late as possible
                    ref = 10000000000000000
                    #time.sleep(10)
                #ref = 10000000
                if ref < loc:
                    names, refs, refs2, code, dcode = self.swap(loc, ref, (names, refs, refs2, code, dcode))
                    checks[loc] = 0
                else:
                    checks[loc] = 0
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
