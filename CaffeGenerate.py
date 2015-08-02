__author__ = 'hugh'
bl_info = {
    "name": "Create Caffe solution",
    "category": "Object",
}

import bpy
import random
import time
import os

def getFillerString(filler, name):
    fillerString = tab3 + 'type: "%s"\n' % filler.type
    
    if filler.type == 'constant':
        fillerString += tab3 + 'value: %f\n' % (filler.value)
    elif filler.type == 'xavier' or filler.type == 'msra':
        fillerString += tab3 + 'variance_norm: %s\n' % (filler.variance_norm)
    elif filler.type == 'gaussian':
        fillerString += tab3 + 'mean: %f\n' % filler.mean
        fillerString += tab3 + 'std: %f\n' % filler.std
        if filler.is_sparse:
            fillerString += tab3 + 'sparse: %i\n' % (filler.sparse)
    elif filler.type == 'uniform':
        fillerString += tab3 + 'min: %f\n' % filler.min
        fillerString += tab3 + 'max: %f\n' % filler.max

    string = '''\
        %s {
%s
        }
''' % (name, fillerString)
    return string

def conv_template(node):
    
    if node.square_padding:
        padding_string = tab2 + 'pad: %i\n' % node.pad
    else:
        padding_string = tab2 + 'pad_h: %i\n' % node.pad_h
        padding_string += tab2 + 'pad_w: %i\n' % node.pad_w

    if node.square_kernel:
        kernel_string = tab2 + 'kernel_size: %i\n' % node.kernel_size
    else:
        kernel_string = tab2 + 'kernel_h: %i\n' % node.kernel_h
        kernel_string += tab2 + 'kernel_w: %i\n' % node.kernel_w

    if node.square_stride:
        stride_string = tab2 + 'stride: %i\n' % node.stride
    else:
        stride_string = tab2 + 'stride_h: %i\n' % node.stride_h
        stride_string += tab2 + 'stride_w: %i\n' % node.stride_w

    weight_filler_string = getFillerString(node.weight_filler, 'weight_filler')
    bias_filler_string = getFillerString(node.bias_filler, 'bias_filler')

    string = '''\
    convolution_param {
        num_output: %i
        bias_term: %i
%s
%s
%s
%s
%s
    }
''' % (node.num_output, node.bias_term, padding_string, kernel_string, stride_string, weight_filler_string, bias_filler_string)
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

def data_param_template(node, source):
    string = '''\
    data_param {
        source: "%s"
        backend: %s
        batch_size: %i
        rand_skip: %i
    }
''' % (source, node.db_type, node.batch_size, node.rand_skip)
    return string

def image_data_param_template(node, source):
    string = '''\
    image_data_param {
        source: "%s"
        batch_size: %i
        rand_skip: %i
        shuffle: %i
        new_height: %i
        new_width: %i
        is_color: %i
    }
''' % (source, node.batch_size, node.rand_skip, node.shuffle, node.new_height, node.new_width, node.is_color)
    return string


#TODO: Finish mean_value and random crop
def transform_param_template(node):
    mean_file_string = ''
    if node.use_mean_file:
        mean_file_string = tab2 + 'mean_file: "%s"\n' % node.mean_file
    
    string = '''\
    transform_param {
        scale: %f
        mirror: %i
%s
    }
''' % (node.scale, node.mirror, mean_file_string)

    return  string

def hdf5_data_template(node, source):
    string = '''\
    hdf5_data_param {
        source: "%s"
        batch_size: %i
        shuffle: %i
    }
''' % (source, node.batch_size, node.shuffle)

    return string


#def pooltemplate(name, kernel, stride, mode, bottom, top):
#    string = '''\
#layer {
#    name: "%s"
#    type: "Pooling"
#    bottom: "%s"
#    top: "%s"
#    pooling_param {
#        pool: %s
#        kernel_size: %i
#        stride: %i
#    }
#}
#''' % (name, bottom, top, mode, kernel, stride)
#    return string

def pool_template(node):
    string = '''\
    pooling_param {
        pool: %s
        kernel_size: %i
        stride: %i
    }
''' % (node.mode, node.kernel_size, node.stride)

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

def FC_template(node):
    weight_filler_string = getFillerString(node.weight_filler, 'weight_filler')
    bias_filler_string = getFillerString(node.bias_filler, 'bias_filler')

    string = '''\
    inner_product_param {
        num_output: %i
        bias_term: %i
%s
%s
        axis: %i
    }
''' % (node.num_output, node.bias_term, weight_filler_string, bias_filler_string, node.axis)

    return string


#TODO: Add to new version.
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


def Concattemplate(node):
    string = '''\
    concat_param {
        axis: %i
    }
    ''' % (node.axis)
    return string

def argmaxtemplate(node):
    string = '''\
    argmax_param {
        out_max_val: %i
        top_k: %i
    }
''' % (node.OutMaxVal, node.TopK)
    return string

def hdf5outputtemplate(node):
    string = '''\
    hdf5_output_param {
        file_name: "%s"
    }
}
''' % (node.filename)
    return string

def logtemplate(node):
    string = '''\
    log_param {
        scale: %f
        shift: %f
        base: %f
    }
''' % (node.scale, node.shift, node.base)
    return string

def powertemplate(node):
    string = '''\
    power_param {
        power: %f
        scale: %f
        shift: %f
    }
''' % (node.power, node.scale, node.shift)
    return string

def reductiontemplate(node):
    string = '''\
    reduction_param {
        operation: %s
        axis: %i
        coeff: %f
    }
''' % (node.operation, node.axis, node.coeff)
    return string

def slicetemplate(node):
    slice_points_string = '\n'.join(map(lambda x: tab2 + 'slice_point: %i' % x.slice_point, node.slice_points))
    
    string = '''\
    slice_param {
        axis: %i
%s
    }
''' % (node.axis, slice_points_string)
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


tab = '    '
tab2 = tab + tab
tab3 = tab2 + tab

def loss_weight_template(loss_weight):
    return tab + 'loss_weight: %f' % loss_weight


def param_template(param):
    string = tab + 'params {\n'
    
    if param.name.strip():
        string += tab2 + 'name: "%s"\n' % param.name

    string += tab2 + 'lr_mult: %f\n' % param.lr_mult
    string += tab2 + 'decay_mult: %f\n' % param.decay_mult
#    string += tab2 + 'share_mode: %s\n' % param.share_mode
    string += tab + '}'
    return string

def get_params(node):
    params = []
    if node.extra_params:
        params.append(param_template(node.weight_params))
        params.append(param_template(node.bias_params))
    return params

def get_include_in(node):
    if node.include_in == "BOTH":
        return ''
    
    string = '''\
    include {
        phase: %s
    }
''' % node.include_in
    
    return string


def layer_template(node, tops, bottoms, special_params):
    tops_string = '\n'.join(map(lambda x: tab + 'top: "%s"' % x, tops))
    bottoms_string = '\n'.join(map(lambda x: tab + 'bottom: "%s"' % x, bottoms))
    params_string = '\n'.join(get_params(node))
    special_params_string = '\n'.join(special_params)
    include_in_string = get_include_in(node)
    
    string = '''\
layer {
    name: "%s"
    type: "%s"
%s
%s
%s
%s
%s
}
''' % (node.name, node.n_type, tops_string, bottoms_string, params_string, special_params_string, include_in_string)
    
    return "\n".join(filter(lambda x: x.strip(), string.splitlines())) + "\n"


def LRNtemplate(node):
    string = '''\
    lrn_param {
        local_size: %i
        alpha: %f
        beta: %f
        norm_region: %s
    }
    ''' % (node.size, node.alpha, node.beta, node.mode)
    return string


def Relutemplate(node):
    string = '''\
    relu_param {
        negative_slope: %f
        engine: %s
    }
    ''' % (node.negative_slope, node.engine)
    return string

def dropouttemplate(node):
    string = '''\
    dropout_param {
        dropout_ratio: %f
    }
    ''' % (node.dropout_ratio)
    return string

class Vertex():
    pass

def reorder(graph):
    res_string = []
    res_dstring = []
    while len(graph) > 0:
        curr = min(graph, key = lambda x: len(x.bottoms))
#        print(curr.string)

        if len(curr.bottoms) != 0:
            print('Cycle in graph?!')
        
        res_string.append(curr.string)
        res_dstring.append(curr.dstring)
        
        for item in graph:
#            print(item.bottoms)
            for top in curr.tops:
                try:
#                    print(item.bottoms)
                    item.bottoms.remove(top)
#                    print("Removed")
                except:
                    pass
#            print("---")
#            print(item.bottoms)
#            print("-----")
        graph.remove(curr)
    return res_string, res_dstring

class Solve(bpy.types.Operator):
    """Generate Caffe solver"""  # blender will use this as a tooltip for menu items and buttons.
    bl_idname = "nodes.make_solver"  # unique identifier for buttons and menu items to reference.
    bl_label = "Create Solution"  # display name in the interface.
    bl_options = {'REGISTER'}  # enable undo for the operator.

    def execute(self, context):  # execute() is called by blender when running the operator.
        graph = []
        ########################################### Main loop
        for node in context.selected_nodes:
            nname = node.name
            string = ''
            bottoms = []
            for input in node.inputs:
                if input.is_linked:
                    bottom = input.links[0].from_socket.output_name
                    bottoms.append(bottom)
        
            tops = [x.output_name for x in node.outputs]
            special_params = []

            ###########################
            if node.bl_idname == 'DataNodeType':
                transform_param = transform_param_template(node)
                node.n_type = node.db_type

                if node.db_type in ('LMDB', 'LEVELDB'):
                    train_params = [data_param_template(node, node.train_path)]
                    test_params = [data_param_template(node, node.test_path)]
                elif node.db_type == 'ImageData':
                    train_params = [image_data_param_template(node, node.train_data)]
                    test_params = [image_data_param_template(node, node.test_data)]
                elif node.db_type == 'HDF5Data':
                    train_params = [hdf5_data_template(node, node.train_data)]
                    test_params = [hdf5_data_template(node, node.test_data)]
                
                train_params.append(transform_param)
                test_params.append(transform_param)
                
                node.include_in = "TRAIN"
                train_string = layer_template(node, tops, bottoms, train_params)
                node.include_in = "TEST"
                test_string = layer_template(node, tops, bottoms, test_params)
                
                string = train_string + test_string
                
                #TODO: Finish dstring
                dstring = ''
                
#                if node.dbtype == 'LMDB':
#                    string = datatemplate(node.name, node.outputs[0].output_name, node.outputs[1].output_name, node.batchsize,
#                                        node.trainpath, node.testpath, node.shuffle, node.supervised,
#                                        node.dbtype, node.usemeanfile, node.imsize, node.maxval, node.mirror,
#                                        node.meanfile, node.silout)
#                    dstring = deploytemplate(node.batchsize, node.channels, node.imsize, node.name)
#                elif node.dbtype == 'Image files':
#                    string = datatemplate(node.name, node.outputs[0].output_name, node.outputs[1].output_name,
#                                        node.batchsize, node.trainfile, node.testfile, node.shuffle, node.supervised,
#                                        node.dbtype, node.usemeanfile, node.imsize, node.maxval, node.mirror,
#                                        node.meanfile, node.silout, channels=node.channels)
#                    dstring = deploytemplate(node.batchsize, node.channels, node.imsize, node.name)
#                elif node.dbtype == 'HDF5Data':
#                    string = datatemplate(node.name, node.outputs[0].output_name, node.outputs[1].output_name,
#                                        node.batchsize, node.trainHDF5, node.trainHDF5, node.shuffle, node.supervised,
#                                        node.dbtype, node.usemeanfile, node.imsize, node.maxval, node.mirror,
#                                        node.meanfile, node.silout, channels=node.channels)
#                    dstring = deploytemplate(node.batchsize, node.channels, node.imsize, node.name)
            elif node.bl_idname == 'PoolNodeType':
                special_params.append(pool_template(node))
            elif node.bl_idname == 'ConvNodeType':
                special_params.append(conv_template(node))
            elif node.bl_idname == 'DeConvNodeType':
                special_params.append(conv_template(node))
            elif node.bl_idname == 'FCNodeType':
                special_params.append(FC_template(node))
            elif node.bl_idname == 'FlattenNodeType':
#                string = layer_template(node.name, "Flatten", tops, bottoms, params, [], include_in)
#                string = flattentemplate(node.name, bottoms[0], node.outputs[0].output_name)
                dstring = string
            elif node.bl_idname == 'SilenceNodeType':
#                string = layer_template(node.name, "Silence", tops, bottoms, params, [], include_in)
#                string = silencetemplate(node.name, bottoms[0])
                dstring = string
            elif node.bl_idname == 'LRNNodeType':
                special_params.append(LRNtemplate(node))
                dstring = string
            elif node.bl_idname == 'AcNodeType':
                node.type = node.mode
            elif node.bl_idname == 'ReluNodeType':
                special_params.append(Relutemplate(node))
                dstring = string
            elif node.bl_idname == 'PReluNodeType':
                string = PRelutemplate(node, in1)
                dstring = string
            elif node.bl_idname == 'DropoutNodeType':
                special_params.append(dropouttemplate(node))
                dstring = string
            elif node.bl_idname == 'SMLossNodeType':
                special_params.append(loss_weight_template(node.w))
                dstring = ''
            elif node.bl_idname == 'SCELossNodeType':
                special_params.append(loss_weight_template(node.w))
                dstring = ''
            elif node.bl_idname == 'EULossNodeType':
                special_params.append(loss_weight_template(node.w))
                dstring = ''
            elif node.bl_idname == 'ConcatNodeType':
                special_params.append(Concattemplate(node))
            elif node.bl_idname == 'AccuracyNodeType':
                dstring = ''
            elif node.bl_idname == 'ArgMaxNodeType':
                special_params.append(argmaxtemplate(node))
                dstring = string
            elif node.bl_idname == 'HDF5OutputNodeType':
                special_params.append(hdf5outputtemplate(node))
                dstring = ''
            elif node.bl_idname == 'LogNodeType':
                special_params.append(logtemplate(node))
                dstring = string;
            elif node.bl_idname == 'PowerNodeType':
                special_params.append(powertemplate(node))
                dstring = string;
            elif node.bl_idname == 'ReductionNodeType':
                special_params.append(reductiontemplate(node))
                dstring = string;
            elif node.bl_idname == 'SliceNodeType':
                special_params.append(slicetemplate(node))
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
                if node.bl_idname != 'DataNodeType':
                    string = layer_template(node, tops, bottoms, special_params)
                    dstring = string
                
                v = Vertex()
                v.string = string
                v.dstring = dstring
                v.bottoms = bottoms
                v.tops = tops
                graph.append(v)
                
        strings, dstrings = reorder(graph)
        solution = ''.join(strings)
        dsolution = ''.join(dstrings)
    
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


def register():
    bpy.utils.register_class(Solve)


def unregister():
    bpy.utils.unregister_class(Solve)


# This allows you to run the script directly from blenders text editor
# to test the addon without having to install it.
if __name__ == "__main__":
    register()
