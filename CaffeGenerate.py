#TODO: Add properties to solver


__author__ = 'hugh'
bl_info = {
    "name": "Create Caffe solution",
    "category": "Object",
}

import bpy
import random
import time
import os

tab = '    '
tab2 = tab + tab
tab3 = tab2 + tab

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

    weight_filler_string = ''
    bias_filler_string = ''
    if not node.use_custom_weight:
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

def data_param_template(node, source, batch_size):
    string = '''\
    data_param {
        source: "%s"
        backend: %s
        batch_size: %i
        rand_skip: %i
    }
''' % (source, node.db_type, batch_size, node.rand_skip)
    return string

def image_data_param_template(node, source, batch_size):
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
''' % (source, batch_size, node.rand_skip, node.shuffle, node.new_height, node.new_width, node.is_color)
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

def hdf5_data_template(node, source, batch_size):
    string = '''\
    hdf5_data_param {
        source: "%s"
        batch_size: %i
        shuffle: %i
    }
''' % (source, batch_size, node.shuffle)

    return string

def pool_template(node):
    string = '''\
    pooling_param {
        pool: %s
        kernel_size: %i
        stride: %i
    }
''' % (node.mode, node.kernel_size, node.stride)

    return string

# TODO: Finish MVN
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

def PReLU_template(node):
    filler_string = getFillerString(node.filler, 'filler')
    string = '''\
    prelu_param {
        channel_shared: %i
%s
    }
''' % (node.channel_shared, filler_string)
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
        tsstring = '''\
lr_policy: "step"
gamma: 0.1
stepsize: 10000
weight_decay: 0.0005
solver_type: ADAGRAD
'''
    elif type == 'NAG':
        tsstring = '''\
lr_policy: "step"\n\
gamma: 0.1
stepsize: 10000
weight_decay: 0.0005
momentum: 0.95
solver_type: NESTEROV
'''
    elif type == 'SGD':
        tsstring = '''\
lr_policy: "step"
gamma: 0.1
stepsize: 10000
weight_decay: 0.0005
momentum: 0.95
'''
    else:
        print ('ERROR')
        time.sleep(1000000)
    genericstring = '''\
net: "%s"
test_iter: %i
test_interval: %i
base_lr: %f
display: %i
max_iter: %i
iter_size: %i
snapshot: %i
snapshot_prefix: "%s"
solver_mode: %s
''' % (netpath, testruns, testinterval, learningrate, displayiter, maxiter, itersize, snapshotiter, snapshotprefix,
        solver)
    solverstring = genericstring + tsstring
    return solverstring


def deploytemplate(batch, channels, size, datain):
    deploystring = '''\
name: "Autogen"
input: "%s"
input_dim: %i
input_dim: %i
input_dim: %i
input_dim: %i
''' % (datain, batch, channels, size, size)
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
    string = "#!/usr/bin/env sh \n '%s' train --solver='%s' %s" % (caffestring, solverstring, extrastring)
    return string

def loss_weight_template(loss_weight):
    return tab + 'loss_weight: %f' % loss_weight


def param_template(param):
    string = tab + 'param {\n'
    
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
    
    custom_weight_string = ''
    if node.use_custom_weight:
        with open(node.custom_weight) as f:
            custom_weight_string = f.read()
    
        custom_weight_string = '\n'.join(tab + r for r in custom_weight_string.split('\n'))
    
    
    string = '''\
layer {
    name: "%s"
    type: "%s"
%s
%s
%s
%s
%s
%s
}
''' % (node.name, node.n_type, tops_string, bottoms_string, params_string, special_params_string, include_in_string, custom_weight_string)
    
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
    }
    ''' % (node.negative_slope)
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
        if len(curr.bottoms) != 0:
            print('Cycle in graph?!')
        
        res_string.append(curr.string)
        res_dstring.append(curr.dstring)
        
        for item in graph:
            for top in curr.tops:
                try:
                    item.bottoms.remove(top)
                except:
                    pass
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
                    train_params = [data_param_template(node, node.train_path, node.train_batch_size)]
                    test_params = [data_param_template(node, node.test_path, node.test_batch_size)]
                    node.n_type = 'Data'
                    train_params.append(transform_param)
                    test_params.append(transform_param)
                elif node.db_type == 'ImageData':
                    train_params = [image_data_param_template(node, node.train_data, node.train_batch_size)]
                    test_params = [image_data_param_template(node, node.test_data, node.test_batch_size)]
                    train_params.append(transform_param)
                    test_params.append(transform_param)
                elif node.db_type == 'HDF5Data':
                    train_params = [hdf5_data_template(node, node.train_data, node.train_batch_size)]
                    test_params = [hdf5_data_template(node, node.test_data, node.test_batch_size)]
                
                node.include_in = "TRAIN"
                train_string = layer_template(node, tops, bottoms, train_params)
                node.include_in = "TEST"
                test_string = layer_template(node, tops, bottoms, test_params)
                
                string = train_string + test_string
                
                #TODO: Finish dstring
                dstring = ''
            elif node.bl_idname == 'PoolNodeType':
                special_params.append(pool_template(node))
            elif node.bl_idname == 'ConvNodeType':
                special_params.append(conv_template(node))
            elif node.bl_idname == 'DeConvNodeType':
                special_params.append(conv_template(node))
            elif node.bl_idname == 'FCNodeType':
                special_params.append(FC_template(node))
            elif node.bl_idname == 'FlattenNodeType':
                dstring = string
            elif node.bl_idname == 'SilenceNodeType':
                dstring = string
            elif node.bl_idname == 'LRNNodeType':
                special_params.append(LRNtemplate(node))
            elif node.bl_idname == 'AcNodeType':
                node.type = node.mode
            elif node.bl_idname == 'ReluNodeType':
                special_params.append(Relutemplate(node))
            elif node.bl_idname == 'PReluNodeType':
                special_params.append(PReLU_template(node))
                dstring = string
            elif node.bl_idname == 'DropoutNodeType':
                special_params.append(dropouttemplate(node))
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
