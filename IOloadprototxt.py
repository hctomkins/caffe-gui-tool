__author__ = 'H'

import os
import random
from .CGTArrangeHelper import ArrangeFunction
from .IOparse import search as findfirstraw
import bpy
import string


def format_filename(s):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ','_') # I don't like spaces in filenames.
    return filename


class fclass(object):
    pass


class nodeclass(object):
    pass


def findfirst(search, string):
    searchresult = findfirstraw(search, string)
    if searchresult:
        toset = searchresult.fixed[0]
        return toset


def findsetbeforecolon(attr, ob, chunk, number=False):
    if not number:
        searchresult = findfirst(attr + ': "{}"\n', chunk)
    else:
        searchresult = findfirst(attr + ': {:g}\n', chunk)
    ob.__setattr__(attr, searchresult)


def findmultiple(search, string):
    outs = []
    while 1:
        out = findfirstraw(search, string)
        # print out
        if out == None:
            break
        endpos = next(iter(out.spans.values()))[1]
        outs.extend(list(out.fixed))
        if out.named:
            for pos in range(len(out.named)):
                outs.append(next(iter(out.named.values())))
        string = string[endpos:]
    if outs == []:
        return None
    return outs


def getsize(deploy):
    deploy = open(deploy, 'r').read()
    if 'input_shape' in deploy:
        startpos = deploy.index('input_shape')
        shapestring = ''
        for letter in deploy[startpos:]:
            shapestring += letter
            if letter == '}':
                break
        dims = findmultiple('dim: {:g}\n', shapestring)
        print (dims)
        return dims[-1], dims[-2]
    else:
        return 0, 0


#####################################


class textlayerob(object):
    def __init__(self, chunkstring):
        self.chunkstring = chunkstring
        self.bottoms = []
        self.tops = []
        self.parse()

    def parse(self):
        node = nodeclass()
        node.weight_params = fclass()
        node.bias_params = fclass()
        chunkstring = self.chunkstring
        self.type = findfirst('type: "{}"',
                              chunkstring)  # the use of the default parse search gets the first instance of type
        findsetbeforecolon('name', node, chunkstring)
        node.include_in = findfirst('phase: {}\n', chunkstring)
        self.bottoms = findmultiple('bottom: "{}"', chunkstring)
        self.tops = findmultiple('top: "{}"', chunkstring)
        #################### Many layer specific
        decaymults = findmultiple('decay_mult: {:g}\n', chunkstring)
        lrmults = findmultiple('lr_mult: {:g}\n', chunkstring)
        if decaymults:
            node.extra_params = True
            node.weight_params.decay_mult = decaymults[0]
            node.bias_params.decay_mult = decaymults[1]
        if lrmults:
            node.extra_params = True
            node.weight_params.lr_mult = lrmults[0]
            node.bias_params.lr_mult = lrmults[1]
        findsetbeforecolon('kernel_size', node, chunkstring, True)
        if not node.kernel_size:
            findsetbeforecolon('kernel_h', node, chunkstring, True)
            findsetbeforecolon('kernel_w', node, chunkstring, True)
            if node.kernel_h and node.kernel_w:
                node.square_kernel = 0
        findsetbeforecolon('pad', node, chunkstring, True)
        if not node.pad:
            findsetbeforecolon('pad_h', node, chunkstring, True)
            findsetbeforecolon('pad_w', node, chunkstring, True)
            if node.pad_h and node.pad_w:
                node.square_padding = 0
        findsetbeforecolon('stride', node, chunkstring, True)
        if not node.stride:
            findsetbeforecolon('stride_h', node, chunkstring, True)
            findsetbeforecolon('stride_w', node, chunkstring, True)
            if node.stride_h and node.stride_w:
                node.square_stride = 0
        ################################ fillers
        node.bias_filler = self.getfiller('bias')
        node.weight_filler = self.getfiller('weight')
        if self.type == 'Pooling':
            node.mode = findfirst('pool: {}\n', chunkstring)
        ################################ Database
        source = findfirst('source: "{}"\n', chunkstring)
        db_end = 'Null'
        for line in chunkstring.split('\n'):
            if 'data_param' in line:
                if 'image' in line:
                    node.db_type = 'ImageData'
                elif 'hdf5' in line:
                    node.db_type = 'HDF5Data'
                else:
                    db_end = 'db'
        if db_end == 'db':
            node.db_type = findfirst('backend: {}\n', chunkstring)
        batch_size = findfirst('batch_size: {:g}\n', chunkstring)
        if batch_size:
            if node.include_in == 'TRAIN':
                node.train_batch_size = batch_size
                node.train_path = source
                node.train_data = source
            else:
                node.test_batch_size = batch_size
                node.test_path = source
                node.test_data = source

        ######################################### Other params
        SimpleNumberProperties = ['test_initialization', 'base_lr', 'display', 'average_loss', 'max_iter', 'iter_size',
                                  'momentum', 'weight_decay', 'snapshot', 'snapshot_diff', 'debug_info',
                                  'snapshot_after_train', 'alpha', 'beta', 'negative_slope', 'dropout_ratio',
                                  'random_seed',
                                  'stepsize', 'gamma', 'delta', 'power', 'base', 'scale', 'shift', 'channel_shared',
                                  'num_output', 'axis', 'stable_prod_grad', 'coeff', 'eps', 'across_channels',
                                  'normalize_variance', 'mirror', 'is_color', 'new_width', 'new_height', 'shuffle',
                                  'rand_skip', 'test_compute_loss', 'test_iter']
        SimpleStringProperties = ['solver_mode', 'lr_policy', 'solver_type', 'regularization_type', 'operation',
                                  'mean_file']
        for prop in SimpleNumberProperties:
            findsetbeforecolon(prop, node, chunkstring, True)
        for prop in SimpleStringProperties:
            findsetbeforecolon(prop, node, chunkstring, False)
        if node.mean_file:
            node.use_mean_file = 1
        if node.random_seed:
            node.use_random_seed = True
        if self.type == 'Solver':
            node.solvername = format_filename(findfirst(os.path.sep + '{}' + '_train', chunkstring))
            if len(node.solvername) > 15:
                node.solvername = node.solvername[15:]
        node.OutMaxVal = findfirst('out_max_val: {:g}\n', chunkstring)  ################
        node.TopK = findfirst('top_k: {:g}\n', chunkstring)  ###################
        node.filename = findfirst('file_name: {}', chunkstring)  #######################
        node.slice_points = findmultiple('slice_point {:g}\n', chunkstring)
        sp = findfirst('snapshot_prefix: "{}"\n', chunkstring)
        if sp:
            node.snapshot_prefix = os.path.split(sp)[0] + os.path.sep
        node.size = findfirst('local_size: {:g}\n', chunkstring)
        node.mode = findfirst('norm_region: {}\n', chunkstring)
        if self.type == 'Concat':
            node.input_amount = len(self.bottoms)
        for parametername in dir(node)[dir(node).index('__weakref__') + 1:]:
            if node.__getattribute__(parametername) == None:
                node.__delattr__(parametername)

        self.node = node

    def getfiller(self, type):
        chunkstring = self.chunkstring
        filler = type + '_filler' in chunkstring
        if filler:
            filler = fclass()
            posf = chunkstring.index(type + '_filler')
            fillerstring = chunkstring[posf:]
            f_type = findfirst('type: "{}"', chunkstring[posf:])
            filler.type = f_type
            if filler.type == 'constant':
                filler.value = findfirst('value: {:g}\n', fillerstring)
            elif filler.type == 'xavier' or filler.type == 'msra':
                filler.variance_norm = findfirst('variance_norm: {:g}\n', fillerstring)
            elif filler.type == 'gaussian':
                filler.mean = findfirst('mean: {:g}\n', fillerstring)
                filler.std = findfirst('std: {:g}\n', fillerstring)
                filler.sparse = findfirst('sparse: {:g}\n', fillerstring)
                filler.is_sparse = bool(filler.sparse)
            elif filler.type == 'uniform':
                filler.min = findfirst('min: {:g}\n', fillerstring)
                filler.max = findfirst('max: {:g}\n', fillerstring)
            for parametername in dir(filler)[dir(filler).index('__weakref__') + 1:]:
                if filler.__getattribute__(parametername) == None:
                    filler.__delattr__(parametername)
            return filler
        else:
            return None


def readprototxt(path):
    with open(path) as f:
        prototxt = []
        for line in f.readlines():
            prototxt.extend([line])
    return prototxt


def getlayers(prototxt):
    # print prototxt
    layers = []
    inlayer = 0  # discard first lines of proto2
    bracketcounter = []  # track brackets in layer
    currentlayer = ''
    for line in prototxt:
        bracketcounter.extend([1 if letter == '{' else -1 if letter == '}' else 0 for letter in line])
        # print 'Bracketpart ' + str(sum(bracketcounter))
        if sum(bracketcounter) == 0 and inlayer:
            inlayer = 0
            if currentlayer:
                layers.append(textlayerob(currentlayer))
            currentlayer = ''
            bracketcounter = []
        elif inlayer:
            currentlayer += line
        elif 'layer' in line:
            inlayer = 1
    return layers


def LoadFunction(prototxt, y, x, nh=False, nw=False, h=False, w=False,operatorself=None):
    nodetypes = {'Pooling': 'PoolNodeType', 'Eltwise': 'EltwiseNodeType', 'Exp': 'ExpNodeType',
                 'Convolution': 'ConvNodeType', 'Deconvolution': 'DeConvNodeType', 'InnerProduct': 'FCNodeType',
                 'Flatten': 'FlattenNodeType', 'Silence': 'SilenceNodeType', 'LRN': 'LRNNodeType',
                 'Sigmoid': 'AcNodeType', 'TanH': 'AcNodeType', 'ReLU': 'ReluNodeType', 'PReLU': 'PReluNodeType',
                 'Dropout': 'DropoutNodeType', 'SoftmaxWithLoss': 'SMLossNodeType','Softmax': 'SMLossNodeType',
                 'SigmoidCrossEntropyLoss': 'SCELossNodeType', 'EuclideanLoss': 'EULossNodeType',
                 'Concat': 'ConcatNodeType', 'Accuracy': 'AccuracyNodeType',
                 'ArgMax': 'ArgMaxNodeType', 'HDF5Output': 'HDF5OutputNodeType', 'Log': 'LogNodeType',
                 'Power': 'PowerNodeType', 'Reduction': 'ReductionNodeType', 'Slice': 'SliceNodeType',
                 'MVN': 'MVNNodeType', 'Solver': 'SolverNodeType', 'Data': 'DataNodeType'}
    textlayers = getlayers(prototxt)
    prevtrees = bpy.data.node_groups.items()
    bpy.ops.node.new_node_tree(type='CaffeNodeTree', name="NodeTree")
    newtrees = bpy.data.node_groups.items()
    tree = list(set(newtrees) - set(prevtrees))[0][1]
    tree.name = 'Loaded'
    links = tree.links
    textlayers = [i for i in textlayers if i.type != None]
    Datanode = False
    # make and set up nodes
    for textlayer in textlayers:
        nodesbefore = bpy.data.node_groups[tree.name].nodes.items()
        tree.nodes.new(nodetypes[textlayer.type])
        nodesafter = bpy.data.node_groups[tree.name].nodes.items()
        node = list(set(nodesafter) - set(nodesbefore))[0][1]
        node.select = True
        if hasattr(textlayer.node, 'name'):
            node.name = textlayer.node.name
        elif textlayer.type == 'Solver':
            node.name = 'Solver'
            textlayer.node.name = 'Solver'
        else:
            textlayer.node.name = str(random.random())
            node.name = textlayer.node.name
        for basicparametername in dir(textlayer.node)[dir(textlayer.node).index('__weakref__') + 1:]:
            if basicparametername not in ['bias_filler', 'weight_filler', 'weight_params', 'bias_params']:
                getattr(node, basicparametername)
                node.__setattr__(basicparametername, getattr(textlayer.node, basicparametername))
            else:
                for parametername in dir(getattr(textlayer.node, basicparametername))[
                                     dir(getattr(textlayer.node, basicparametername)).index('__weakref__') + 1:]:
                    setattr(getattr(node, basicparametername), parametername,
                            getattr(getattr(textlayer.node, basicparametername), parametername))
        if 'Data' in textlayer.type:
            Datanode = True
            for pos in range(2):
                node.outputs[pos].output_name = textlayer.tops[pos]
            if nh or nw or h or w:
                node.new_height = nh
                node.height = h
                node.new_width = nw
                node.width = w
            else:
                node.new_height = y
                node.height = y
                node.new_width = x
                node.width = x



    # Get inplaces
    inplaceafternodes = {}  ### {Nodebefore: [inplace,inplace,inplace] }
    for textlayer in textlayers:
        if textlayer.bottoms:
            if textlayer.bottoms == textlayer.tops:
                # print (textlayer.node.name)
                # print(textlayer.bottoms)
                # print (textlayer.tops)
                textlayer.inplace = 1
                if textlayer.bottoms[0] not in inplaceafternodes:
                    inplaceafternodes[textlayer.bottoms[0]] = []
                inplaceafternodes[textlayer.bottoms[0]].append(textlayer)
            else:
                textlayer.inplace = 0
        else:
            textlayer.inplace = 0


    # Join Not in place Nodes
    for textlayer in textlayers:
        if textlayer.tops and not textlayer.inplace:
            for startpos, top in enumerate(textlayer.tops):
                startnode = bpy.data.node_groups[tree.name].nodes[textlayer.node.name]
                found = False
                for othertextlayer in textlayers:
                    if othertextlayer.bottoms and not othertextlayer.inplace and othertextlayer != textlayer:
                        if top in othertextlayer.bottoms:
                            # if len(othertextlayer.bottoms) > 2:
                            #     print (othertextlayer.chunkstring)
                            endpos = othertextlayer.bottoms.index(top)
                            endnodename = othertextlayer.node.name
                            endnode = bpy.data.node_groups[tree.name].nodes[endnodename]
                            # print('Link')
                            link(startnode, endnode, startpos, endpos, links)
                            found = True
                if not found:
                    print(top)



    # Join in place nodes
    for basetop in inplaceafternodes:
        for textlayer in textlayers:
            if textlayer.tops:
                if basetop in textlayer.tops and not textlayer.inplace:
                    baselayer = textlayer
        for position, inplacetextlayer in enumerate(inplaceafternodes[basetop]):
            if position == 0:
                # print (bpy.data.node_groups[tree.name].nodes.items())
                startnode = bpy.data.node_groups[tree.name].nodes[baselayer.node.name]
                endnode = bpy.data.node_groups[tree.name].nodes[inplacetextlayer.node.name]
                startpos = baselayer.tops.index(inplacetextlayer.bottoms[0])
                endpos = 0
                link(startnode, endnode, startpos, endpos, links)
            else:
                startnode = bpy.data.node_groups[tree.name].nodes[
                    inplaceafternodes[basetop][position - 1].node.name]
                endnode = bpy.data.node_groups[tree.name].nodes[inplaceafternodes[basetop][position].node.name]
                link(startnode, endnode, 0, 0, links)

    # Connect end of in place node chain
    for basename in inplaceafternodes:
        lastlayer = inplaceafternodes[basename][-1]
        for textlayer in textlayers:
            if textlayer.bottoms:
                if not textlayer.inplace and lastlayer.tops[0] in textlayer.bottoms:
                    endnode = bpy.data.node_groups[tree.name].nodes[textlayer.node.name]
                    startnode = bpy.data.node_groups[tree.name].nodes[lastlayer.node.name]
                    endpos = textlayer.bottoms.index(lastlayer.tops[0])
                    startpos = 0
        link(startnode, endnode, startpos, endpos, links)

    if not Datanode:
        if operatorself:
            operatorself.report({'ERROR'}, "No Data Node in protoxt. Please add data node manually.")


class Load(bpy.types.Operator):
    """Load Caffe solver"""
    bl_idname = "nodes.load_solver"
    bl_label = "Load solution"
    bl_options = {'REGISTER'}

    def execute(self, context):

        if 'deploy' in bpy.context.scene:
            deploy = bpy.context.scene['deploy']
        else:
            deploy = False
        if deploy:
            x, y = getsize(deploy)
        else:
            x = 0
            y = 0
        wholefile = readprototxt(bpy.context.scene['traintest'])
        solve = readprototxt(bpy.context.scene['solver'])
        prototxt = wholefile + ['\nlayer {\n'] + ['type: "Solver"\n'] + solve + ['\n}\n']
        prevtrees = bpy.data.node_groups.items()
        LoadFunction(prototxt, y, x,operatorself=self)
        newtrees = bpy.data.node_groups.items()
        tree = list(set(newtrees) - set(prevtrees))[0][1]
        tree.name = 'Loaded'
        ArrangeFunction(context, treename=tree.name)

        return {'FINISHED'}  # this lets blender know the operator finished successfully.


def link(start, end, startpos, endpos, links):
    start = start.outputs[startpos]
    end = end.inputs[endpos]
    links.new(start, end)
    return True


    # load('C:\Users\H\Documents\caffeconfigs\_train_test.prototxt','C:\Users\H\Documents\caffeconfigs\_solver.prototxt')


def register():
    bpy.utils.register_class(Load)


def unregister():
    bpy.utils.unregister_class(Load)
