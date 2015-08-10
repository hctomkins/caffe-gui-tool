bl_info = {
    "name": "Caffe Nodes",
    "category": "Object",
}
import bpy
import subprocess
from bpy.types import NodeTree, Node, NodeSocket

def calcsize(self, context,axis='x'):
    '''NOTE - this function works out the dimensions of an image by the time it has reached a certain layer.
    It traverses all the layers, builds up several lists about the properties of each layer, then computes the
    size up to a given layer.'''
    x = 0.0
    node = self
    # print (node.bl_idname)
    try:
        node.inputs[0].links[0].from_node
    except IndexError:
        return 0
    #These are the lists to be populated
    kernelsizes = []
    strides = []
    paddings = []
    poolstrides = []
    poolsizes = []
    offsets = []
    fcsizes = []
    reversals = []
    passes = []
    while 1 == 1:
        if node.bl_idname == "ConvNodeType":
            #print (node.inputs[0])
            #print(dir(node.inputs[0]))
            #print(type(node.inputs[0]))
            if not node.nonsquare:
                kernelsizes.extend([node.kernelsize])
            elif axis == 'x':
                kernelsizes.extend([node.kernelsizex])
            elif axis == 'y':
                kernelsizes.extend([node.kernelsizey])
            else:
                raise RuntimeError
            strides.extend([node.Stride])
            paddings.extend([node.Padding])
            poolsizes.extend([1])
            poolstrides.extend([1])
            offsets.extend([0])
            fcsizes.extend([0])
            passes.extend([0])
            reversals.extend([0])
            node = node.inputs[0].links[0].from_node
        if node.bl_idname == "DeConvNodeType":
            #print (node.inputs[0])
            #print(dir(node.inputs[0]))
            #print(type(node.inputs[0]))
            if not node.nonsquare:
                kernelsizes.extend([node.kernelsize])
            elif axis == 'x':
                kernelsizes.extend([node.kernelsizex])
            elif axis == 'y':
                kernelsizes.extend([node.kernelsizey])
            else:
                raise RuntimeError
            strides.extend([node.Stride])
            paddings.extend([node.Padding])
            poolsizes.extend([1])
            poolstrides.extend([1])
            offsets.extend([0])
            fcsizes.extend([0])
            passes.extend([0])
            reversals.extend([1])
            node = node.inputs[0].links[0].from_node
        elif node.bl_idname == "FCNodeType":
            #print (node.inputs[0])
            #print(dir(node.inputs[0]))
            #print(type(node.inputs[0]))
            kernelsizes.extend([0])
            strides.extend([0])
            paddings.extend([0])
            poolsizes.extend([0])
            poolstrides.extend([0])
            offsets.extend([0])
            passes.extend([0])
            reversals.extend([0])
            fcsizes.extend([node.outputnum])
            node = node.inputs[0].links[0].from_node
            #print (node)
        elif node.bl_idname == "PoolNodeType":
            kernelsizes.extend([0])
            paddings.extend([0])
            strides.extend([1])
            fcsizes.extend([0])
            passes.extend([0])
            reversals.extend([0])
            poolsizes.extend([node.kernel])
            poolstrides.extend([node.stride])
            offsets.extend([1])
            node = node.inputs[0].links[0].from_node
        elif node.bl_idname == "DataNodeType":
            # When the data node is reached, we must be at the back of the nodetree, so start to work forwards
            if not node.rim:
                x = float(node.imsize)
            elif axis=='x':
                x = float(node.imsizex)
            else:
                x = float(node.imsizey)
            # work forwards
            numofnodes = len(passes)
            for node in range(numofnodes):
                # - 1 as starts from 0
                node = (numofnodes - 1) - node
                padding = paddings[node]
                stride = strides[node]
                ksize = kernelsizes[node]
                offset = offsets[node]
                poolstride = poolstrides[node]
                poolsize = poolsizes[node]
                reversal = reversals[node]
                if passes[node] == 0:
                    if fcsizes[node] == 0:
                        if reversal ==0:
                            #########################
                            x = ((x + (2 * padding) - ksize) / stride + 1 - offset)
                            x = (x - poolsize) / poolstride + 1
                            ###################
                        else:
                            x = (x*stride - stride) + ksize - 2*padding
                    else:
                        x = fcsizes[node]
            break
        else:
            kernelsizes.extend([0])
            strides.extend([0])
            paddings.extend([0])
            poolsizes.extend([0])
            poolstrides.extend([0])
            offsets.extend([0])
            reversals.extend([0])
            fcsizes.extend([0])
            passes.extend([1])
            node = node.inputs[0].links[0].from_node
    return str(round(x, 2))

############################## Function for determining number of gpus
def getgpus():
    command = ['nvidia-smi','-L']
    try:
        proc = subprocess.Popen(command, bufsize=1,stdout=subprocess.PIPE, stderr=subprocess.STDOUT,universal_newlines=True)
    except OSError:
        return 'Error'
    lines = []
    while proc.poll() is None: # while alive
        line = proc.stdout.readline()
        if line:
            # Process output here
            lines.append(line)
    return len(lines)
##################################

# Derived from the NodeTree base type, similar to Menu, Operator, Panel, etc.
class CaffeTree(NodeTree):
    # Description string
    '''A custom node tree type that will show up in the node editor header'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'CaffeNodeTree'
    # Label for nice name display
    bl_label = 'Caffe Node Tree'
    bl_icon = 'NODETREE'

# Custom socket type
class ImageSocket(NodeSocket):
    # Description string
    '''Blob socket type'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'ImageSocketType'
    # Label for nice name display
    bl_label = 'Image Socket'

    # Enum items list

    # Optional function for drawing the socket input value
    def draw(self, context, layout, node, text):
        layout.label(text)

    # Socket color
    def draw_color(self, context, node):
        return (0.0, 1.0, 1.0, 0.5)


class OutputSocket(NodeSocket):
    # Description string
    '''Custom node socket type'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'OutputSocketType'
    # Label for nice name display
    bl_label = 'Output Socket'
    # Enum items list
    
    output_name = bpy.props.StringProperty(name='')
    
    # Optional function for drawing the socket input value
    def draw(self, context, layout, node, text):
        layout.prop(self, "output_name")
    
    # Socket color
    def draw_color(self, context, node):
        return (0.0, 1.0, 1.0, 0.5)


class LabelSocket(NodeSocket):
    # Description string
    '''Label socket type'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'LabelSocketType'
    # Label for nice name display
    bl_label = 'Label Socket'

    # Enum items list

    # Optional function for drawing the socket input value
    def draw(self, context, layout, node, text):
        layout.label(text)

    # Socket color
    def draw_color(self, context, node):
        return (0.5, 1.0, 0.2, 0.5)

class LossSocket(NodeSocket):
    # Description string
    '''Loss socket type'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'LossSocketType'
    # Label for nice name display
    bl_label = 'Loss Socket'

    # Enum items list

    # Optional function for drawing the socket input value
    def draw(self, context, layout, node, text):
        layout.label(text)

    # Socket color
    def draw_color(self, context, node):
        return (1.0, 0.3, 1.0, 0.5)


class NAFlatSocket(NodeSocket):
    # Description string
    '''NAFlat socket type'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'NAFlatSocketType'
    # Label for nice name display
    bl_label = 'Linear Flat Socket'

    # Enum items list

    # Optional function for drawing the socket input value
    def draw(self, context, layout, node, text):
        layout.label(text)

    # Socket color
    def draw_color(self, context, node):
        return (1.0, 0.2, 0.2, 0.5)

class AFlatSocket(NodeSocket):
    # Description string
    '''AFlat socket type'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'AFlatSocketType'
    # Label for nice name display
    bl_label = 'Non linear Flat Socket'

    # Enum items list

    # Optional function for drawing the socket input value
    def draw(self, context, layout, node, text):
        layout.label(text)

    # Socket color
    def draw_color(self, context, node):
        return (0.0, 0.8, 0.8, 0.5)


class params_p_g(bpy.types.PropertyGroup):
    name = bpy.props.StringProperty(name='Shared name')
    lr_mult = bpy.props.FloatProperty(default=1.0)
    decay_mult = bpy.props.FloatProperty(default=1.0)
    
    def draw(self, context, layout):
        layout.prop(self, "name")
        layout.prop(self, "lr_mult")
        layout.prop(self, "decay_mult")

class CaffeTreeNode:
    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == 'CaffeNodeTree'

    extra_params = bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params = bpy.props.PointerProperty(type=params_p_g)
    bias_params = bpy.props.PointerProperty(type=params_p_g)
    
    phases = [("TRAIN", "TRAIN", "Train only"),
              ("TEST", "TEST", "Test only"),
              ("BOTH", "BOTH", "Both")]
    include_in = bpy.props.EnumProperty(items=phases, default="BOTH")
    
    use_custom_weight = bpy.props.BoolProperty(name="Use custom weights", default=False)
    custom_weight = bpy.props.StringProperty(name="Custom weights",
                                             default="",
                                             description="Custom weights and bias from file",
                                             subtype='FILE_PATH')
    
    def draw_include_in(self, layout):
        layout.prop(self, "include_in")

    def draw_extra_params(self, context, layout):
        layout.prop(self, "extra_params")
        if self.extra_params:
            layout.label("Weight Params")
            self.weight_params.draw(context, layout)
            layout.label("Bias Params")
            self.bias_params.draw(context, layout)


class DataNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''A data node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'DataNodeType'
    # Label for nice name display
    bl_label = 'Data input'
    # Icon identifier
    bl_icon = 'SOUND'
    DBs = [
        ("LMDB", "LMDB", "Lmdb database"),
        ("LEVELDB", "LEVELDB", "LevelDB database"),
        ("ImageData","ImageData","Image files"),
        ("HDF5Data", "HDF5Data", "HDF5 Data")
    ]
    # === Custom Properties ===
    
    db_type = bpy.props.EnumProperty(name="Database type", description="Type of Data", items=DBs, default='HDF5Data')
    train_batch_size = bpy.props.IntProperty(min=1, default=100)
    test_batch_size = bpy.props.IntProperty(min=1, default=100)
    
    def update_tops(self, context):
        while len(self.outputs) < self.output_amount:
            self.outputs.new('OutputSocketType', "Out%i" % len(self.outputs))
        while len(self.outputs) > self.output_amount:
            self.outputs.remove(self.outputs[len(self.outputs)-1])

    output_amount = bpy.props.IntProperty(min=1, default=2, update=update_tops)
    
    train_path = bpy.props.StringProperty (
        name="Train Data Path",
        default="",
        description="Get the path to the data",
        subtype='DIR_PATH'
    )

    test_path = bpy.props.StringProperty (
        name="Test Data Path",
        default="",
        description="Get the path to the data",
        subtype='DIR_PATH'
    )
    
    train_data = bpy.props.StringProperty (
        name="Train Data Path",
        default="",
        description="Get the path to the data",
        subtype='FILE_PATH'
    )
                                           
    test_data = bpy.props.StringProperty (
         name="Test Data Path",
         default="",
         description="Get the path to the data",
         subtype='FILE_PATH'
    )

    # Transformation params
    scale = bpy.props.FloatProperty(default=1.0, min=0)
    mirror = bpy.props.BoolProperty(name='Random Mirror',default=False)
    use_mean_file = bpy.props.BoolProperty(name='Use mean file',default=False)
    mean_file = bpy.props.StringProperty (
        name="Mean File Path",
        default="",
        description="Mean file location",
        subtype='FILE_PATH'
    )
    # TODO: Add Mean Value and random crop
    
    # Image data params
    new_height = bpy.props.IntProperty(name="New image height",min=0, default=0, soft_max=1000)
    new_width = bpy.props.IntProperty(name="New image width",min=0, default=0, soft_max=1000)
    is_color = bpy.props.BoolProperty(name="Is color image",default=True)
    
    # For Image data + HDF5 data
    shuffle = bpy.props.BoolProperty(name='Shuffle', default=False)
    
    # For Data + Image data
    rand_skip = bpy.props.IntProperty(name="Random skip",min=0, default=0, soft_max=1000)

    # TODO: Add non supervised property

    # === Optional Functions ===
    def init(self, context):
        self.outputs.new('OutputSocketType', "Image Stack")
        self.outputs.new('OutputSocketType', "Label")

    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "output_amount")
        layout.prop(self, "db_type")
        
        if self.db_type in ('ImageData', 'HDF5Data'):
            layout.prop(self, "train_data")
            layout.prop(self, "test_data")
        else:
            layout.prop(self, "train_path")
            layout.prop(self, "test_path")

        layout.prop(self, "train_batch_size")
        layout.prop(self, "test_batch_size")
        
        
        if self.db_type in ('ImageData', 'LMDB', 'LEVELDB'):
            layout.label("Transformation Parameters")
            layout.prop(self, "scale")
            layout.prop(self, "mirror")
            layout.prop(self, "use_mean_file")
            if self.use_mean_file:
                layout.prop(self, "mean_file")
        
        layout.label("Special Parameters")
        if self.db_type == 'ImageData':
            layout.prop(self, "shuffle")
            layout.prop(self, "new_height")
            layout.prop(self, "new_width")
            layout.prop(self, "is_color")
            layout.prop(self, "rand_skip")
        elif self.db_type == 'HDF5Data':
            layout.prop(self, "shuffle")
        else:
            layout.prop(self, "rand_skip")

    def draw_label(self):
        return "Data Node"


class filler_p_g(bpy.types.PropertyGroup):
    types = [("constant", "constant", "Constant val"),
             ("uniform", "uniform", "Uniform dist"),
             ("gaussian", "gaussian", "Gaussian dist"),
             ("positive_unitball", "positive_unitball", "Positive unit ball dist"),
             ("xavier", "xavier", "Xavier dist"),
             ("msra", "msra", "MSRA dist"),
             ("bilinear", "bilinear", "Bi-linear upsample weights")]
             
    vnormtypes = [("FAN_IN", "FAN_IN", "Constant val"),
                  ("FAN_OUT", "FAN_OUT", "Uniform dist"),
                  ("AVERAGE", "AVERAGE", "Gaussian dist")]
    
    type = bpy.props.EnumProperty(name='Type', items=types, default='xavier')
    value = bpy.props.FloatProperty(default=0.0, soft_max=1000.0, soft_min=-1000.0)
    min = bpy.props.FloatProperty(default=0.0, soft_max=1000.0, soft_min=-1000.0)
    max = bpy.props.FloatProperty(default=1.0, soft_max=1000.0, soft_min=-1000.0)
    mean = bpy.props.FloatProperty(default=0.0, soft_max=1000.0, soft_min=-1000.0)
    std = bpy.props.FloatProperty(default=1.0, soft_max=1000.0, soft_min=-1000.0)
    variance_norm = bpy.props.EnumProperty(name='Weight variance norm', default='FAN_IN', items=vnormtypes)
    is_sparse = bpy.props.BoolProperty(name="Use Sprasity", default=False)
    sparse = bpy.props.IntProperty(default=100, min=1)

    def draw(self, context, layout):
        layout.prop(self, "type")
        
        if self.type == 'constant':
            layout.prop(self, "value")
        elif self.type in ('xavier', 'msra'):
            layout.prop(self, "variance_norm")
        elif self.type == 'gaussian':
            layout.prop(self, "mean")
            layout.prop(self, "std")
            layout.prop(self, "is_sparse")
            if self.is_sparse:
                layout.prop(self, "sparse")
        elif self.type == 'uniform':
            layout.prop(self, "min")
            layout.prop(self, "max")

class PoolNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''A pooling node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'PoolNodeType'
    # Label for nice name display
    bl_label = 'Pooling Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'Pooling'

    # === Custom Properties ===
    modes = [
        ("MAX", "MAX", "Max pooling"),
        ("AVE", "AVE", "Average pooling"),
        ("STOCHASTIC", "SGD", "Stochastic pooling"),
    ]

    kernel_size = bpy.props.IntProperty(default=2, min=1, soft_max=5)
    stride = bpy.props.IntProperty(default=2, min=1, soft_max=5)
    mode = bpy.props.EnumProperty(name='Mode', default='MAX', items=modes)
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input image")
        self.outputs.new('OutputSocketType', "Output image")

    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
#        if calcsize(self, context,axis='x') != calcsize(self, context,axis='y'):
#            layout.label("image x,y output is %s,%s pixels" %
#                        (calcsize(self, context,axis='x'),calcsize(self, context,axis='y')))
#        else:
#            layout.label("image output is %s pixels" %calcsize(self, context,axis='x'))
        layout.prop(self, "kernel_size")
        layout.prop(self, "stride")
        layout.prop(self, "mode")

class EltwiseNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''An element-wise node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'EltwiseNodeType'
    # Label for nice name display
    bl_label = 'Element-wise Node'
    # Icon identifier
    bl_icon = 'SOUND'

    # === Custom Properties ===
    eltwiseOps  = [
        ("PROD", "PROD", "Eltwise prod: c(i) -> a(i)*b(i)"),
        ("SUM", "SUM", "Eltwise sum: c(i) -> a(i)+b(i)"),
        ("MAX", "MAX", "Eltwise max: c(i) -> max [a(i),b(i)]"),
    ]
    coeff = bpy.props.FloatProperty(default=2.0,soft_max=10.0,min=0)
    stable_prod_grad = bpy.props.BoolProperty(name='Stable(slower) gradient',default=1)
    operation  = bpy.props.EnumProperty(name='Operation', default='SUM', items=eltwiseOps)
    
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input blob A")
        self.inputs.new('ImageSocketType', "Input blob B")
        self.outputs.new('OutputSocketType', "Output blob C")

    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "operation")    
        if self.operation == 'PROD':
            layout.prop(self, "stable_prod_grad ")
        elif self.operation == 'SUM':
            layout.prop(self, "coeff  ")
            
class ExpNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''An exponential node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'ExpNodeType'
    # Label for nice name display
    bl_label = 'Exponential Node'
    # Icon identifier
    bl_icon = 'SOUND'

    n_type = 'Exp'

    # === Custom Properties ===
    base = bpy.props.FloatProperty(default=-1.0,soft_max=10.0,min=0)
    scale = bpy.props.FloatProperty(default=1.0,soft_max=10.0,min=0)
    shift = bpy.props.FloatProperty(default=0.0,soft_max=10.0,min=-10)
    
    # === Optional Functions ===
    def init(self, context):
        
        
        self.inputs.new('ImageSocketType', "Input blob")
        self.outputs.new('OutputSocketType', "Output blob")

    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "base")    
        layout.prop(self, "scale")    
        layout.prop(self, "shift")

        self.draw_extra_params(context, layout)

class MVNNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''Mean variance normalization node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'MVNNodeType'
    # Label for nice name display
    bl_label = 'MVN Node'
    # Icon identifier
    bl_icon = 'SOUND'

    # === Custom Properties ===
    normalize_variance  = bpy.props.BoolProperty(default=True)
    across_channels  = bpy.props.BoolProperty(default=False)
    eps  = bpy.props.FloatProperty(default=1e-9,soft_max=1.0,min=1e-20)
    
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input blob")
        self.outputs.new('OutputSocketType', "Output blob")

    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "normalize_variance")    
        layout.prop(self, "across_channels")    
        layout.prop(self, "eps")
		
        self.draw_extra_params(context, layout)
        
class ConvNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'ConvNodeType'
    # Label for nice name display
    bl_label = 'Convolution Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = "Convolution"

    # === Custom Properties ===
    
    num_output = bpy.props.IntProperty(name="Number of outputs", default=20, min=1, soft_max=300)
    bias_term = bpy.props.BoolProperty(name='Include Bias term',default=True)
    
    square_padding = bpy.props.BoolProperty(name="Equal x,y padding", default=True)
    pad = bpy.props.IntProperty(name="Padding", default=0, min=0, soft_max=5)
    pad_h = bpy.props.IntProperty(name="Padding height", default=0, min=0, soft_max=5)
    pad_w = bpy.props.IntProperty(name="Padding width", default=0, min=0, soft_max=5)
    
    square_kernel = bpy.props.BoolProperty(name="Equal x,y kernel", default=True)
    kernel_size = bpy.props.IntProperty(name="Kernel size", default=5, min=1, soft_max=25)
    kernel_h = bpy.props.IntProperty(name="Kernel height", default=5, min=1, soft_max=25)
    kernel_w = bpy.props.IntProperty(name="Kernel width", default=5, min=1, soft_max=25)
    
    #TODO: Maybe add group
    
    square_stride = bpy.props.BoolProperty(name="Equal x,y stride", default=True)
    stride = bpy.props.IntProperty(name="Stride", default=1, min=1, soft_max=5)
    stride_h = bpy.props.IntProperty(name="Stride height", default=1, min=1, soft_max=5)
    stride_w = bpy.props.IntProperty(name="Stride width", default=1, min=1, soft_max=5)
    
    weight_filler = bpy.props.PointerProperty(type=filler_p_g)
    bias_filler = bpy.props.PointerProperty(type=filler_p_g)

    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input image")
        self.outputs.new('OutputSocketType', "Output image")
        self.color = [1, 0 ,1]
        self.use_custom_color = True

    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        #TODO: Finish calcsize
        
#        if calcsize(self, context,axis='x') != calcsize(self, context,axis='y'):
#            layout.label("image x,y output is %s,%s pixels" %
#                        (calcsize(self, context,axis='x'),calcsize(self, context,axis='y')))
#        else:
#            layout.label("image output is %s pixels" %calcsize(self, context,axis='x'))
        
        layout.prop(self, "num_output")
        layout.prop(self, "bias_term")
        
        layout.prop(self, "square_padding")
        if self.square_padding:
            layout.prop(self, "pad")
        else:
            layout.prop(self, "pad_h")
            layout.prop(self, "pad_w")
        
        layout.prop(self, "square_kernel")
        if self.square_kernel:
            layout.prop(self, "kernel_size")
        else:
            layout.prop(self, "kernel_h")
            layout.prop(self, "kernel_w")

        layout.prop(self, "square_stride")
        if self.square_stride:
            layout.prop(self, "stride")
        else:
            layout.prop(self, "stride_h")
            layout.prop(self, "stride_w")
        
        layout.prop(self, "use_custom_weight")
        if self.use_custom_weight:
            layout.prop(self, "custom_weight")
        else:
            layout.label("Weight Filler")
            self.weight_filler.draw(context, layout)
        
            if self.bias_term:
                layout.label("bias Filler")
                self.bias_filler.draw(context, layout)

        self.draw_extra_params(context, layout)


class DeConvNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''A DeConvolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'DeConvNodeType'
    # Label for nice name display
    bl_label = 'DeConvolution Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = "Deconvolution"
    
    # === Custom Properties ===
    
    num_output = bpy.props.IntProperty(name="Number of outputs", default=20, min=1, soft_max=300)
    bias_term = bpy.props.BoolProperty(name='Include Bias term',default=True)
    
    square_padding = bpy.props.BoolProperty(name="Equal x,y padding", default=True)
    pad = bpy.props.IntProperty(name="Padding", default=0, min=0, soft_max=5)
    pad_h = bpy.props.IntProperty(name="Padding height", default=0, min=0, soft_max=5)
    pad_w = bpy.props.IntProperty(name="Padding width", default=0, min=0, soft_max=5)
    
    square_kernel = bpy.props.BoolProperty(name="Equal x,y kernel", default=True)
    kernel_size = bpy.props.IntProperty(name="Kernel size", default=5, min=1, soft_max=25)
    kernel_h = bpy.props.IntProperty(name="Kernel height", default=5, min=1, soft_max=25)
    kernel_w = bpy.props.IntProperty(name="Kernel width", default=5, min=1, soft_max=25)
    
    #TODO: Maybe add group
    
    square_stride = bpy.props.BoolProperty(name="Equal x,y stride", default=True)
    stride = bpy.props.IntProperty(name="Stride", default=1, min=1, soft_max=5)
    stride_h = bpy.props.IntProperty(name="Stride height", default=1, min=1, soft_max=5)
    stride_w = bpy.props.IntProperty(name="Stride width", default=1, min=1, soft_max=5)
    
    weight_filler = bpy.props.PointerProperty(type=filler_p_g)
    bias_filler = bpy.props.PointerProperty(type=filler_p_g)
    
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input image")
        self.outputs.new('OutputSocketType', "Output image")
        self.color = [1, 0 ,1]
        self.use_custom_color = True
    
    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)
    
    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")
    
    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        #TODO: Finish calcsize
        
        #        if calcsize(self, context,axis='x') != calcsize(self, context,axis='y'):
        #            layout.label("image x,y output is %s,%s pixels" %
        #                        (calcsize(self, context,axis='x'),calcsize(self, context,axis='y')))
        #        else:
        #            layout.label("image output is %s pixels" %calcsize(self, context,axis='x'))
        
        layout.prop(self, "num_output")
        layout.prop(self, "bias_term")
        
        layout.prop(self, "square_padding")
        if self.square_padding:
            layout.prop(self, "pad")
        else:
            layout.prop(self, "pad_h")
            layout.prop(self, "pad_w")
        
        layout.prop(self, "square_kernel")
        if self.square_kernel:
            layout.prop(self, "kernel_size")
        else:
            layout.prop(self, "kernel_h")
            layout.prop(self, "kernel_w")
        
        layout.prop(self, "square_stride")
        if self.square_stride:
            layout.prop(self, "stride")
        else:
            layout.prop(self, "stride_h")
            layout.prop(self, "stride_w")
        
        layout.label("Weight Filler")
        self.weight_filler.draw(context, layout)
        
        layout.label("bias Filler")
        self.bias_filler.draw(context, layout)
        
        self.draw_extra_params(context, layout)

class FCNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''An inner product node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'FCNodeType'
    # Label for nice name display
    bl_label = 'Fully connected Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'InnerProduct'

    # === Custom Properties ===
    num_output = bpy.props.IntProperty(name="Number of outputs", default=10, min=1)
    bias_term = bpy.props.BoolProperty(name='Include Bias term',default=True)
    weight_filler = bpy.props.PointerProperty(type=filler_p_g)
    bias_filler = bpy.props.PointerProperty(type=filler_p_g)
    axis = bpy.props.IntProperty(name="Starting axis", default=1)
    
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input image")
        self.outputs.new('OutputSocketType', "Output Activations")
        self.color = [1, 0 ,0]
        self.use_custom_color = True

    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
#        layout.label("Network is now %s neurons" % calcsize(self, context))
        layout.prop(self, "num_output")
        layout.prop(self, "bias_term")
        layout.prop(self, "axis")
        
        layout.label("Weight Filler")
        self.weight_filler.draw(context, layout)
        
        layout.label("bias Filler")
        self.bias_filler.draw(context, layout)
        
        self.draw_extra_params(context, layout)

class FlattenNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''Flatten layer node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'FlattenNodeType'
    # Label for nice name display
    bl_label = 'Flatten Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'Flatten'
    
    # === Custom Properties ===
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input image")
        self.outputs.new('OutputSocketType', "Flat output")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.label("Flatten")
        self.draw_extra_params(context, layout)
        
class SilenceNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''A Silence node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'SilenceNodeType'
    # Label for nice name display
    bl_label = 'Silence Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'Silence'
    
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input")

    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.label("Silence")

class LRNNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'LRNNodeType'
    # Label for nice name display
    bl_label = 'LRN Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'LRN'
    
    modes = [
        ("ACROSS_CHANNELS", "ACROSS_CHANNELS", "Go across Channels"),
        ("WITHIN_CHANNEL", "WITHIN_CHANNEL", "Go by location"),
    ]
    # === Custom Properties ===
    alpha = bpy.props.FloatProperty(default=1, min=0, soft_max=50)
    beta = bpy.props.FloatProperty(default=5, min=0, soft_max=50)
    size = bpy.props.IntProperty(default=5, min=1, soft_max=50)
    mode = bpy.props.EnumProperty(name="Mode", default='ACROSS_CHANNELS', items=modes)
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input image")
        self.outputs.new('OutputSocketType', "Normalized output")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "alpha")
        layout.prop(self, "beta")
        layout.prop(self, "size")
        layout.prop(self, "mode")
		
        self.draw_extra_params(context, layout)

class ActivationNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'AcNodeType'
    # Label for nice name display
    bl_label = 'Activation Node'
    # Icon identifier
    bl_icon = 'SOUND'
    modes = [
        ('"Sigmoid"', "Sigmoid", "Sigmoid"),
        ('"TanH"', "TanH", "TanH"),
    ]
    # === Custom Properties ===
    mode = bpy.props.EnumProperty(name="Mode", default='"Sigmoid"', items=modes)
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('NAFlatSocketType', "Linear input")
        self.outputs.new('OutputSocketType', "Non Linear output")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "mode")
        self.draw_extra_params(context, layout)

class ReLuNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''A ReLU node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'ReluNodeType'
    # Label for nice name display
    bl_label = 'ReLu Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'ReLU'

    engines = [("DEFAULT", "DEFAULT", "Default"),
               ("CAFFE", "CAFFE", "Caffe"),
               ("CUDNN", "CUDNN", "CUDNN")]

    # === Custom Properties ===
    negative_slope = bpy.props.FloatProperty(default=0)
    #engine = bpy.props.EnumProperty(items=engines, default='DEFAULT')
    
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input image")
        self.outputs.new('OutputSocketType', "Rectified output")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "negative_slope")
        #layout.prop(self, "engine")
        self.draw_extra_params(context, layout)

class PReLuNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''A PReLU node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'PReluNodeType'
    # Label for nice name display
    bl_label = 'PReLu Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'PReLU'

    # === Custom Properties ===
    channel_shared = bpy.props.BoolProperty(default=False)
    filler = bpy.props.PointerProperty(type=filler_p_g)
    
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input image")
        self.outputs.new('OutputSocketType', "Rectified output")

    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "channel_shared")
        self.filler.draw(context, layout)
        self.draw_extra_params(context, layout)    
                
class SMLossNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'SMLossNodeType'
    # Label for nice name display
    bl_label = 'Softmax Loss Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'SoftmaxWithLoss'
    
    w = bpy.props.FloatProperty(default=1)
    
    def init(self, context):
        self.inputs.new('NAFlatSocketType', "Input Probabilities")
        self.inputs.new('LabelSocketType', "Input Label")
        self.outputs.new('OutputSocketType', "Loss output")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.label("Softmax Loss")
        layout.prop(self, "w")
        self.draw_extra_params(context, layout)


class SCELossNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'SCELossNodeType'
    # Label for nice name display
    bl_label = 'SCE Loss Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'SigmoidCrossEntropyLoss'
    
    w = bpy.props.FloatProperty(default=0)
    # === Custom Properties ===
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('NAFlatSocketType', "Input values")
        self.inputs.new('AFlatSocketType', "Input values 2")
        self.outputs.new('OutputSocketType', "Loss output")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.label("SCE Loss")
        layout.prop(self, "w")
        self.draw_extra_params(context, layout)

class EULossNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'EULossNodeType'
    # Label for nice name display
    bl_label = 'EU Loss Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'EuclideanLoss'
    
    w = bpy.props.FloatProperty(default=0)
    # === Custom Properties ===
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('AFlatSocketType', "Input values")
        self.inputs.new('AFlatSocketType', "Input values 2")
        self.outputs.new('OutputSocketType', "Loss output")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.label("EU Loss")
        layout.prop(self, "w")
        self.draw_extra_params(context, layout)

class DropoutNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'DropoutNodeType'
    # Label for nice name display
    bl_label = 'Dropout Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'Dropout'

    # === Custom Properties ===
    dropout_ratio = bpy.props.FloatProperty(default=0.5,min=0,max=1)
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('NAFlatSocketType', "Input image")
        self.outputs.new('OutputSocketType', "Output image")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.label("Dropout factor")
        layout.prop(self, "dropout_ratio")
        self.draw_extra_params(context, layout)

class ConcatNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''A Concatination node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'ConcatNodeType'
    # Label for nice name display
    bl_label = 'Concatanation Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'Concat'
    
    def update_bottoms(self, context):
        while len(self.inputs) < self.input_amount:
            self.inputs.new('ImageSocketType', "Input%i" % (len(self.inputs)+1))
        while len(self.inputs) > self.input_amount:
            self.inputs.remove(self.inputs[len(self.inputs)-1])

    input_amount = bpy.props.IntProperty(min=1, default=2, update=update_bottoms)

    # === Custom Properties ===
    axis = bpy.props.IntProperty(default=1)
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input1")
        self.inputs.new('ImageSocketType', "Input2")
        self.outputs.new('OutputSocketType', "Output image")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "input_amount")
        layout.prop(self, "axis")
        self.draw_extra_params(context, layout)


class AccuracyNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'AccuracyNodeType'
    # Label for nice name display
    bl_label = 'Accuracy Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'Accuracy'

    # === Custom Properties ===
    Testonly = bpy.props.BoolProperty(default=True)
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('NAFlatSocketType', "Input class")
        self.inputs.new('LabelSocketType', "Input label")
        self.outputs.new('OutputSocketType', "Output Accuracy")
    
        self.include_in = "TEST"


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.label("Tick for only testing")
        layout.prop(self, "Testonly")

        self.draw_include_in(layout)
        self.draw_extra_params(context, layout)

class ArgMaxNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''Arg Max Node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'ArgMaxNodeType'
    # Label for nice name display
    bl_label = 'Arg Max Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'ArgMax'
    
    # === Custom Properties ===
    OutMaxVal = bpy.props.BoolProperty(name='Output max value', default=False)
    TopK = bpy.props.IntProperty(name='Top k',default=1, min=1, soft_max=200)
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('LossSocketType', "Input loss")
        self.outputs.new('OutputSocketType', "Output Arg Max")
    
    
    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)
    
    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")
    
    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "OutMaxVal")
        layout.prop(self, "TopK")
        self.draw_extra_params(context, layout)
        
class HDF5OutputNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''HDF5 Output Node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'HDF5OutputNodeType'
    # Label for nice name display
    bl_label = 'HDF 5 Output Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'HDF5Output'
    
    # === Custom Properties ===
    filename = bpy.props.StringProperty \
        (
         name="HDF5 output File",
         default="",
         description="The path to the data file",
         subtype='FILE_PATH'
         )
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input Image")
    
    
    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)
    
    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")
    
    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "filename")


class LogNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''Log Node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'LogNodeType'
    # Label for nice name display
    bl_label = 'Log Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'Log'
    
    # === Custom Properties ===
    scale = bpy.props.FloatProperty(name='Scale', default=1, min=0, soft_max=200)
    shift = bpy.props.FloatProperty(name='Shift',default=0, soft_min=-200, soft_max=200)
    base = bpy.props.FloatProperty(name='base',default=-1, min=-1, soft_max=200)
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input data")
        self.outputs.new('OutputSocketType', "Output data")
    
    
    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)
    
    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")
    
    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "scale")
        layout.prop(self, "shift")
        layout.prop(self, "base")
        self.draw_extra_params(context, layout)

class PowerNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''Power Node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'PowerNodeType'
    # Label for nice name display
    bl_label = 'Power Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'Power'
    
    # === Custom Properties ===
    power = bpy.props.FloatProperty(name='Power', default=1)
    scale = bpy.props.FloatProperty(name='Scale', default=1)
    shift = bpy.props.FloatProperty(name='Shift', default=0)
    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input data")
        self.outputs.new('OutputSocketType', "Output data")
    
    
    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)
    
    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")
    
    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "power")
        layout.prop(self, "scale")
        layout.prop(self, "shift")
        self.draw_extra_params(context, layout)


class ReductionNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''Reduction Node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'ReductionNodeType'
    # Label for nice name display
    bl_label = 'Reduction Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'Reduction'
    
    ops = [ ("SUM", "SUM", "Sum"),
           ("ASUM", "ASUM", "Absolute Sum"),
           ("SUMSQ", "SUMSQ", "Sum of squares"),
           ("MEAN", "MEAN", "Mean")
    ]
        
    # === Custom Properties ===
    operation = bpy.props.EnumProperty(name='Operation', default='SUM', items=ops)
    axis = bpy.props.IntProperty(name='Axis', default=0)
    coeff = bpy.props.FloatProperty(name='Coeff', default=1)
    # === Optional Functions ===
    def init(self, context):
       self.inputs.new('ImageSocketType', "Input data")
       self.outputs.new('OutputSocketType', "Output data")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)
    
    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")
    
    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "operation")
        layout.prop(self, "axis")
        layout.prop(self, "coeff")
        self.draw_extra_params(context, layout)

class slice_point_p_g(bpy.types.PropertyGroup):
    slice_point = bpy.props.IntProperty(min=0)

    def draw(self, context, layout):
        layout.prop(self, "slice_point")


class SliceNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''Slice Node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'SliceNodeType'
    # Label for nice name display
    bl_label = 'Slice Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    n_type = 'Slice'
        
    # === Custom Properties ===
    axis = bpy.props.IntProperty(name='Axis', default=0)
    slice_points = bpy.props.CollectionProperty(type=slice_point_p_g)
    
    def update_slices(self, context):
        while len(self.slice_points) < self.num_of_slices:
            self.slice_points.add()
            self.outputs.new('OutputSocketType', "Out%i" % len(self.slice_points))
        while len(self.slice_points) > self.num_of_slices:
            self.slice_points.remove(len(self.slice_points)-1)
            self.outputs.remove(self.outputs[len(self.slice_points)+1])
    
    num_of_slices = bpy.props.IntProperty(default=1, min=1, update=update_slices)

    # === Optional Functions ===
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input data")
        self.outputs.new('OutputSocketType', "Out1")
        self.outputs.new('OutputSocketType', "Out2")
        self.slice_points.add()

    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "axis")
        layout.prop(self, "num_of_slices")
        
        for slice_point in self.slice_points:
            slice_point.draw(context, layout)
			
        self.draw_extra_params(context, layout)


#// Return the current learning rate. The currently implemented learning rate
#// policies are as follows:
#//    - fixed: always return base_lr.
#//    - step: return base_lr * gamma ^ (floor(iter / step))
#//    - exp: return base_lr * gamma ^ iter
#//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
#//    - multistep: similar to step but it allows non uniform steps defined by
#//      stepvalue
#//    - poly: the effective learning rate follows a polynomial decay, to be
#//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
#//    - sigmoid: the effective learning rate follows a sigmod decay
#//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
#//
#// where base_lr, max_iter, gamma, step, stepvalue and power are defined
#// in the solver parameter protocol buffer, and iter is the current iteration.

class SolverNode(Node, CaffeTreeNode):
    # === Basics ===
    # Description string
    '''A Solver node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'SolverNodeType'
    # Label for nice name display
    bl_label = 'Solver Node'
    # Icon identifier
    bl_icon = 'SOUND'
    
    lr_policies = [("fixed", "fixed", "Fixed"),
                   ("step", "step", "Step"),
                   ("exp", "exp", "Exponential"),
                   ("inv", "inv", "Inverse"),
                   ("multistep", "multistep", "Multi Step"),
                   ("poly", "poly", "Polinomial"),
                   ("sigmoid", "sigmoid", "Sigmoid")]
                   
    regularization_types = [("L1", "L1", "L1"), ("L2", "L2", "L2")]
    
    
    gputoggles = []
    gpunum = getgpus()
    gpufailed = 0

    if gpunum == 'Error':
        gpufailed = 1
    else:
        for gpu in range(gpunum):
            gpu_name = 'GPU %i'%gpu
            gputoggles.append((gpu_name, gpu_name, gpu_name))



    # === Custom Properties ===
    solvername = bpy.props.StringProperty()

    test_iter = bpy.props.IntProperty(name='Test Iterations',default=100, min=1, description="How many forward passes the test should carry out")
    test_interval = bpy.props.IntProperty(name='Test Interval', default=500, min=1, description="Carry out testing every test interval iterations")
    test_compute_loss = bpy.props.BoolProperty(name='Test Compute Loss',default=False, description="Compute loss in testing")
    test_initialization = bpy.props.BoolProperty(name='Test Initialization',default=True,
                          description="run an initial test pass before the first iteration, ensuring memory availability and printing the starting value of the loss.")
    base_lr = bpy.props.FloatProperty(name='Base Learning rate',default=0.01, min=0)
    display = bpy.props.IntProperty(name='Display',default=100, min=0, description="The number of iterations between displaying info. If display = 0, no info will be displayed")
    average_loss = bpy.props.IntProperty(name='Average Loss',default=1, min=1, description="Display the loss averaged over the last average_loss iterations")
    max_iter = bpy.props.IntProperty(name='Maximum Iterations', default=50000, min=1)
    iter_size = bpy.props.IntProperty(name='Iteration Size', default=1, min=1, description="Accumulate gradients over iter_size x batch_size instances")

    lr_policy = bpy.props.EnumProperty(name='Learning rate Policy', items=lr_policies, default='step')
    gamma = bpy.props.FloatProperty(name='Gamma', default=0.0001, min=0)
    power = bpy.props.FloatProperty(name='Power', default=0.75)
    momentum = bpy.props.FloatProperty(name='Momentum', default=0.9, min=0)
    weight_decay = bpy.props.FloatProperty(name='Weight Decay', default=0.0005, min=0)

    regularization_type = bpy.props.EnumProperty(name='Regularization type', items=regularization_types, default='L2')

    stepsize = bpy.props.IntProperty(name='Step size', default=5000, min=1)

    #TODO: Finish stepvalue and multistep
    #stepvalue

    #TODO: Maybe add clip gradients

    snapshot = bpy.props.IntProperty(name='Snapshot Interval', default=0, min=0, description="The snapshot interval. 0 for no snapshot")
    snapshot_prefix = bpy.props.StringProperty \
        (
         name="Snapshot Path",
         default="",
         description="Give the path to the snapshot data",
         subtype='DIR_PATH'
         )

    snapshot_diff = bpy.props.BoolProperty(name='Snapshot diff',default=False, description="Whether to snapshot diff in the results or not")

#    snapshot_formats = [("HDF5", "HDF5", "HDF5"), ("BINARYPROTO", "BINARYPROTO", "BINARYPROTO")]
#    snapshot_format = bpy.props.EnumProperty(name='Snapshot format', items=snapshot_formats, default='BINARYPROTO')

    solver_modes = [("GPU", "GPU", "GPU"), ("CPU", "CPU", "CPU")]
    solver_mode =  bpy.props.EnumProperty(name='Solver mode', items=solver_modes, default='GPU')
    gpus = bpy.props.EnumProperty(name="GPU",description="GPU to use", items=gputoggles)

    use_random_seed = bpy.props.BoolProperty(name='Use Random seed',default=False)
    random_seed = bpy.props.IntProperty(name='Random seed', default=10, description="The seed with which the Solver will initialize the Caffe random number generator")

    solver_types = [("NESTEROV", "NESTEROV", "Nesterovs Accelerated Gradient"),
                    ("ADAGRAD", "ADAGRAD", "Adaptive gradient descent"),
                    ("SGD", "SGD", "Stochastic Gradient Descent")]
    solver_type = bpy.props.EnumProperty(name='Solver type', items=solver_types, default='SGD')

    delta = bpy.props.FloatProperty(name='Delta', default=1e-8, min=0, description="Numerical stability for AdaGrad")

    debug_info = bpy.props.BoolProperty(name='Debug info', default=False)

    snapshot_after_train = bpy.props.BoolProperty(name='Snapshot after train', default=True, description="If false, don't save a snapshot after training finishes")



#    solver = bpy.props.EnumProperty(name="Mode", default='SGD', items=modes)
#    compmode = bpy.props.EnumProperty(name="Compute Mode", default='GPU', items=computemodes)
#
#    accum = bpy.props.BoolProperty(name='Accumulate Gradients',default=True)
#    accumiters = bpy.props.IntProperty(name='Number of minibatches to Accumulate',default=1 ,min=1,soft_max=10)
#    testinterval = bpy.props.IntProperty(name='Test Interval',default=500, min=1, soft_max=2000)
#    testruns = bpy.props.IntProperty(name='Test Batches',default=50, min=1, soft_max=200)
#    displayiter = bpy.props.IntProperty(name='Display iter.',default=100, min=1, soft_max=5000)
#    maxiter = bpy.props.IntProperty(name='Final iteration',default=50000, min=5, soft_max=100000)
#    learningrate = bpy.props.FloatProperty(name = 'Learning rate',default=0.01, min=0.001, soft_max=1)
#    snapshotiter = bpy.props.IntProperty(name = 'Snapshot iteration',default=10000, min=10, soft_max=50000)
#    snapshotpath = bpy.props.StringProperty \
#        (
#            name="Snapshot Data Path",
#            default="",
#            description="Give the path to the snapshot data",
#            subtype='DIR_PATH'
#        )



    config_path = bpy.props.StringProperty \
        (
            name="Configuration Data Path",
            default="",
            description="Give the path to the config data",
            subtype='DIR_PATH'
        )
    caffe_exec = bpy.props.StringProperty \
        (
        name="Caffe Tools Folder",
        default="",
        description="Give the path to the caffe executable",
        subtype='DIR_PATH'
        )

#    def init(self, context):
#        self.inputs.new('LossSocketType', "Input Loss")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

#    def update(self):
#        x = 0
#        for input in self.inputs:
#            if input.is_linked == False:
#                x += 1
#                if x > 1:
#                    self.inputs.remove(input)
#        if x == 0:
#            self.inputs.new('LossSocketType', "Input Loss")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        
        layout.prop(self, "solvername")
        layout.prop(self, "test_iter")
        layout.prop(self, "test_interval")
        layout.prop(self, "test_compute_loss")
        layout.prop(self, "test_initialization")
        layout.prop(self, "base_lr")
        layout.prop(self, "display")
        layout.prop(self, "average_loss")
        layout.prop(self, "max_iter")
        layout.prop(self, "iter_size")
        
        layout.prop(self, "lr_policy")
        
        if self.lr_policy == 'step':
            layout.prop(self, "gamma")
            layout.prop(self, "stepsize")
        elif self.lr_policy == 'exp':
            layout.prop(self, "gamma")
        elif self.lr_policy == 'inv':
            layout.prop(self, "gamma")
            layout.prop(self, "power")
        elif self.lr_policy == 'multistep':
            layout.label("NOT IMPLEMENTED", icon='ERROR')
        elif self.lr_policy == 'poly':
            layout.prop(self, "power")
        elif self.lr_policy == 'sigmoid':
            layout.prop(self, "gamma")
            layout.prop(self, "stepsize")

        layout.prop(self, "momentum")
        layout.prop(self, "weight_decay")
        layout.prop(self, "regularization_type")

        layout.prop(self, "snapshot")
        layout.prop(self, "snapshot_prefix")
        layout.prop(self, "snapshot_diff")
#        layout.prop(self, "snapshot_format")
        layout.prop(self, "snapshot_after_train")
        
        layout.prop(self, "solver_mode")
        if self.solver_mode == 'GPU':
            if self.gpufailed:
                layout.label("WARNING: GPU NOT DETECTED",icon='ERROR')
                layout.label("Check 'nvidia-smi' command can be run",icon='ERROR')
            else:
                layout.prop(self, "gpus")

        layout.prop(self, "use_random_seed")
        if self.use_random_seed:
            layout.prop(self, "random_seed")

        layout.prop(self, "solver_type")

        if self.solver_type == 'ADAGRAD':
            layout.prop(self, "delta")
                
        layout.prop(self, "debug_info")

        layout.prop(self, "config_path")
        layout.prop(self, "caffe_exec")

#        layout.prop(self, "display")
#        layout.prop(self, "display")
#        
#        
#        
#        layout.prop(self, "solvername")
#        layout.prop(self, "solver")
#        layout.prop(self, "compmode")
#
#        ########################GPUS
#        if not self.gpufailed:
#            layout.label("Multiple GPUs req. parallel caffe branch",icon='ERROR')
#        else:
#            layout.label("WARNING: GPU NOT DETECTED",icon='ERROR')
#            layout.label("Check 'nvidia-smi' command can be run",icon='ERROR')
#        if self.compmode == 'GPU':
#            for i,name in enumerate(self.gputoggles):
#                layout.prop(self, "gpus",index=i,text=name,toggle=True)
#        ###############Accumulate batches
#        layout.prop(self, "accum")
#        if self.accum:
#            layout.prop(self,"accumiters")
#        #layout.prop(self, "gpu")
#        layout.prop(self, "testinterval")
#        layout.prop(self, "testruns")
#        layout.prop(self, "displayiter")
#        layout.prop(self, "maxiter")
#        layout.prop(self, "learningrate")
#        layout.prop(self, "snapshotiter")
#        layout.prop(self, "snapshotpath")
#        layout.prop(self, "configpath")
#        layout.prop(self, "caffexec")

import nodeitems_utils
from nodeitems_utils import NodeCategory, NodeItem

# our own base class with an appropriate poll function,
# so the categories only show up in our own tree type
class CaffeNodeCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == 'CaffeNodeTree'

# all categories in a list
node_categories = [
    # identifier, label, items list
    CaffeNodeCategory("PNODES", "Processing Nodes", items=[
        # our basic node
        NodeItem("PoolNodeType"),
        NodeItem("ConvNodeType"),
        NodeItem("DeConvNodeType"),
        NodeItem("LRNNodeType"),
        NodeItem("ConcatNodeType"),
        NodeItem("SliceNodeType")
    ]),
    CaffeNodeCategory("NNODES", "Neuron Nodes", items=[
        # our basic node
        NodeItem("MVNNodeType"),
        NodeItem("ExpNodeType"),
        NodeItem("EltwiseNodeType"),
        NodeItem("ArgMaxNodeType"),
        NodeItem("FCNodeType"),
        NodeItem("FlattenNodeType"),
        NodeItem("AcNodeType"),
        NodeItem("ReluNodeType"),
        NodeItem("PReluNodeType"),
        NodeItem("DropoutNodeType"),
        NodeItem("LogNodeType"),
        NodeItem("PowerNodeType")
    ]),
    CaffeNodeCategory("SNODES", "Solver Nodes", items=[
        # our basic node
        NodeItem("SolverNodeType"),
        NodeItem("AccuracyNodeType"),
        NodeItem("EULossNodeType"),
        NodeItem("SCELossNodeType"),
        NodeItem("SMLossNodeType"),        
        NodeItem("ReductionNodeType")
        
    ]),
    CaffeNodeCategory("DNODES", "Data Nodes", items=[
        # our basic node
        NodeItem("DataNodeType"),
        NodeItem("HDF5OutputNodeType")
    ]),
    CaffeNodeCategory("MNODES", "Misc Nodes", items=[
        # our basic node
        NodeItem("SilenceNodeType")
    ]),
]

def register():
    bpy.utils.register_class(filler_p_g)
    bpy.utils.register_class(params_p_g)
    bpy.utils.register_class(slice_point_p_g)
    bpy.utils.register_class(OutputSocket)
    bpy.utils.register_class(CaffeTree)
    bpy.utils.register_class(DataNode)
    bpy.utils.register_class(DropoutNode)
    bpy.utils.register_class(PoolNode)
    bpy.utils.register_class(EltwiseNode)
    bpy.utils.register_class(MVNNode)
    bpy.utils.register_class(ExpNode)
    bpy.utils.register_class(ConvNode)
    bpy.utils.register_class(DeConvNode)
    bpy.utils.register_class(FCNode)
    bpy.utils.register_class(FlattenNode)
    bpy.utils.register_class(LRNNode)
    bpy.utils.register_class(ActivationNode)
    bpy.utils.register_class(ReLuNode)
    bpy.utils.register_class(PReLuNode)
    bpy.utils.register_class(SMLossNode)
    bpy.utils.register_class(SCELossNode)
    bpy.utils.register_class(EULossNode)
    bpy.utils.register_class(ConcatNode)
    bpy.utils.register_class(AccuracyNode)
    bpy.utils.register_class(ArgMaxNode)
    bpy.utils.register_class(SolverNode)
    bpy.utils.register_class(ImageSocket)
    bpy.utils.register_class(LabelSocket)
    bpy.utils.register_class(LossSocket)
    bpy.utils.register_class(AFlatSocket)
    bpy.utils.register_class(NAFlatSocket)
    bpy.utils.register_class(SilenceNode)
    bpy.utils.register_class(HDF5OutputNode)
    bpy.utils.register_class(LogNode)
    bpy.utils.register_class(PowerNode)
    bpy.utils.register_class(ReductionNode)
    bpy.utils.register_class(SliceNode)

    nodeitems_utils.register_node_categories("CUSTOM_NODES", node_categories)

def unregister():
    nodeitems_utils.unregister_node_categories("CUSTOM_NODES")

    bpy.utils.unregister_class(filler_p_g)
    bpy.utils.unregister_class(params_p_g)
    bpy.utils.unregister_class(slice_point_p_g)
    bpy.utils.unregister_class(OutputSocket)
    bpy.utils.unregister_class(CaffeTree)
    bpy.utils.unregister_class(DataNode)
    bpy.utils.unregister_class(DropoutNode)
    bpy.utils.unregister_class(PoolNode)
    bpy.utils.unregister_class(EltwiseNode)
    bpy.utils.unregister_class(MVNNode)
    bpy.utils.unregister_class(ExpNode)
    bpy.utils.unregister_class(ConvNode)
    bpy.utils.unregister_class(DeConvNode)
    bpy.utils.unregister_class(FCNode)
    bpy.utils.unregister_class(FlattenNode)
    bpy.utils.unregister_class(LRNNode)
    bpy.utils.unregister_class(ActivationNode)
    bpy.utils.unregister_class(ReLuNode)
    bpy.utils.unregister_class(PReLuNode)
    bpy.utils.unregister_class(SMLossNode)
    bpy.utils.unregister_class(SCELossNode)
    bpy.utils.unregister_class(EULossNode)
    bpy.utils.unregister_class(ConcatNode)
    bpy.utils.unregister_class(AccuracyNode)
    bpy.utils.unregister_class(ArgMaxNode)
    bpy.utils.unregister_class(SolverNode)
    bpy.utils.unregister_class(ImageSocket)
    bpy.utils.unregister_class(LabelSocket)
    bpy.utils.unregister_class(LossSocket)
    bpy.utils.unregister_class(AFlatSocket)
    bpy.utils.unregister_class(NAFlatSocket)
    bpy.utils.unregister_class(SilenceNode)
    bpy.utils.unregister_class(HDF5OutputNode)
    bpy.utils.unregister_class(LogNode)
    bpy.utils.unregister_class(PowerNode)
    bpy.utils.unregister_class(ReductionNode)
    bpy.utils.unregister_class(SliceNode)

if __name__ == "__main__":
    register()
