bl_info = {
    "name": "Caffe Nodes",
    "category": "Object",
}
import bpy
from bpy.types import NodeTree, Node, NodeSocket

# Implementation of custom nodes from Python
def calcsize(self, context):
    x = 0.0
    node = self
    # print (node.bl_idname)
    try:
        node.inputs[0].links[0].from_node
    except IndexError:
        return 0
    kernelsizes = []
    strides = []
    paddings = []
    poolstrides = []
    poolsizes = []
    offsets = []
    fcsizes = []
    passes = []
    while 1 == 1:
        if node.bl_idname == "ConvNodeType":
            #print (node.inputs[0])
            #print(dir(node.inputs[0]))
            #print(type(node.inputs[0]))
            kernelsizes.extend([node.kernelsize])
            strides.extend([node.Stride])
            paddings.extend([node.Padding])
            poolsizes.extend([1])
            poolstrides.extend([1])
            offsets.extend([0])
            fcsizes.extend([0])
            passes.extend([0])
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
            fcsizes.extend([node.outputnum])
            node = node.inputs[0].links[0].from_node

            #print (node)
        elif node.bl_idname == "PoolNodeType":
            kernelsizes.extend([0])
            paddings.extend([0])
            strides.extend([1])
            fcsizes.extend([0])
            passes.extend([0])
            poolsizes.extend([node.kernel])
            poolstrides.extend([node.stride])
            offsets.extend([1])
            node = node.inputs[0].links[0].from_node
        elif node.bl_idname == "DataNodeType":
            x = float(node.imsize)
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
                if passes[node] == 0:
                    if fcsizes[node] == 0:
                        #########################
                        x = ((x + (2 * padding) - ksize) / stride + 1 - offset)
                        x = (x - poolsize) / poolstride + 1
                        ###################
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
            fcsizes.extend([0])
            passes.extend([1])
            node = node.inputs[0].links[0].from_node
    return str(round(x, 2))


# Derived from the NodeTree base type, similar to Menu, Operator, Panel, etc.
class MyCustomTree(NodeTree):
    # Description string
    '''A custom node tree type that will show up in the node editor header'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'CustomTreeType'
    # Label for nice name display
    bl_label = 'Custom Node Tree'
    # Icon identifier
    # NOTE: If no icon is defined, the node tree will not show up in the editor header!
    # This can be used to make additional tree types for groups and similar nodes (see below)
    #       Only one base tree class is needed in the editor for selecting the general category
    bl_icon = 'NODETREE'


# Custom socket type
class ImageSocket(NodeSocket):
    # Description string
    '''Custom node socket type'''
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


class LabelSocket(NodeSocket):
    # Description string
    '''Custom node socket type'''
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
    '''Custom node socket type'''
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
    '''Custom node socket type'''
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
    '''Custom node socket type'''
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


# Mix-in class for all custom nodes in this tree type.
# Defines a poll function to enable instantiation.
class MyCustomTreeNode:
    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == 'CustomTreeType'


# Derived from the Node base type.
class MyCustomNode(Node, MyCustomTreeNode):
    # === Basics ===
    # Description string
    '''A custom node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'CustomNodeType'
    # Label for nice name display
    bl_label = 'Custom Node'
    # Icon identifier
    bl_icon = 'SOUND'

    # === Custom Properties ===
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    myStringProperty = bpy.props.StringProperty()
    myFloatProperty = bpy.props.FloatProperty(default=3.1415926)

    # === Optional Functions ===
    # Initialization function, called when a new node is created.
    # This is the most common place to create the sockets for a node, as shown below.
    # NOTE: this is not the same as the standard __init__ function in Python, which is
    # a purely internal Python method and unknown to the node system!
    def init(self, context):
        self.inputs.new('CustomSocketType', "Hello")
        self.inputs.new('NodeSocketFloat', "World")
        self.inputs.new('NodeSocketVector', "!")

        self.outputs.new('NodeSocketColor', "How")
        self.outputs.new('NodeSocketColor', "are")
        self.outputs.new('NodeSocketFloat', "you")

    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.label("Node settings")
        layout.prop(self, "myFloatProperty")

    # Detail buttons in the sidebar.
    # If this function is not defined, the draw_buttons function is used instead
    def draw_buttons_ext(self, context, layout):
        layout.prop(self, "myFloatProperty")
        # myStringProperty button will only be visible in the sidebar
        layout.prop(self, "myStringProperty")

    # Optional: custom label
    # Explicit user label overrides this, but here we can define a label dynamically
    def draw_label(self):
        return "I am a custom node"


class DataNode(Node, MyCustomTreeNode):
    # === Basics ===
    # Description string
    '''A data node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'DataNodeType'
    # Label for nice name display
    bl_label = 'data'
    # Icon identifier
    bl_icon = 'SOUND'
    DBs = [
        ("LMDB", "LMDB", "Lmdb database"), ("Image files","Image files","Image files"),
]
    # === Custom Properties ===
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    dbtype = bpy.props.EnumProperty(name="Database type", description="Type of Data", items=DBs, default='LMDB')
    imsize = bpy.props.IntProperty(name="Image targetsize",min=1, default=28, soft_max=1000)
    channels = bpy.props.IntProperty(min=1, default=3, soft_max=250)
    maxval = bpy.props.IntProperty(min=1, default=255, soft_max=255)
    batchsize = bpy.props.IntProperty(min=1, default=100, soft_max=500)
    supervised = bpy.props.BoolProperty(default=True)
    mirror = bpy.props.BoolProperty(name='Random Mirror',default=False)
    silout = bpy.props.BoolProperty(name='Silence label (sil. node doesnt work on labels)',default=False)
    usemeanfile = bpy.props.BoolProperty(name='Use mean file',default=True)
    meanfile = bpy.props.StringProperty \
        (
            name="Mean File Path",
            default="",
            description="Mean file location",
            subtype='FILE_PATH'
        )
    trainpath = bpy.props.StringProperty \
        (
            name="Train Data Path",
            default="",
            description="Get the path to the data",
            subtype='DIR_PATH'
        )
    testpath = bpy.props.StringProperty \
        (
            name="Test Data Path",
            default="",
            description="Get the path to the data",
            subtype='DIR_PATH'
        )
    trainfile = bpy.props.StringProperty \
        (
            name="Train image txt",
            default="",
            description="Get the path to the data",
            subtype='FILE_PATH'
        )
    testfile = bpy.props.StringProperty \
        (
            name="Test image txt",
            default="",
            description="Get the path to the data",
            subtype='FILE_PATH'
        )
    # === Optional Functions ===
    # Initialization function, called when a new node is created.
    # This is the most common place to create the sockets for a node, as shown below.
    # NOTE: this is not the same as the standard __init__ function in Python, which is
    # a purely internal Python method and unknown to the node system!
    def init(self, context):
        self.outputs.new('ImageSocketType', "Image Stack")
        self.outputs.new('LabelSocketType', "Label")

    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "dbtype")
        if self.dbtype == 'LMDB':
            layout.prop(self, "trainpath")
            layout.prop(self, "testpath")
            layout.prop(self, "supervised")
        elif self.dbtype == 'Image files':
            if self.supervised == 0:
                layout.label("WARNING: Check the supervised box",icon='ERROR')
                layout.prop(self,"supervised")
            layout.prop(self, "trainfile")
            layout.prop(self, "testfile")
        else:
            print(self.dbtype)
        layout.prop(self, "batchsize")
        layout.prop(self, "channels")
        layout.prop(self, "imsize")
        layout.prop(self, "maxval")
        layout.prop(self, "mirror")
        if self.supervised:
            layout.prop(self, "silout")
        elif self.supervised ==0 and self.silout ==1 and self.dbtype=='LMDB':
            layout.label("WARNING: Uncheck the silence box",icon='ERROR')
            layout.prop(self,"silout")
        layout.prop(self, "usemeanfile")
        if self.usemeanfile:
            layout.prop(self,"meanfile")
    # Detail buttons in the sidebar.
    # If this function is not defined, the draw_buttons function is used instead

    # Optional: custom label
    # Explicit user label overrides this, but here we can define a label dynamically
    def draw_label(self):
        return "Data Node"


class PoolNode(Node, MyCustomTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'PoolNodeType'
    # Label for nice name display
    bl_label = 'Pooling Node'
    # Icon identifier
    bl_icon = 'SOUND'

    # === Custom Properties ===
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    modes = [
        ("MAX", "MAX", "Max pooling"),
        ("AVE", "AVE", "Average pooling"),
        ("STOCHASTIC", "SGD", "Stochastic pooling"),
    ]
    kernel = bpy.props.IntProperty(default=2, min=1, soft_max=5)
    stride = bpy.props.IntProperty(default=2, min=1, soft_max=5)
    mode = bpy.props.EnumProperty(name='Mode', default='MAX', items=modes)
    # === Optional Functions ===
    # Initialization function, called when a new node is created.
    # This is the most common place to create the sockets for a node, as shown below.
    # NOTE: this is not the same as the standard __init__ function in Python, which is
    # a purely internal Python method and unknown to the node system!
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input image")
        self.outputs.new('ImageSocketType', "Output image")

    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.label("image output is %s pixels" % calcsize(self, context))
        layout.prop(self, "kernel")
        layout.prop(self, "stride")
        layout.prop(self, "mode")
        # Detail buttons in the sidebar.
        # If this function is not defined, the draw_buttons function is used instead

        # Optional: custom label
        # Explicit user label overrides this, but here we can define a label dynamically


class ConvNode(Node, MyCustomTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'ConvNodeType'
    # Label for nice name display
    bl_label = 'Convolution Node'
    # Icon identifier
    bl_icon = 'SOUND'

    # === Custom Properties ===
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    modes = [
        ("xavier", "xavier", "Xavier dist"),
        ("gaussian", "gaussian", "Gaussian dist"),
    ]
    weights = bpy.props.EnumProperty(name='Weights', default='gaussian', items=modes)
    kernelsize = bpy.props.IntProperty(default=5, min=0, soft_max=25)
    Stride = bpy.props.IntProperty(default=1, min=1, soft_max=5)
    Padding = bpy.props.IntProperty(default=0, min=0, soft_max=5)
    OutputLs = bpy.props.IntProperty(default=20, min=1, soft_max=300)
    std = bpy.props.FloatProperty(default=0.01,soft_max=0.1,min=0)
    filterlr = bpy.props.IntProperty(default=1, max=5, min=0)
    biaslr = bpy.props.IntProperty(default=2, max=5, min=0)
    filterdecay = bpy.props.IntProperty(default=1, max=5, min=0)
    biasdecay = bpy.props.IntProperty(default=0, max=5, min=0)
    biasfill = bpy.props.FloatProperty(default=0.4, soft_max=1.0, min=0)
    # === Optional Functions ===
    # Initialization function, called when a new node is created.
    # This is the most common place to create the sockets for a node, as shown below.
    # NOTE: this is not the same as the standard __init__ function in Python, which is
    # a purely internal Python method and unknown to the node system!
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input image")
        self.outputs.new('ImageSocketType', "Output image")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.label("image output is %s pixels" % calcsize(self, context))
        layout.prop(self, "kernelsize")
        layout.prop(self, "Stride")
        layout.prop(self, "Padding")
        layout.prop(self, "OutputLs")
        layout.prop(self, "filterlr")
        layout.prop(self, "biaslr")
        layout.prop(self, "filterdecay")
        layout.prop(self, "biasdecay")
        layout.prop(self, "weights")
        layout.prop(self, "std")
        layout.prop(self, "biasfill")
        # Detail buttons in the sidebar.
        # If this function is not defined, the draw_buttons function is used instead

        # Optional: custom label
        # Explicit user label overrides this, but here we can define a label dynamically


class FCNode(Node, MyCustomTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'FCNodeType'
    # Label for nice name display
    bl_label = 'Fully connected Node'
    # Icon identifier
    bl_icon = 'SOUND'

    # === Custom Properties ===
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    myStringProperty = bpy.props.StringProperty()
    outputnum = bpy.props.IntProperty(default=1000, min=1, soft_max=10000)
    sparse = bpy.props.IntProperty(default=100, min=1, max=500)
    sparsity = bpy.props.BoolProperty(default=False)
    std = bpy.props.FloatProperty(default=0.1, min=0, max=1)
    modes = [
        ("xavier", "xavier", "Xavier dist"),
        ("gaussian", "gaussian", "Gaussian dist"),
    ]
    kernel = bpy.props.IntProperty(default=2, min=1, soft_max=5)
    stride = bpy.props.IntProperty(default=2, min=1, soft_max=5)
    weights = bpy.props.EnumProperty(name='Weights', default='gaussian', items=modes)
    filterlr = bpy.props.IntProperty(default=1, max=5, min=0)
    biaslr = bpy.props.IntProperty(default=2, max=5, min=0)
    filterdecay = bpy.props.IntProperty(default=1, max=5, min=0)
    biasdecay = bpy.props.IntProperty(default=0, max=5, min=0)
    biasfill = bpy.props.FloatProperty(default=0.4, soft_max=1.0, min=0)
    # === Optional Functions ===
    # Initialization function, called when a new node is created.
    # This is the most common place to create the sockets for a node, as shown below.
    # NOTE: this is not the same as the standard __init__ function in Python, which is
    # a purely internal Python method and unknown to the node system!
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input image")
        self.outputs.new('NAFlatSocketType', "Output Activations")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.label("Network is now %s neurons" % calcsize(self, context))
        layout.prop(self, "outputnum")
        layout.prop(self, "filterlr")
        layout.prop(self, "biaslr")
        layout.prop(self, "filterdecay")
        layout.prop(self, "biasdecay")
        layout.prop(self, "weights")
        layout.prop(self, "std")
        layout.prop(self, "sparsity")
        layout.prop(self, "sparse")
        layout.prop(self, "biasfill")
        # Detail buttons in the sidebar.
        # If this function is not defined, the draw_buttons function is used instead

        # Optional: custom label
        # Explicit user label overrides this, but here we can define a label dynamically


class FlattenNode(Node, MyCustomTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'FlattenNodeType'
    # Label for nice name display
    bl_label = 'Flatten Node'
    # Icon identifier
    bl_icon = 'SOUND'

    # === Custom Properties ===
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    # === Optional Functions ===
    # Initialization function, called when a new node is created.
    # This is the most common place to create the sockets for a node, as shown below.
    # NOTE: this is not the same as the standard __init__ function in Python, which is
    # a purely internal Python method and unknown to the node system!
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input image")
        self.outputs.new('AFlatSocketType', "Flat output")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.label("Flatten")
        # Detail buttons in the sidebar.
        # If this function is not defined, the draw_buttons function is used instead

        # Optional: custom label
        # Explicit user label overrides this, but here we can define a label dynamically


class SilenceNode(Node, MyCustomTreeNode):
    # === Basics ===
    # Description string
    '''A Silence node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'SilenceNodeType'
    # Label for nice name display
    bl_label = 'Silence Node'
    # Icon identifier
    bl_icon = 'SOUND'

    # === Custom Properties ===
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    # === Optional Functions ===
    # Initialization function, called when a new node is created.
    # This is the most common place to create the sockets for a node, as shown below.
    # NOTE: this is not the same as the standard __init__ function in Python, which is
    # a purely internal Python method and unknown to the node system!
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
        # Detail buttons in the sidebar.
        # If this function is not defined, the draw_buttons function is used instead

        # Optional: custom label
        # Explicit user label overrides this, but here we can define a label dynamically


class LRNNode(Node, MyCustomTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'LRNNodeType'
    # Label for nice name display
    bl_label = 'LRN Node'
    # Icon identifier
    bl_icon = 'SOUND'
    modes = [
        ("ACROSS_CHANNELS", "ACROSS_CHANNELS", "Go across Channels"),
        ("WITHIN_CHANNEL", "WITHIN_CHANNEL", "Go by location"),
    ]
    # === Custom Properties ===
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    alpha = bpy.props.FloatProperty(default=1, min=0, soft_max=50)
    beta = bpy.props.FloatProperty(default=5, min=0, soft_max=50)
    size = bpy.props.IntProperty(default=5, min=1, soft_max=50)
    mode = bpy.props.EnumProperty(name="Mode", default='ACROSS_CHANNELS', items=modes)
    # === Optional Functions ===
    # Initialization function, called when a new node is created.
    # This is the most common place to create the sockets for a node, as shown below.
    # NOTE: this is not the same as the standard __init__ function in Python, which is
    # a purely internal Python method and unknown to the node system!
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input image")
        self.outputs.new('ImageSocketType', "Normalized output")


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
        # Detail buttons in the sidebar.
        # If this function is not defined, the draw_buttons function is used instead

        # Optional: custom label
        # Explicit user label overrides this, but here we can define a label dynamically


class ActivationNode(Node, MyCustomTreeNode):
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
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    mode = bpy.props.EnumProperty(name="Mode", default='"Sigmoid"', items=modes)
    # === Optional Functions ===
    # Initialization function, called when a new node is created.
    # This is the most common place to create the sockets for a node, as shown below.
    # NOTE: this is not the same as the standard __init__ function in Python, which is
    # a purely internal Python method and unknown to the node system!
    def init(self, context):
        self.inputs.new('NAFlatSocketType', "Linear input")
        self.outputs.new('AFlatSocketType', "Non Linear output")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "mode")
        # Detail buttons in the sidebar.
        # If this function is not defined, the draw_buttons function is used instead

        # Optional: custom label
        # Explicit user label overrides this, but here we can define a label dynamically


class ReLuNode(Node, MyCustomTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'ReluNodeType'
    # Label for nice name display
    bl_label = 'ReLu Node'
    # Icon identifier
    bl_icon = 'SOUND'

    # === Custom Properties ===
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    Negativeg = bpy.props.BoolProperty(default=False)
    # === Optional Functions ===
    # Initialization function, called when a new node is created.
    # This is the most common place to create the sockets for a node, as shown below.
    # NOTE: this is not the same as the standard __init__ function in Python, which is
    # a purely internal Python method and unknown to the node system!
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input image")
        self.outputs.new('ImageSocketType', "Rectified output")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "Negativeg")
        # Detail buttons in the sidebar.
        # If this function is not defined, the draw_buttons function is used instead

        # Optional: custom label
        # Explicit user label overrides this, but here we can define a label dynamically


class SMLossNode(Node, MyCustomTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'SMLossNodeType'
    # Label for nice name display
    bl_label = 'Softmax Loss Node'
    # Icon identifier
    bl_icon = 'SOUND'
    w = bpy.props.FloatProperty(default=0)
    # === Custom Properties ===
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    # === Optional Functions ===
    # Initialization function, called when a new node is created.
    # This is the most common place to create the sockets for a node, as shown below.
    # NOTE: this is not the same as the standard __init__ function in Python, which is
    # a purely internal Python method and unknown to the node system!
    def init(self, context):
        self.inputs.new('NAFlatSocketType', "Input Probabilities")
        self.inputs.new('LabelSocketType', "Input Label")
        self.outputs.new('LossSocketType', "Loss output")


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
        # Detail buttons in the sidebar.
        # If this function is not defined, the draw_buttons function is used instead

        # Optional: custom label
        # Explicit user label overrides this, but here we can define a label dynamically


class SCELossNode(Node, MyCustomTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'SCELossNodeType'
    # Label for nice name display
    bl_label = 'SCE Loss Node'
    # Icon identifier
    bl_icon = 'SOUND'
    w = bpy.props.FloatProperty(default=0)
    # === Custom Properties ===
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    # === Optional Functions ===
    # Initialization function, called when a new node is created.
    # This is the most common place to create the sockets for a node, as shown below.
    # NOTE: this is not the same as the standard __init__ function in Python, which is
    # a purely internal Python method and unknown to the node system!
    def init(self, context):
        self.inputs.new('NAFlatSocketType', "Input values")
        self.inputs.new('AFlatSocketType', "Input values 2")
        self.outputs.new('LossSocketType', "Loss output")


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
        # Detail buttons in the sidebar.
        # If this function is not defined, the draw_buttons function is used instead

        # Optional: custom label
        # Explicit user label overrides this, but here we can define a label dynamically


class EULossNode(Node, MyCustomTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'EULossNodeType'
    # Label for nice name display
    bl_label = 'EU Loss Node'
    # Icon identifier
    bl_icon = 'SOUND'
    w = bpy.props.FloatProperty(default=0)
    # === Custom Properties ===
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    # === Optional Functions ===
    # Initialization function, called when a new node is created.
    # This is the most common place to create the sockets for a node, as shown below.
    # NOTE: this is not the same as the standard __init__ function in Python, which is
    # a purely internal Python method and unknown to the node system!
    def init(self, context):
        self.inputs.new('AFlatSocketType', "Input values")
        self.inputs.new('AFlatSocketType', "Input values 2")
        self.outputs.new('LossSocketType', "Loss output")


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
        # Detail buttons in the sidebar.
        # If this function is not defined, the draw_buttons function is used instead

        # Optional: custom label
        # Explicit user label overrides this, but here we can define a label dynamically


class DropoutNode(Node, MyCustomTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'DropoutNodeType'
    # Label for nice name display
    bl_label = 'Dropout Node'
    # Icon identifier
    bl_icon = 'SOUND'

    # === Custom Properties ===
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    fac = bpy.props.FloatProperty(default=0.5,min=0,max=1)
    # === Optional Functions ===
    # Initialization function, called when a new node is created.
    # This is the most common place to create the sockets for a node, as shown below.
    # NOTE: this is not the same as the standard __init__ function in Python, which is
    # a purely internal Python method and unknown to the node system!
    def init(self, context):
        self.inputs.new('NAFlatSocketType', "Input image")
        self.outputs.new('NAFlatSocketType', "Output image")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.label("Dropout factor")
        layout.prop(self, "fac")
        # Detail buttons in the sidebar.
        # If this function is not defined, the draw_buttons function is used instead

        # Optional: custom label
        # Explicit user label overrides this, but here we can define a label dynamically


class ConcatNode(Node, MyCustomTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'ConcatNodeType'
    # Label for nice name display
    bl_label = 'Concatanation Node'
    # Icon identifier
    bl_icon = 'SOUND'

    # === Custom Properties ===
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    dim = bpy.props.BoolProperty(default=True)
    # === Optional Functions ===
    # Initialization function, called when a new node is created.
    # This is the most common place to create the sockets for a node, as shown below.
    # NOTE: this is not the same as the standard __init__ function in Python, which is
    # a purely internal Python method and unknown to the node system!
    def init(self, context):
        self.inputs.new('ImageSocketType', "Input image")
        self.inputs.new('ImageSocketType', "Input image")
        self.outputs.new('ImageSocketType', "Output image")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.label("Tick for channels, no tick for batch")
        layout.prop(self, "dim")
        # Detail buttons in the sidebar.
        # If this function is not defined, the draw_buttons function is used instead

        # Optional: custom label
        # Explicit user label overrides this, but here we can define a label dynamically


class AccuracyNode(Node, MyCustomTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'AccuracyNodeType'
    # Label for nice name display
    bl_label = 'Accuracy Node'
    # Icon identifier
    bl_icon = 'SOUND'

    # === Custom Properties ===
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    Testonly = bpy.props.BoolProperty(default=True)
    # === Optional Functions ===
    # Initialization function, called when a new node is created.
    # This is the most common place to create the sockets for a node, as shown below.
    # NOTE: this is not the same as the standard __init__ function in Python, which is
    # a purely internal Python method and unknown to the node system!
    def init(self, context):
        self.inputs.new('NAFlatSocketType', "Input class")
        self.inputs.new('LabelSocketType', "Input label")
        self.outputs.new('LossSocketType', "Output Accuracy")


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
        # Detail buttons in the sidebar.
        # If this function is not defined, the draw_buttons function is used instead

        # Optional: custom label
        # Explicit user label overrides this, but here we can define a label dynamically


class SolverNode(Node, MyCustomTreeNode):
    # === Basics ===
    # Description string
    '''A Convolution node'''
    # Optional identifier string. If not explicitly defined, the python class name is used.
    bl_idname = 'SolverNodeType'
    # Label for nice name display
    bl_label = 'Solver Node'
    # Icon identifier
    bl_icon = 'SOUND'
    modes = [
        ("NAG", "NAG", "Nesterovs Accelerated Gradient"),
        ("ADAGRAD", "ADAGRAD", "Adaptive gradient descent"),
        ("SGD", "SGD", "Stochastic Gradient Descent")
    ]
    computemodes = [
        ("GPU", "GPU", "GPU"),
        ("CPU", "CPU", "CPU")
    ]
    # === Custom Properties ===
    # These work just like custom properties in ID data blocks
    # Extensive information can be found under
    # http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
    solvername = bpy.props.StringProperty()
    solver = bpy.props.EnumProperty(name="Mode", default='SGD', items=modes)
    compmode = bpy.props.EnumProperty(name="Compute Mode", default='GPU', items=computemodes)
    gpu = bpy.props.IntProperty(default=0 , soft_max=1)
    testinterval = bpy.props.IntProperty(default=500, min=1, soft_max=2000)
    testruns = bpy.props.IntProperty(default=50, min=1, soft_max=200)
    displayiter = bpy.props.IntProperty(default=100, min=1, soft_max=5000)
    maxiter = bpy.props.IntProperty(default=50000, min=5, soft_max=100000)
    learningrate = bpy.props.FloatProperty(default=0.01, min=0.001, soft_max=1)
    snapshotiter = bpy.props.IntProperty(default=10000, min=10, soft_max=50000)
    snapshotpath = bpy.props.StringProperty \
        (
            name="Snapshot Data Path",
            default="",
            description="Give the path to the snapshot data",
            subtype='DIR_PATH'
        )
    configpath = bpy.props.StringProperty \
        (
            name="Configuration Data Path",
            default="",
            description="Give the path to the config data",
            subtype='DIR_PATH'
        )
    caffexec = bpy.props.StringProperty \
        (
        name="Caffe Path",
        default="",
        description="Give the path to the config data",
        subtype='DIR_PATH'
        )

    def init(self, context):
        self.inputs.new('LossSocketType', "Input Loss")


    # Copy function to initialize a copied node from an existing one.
    def copy(self, node):
        print("Copying from node ", node)

    # Free function to clean up on removal.
    def free(self):
        print("Removing node ", self, ", Goodbye!")

    def update(self):
        x = 0
        for input in self.inputs:
            if input.is_linked == False:
                x += 1
                if x > 1:
                    self.inputs.remove(input)
        if x == 0:
            self.inputs.new('LossSocketType', "Input Loss")

    # Additional buttons displayed on the node.
    def draw_buttons(self, context, layout):
        layout.prop(self, "solvername")
        layout.prop(self, "solver")
        layout.prop(self, "compmode")
        layout.prop(self, "gpu")
        layout.prop(self, "testinterval")
        layout.prop(self, "testruns")
        layout.prop(self, "displayiter")
        layout.prop(self, "maxiter")
        layout.prop(self, "learningrate")
        layout.prop(self, "snapshotiter")
        layout.prop(self, "snapshotpath")
        layout.prop(self, "configpath")
        layout.prop(self, "caffexec")
        # Detail buttons in the sidebar.
        # If this function is not defined, the draw_buttons function is used instead

        # Optional: custom label
        # Explicit user label overrides this, but here we can define a label dynamically

# ## Node Categories ###
# Node categories are a python system for automatically
# extending the Add menu, toolbar panels and search operator.
# For more examples see release/scripts/startup/nodeitems_builtins.py

import nodeitems_utils
from nodeitems_utils import NodeCategory, NodeItem

# our own base class with an appropriate poll function,
# so the categories only show up in our own tree type
class MyNodeCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == 'CustomTreeType'

# all categories in a list
node_categories = [
    # identifier, label, items list
    MyNodeCategory("PNODES", "Processing Nodes", items=[
        # our basic node
        NodeItem("PoolNodeType"),
        NodeItem("ConvNodeType"),
        NodeItem("LRNNodeType"),
        NodeItem("ConcatNodeType")
    ]),
    MyNodeCategory("NNODES", "Neuron Nodes", items=[
        # our basic node
        NodeItem("FCNodeType"),
        NodeItem("FlattenNodeType"),
        NodeItem("AcNodeType"),
        NodeItem("ReluNodeType"),
        NodeItem("DropoutNodeType")
    ]),
    MyNodeCategory("SNODES", "Solver Nodes", items=[
        # our basic node
        NodeItem("SolverNodeType"),
        NodeItem("AccuracyNodeType"),
        NodeItem("EULossNodeType"),
        NodeItem("SCELossNodeType"),
        NodeItem("SMLossNodeType")
    ]),
    MyNodeCategory("DNODES", "Data Nodes", items=[
        # our basic node
        NodeItem("DataNodeType")
    ]),
    MyNodeCategory("MNODES", "Misc Nodes", items=[
        # our basic node
        NodeItem("SilenceNodeType")
    ]),
]


def register():
    bpy.utils.register_class(MyCustomTree)
    bpy.utils.register_class(DataNode)
    bpy.utils.register_class(DropoutNode)
    bpy.utils.register_class(PoolNode)
    bpy.utils.register_class(ConvNode)
    bpy.utils.register_class(FCNode)
    bpy.utils.register_class(FlattenNode)
    bpy.utils.register_class(LRNNode)
    bpy.utils.register_class(ActivationNode)
    bpy.utils.register_class(ReLuNode)
    bpy.utils.register_class(SMLossNode)
    bpy.utils.register_class(SCELossNode)
    bpy.utils.register_class(EULossNode)
    bpy.utils.register_class(ConcatNode)
    bpy.utils.register_class(AccuracyNode)
    bpy.utils.register_class(SolverNode)
    bpy.utils.register_class(ImageSocket)
    bpy.utils.register_class(LabelSocket)
    bpy.utils.register_class(LossSocket)
    bpy.utils.register_class(AFlatSocket)
    bpy.utils.register_class(NAFlatSocket)
    bpy.utils.register_class(SilenceNode)

    nodeitems_utils.register_node_categories("CUSTOM_NODES", node_categories)


def unregister():
    nodeitems_utils.unregister_node_categories("CUSTOM_NODES")

    bpy.utils.unregister_class(MyCustomTree)
    bpy.utils.unregister_class(MyCustomSocket)
    bpy.utils.unregister_class(MyCustomNode)


if __name__ == "__main__":
    register()
