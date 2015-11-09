import bpy


def forceupdate(nodes):
    for node in nodes:
        if node.inputs:
            for inpt in node.inputs:
                try:
                    inpt.default_value = inpt.default_value  # set value to itself to force update
                    return True
                except:
                    pass
    return False


def get_nodes_links(treename):
    if not treename:
        for space in bpy.context.area.spaces:
            if space.type == 'NODE_EDITOR':
                treename = space.edit_tree.name
    tree = bpy.data.node_groups[treename]
    nodes = tree.nodes
    links = tree.links
    return nodes, links


def get_nodes_links_withsel(treename):
    if not treename:
        for space in bpy.context.area.spaces:
            if space.type == 'NODE_EDITOR':
                treename = space.edit_tree.name
    tree = bpy.data.node_groups[treename]
    nodes = tree.nodes
    links = tree.links
    all_nodes = nodes
    newnodes = []
    for node in nodes:
        if node.select == True:
            newnodes.append(node)
    if len(newnodes) == 0:
        newnodes = all_nodes
    nodes_sorted = sorted(newnodes, key=lambda x: x.name)  # Sort the nodes list to achieve consistent
    links_sorted = sorted(links, key=lambda x: x.from_node.name)  # results (order was changed based on selection).
    return nodes_sorted, links_sorted


def isStartNode(node):
    bool = True
    if len(node.inputs):
        for input in node.inputs:
            if input.links != ():
                bool = False
    return bool


def isEndNode(node):
    bool = True
    if len(node.outputs):
        for output in node.outputs:
            if output.links != ():
                bool = False
    return bool


def between(b1, a, b2):
    #   b1 MUST be smaller than b2!
    bool = False
    if a >= b1 and a <= b2:
        bool = True
    return bool


def overlaps(node1, node2):
    dim1x = node1.dimensions.x
    dim1y = node1.dimensions.y
    dim2x = node2.dimensions.x
    dim2y = node2.dimensions.y
    boolx = False
    booly = False
    boolboth = False

    # check for x overlap
    if between(node2.location.x, node1.location.x, (node2.location.x + dim2x)) or between(node2.location.x,
                                                                                          (node1.location.x + dim1x), (
                                                                                                      node2.location.x + dim2x)):  # if either edges are inside the second node
        boolx = True
    if between(node1.location.x, node2.location.x, node1.location.x + dim1x) and between(node1.location.x,
                                                                                         (node2.location.x + dim2x),
                                                                                         node1.location.x + dim1x):  # if each edge is on either side of the second node
        boolx = True

    # check for y overlap
    if between((node2.location.y - dim2y), node1.location.y, node2.location.y) or between((node2.location.y - dim2y),
                                                                                          (node1.location.y - dim1y),
                                                                                          node2.location.y):
        booly = True
    if between((node1.location.y - dim1y), node2.location.y, node1.location.y) and between((node1.location.y - dim1y),
                                                                                           (node2.location.y - dim2y),
                                                                                           node1.location.y):
        booly = True

    if boolx == True and booly == True:
        boolboth = True
    return boolboth


def treeMidPt(nodes):
    minx = (sorted(nodes, key=lambda k: k.location.x))[0].location.x
    miny = (sorted(nodes, key=lambda k: k.location.y))[0].location.y
    maxx = (sorted(nodes, key=lambda k: k.location.x, reverse=True))[0].location.x
    maxy = (sorted(nodes, key=lambda k: k.location.y, reverse=True))[0].location.y

    midx = minx + ((maxx - minx) / 2)
    midy = miny + ((maxy - miny) / 2)

    return midx, midy


def ArrangeFunction(context, treename=False):
    nodes, links = get_nodes_links_withsel(treename)
    margin = context.scene.NWSpacing

    oldmidx, oldmidy = treeMidPt(nodes)

    if context.scene.NWDelReroutes:
        # Store selection
        selection = []
        for node in nodes:
            if node.select == True and node.type != "REROUTE":
                selection.append(node.name)
        # Delete Reroutes
        for node in nodes:
            node.select = False  # deselect all nodes
        for node in nodes:
            if node.type == 'REROUTE':
                node.select = True
                bpy.ops.node.delete_reconnect()
        # Restore selection
        nodes, links = get_nodes_links(treename)
        nodes = list(nodes)
        for node in nodes:
            if node.name in selection:
                node.select = True
    else:
        # Store selection anyway
        selection = []
        for node in nodes:
            if node.select == True:
                selection.append(node.name)

    if context.scene.NWFrameHandling == "delete":
        # Store selection
        selection = []
        for node in nodes:
            if node.select == True and node.type != "FRAME":
                selection.append(node.name)
        # Delete Frames
        for node in nodes:
            node.select = False  # deselect all nodes
        for node in nodes:
            if node.type == 'FRAME':
                node.select = True
                bpy.ops.node.delete()
        # Restore selection
        nodes, links = get_nodes_links(treename)
        nodes = list(nodes)
        for node in nodes:
            if node.name in selection:
                node.select = True

    layout_iterations = len(nodes) * 2
    backward_check_iterations = len(nodes)
    overlap_iterations = len(nodes)
    for it in range(0, layout_iterations):
        print (
            'Layout Iteration %i / %i' % (it, layout_iterations + overlap_iterations + backward_check_iterations - 1))
        for node in nodes:
            isframe = False
            if node.type == "FRAME" and context.scene.NWFrameHandling == 'ignore':
                isframe = True
            if not isframe:
                if isStartNode(node) and context.scene.NWStartAlign:  # line up start nodes
                    node.location.x = node.dimensions.x / -2
                    node.location.y = node.dimensions.y / 2
                for link in links:
                    if link.from_node == node and link.to_node in nodes:
                        link.to_node.location.x = node.location.x + node.dimensions.x + margin
                        link.to_node.location.y = node.location.y - (node.dimensions.y / 2) + (
                            link.to_node.dimensions.y / 2)
            else:
                node.location.x = 0
                node.location.y = 0

    for it in range(0, backward_check_iterations):
        print ('Layout Iteration %i / %i' % (
            layout_iterations + it, layout_iterations + overlap_iterations + backward_check_iterations - 1))
        for link in links:
            if link.from_node.location.x + link.from_node.dimensions.x >= link.to_node.location.x and link.to_node in nodes:
                link.to_node.location.x = link.from_node.location.x + link.from_node.dimensions.x + margin

    # line up end nodes
    if context.scene.NWEndAlign:
        for node in nodes:
            max_loc_x = (sorted(nodes, key=lambda x: x.location.x, reverse=True))[0].location.x
            if isEndNode(node) and not isStartNode(node):
                node.location.x = max_loc_x

    for it in range(0, overlap_iterations):
        print ('Layout Iteration %i / %i' % (layout_iterations + overlap_iterations + it,
                                             layout_iterations + overlap_iterations + backward_check_iterations - 1))
        for node in nodes:
            isframe = False
            if node.type == "FRAME" and context.scene.NWFrameHandling == 'ignore':
                isframe = True
            if not isframe:
                for nodecheck in nodes:
                    isframe = False
                    if nodecheck.type == "FRAME" and context.scene.NWFrameHandling == 'ignore':
                        isframe = True
                    if not isframe:
                        if (node != nodecheck):  # dont look for overlaps with self
                            if overlaps(node, nodecheck):
                                node.location.y = nodecheck.location.y - nodecheck.dimensions.y - 0.5 * margin

    newmidx, newmidy = treeMidPt(nodes)
    middiffx = newmidx - oldmidx
    middiffy = newmidy - oldmidy

    # put nodes back to the center of the old center
    datacounter = 0
    for node in nodes:
        node.location.x = node.location.x - middiffx
        node.location.y = node.location.y - middiffy
        if node.bl_idname == "DataNodeType":
            node.location.x -= 200
            node.location.y += 320 * datacounter
            datacounter += 1


class ArrangeNodes(bpy.types.Operator):
    'Automatically layout the selected nodes in a linear and non-overlapping fashion.'
    bl_idname = 'nodes.layout'
    bl_label = 'Arrange Nodes'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        valid = False
        if context.space_data:
            if context.space_data.node_tree:
                if context.space_data.node_tree.nodes:
                    valid = True
        return valid

    def execute(self, context):
        ArrangeFunction(context)
        return {'FINISHED'}


def register():
    # props
    bpy.types.Scene.NWStartAlign = bpy.props.BoolProperty(
        name="Align Start Nodes",
        default=False,
        description="Put all nodes with no inputs on the left of the tree")
    bpy.types.Scene.NWEndAlign = bpy.props.BoolProperty(
        name="Align End Nodes",
        default=True,
        description="Put all nodes with no outputs on the right of the tree")
    bpy.types.Scene.NWSpacing = bpy.props.FloatProperty(
        name="Spacing",
        default=190.0,
        min=0.0,
        description="The horizonal space between nodes (vertical is half this)")
    bpy.types.Scene.NWDelReroutes = bpy.props.BoolProperty(
        name="Delete Reroutes",
        default=True,
        description="Delete all Reroute nodes to avoid unexpected layouts")
    bpy.types.Scene.NWFrameHandling = bpy.props.EnumProperty(
        name="Frames",
        items=(("ignore", "Ignore", "Do nothing about Frame nodes (can be messy)"),
               ("delete", "Delete", "Delete Frame nodes")),
        default='ignore',
        description="How to handle Frame nodes")

    bpy.utils.register_class(ArrangeNodes)


def unregister():
    # props
    del bpy.types.Scene.NWStartAlign
    del bpy.types.Scene.NWEndAlign
    del bpy.types.Scene.NWSpacing
    del bpy.types.Scene.NWDelReroutes
    del bpy.types.Scene.NWFrameHandling

    bpy.utils.unregister_class(ArrangeNodes)
