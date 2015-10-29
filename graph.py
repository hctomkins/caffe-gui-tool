__author__ = 'Hugh'

import os
from .parse import search as findfirstraw
import time
import bpy
import random
import subprocess


class Load(bpy.types.Operator):
    """Load Caffe solver"""
    bl_idname = "nodes.load_solver"
    bl_label = "Load solution"
    bl_options = {'REGISTER'}

    def execute(self, context):

        return {'FINISHED'}  # this lets blender know the operator finished successfully.
