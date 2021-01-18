# Note
Caffe V1 is now well over 6 years old, and superceded by a mutlitude of tools. I have archived this project to be used as reference. I hope it proved useful. 

# Caffe Gui Tool (V2)
This a node based tool for creating caffe networks. It works inside the graphics application 'blender' as a plugin. The reason for this is blender's highly stable, and universally compatible node editor.

## Important (Current users)
* The latest commits add a huge number of features. Make sure you read the [**Wiki**](http://bit.ly/1HCES6r)! If you haven't already done so, you'll need to reinstall the addon once following the installation instructions in the wiki. This allows you to update with just a git pull. With huge features do sometimes come bugs. Drop any in the issues tracker and we will try and fix same day.

## [**Installation and usage instructions on the wiki**](http://bit.ly/1HCES6r)

## Useful features:
* .prototxt generation - instead of typing 700 lines of prototxt, create a nodegraph and have it converted to prototxt.
* .prototxt import - load networks up to and including googlenet, edit them, then regenerate the prototxt.
* Train networks asyncronously within blender, plotting loss in real time
* Manages different network structures, and stores networks with performance data and comments under a single file
* Simultaneous graphing of multiple networks
* Automatic naming of top,bottom, and layer names if needed
* Visual editing of layer properties
* Reliable duplication of layers with SHIFT+D
* Saving & Loading nodetrees, along with their performance and info data for later editing
* Generation of train_test, _deploy, _solver, prototxts, and the training bash script


## UI screenshots

![Nodes](https://image.ibb.co/c6RiTa/Selection_032.png)
![Graph](https://image.ibb.co/cc6TuF/Selection_036.png)
