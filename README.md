#Caffe Gui Tool
==============
This a node based tool for creating caffe networks. It works inside the graphics application 'blender' as a plugin. The reason for this is blender's highly stable, and universally compatible node editor.
##Important
**CGT features change regularly. If you update and your nodetree no longer compiles, all the nodes in your tree must be Re-added. A prototxt load feature will be added in the next update to make this process redundant.**

###Potentially useful features:
* Autonaming of top,bottom, and layer names
* Visual editing of later properties
* Reliable duplication of layers with SHIFT+D
* Saving & Loading nodetrees in blender's native 'xxxx.blend' format for later editing
* Generation of Train_test, _deploy, _solver, and the training .sh script

###WIP features:
* .prototxt import
* Training network within blender, plotting train & test error

For [**installation instructions**](http://bit.ly/1AnTVD2) please see [http://chasvortex.github.io/caffe-gui-tool/](http://bit.ly/1AnTVD2)

####ConvNet

![alt tag](https://dl.dropboxusercontent.com/u/10860244/Selection_001.png)
