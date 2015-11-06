#Caffe Gui Tool
==============
This a node based tool for creating caffe networks. It works inside the graphics application 'blender' as a plugin. The reason for this is blender's highly stable, and universally compatible node editor.
##Important
**The latest commits bundle the application into one single extension for blender. You will need to disable/remove all the old CGT addons, and then install the latest version.**

On the plus side - CGT now has .prototxt import features, and asyncrounous network training/plotting.

[**Installation Tutorial Video**](http://bit.ly/1AnTVD2)


[**Installation and usage instructions on the wiki**](http://bit.ly/1AnTVD2)

###Useful features:
* .prototxt import - load networks up to and including googlenet
* Train networks asyncronously within blender
* Plot loss in real time
* Manage different network structures, store networks with attached comments
* Plot and compare Multiple networks' loss
* Autonaming of top,bottom, and layer names
* Visual editing of layer properties
* Reliable duplication of layers with SHIFT+D
* Saving & Loading nodetrees, along with their performance and info data for later editing
* Generation of train_test, _deploy, _solver, prototxts, and the training bash script


####ConvNet

![alt tag](https://dl.dropboxusercontent.com/u/10860244/Selection_001.png)
