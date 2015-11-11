# Caffe Gui Tool (V2)
This a node based tool for creating caffe networks. It works inside the graphics application 'blender' as a plugin. The reason for this is blender's highly stable, and universally compatible node editor.
## Important (Current users)

The latest commits add a huge number of features, and bundle the application into one single extension for blender. Pretty much everything has changed. If you haven't already done so, make sure you remove any old copies and then install the new version with the instructions on the wiki. You only have to do this once, and from then on you can update with a git-pull.

The way everything works has also changed a lot. Make sure you read the wiki for all the new features.
## [**Installation and usage instructions on the wiki**](http://bit.ly/1HCES6r)

## Useful features:
* .prototxt import - load networks up to and including googlenet
* Train networks asyncronously within blender
* Plot loss in real time
* Manage different network structures, store networks with attached comments
* Storage in native .cexp format, enabling node trees to be stored and recovered together with the performance data of that network
* Plot and compare Multiple networks' loss
* Autonaming of top,bottom, and layer names
* Visual editing of layer properties
* Reliable duplication of layers with SHIFT+D
* Saving & Loading nodetrees, along with their performance and info data for later editing
* Generation of train_test, _deploy, _solver, prototxts, and the training bash script


## UI

![Nodes](https://dl.dropboxusercontent.com/u/10860244/CGT/Selection_032.png)
![Graph](https://dl.dropboxusercontent.com/u/10860244/CGT/Selection_031.png)
