# Caffe Gui Tool
This a node based tool for creating caffe networks. It works inside the graphics application 'blender' as a plugin. The reason for this is blender's highly stable, and universally compatible node editor.
## Important

* The latest commits add a huge number of features, and bundle the application into one single extension for blender. You will need to disable/remove all the old CGT addons, and then install the latest version if you have not already done so. You only have to do this once.
* These Commits also MASSIVELY change how the addon is used. No more spacebar --> create solution. Make sure you read the [**Wiki**](http://bit.ly/1HCES6r)!
On the plus side - CGT now has .prototxt import features, and asyncronous network training/plotting. Everything has also been hugely refactored for clarity.
* Huge changes such as the recent commits mean bugs will slip through. Please feel free to report any in the issues manager and I will try and patch them same-day.

[**Installation and usage instructions on the wiki**](http://bit.ly/1HCES6r)

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
