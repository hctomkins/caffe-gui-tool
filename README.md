caffe-gui-tool
==============

Node based Gui for creating caffe networks
NB - Only clone the prebuilt folder if absolutely neccessary!!

Installation
==============

Apt-get Binaries (~5kb download, and quicker apt-get download)
0. sudo add-apt-repository ppa:irie/blender && sudo apt-get update
1. sudo apt-get install blender, and launch blender with 'blender' command
2. download two python scripts
3. file --> user preferences --> addons --> install from file --> select Caffe Nodes script --> install
4. tick the checkbox in the newly appeared boxe's top right corner
5. file --> user preferences --> addons --> install from file --> select Caffe Generate script --> install
6. tick the checkbox in the newly appeared boxe's top right corner
7. user preferences --> file --> uncheck relative paths (IMPORTANT)
8. click on the cube icon in the bottom left
9. click 'node editor' in the popup menu
10. in the newly appeared node editor, look at the bottom panel, it should read 'view - select - add - node'
11. immediately right of this panel there should be a set of four icons joined together. select the THIRD.
12. file --> save startup file
13. file --> user preferences --> save user settings (on the bottom of the window(
14. restart blender

Git Binaries (Simpler, but less reliable)
1. launch blender with ./blender in directory

Use
============
Click the 'New' button on the bottom part of the screen to create a new network
Name your network in this box, and ENSURE THE 'F' BUTTON NEXT TO THE NAME IS CHECKED
Use shift+a to start adding nodes
Join up, and fill in all the required fields to your taste. All networks require a data node, and solver node.
When ready, press spacebar.
type 'solution' in the search bar
select 'create solution' and press enter
This will create the required prototxt files in the config directory.
Mark the train file as executable
run the train file


Limitations
=============
- Negative gradient on ReLu not yet supported
- ReLu not calculated in place
- Many nodes not yet supported
