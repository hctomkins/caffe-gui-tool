Caffe Gui Tool
==============

Node based Gui for creating caffe networks

Master branch requires manual setup, although no compilation, and prebuilt is ready to go, but ~100mb compressed

ConvNet

![alt tag](https://camo.githubusercontent.com/31e4f9dd627afa62bb0366eae79e9254dfb94934/68747470733a2f2f646c2e64726f70626f7875736572636f6e74656e742e636f6d2f752f31303836303234342f676875622f53637265656e73686f7425323066726f6d253230323031342d31302d31392532303133253341333025334132312e706e67)

Autoencoder

![alt tag](https://camo.githubusercontent.com/d28a002ce2c7139877f2c1e8fa3f935ed57fa055/68747470733a2f2f646c2e64726f70626f7875736572636f6e74656e742e636f6d2f752f31303836303234342f676875622f53637265656e73686f7425323066726f6d253230323031342d31302d31392532303133253341333025334131352e706e67)


Installation
==============

Apt-get Binaries (~5kb download, and quicker apt-get download)
- sudo add-apt-repository ppa:irie/blender && sudo apt-get update
- sudo apt-get install blender, and launch blender with 'blender' command
- download two python scripts
- file --> user preferences --> addons --> install from file --> select Caffe Nodes script --> install
- tick the checkbox in the newly appeared box's top right corner
- file --> user preferences --> addons --> install from file --> select Caffe Generate script --> install
- tick the checkbox in the newly appeared box's top right corner
- user preferences --> file --> uncheck relative paths (IMPORTANT)
- click on the cube icon in the bottom left
- click 'node editor' in the popup menu
- in the newly appeared node editor, look at the bottom panel, it should read 'view - select - add - node'
- immediately right of this panel there should be a set of four icons joined together. select the THIRD.
- file --> save startup file
- file --> user preferences --> save user settings (on the bottom of the window(
- restart blender

Use
============
- Click the 'New' button on the bottom part of the screen to create a new network
- Name your network in this box, and ENSURE THE 'F' BUTTON NEXT TO THE NAME IS CHECKED
- Use shift+a to start adding nodes
- Join up, and fill in all the required fields to your taste. All networks require a data node, and solver node.
- When ready, select all nodes by single, (or double if required) tapping 'a', then press spacebar.
- type 'solution' in the search bar
- select 'create solution' and press enter
- This will create the required prototxt files in the config directory.
- Mark the train file as executable
- run the train file


Limitations
=============
- Negative gradient on ReLu not yet supported
- ReLu not calculated in place
- Many nodes not yet supported
