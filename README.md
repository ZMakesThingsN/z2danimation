# z2danimation
Blender 2.82+ plugin to aid in 2D animation

I am trying to make this more useful for practical purposes,
The intent is to mimic existing tools that minimize interference with the artist,
giving you more powerful tools that keep you from fiddling with UI's or wasting time managing abstractions.
Flash is still the best thing I have used for this, so we should make sure blender behaves at least as well.

I would eventually like to use this to create quality animated characters for videogames, cite Cuphead or Monster Boy and the Cursed Kingdom or many other 2D art driven games.
Godot (https://godotengine.org/) is the target game development platform.

## Installing the plugin

There are a few ways to do this;

Default methods are https://docs.blender.org/manual/en/latest/editors/preferences/addons.html

HOWEVER,

I strongly recommend creating a dedicated folder for your custom blender addons.
This helps defend against version changes / platform changes.

First, open blender and open the preferences in Edit -> Preferences
![Alt text](help/how_to_plugins_1.png?raw=true "How to get to preferences")

Second, make sure you have a custom path to addons
![Alt text](help/how_to_plugins_2.png?raw=true "How to set addon path")

Third, enable the new addon (You may need to click refresh)
![Alt text](help/how_to_plugins_3.png?raw=true "How to enable plugin")

## Using the plugin

TBD;

It helps immensely if you bind custom hotkeys to things.

Open the preferences, select Keymap on the left.

Here, you will be presented with a mess of information.

Since blender decided (foolishly) to hide scrollbars, you have to expand the 3D View -> 3D View (Global), and scroll down to find the "+ Add New" button.
Click it a few times, and plugin the information as shown.

There will be more operators as the plugin improves, however being able to:
	Rotate the view
	Mirror the view
	Insert blank frames
	Remove blank frames (until keyframe)

Are very valuable for workflow (TBD in a video).

Here are 3 example key bindings:

![Alt text](help/how_to_z2d_1.png?raw=true "Adding keybinds")

## Using blender's 2D animation to make a sketchy animatic

TBD

## Using blender to make a flat shaded animation

TBD