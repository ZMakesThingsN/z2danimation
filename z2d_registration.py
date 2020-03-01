# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

import bpy

class Z2D_REG:

	@staticmethod
	def init():
		
		wm = bpy.types.WindowManager
		wm.Z2D_state = bpy.props.IntProperty( default=0 )
		
		

		"""
		addon_keymaps = []

		bpy.types.Scene.show_buttons = bpy.props.BoolProperty(
			name="Show Buttons", 
			description="Show or hide the mouse buttons", 
			default=True)

		h_dock = [ ("0",  "Left",  "Dock to the left side"),
				   ("1",  "Right", "Dock to the right side"),
				   ("2",  "Center", "Dock to the center"),
				   ("3",  "Cursor", "Attach to mouse cursor")
				 ]

		bpy.types.Scene.h_dock = bpy.props.EnumProperty(
			items = h_dock, name="Dock", 
			description="Dock to left, center, right or to the cursor", 
			default="1")

		bpy.types.Scene.cursor_offset_x = IntProperty(
											  name="Offset X", 
											  description="Offset X to cursor",
											  default = 0)

		bpy.types.Scene.cursor_offset_y = IntProperty(
											  name="Offset Y", 
											  description="Offset Y to cursor",
											  default = 0)

		bpy.types.Scene.font_color = bpy.props.FloatVectorProperty(  
		   name="Text Color",
		   subtype='COLOR',
		   default=(1.0, 1.0, 1.0),
		   min=0.0, max=1.0,
		   description="Color for the text"
		   )

		bpy.types.Scene.color_buttons = bpy.props.FloatVectorProperty(  
		   name="Color Buttons",
		   subtype='COLOR',
		   default=(0.1, 0.1, 0.1),
		   min=0.0, max=1.0,
		   description="Color for mouse buttons"
		   )

		bpy.types.Scene.color_buttons_active = bpy.props.FloatVectorProperty(  
		   name="Color Buttons active",
		   subtype='COLOR',
		   default=(1.0, 1.0, 1.0),
		   min=0.0, max=1.0,
		   description="Color for mouse active buttons"
		   )

		"""
