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
from bpy.types import Panel

class Z2D_PT_panel(Panel):
	bl_space_type = "VIEW_3D"
	bl_region_type = "UI"
	bl_label = "Z2D Panel"
	bl_category = "Z2D Panel"
	
	def draw(self, context):
		
		layout = self.layout	#https://docs.blender.org/api/current/bpy.types.UILayout.html#bpy.types.UILayout
		scene = context.scene
						  
		row = layout.row()
		
		row = layout.row()
		layout.label(text="Brushes")
		
		row = layout.row()
		layout.label(text="View")
		
		row = layout.row()
		
		row.operator('object.z2d_orbit_ccw_operator', text="CCW", icon="TRACKING_REFINE_BACKWARDS")
		row.operator('object.z2d_orbit_reset_operator', text="Reset", icon="HIDE_OFF")
		row.operator('object.z2d_orbit_cw_operator', text="CW", icon="TRACKING_REFINE_FORWARDS")
		
		row = layout.row()
		row.operator('object.z2d_orbit_mirror_operator', text="Mirror", icon="MOD_MIRROR")
		
		
		#col = layout.column()
		
		row = layout.row()
		layout.label(text="Layers")
		
		row = layout.row()
		layout.label(text="Frames")
		
		row = layout.row()
		row.operator('object.z2d_frame_rem_shift_rem', text="", icon="TRIA_LEFT")
		row.operator('object.z2d_frame_add_shift_add', text="", icon="TRIA_RIGHT")
		
		row = layout.row()
		layout.label(text="Setup")
		
			#-> create new setup automagically,
			#-> add in shortcuts (hm, dangerous) and setup everything for fast sketch drawing?
			
	
		#layout.operator(path_to_operator, text="CCW", icon="CANCEL")
		
		"""
		if(context.window_manager.SCV_started):
			row.operator('object.scv_ot_draw_operator', text="Stop Shortcut VUr", icon="CANCEL")
		else:
			row.operator('object.scv_ot_draw_operator', text="Start Shortcut VUr", icon="PLAY")

		row = layout.row()
		layout.prop(context.scene, "h_dock")
		
		if (scene.h_dock == "3"):
			row = layout.row()
			layout.prop(context.scene, "cursor_offset_x")

			row = layout.row()
			layout.prop(context.scene, "cursor_offset_y")

		row = layout.row()
		layout.prop(context.scene, "font_color")

		row = layout.row()
		layout.prop(context.scene, "color_buttons")

		row = layout.row()
		layout.prop(context.scene, "color_buttons_active")

		row = layout.row()
		layout.prop(context.scene, "show_buttons")
		"""