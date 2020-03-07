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

class Z2D_PT_panel_View(Panel):
	bl_space_type = "VIEW_3D"
	bl_region_type = "UI"
	bl_label = "View"
	bl_category = "Z2D Panel"
	bl_parent_id = "Z2D_PT_panel"

	def draw(self, context):
		
		layout = self.layout
			
		row = layout.row()
		
		row.operator('object.z2d_orbit_ccw_operator', text="CCW", icon="TRACKING_REFINE_BACKWARDS")
		row.operator('object.z2d_orbit_reset_operator', text="Reset", icon="HIDE_OFF")
		row.operator('object.z2d_orbit_cw_operator', text="CW", icon="TRACKING_REFINE_FORWARDS")
		
		row = layout.row()
		row.operator('object.z2d_orbit_mirror_operator', text="Mirror", icon="MOD_MIRROR")
		
class Z2D_PT_panel_Frames(Panel):
	bl_space_type = "VIEW_3D"
	bl_region_type = "UI"
	bl_label = "Frames"
	bl_category = "Z2D Panel"
	bl_parent_id = "Z2D_PT_panel"

	def draw(self, context):
		
		layout = self.layout
			
		row = layout.row()
		row.operator('object.z2d_frame_rem_shift_rem', text="", icon="TRIA_LEFT")
		row.operator('object.z2d_frame_add_shift_add', text="", icon="TRIA_RIGHT")
		
class Z2D_PT_panel_Brushes(Panel):
	bl_space_type = "VIEW_3D"
	bl_region_type = "UI"
	bl_label = "Brushes"
	bl_category = "Z2D Panel"
	bl_parent_id = "Z2D_PT_panel"

	def draw(self, context):
		layout = self.layout
		layout.use_property_split = True
		layout.use_property_decorate = False

		#VIEW3D_PT_tools_grease_pencil_brush_settings  (this one SUCKS for usability.)
		tool_settings = context.scene.tool_settings
		gp_settings = tool_settings.gpencil_paint

		row = layout.row()
		row.template_ID_preview(gp_settings, "brush", hide_buttons=True );#, new="brush.add_gpencil" );#, rows=3, cols=8)

		
		paint = tool_settings.gpencil_paint
		brush = paint.brush
		if brush is None:
			pass;
		else:
				
			gp_brush_settings = brush.gpencil_settings

			ma = gp_brush_settings.material
			row = layout.row(align=True)
			if not gp_brush_settings.use_material_pin:
				ma = context.object.active_material
			icon_id = 0
			if ma:
				icon_id = ma.id_data.preview.icon_id
				txt_ma = ma.name
				maxw = 25
				if len(txt_ma) > maxw:
					txt_ma = txt_ma[:maxw - 5] + '..' + txt_ma[-3:]
			else:
				txt_ma = ""

			sub = row.row()
			sub.ui_units_x = 8
			sub.popover(
				panel="TOPBAR_PT_gpencil_materials",
				text=txt_ma,
				icon_value=icon_id,
			)

			#row.prop(gp_brush_settings, "use_material_pin", text="")

			#row.prop(brush, "use_material_pin", text="")
			
		#VIEW3D_PT_tools_grease_pencil_brush_settings
		gp_settings = brush.gpencil_settings
		tool = context.workspace.tools.from_space_view3d_mode(context.mode, create=False)
		if gp_settings is None:
			pass;
		else:

			# Brush details
			if brush.gpencil_tool == 'ERASE':
				row = layout.row(align=True)
				row.prop(brush, "size", text="Radius")
				row.prop(gp_settings, "use_pressure", text="", icon='STYLUS_PRESSURE')
				row.prop(gp_settings, "use_occlude_eraser", text="", icon='XRAY')

				row = layout.row(align=True)
				row.prop(gp_settings, "eraser_mode", expand=True)
				if gp_settings.eraser_mode == 'SOFT':
					row = layout.row(align=True)
					row.prop(gp_settings, "pen_strength", slider=True)
					row.prop(gp_settings, "use_strength_pressure", text="", icon='STYLUS_PRESSURE')
					row = layout.row(align=True)
					row.prop(gp_settings, "eraser_strength_factor")
					row = layout.row(align=True)
					row.prop(gp_settings, "eraser_thickness_factor")

				row = layout.row(align=True)
				row.prop(gp_settings, "use_cursor", text="Display Cursor")

			# FIXME: tools must use their own UI drawing!
			elif brush.gpencil_tool == 'FILL':
				row = layout.row(align=True)
				row.prop(gp_settings, "fill_leak", text="Leak Size")
				row = layout.row(align=True)
				row.prop(brush, "size", text="Thickness")
				row = layout.row(align=True)
				row.prop(gp_settings, "fill_simplify_level", text="Simplify")

			else:  # brush.gpencil_tool == 'DRAW':
				row = layout.row(align=True)
				row.prop(brush, "size", text="Radius")
				row.prop(gp_settings, "use_pressure", text="", icon='STYLUS_PRESSURE')
				row = layout.row(align=True)
				row.prop(gp_settings, "pen_strength", slider=True)
				row.prop(gp_settings, "use_strength_pressure", text="", icon='STYLUS_PRESSURE')

			# FIXME: tools must use their own UI drawing!
			if tool.idname in {
					"builtin.arc",
					"builtin.curve",
					"builtin.line",
					"builtin.box",
					"builtin.circle",
					"builtin.polyline"
			}:
				settings = context.tool_settings.gpencil_sculpt
				if compact:
					row = layout.row(align=True)
					row.prop(settings, "use_thickness_curve", text="", icon='CURVE_DATA')
					sub = row.row(align=True)
					sub.active = settings.use_thickness_curve
					sub.popover(
						panel="TOPBAR_PT_gpencil_primitive",
						text="Thickness Profile",
					)
				else:
					row = layout.row(align=True)
					row.prop(settings, "use_thickness_curve", text="Use Thickness Profile")
					sub = row.row(align=True)
					if settings.use_thickness_curve:
						# Curve
						layout.template_curve_mapping(settings, "thickness_primitive_curve", brush=True)
		
class Z2D_PT_panel_Layers(Panel):
	bl_space_type = "VIEW_3D"
	bl_region_type = "UI"
	bl_label = "Layers"
	bl_category = "Z2D Panel"
	bl_parent_id = "Z2D_PT_panel"

	def draw(self, context):
		pass;
		
class Z2D_PT_panel_Setup(Panel):
	bl_space_type = "VIEW_3D"
	bl_region_type = "UI"
	bl_label = "Setup"
	bl_category = "Z2D Panel"
	bl_parent_id = "Z2D_PT_panel"

	def draw(self, context):
		pass;

class Z2D_PT_panel(Panel):
	bl_space_type = "VIEW_3D"
	bl_region_type = "UI"
	bl_label = "Z2D Panel"
	bl_category = "Z2D Panel"
	
	def draw(self, context):
		scene = context.scene
			
		layout = self.layout	#https://docs.blender.org/api/current/bpy.types.UILayout.html#bpy.types.UILayout
		
		#layout.use_property_split = True;
		#layout.use_property_decorate = False;
		
		#Special updates? Hm...
		
		"""
		
		flow = layout.grid_flow(row_major=True, columns=0, even_columns=False, even_rows=False, align=True)
			
		col = flow.column();  #column_flow didnt work, box();#
		
		#col.label(text="Brushes")
		
		row = flow.row();  #column_flow didnt work, box();#
		
		row.label(text="Brushes")
		
		row = subcol.row()
		
		#sketch, Ink, Fill, ?
		#	GPBRUSH_PENCIL, GPBRUSH_INK, GPBRUSH_FILL
		#<size slider>
		#	(sketch just has size, ink has smoothing amount, Fill?)
		#Last used color swatches
		#Color palette memory (hm... separate window? Not sure here.)
		
		row = col.row()
		row.label(text="View")
		
		row = col.row()
		
		row.operator('object.z2d_orbit_ccw_operator', text="CCW", icon="TRACKING_REFINE_BACKWARDS")
		row.operator('object.z2d_orbit_reset_operator', text="Reset", icon="HIDE_OFF")
		row.operator('object.z2d_orbit_cw_operator', text="CW", icon="TRACKING_REFINE_FORWARDS")
		
		row = col.row()
		row.operator('object.z2d_orbit_mirror_operator', text="Mirror", icon="MOD_MIRROR")
		
		#Background color (context sensitive to current display mode)
		
		#Onion skin settings (current layer(s) ?)
		
		#col = layout.column()
		
		row = col.row()
		row.label(text="Layers")
		
		#this one is tricky... we ALREADY KNOW that ink lines & fills should be SEPARATE layers...
			#If it worked like flash, the two would be linked for deformations...
			#And we'd try and keep track of that.
		#But a "logical" layer should adjust if you are in sketch, ink, fill modes,
		#	so that each layer has 3 - 4 actual layers depending (adds as needed)
		#internal naming convention prefix with username postfix?
		#And every layer AUTOMATICALLY has a transform & transform animation attached to it
		#	(allows you to animate each layer as a normal transform... use linear only? what can Godot support?)
		#
		#RESTRICT_SELECT_OFF  (filled arrow)
		#RESTRICT_SELECT_ON   (hollow arrow)
		#
		row = col.row()
		row.label(text="Frames")
		
		row = col.row()
		row.operator('object.z2d_frame_rem_shift_rem', text="", icon="TRIA_LEFT")
		row.operator('object.z2d_frame_add_shift_add', text="", icon="TRIA_RIGHT")
		
		#Add a keyframe KEY_HLT  (blank? or transform keyframe?)
		#move/translate, rotate, scale, shear current layer? (hm)
		#	for this to WORK, each "layer" is a gpencil object... Thats pretty annoying.
		#
		
		row = col.row()
		row.label(text="Setup")
		
			#-> create new setup automagically,  ERROR  icon is a good warning
			#-> add in shortcuts (hm, dangerous) and setup everything for fast sketch drawing?
			#-> simplified rendering presets? OUTLINER_OB_CAMERA  + size? can we thread this out??
			#ORPHAN_DATA (broken heart, lol)
	
		#layout.operator(path_to_operator, text="CCW", icon="CANCEL")
		"""
		
		
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