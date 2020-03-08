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

bl_info = {
	"name": "Z2D Animation",
	"description": "2D Animation tools",
	"author": "ZMakesThingsGo",
	"version": (0, 0, 0, 0),
	"blender": (2, 81, 0),
	"location": "View3D",
	"category": "Object"}

import bpy
import bpy.props  #from bpy.props import *

if True: #attempting to make testing/debugging easier and faster
	import importlib
	from . import z2d_registration; importlib.reload( z2d_registration )
	from . import z2d_panel; importlib.reload( z2d_panel )
	from . import z2d_op_orbit; importlib.reload( z2d_op_orbit )

from . z2d_registration import Z2D_REG

from . z2d_panel import Z2D_PT_panel
from . z2d_panel import Z2D_PT_panel_View
from . z2d_panel import Z2D_PT_panel_Frames
from . z2d_panel import Z2D_PT_panel_Brushes
from . z2d_panel import Z2D_PT_panel_Layers
from . z2d_panel import Z2D_PT_panel_Setup


from . z2d_op_orbit import Z2D_OT_orbit_ccw_operator
from . z2d_op_orbit import Z2D_OT_orbit_cw_operator
from . z2d_op_orbit import Z2D_OT_orbit_reset_operator
from . z2d_op_orbit import Z2D_OT_orbit_mirror_operator

from . z2d_op_frames import Z2D_OT_frame_add_shift_add
from . z2d_op_frames import Z2D_OT_frame_rem_shift_rem

from . import z2d_fake_preview;
from . z2d_fake_preview import Z2D_OT_fake_preview_operator
from . z2d_fake_preview import Z2D_PT_fakepreview_context_z2d

#classlist = []
#for uclass in z2d_fake_preview.classes:
#	classlist.append( uclass );


#https://docs.blender.org/api/current/index.html
Z2D_REG.init();

classes = ( 
	Z2D_PT_panel
	, Z2D_PT_panel_View
	, Z2D_PT_panel_Frames
	, Z2D_PT_panel_Brushes
	, Z2D_PT_panel_Layers
	, Z2D_PT_panel_Setup
	, Z2D_OT_orbit_ccw_operator
	, Z2D_OT_orbit_cw_operator
	, Z2D_OT_orbit_reset_operator 
	, Z2D_OT_orbit_mirror_operator
	, Z2D_OT_frame_add_shift_add
	, Z2D_OT_frame_rem_shift_rem
	, Z2D_OT_fake_preview_operator
	, Z2D_PT_fakepreview_context_z2d
)

cls_register, cls_unregister = bpy.utils.register_classes_factory( classes )

def register():
	cls_register();
	
	#TIMERS exist. Nevermind, they should NOT exist (can't draw this way (yet))
	bpy.app.timers.register( z2d_fake_preview.z2d_fakepreview_timer_update, persistent=True )

	
def unregister():

	if bpy.app.timers.is_registered( z2d_fake_preview.z2d_fakepreview_timer_update ):
		bpy.app.timers.unregister( z2d_fake_preview.z2d_fakepreview_timer_update )
		
	cls_unregister();
	
if __name__ == "__main__":
	register()