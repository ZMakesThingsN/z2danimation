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
import mathutils
import math

from bpy.types import Operator

M_DTR = 0.01745329251994329576923690768489	# pi / 180.0

def ported_matrix_multiply( A, B ):
	return A @ B

def z2dmath_roll( M, angle_deg, axis='Z' ):
	return ported_matrix_multiply( M, mathutils.Matrix.Rotation( angle_deg * M_DTR, 4, axis ) );
	
def z2dmath_mirror( M, axis=[1.0,0.0,0.0] ):
	return ported_matrix_multiply( M, mathutils.Matrix.Scale( -1, 4, axis ) );
	
	
def z2dmath_roll_camera_view( context, angle_deg ):

	angle_rad = angle_deg * M_DTR
	
	if True:
		rollmatrix = mathutils.Matrix.Rotation( angle_rad, 4, 'Z' )
		context.scene.camera.matrix_basis = ported_matrix_multiply( context.scene.camera.matrix_basis, rollmatrix ); #Pure rotation
	
	#buuuut actually CHANGING THE CAMERA is a BAD IDEA (because we dont know the VIEW rotation anymore)
	#Can we do something about this?!??!
	#AUGH. how???
	#
	#layout.operator("view3d.view_lock_to_active")
	#layout.operator("view3d.view_lock_clear")
	#SpaceView3D.lock_camera
	#
	
	#BUT this involves a TRANSLATION of the view_camera_offset... (rotate AROUND it)
	#BUT the offset is RELATIVE TO THE REGION RATIO...
	ca = math.cos( angle_rad );
	sa = math.sin( angle_rad );
	
	regionsize = ( context.region.width, context.region.height );
	
	ox = regionsize[0]*context.region_data.view_camera_offset[0];  #Apparently not really even, more of... scaled to camera?? sick.
	oy = regionsize[1]*context.region_data.view_camera_offset[1];
	
	newx = ca*ox + sa*oy;
	newy = -sa*ox + ca*oy;
	
	context.region_data.view_camera_offset = ( newx/regionsize[0], newy/regionsize[1] );	#view_camera_zoom,  view_distance,   view_location (pivot location)
	
	#Does Nothing:
	#context.region_data.view_rotation = context.region_data.view_rotation @ mathutils.Quaternion( (1.0, 0.0, 0.0), angle_rad )
	#context.region_data.update()
	
	#Does Nothing:
	#rollmatrix = mathutils.Matrix.Rotation( angle_rad, 4, 'Z' )
	#context.region_data.view_matrix = ported_matrix_multiply( context.region_data.view_matrix, rollmatrix ); #Pure rotation
	#context.region_data.update()
	#print( context.region_data.window_matrix ) <- just the offset really.
	#
	
	#Cannot alter perspective matrix.
	#rollmatrix = mathutils.Matrix.Rotation( angle_rad, 4, 'Z' )
	#context.region_data.perspective_matrix = context.region_data.window_matrix @ ported_matrix_multiply( context.region_data.view_matrix, rollmatrix );
	
	
	#rendersize = ( context.scene.render.resolution_x, context.scene.render.resolution_y )
	#framepoints = context.scene.camera.data.view_frame();
	
	#print( framepoints )
	#print( regionsize, ox, oy );
	
	#print( "cw camera" );

def z2dmath_roll_context_view( context, angle_deg ):

	if context.region_data.view_perspective == 'CAMERA':	#If we are in camera view:
		z2dmath_roll_camera_view( context, angle_deg );
		#context.scene.camera.matrix_basis = z2dmath_roll( context.scene.camera.matrix_basis, angle_deg );
	elif context.region_data.view_perspective == 'ORTHO':
		#Not sure what this means...
	
		transmat = context.region_data.view_matrix;
		context.region_data.view_matrix = z2dmath_roll( transmat, angle_deg, axis = context.region_data.perspective_matrix.row[2][:3] );
		context.region_data.update()
		
		pass;#context.region_data.
		#	scene.view_layers.
		#	context.region_data
	elif context.region_data.view_perspective == 'PERSP':
		#perspective_matrix = window_matrix * view_matrix
		#context.region_data.perspective_matrix.row[2]
		#context.region_data.perspective_matrix.col[2]
		#context.region_data.view_matrix = z2dmath_roll( context.region_data.view_matrix, angle_deg, axis = context.region_data.perspective_matrix.col[2][:3] );
		
		#This seems to be the right roll but... we need to roll AROUND a point... specifically, the middle of the window point so. Hm...
		
		transmat = context.region_data.view_matrix;#ported_matrix_multiply( context.region_data.view_matrix, mathutils.Matrix.Translation( -context.region_data.perspective_matrix.translation ) );
		context.region_data.view_matrix = z2dmath_roll( transmat, angle_deg, axis = context.region_data.perspective_matrix.row[2][:3] );
		
		#context.region_data.window_matrix = z2dmath_roll( context.region_data.window_matrix, angle_deg );
		context.region_data.update()
		


class Z2D_OT_orbit_ccw_operator(Operator):
	bl_idname = "object.z2d_orbit_ccw_operator"
	bl_label = "Orbit CCW"
	bl_description = "Orbit CCW" 
	bl_options = {'REGISTER'}
	
	delta : bpy.props.FloatProperty()
	
	@classmethod
	def poll(cls, context):
		ob = context.object
		if ob and ob.type == 'GPENCIL':
			pass;
		return True;

	def execute(self, context):
		scene = context.scene
	
		if(context.window_manager.Z2D_state == 0):
			pass;
		
		z2dmath_roll_context_view( context, -15 );
		
		return {'FINISHED'};
	
class Z2D_OT_orbit_cw_operator(Operator):
	bl_idname = "object.z2d_orbit_cw_operator"
	bl_label = "Orbit CW"
	bl_description = "Orbit CW" 
	bl_options = {'REGISTER'}
	
	delta : bpy.props.FloatProperty()
	
	@classmethod
	def poll(cls, context):
		ob = context.object
		if ob and ob.type == 'GPENCIL':
			pass;
		return True;

	def execute(self, context):
		scene = context.scene
	
		if(context.window_manager.Z2D_state == 0):
			pass;
			
		z2dmath_roll_context_view( context, 15 );
		
		return {'FINISHED'};
	
class Z2D_OT_orbit_reset_operator(Operator):
	bl_idname = "object.z2d_orbit_reset_operator"
	bl_label = "Orbit Reset"
	bl_description = "Orbit Reset" 
	bl_options = {'REGISTER'}
	
	delta : bpy.props.FloatProperty()
	
	@classmethod
	def poll(cls, context):
		ob = context.object
		if ob and ob.type == 'GPENCIL':
			pass;
		return True;

	def execute(self, context):
		scene = context.scene
	
		if(context.window_manager.Z2D_state == 0):
			pass;
			
		#z2dmath_roll_context_view( context, 15 );
		if context.region_data.view_perspective == 'CAMERA':	#If we are in camera view:
			oldtrans = context.scene.camera.matrix_basis.translation.copy();
			context.scene.camera.matrix_basis.identity();# = z2dmath_roll( context.scene.camera.matrix_basis, angle_deg );
			context.scene.camera.matrix_basis.translation = oldtrans;
			context.region_data.view_camera_offset = ( 0, 0 );	#Also important for cameras
			
		elif context.region_data.view_perspective == 'ORTHO':
		
			#Not sure what this means...
			oldtrans = context.region_data.view_matrix.copy();
			oldorigin = oldtrans.translation.copy();
			oldtrans.identity();# = z2dmath_roll( context.scene.camera.matrix_basis, angle_deg );
			oldtrans.translation = oldorigin;
			context.region_data.view_matrix = oldtrans
			context.region_data.update()
			
		elif context.region_data.view_perspective == 'PERSP':
			oldtrans = context.region_data.view_matrix.copy();
			oldorigin = oldtrans.translation.copy();
			oldtrans.identity();# = z2dmath_roll( context.scene.camera.matrix_basis, angle_deg );
			oldtrans.translation = oldorigin;
			context.region_data.view_matrix = oldtrans
			context.region_data.update()
			

		return {'FINISHED'};
	
class Z2D_OT_orbit_mirror_operator(Operator):
	bl_idname = "object.z2d_orbit_mirror_operator"
	bl_label = "Orbit Mirror"
	bl_description = "Orbit Mirror" 
	bl_options = {'REGISTER'}
	
	delta : bpy.props.FloatProperty()
	
	@classmethod
	def poll(cls, context):
		ob = context.object
		if ob and ob.type == 'GPENCIL':
			pass;
		return True;

	def execute(self, context):
		scene = context.scene
	
		if(context.window_manager.Z2D_state == 0):
			pass;
			
		#z2dmath_roll_context_view( context, 15 );
		if context.region_data.view_perspective == 'CAMERA':	#If we are in camera view:
		
			#This needs to respect view rotation...
		
			oldcamera = context.scene.camera.matrix_basis.copy();
			oldoffset = context.region_data.view_camera_offset;	#view_camera_zoom,  view_distance,   view_location (pivot location)
			
			mirroredcam = z2dmath_mirror( oldcamera );
			
			context.scene.camera.matrix_basis = mirroredcam
			context.region_data.view_camera_offset = ( -oldoffset[0], oldoffset[1] )
			
			
			#oldtrans = context.scene.camera.matrix_basis.translation.copy();
			#context.scene.camera.matrix_basis.identity();# = z2dmath_roll( context.scene.camera.matrix_basis, angle_deg );
			#context.scene.camera.matrix_basis.translation = oldtrans;
			
			#Interesting problems:
			#The camera object might be responsible for mirroring, but its POSITION needs to be mirrored somehow about the current view...
			#Difficult to fix.
			
			#ADJUST OFFSET of view from camera... Huh... "Mirror about" in camera view... (respecting rotation, scale and position)
			#view_camera_offset = context.region_data.view_camera_offset;
			
			
			#oldviewmatrix = context.region_data.view_matrix.copy();
			
			#newviewmatrix = z2dmath_mirror( context.scene.camera.matrix_basis );
			
			#newviewmatrix.translation = oldviewmatrix.translation;
			
			#context.region_data.view_matrix = newviewmatrix;
			
			#context.region_data.update()
			
		elif context.region_data.view_perspective == 'ORTHO':
		
			#Okay. Now that the CAMERA roll/mirror is perfect,
			#	Do we base THIS mirror relative to THAT direction/axis?
			#	That would make sense... but we can't "mirror" outside of the camera.
			#	Huh. Can rotate though? Why can't we mirror???
			#
			#
		
			#Not sure what this means...
			#oldtrans = context.region_data.view_matrix.copy();#ported_matrix_multiply( context.region_data.view_matrix, mathutils.Matrix( ( (-1.0,0.0,0.0,0.0),(0.0,1.0,0.0,0.0),(0.0,0.0,1.0,0.0), (0.0,0.0,0.0,1.0) ) ) )
			
			#oldtrans = ported_matrix_multiply( context.region_data.view_matrix, mathutils.Matrix( ( (-1.0,0.0,0.0,0.0),(0.0,1.0,0.0,0.0),(0.0,0.0,1.0,0.0), (0.0,0.0,0.0,1.0) ) ) )
			#oldtrans[0] *= -1.0;
			
			#oldtrans[0][0] *= -1.0;
			#oldtrans[0][1] *= -1.0;
			#oldtrans[0][2] *= -1.0;
			
			# window_matrix ???
			
			# perspective_matrix = window_matrix @ view_matrix
			
			#
			
			context.region_data.view_matrix = z2dmath_mirror( context.region_data.view_matrix );
			context.region_data.update()	#Huh...
			print( "yeah mirrorO!" );
			
		elif context.region_data.view_perspective == 'PERSP':
			#oldtrans = ported_matrix_multiply( context.region_data.view_matrix, mathutils.Matrix( ( (-1.0,0.0,0.0,0.0),(0.0,1.0,0.0,0.0),(0.0,0.0,1.0,0.0), (0.0,0.0,0.0,1.0) ) ) )
			#oldtrans = context.region_data.view_matrix.Scale( -1, 4, context.region_data.view_matrix.row[0] );

			context.region_data.view_matrix = z2dmath_mirror( context.region_data.view_matrix ); 
			context.region_data.update()
			
			print( "yeah mirrorP!" );

		return {'FINISHED'};