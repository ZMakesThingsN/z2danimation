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

def move_frames( frames_to_move=0, move_direction=1, context=None ):
	"""
		Move frames toward/away (move_direction) to current frame by frames_to_move
		
		move_frames( frames_to_move=1, move_direction=1, context=context );#Shift right, AWAY from  (flash's F5)
		move_frames( frames_to_move=-1, move_direction=1, context=context );#Shift left, AWAY from
	
		move_frames( frames_to_move=-1, move_direction=-1, context=context );#Shift left, INTO frame  (flash's shift+F5)
		move_frames( frames_to_move=1, move_direction=-1, context=context );#Shift right, INTO frame
	"""

	#Options:
	#"replace collisions" vs keep current keyframe
	#	If you shift frames into current keyframe, does it get replaced or deletes the other keyframe?
	#"stop ALL on collision" vs "stop each on collision"
	#	If a collision will occur when you shift keyframes, STOP shifting.
	#	Otherwise, it continues stacking the keyframes...
	#
	
	
	"""
	#https://docs.blender.org/api/current/bpy.types.UILayout.html#bpy.types.UILayout
	#https://docs.blender.org/api/current/bpy.context.html
	#bpy.context.selected_objects
	#bpy.context.objects_in_mode
	#
	#operator methodology? problem is, "locking" is respected... gross!
	#
	#override = bpy.context.copy()
	#override[ 'editable_gpencil_layers' ] = []
	#	bpy.context.editable_fcurves ??
	#	bpy.context.selected_editable_fcurves
	#	
	#
	#	bpy.ops.gpencil.blank_frame_add(all_layers=False)
	#
	#	bpy.ops.gpencil.draw(mode='DRAW', stroke=None, wait_for_input=True, disable_straight=False, disable_fill=False, guide_last_angle=0.0)
	#
	#	bpy.ops.gpencil.stroke_merge_by_distance(threshold=0.001, use_unselected=False)
	#
	#	bpy.ops.gpencil.stroke_trim()
	#
	#	bpy.ops.action.select_leftright(mode='CHECK', extend=False)
	#	bpy.ops.action.select_leftright(mode='LEFT', extend=False)
	#	bpy.ops.action.select_leftright(mode='RIGHT', extend=False)
	#
	#	bpy.ops.transform.translate( value = (1.0, 0.0, 0.0), snap=True, gpencil_strokes=False )
	#
	#gpenc.animation_data.frames = GPencilFrames -> .frame_number,  .
	#gpenc.layers = https://docs.blender.org/api/current/bpy.types.GreasePencilLayers.html#bpy.types.GreasePencilLayers
	
	#bpy.ops.gpencil.blank_frame_add() the heck.
	
		override = bpy.context.copy()
		
		override[ 'active_object' ] = gpenc
		override[ 'edit_object' ] = gpenc
		override[ 'object' ] = gpenc
		override[ 'gpencil' ] = gpenc.data
		override[ 'gpencil_data' ] = gpenc.data
		override[ 'gpencil_data_owner' ] = gpenc
		override[ 'visible_gpencil_layers' ] = [ layer ]
		override[ 'editable_gpencil_layers' ] = [ layer ]
		override[ 'active_gpencil_layer' ] = layer
		override[ 'active_gpencil_frame' ] = None
		
		bpy.ops.action.select_leftright( override, mode='RIGHT', extend=False )
			
		#gpenc.layers.frames = https://docs.blender.org/api/current/bpy.types.GPencilFrames.html#bpy.types.GPencilFrames
	
	#
	"""
		
	
	zeroframe = context.scene.frame_current
	
	gpencils = context.objects_in_mode
	
	if len( gpencils ) > 0:
	
		for gpenc in gpencils:
		
			#gpenc.layers
			if frames_to_move != 0:
			
				#layer_move_list = [];
				
				frame_relative_check = {};
				frame_relative_min = 0;
				frame_relative_max = 0;
				
				layer_index = 0;
				for layer in gpenc.data.layers:
							
						
					#remember if locked for editing.
					#Unlock layer, select NO keyframes.
					prevlock = layer.lock;
					layer.lock = False;
					
					#Make sure we have frames that are sorted. collections are NOT GUARANTEED to be in any order.
					frame_to_sorted = [];
					framei = 0;
					for frame in layer.frames:
						frame_to_sorted.append( ( frame.frame_number, framei ) )
						framei += 1;
						
					frame_to_sorted.sort( key=lambda x:x[0] );
					
					if move_direction > 0:	#Away from == safe in all cases
						
						if frames_to_move < 0:
						
							fsforwardi = 0; 
							fsforwardimax = len( frame_to_sorted )
							if fsforwardimax:
								while fsforwardi < fsforwardimax:
									fs = frame_to_sorted[ fsforwardi ];
									if fs[0] < zeroframe:
										
										#Shift frame in timeline??? how do we do this?
										layer.frames[ fs[1] ].frame_number += frames_to_move;
									else:
										break;
										
									fsforwardi += 1;
							
						elif frames_to_move > 0:
						
							#Find ALL FRAMES and MOVE THEM UP? uh...
							fsreversei = len( frame_to_sorted )
							if fsreversei:
								while fsreversei > 0:
									fsreversei -= 1;
									fs = frame_to_sorted[ fsreversei ];
									if fs[0] > zeroframe:
										
										#Shift frame in timeline??? how do we do this?
										layer.frames[ fs[1] ].frame_number += frames_to_move;
									else:
										break;
									
					elif move_direction < 0:	#Move into / vacuum == dangerous because of possible overrun. Difficult options here.
					
						#Collapsing is more... difficult.
						
						if frames_to_move < 0:
						
							#This is a LEFT SHIFT
							fsreversei = len( frame_to_sorted )
							if fsreversei:
								while fsreversei > 0:
									fsreversei -= 1;
									fs = frame_to_sorted[ fsreversei ];
									if fs[0] >= zeroframe:
										
										frame_relative_check[ zeroframe - fs[0] ] = 1;
										frame_relative_min = min( zeroframe - fs[0], frame_relative_min );
									else:
										break;
										
						elif frames_to_move > 0:
						
							#This is a RIGHT SHIFT
							fsforwardi = 0; 
							fsforwardimax = len( frame_to_sorted )
							if fsforwardimax:
								while fsforwardi < fsforwardimax:
									fs = frame_to_sorted[ fsforwardi ];
									if fs[0] <= zeroframe:
										
										frame_relative_check[ zeroframe - fs[0] ] = 1;
										#frame_relative_min = min( zeroframe - fs[0], frame_relative_min );
										frame_relative_max = max( 1 + zeroframe - fs[0], frame_relative_max );
									else:
										break;
									fsforwardi += 1;
						
						pass;
					
					else:
						
						pass;
					
							
										
					layer.lock = prevlock;
					layer_index += 1;
				
				if move_direction < 0:
					if frames_to_move < 0:
						
						#This is a LEFT SHIFT so relative checks will be 0, -1, -2, -3, because left shift.
						nearest_space = 0;
						while not nearest_space in frame_relative_check:
							nearest_space -= 1;
							if nearest_space < frame_relative_min:
								break;
							
						nearest_move = -1;
						while not nearest_move in frame_relative_check:
							nearest_move -= 1;
							if nearest_move < frame_relative_min:
								break;
							
						possible_move = frames_to_move;
						
						if nearest_space == 0: #occupied 0 space
							if nearest_move+1 > possible_move:
								possible_move = nearest_move+1;
						else:
							if nearest_move > possible_move:
								possible_move = nearest_move;
						
						if possible_move != 0:
							for layer in gpenc.data.layers:
								for frame in layer.frames:
									if frame.frame_number > zeroframe:
										frame.frame_number += possible_move;
						
					elif frames_to_move > 0:
					
						#This is a RIGHT SHIFT
						nearest_space = 0;
						while not nearest_space in frame_relative_check:
							nearest_space += 1;
							if nearest_space > frame_relative_max:
								break;
							
						nearest_move = 1;
						while not nearest_move in frame_relative_check:
							nearest_move += 1;
							if nearest_move > frame_relative_max:
								break;
						
						possible_move = frames_to_move;
						
						if nearest_space == 0: #occupied 0 space
							if nearest_move-1 < possible_move:  #we can't shift right INTO a existing keyframe.
								possible_move = nearest_move-1;
						else:
							if nearest_move < possible_move:
								possible_move = nearest_move;
								
						print( possible_move, nearest_space, nearest_move );
						
						if possible_move != 0:
							for layer in gpenc.data.layers:
								for frame in layer.frames:
									if frame.frame_number < zeroframe:
										frame.frame_number += possible_move;
					
						pass;
					


class Z2D_OT_frame_add_shift_add(Operator):
	bl_idname = "object.z2d_frame_add_shift_add"
	bl_label = "Frame add"
	bl_description = "Frame add" 
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
		
		move_frames( frames_to_move=1, move_direction=1, context=context );#Shift right, AWAY from
		#move_frames( frames_to_move=-1, move_direction=1, context=context );#Shift left, AWAY from
		
		return {'FINISHED'};
	
class Z2D_OT_frame_rem_shift_rem(Operator):
	bl_idname = "object.z2d_frame_rem_shift_rem"
	bl_label = "Frame remove"
	bl_description = "Frame remove" 
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
			
		move_frames( frames_to_move=-1, move_direction=-1, context=context );#Shift left, INTO frame
		#move_frames( frames_to_move=1, move_direction=-1, context=context );#Shift right, INTO frame
		
		return {'FINISHED'};
