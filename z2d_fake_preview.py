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
import time
import math
from bisect import bisect_left

from bpy.types import Operator
from bpy.types import Panel

#THE OBJECTIVE:
#
#	Have a moveable, resizeable PANEL that is OVER EVERYTHING
#	It must display its OWN grease pencil timeline animation
#	With options to adjust start/end frame to loop on
#	and playback speed factor ( from scene FPS of course )
#	It does NOT need to be synchronized to real time (it can pre-cache via threads and update as needed?)
#		A separate thread can prepare data for the GPU copying,
#		then when our tick happens, we can copy some over (time based?)
#		until exhausted
#	We want to be able to see a LIVE PREVIEW of our current animation at ALL TIMES,
#		so that we can make adjustments AS the animation is playing
#		This is VERY BENEFICIAL and massively reduces flipping around the timeline.
#		(MakerL did this extremely well)
#		Plus, being able to position the preview anywhere lets us compare and adjust.
#
#

#
#Okay, gizmo's are WAY not OK for this.
#Are we back to a "object type" that IS a grease pencil preview that constantly listens to context???
#Probably the best option since updating it is fairly simple (reuse grease pencil data blocks?)
#HM.
#	property driven?
#	Maybe add a gizmo to move it around easily? Hm...
#
#
#

"""
def blookup_object_get_namepath( ob ):

	rootcoll = bpy.context.scene.collection;
	
	#Walk tree to ROOT collection:
	
	#Warning: ob.users_collection is ALL collections, so  thats insane slow and NOT scene specific)
	#
	allob = rootcoll.all_objects
	allidx = allob.find( ob );
	if allidx != -1:
	
		#okay fine.
		layer = rootcoll;
		for coll in layer.children:
			if 
		
	
	return None;
	#all_collections = ob.users_collection;
	#
	


def blookup_object_by_namepath( namepath ):
	#
	#	namepath is a list of tuples:
	#		( type, string )
	#	Where
	#		type == 0 collection (find collection in collection)
	#		type == 1 object (find object in collection)
	#		type == 2 all_objects (find object in ALL sub collections. used when parenting issues exist)
	#		type == 4 object children (find object in object children) NO this is EXTREMELY expensive so dont do it.
	
	rootcoll = bpy.context.scene.collection;
	
	R = None;
	layer = rootcoll;
	
	for elm in namepath:
		names = [];
		name_idx = -1;
		
		if elm[0] == 0:
			#layer.children ?	#find by property (name)
			names = [ ob.name for ob in layer.children ]
			name_idx = names.find( elm[1] );
			if name_idx >= 0:
				layer = layer.children[ name_idx ]
				
		elif elm[0] == 1:
			#layer.objects ?		#find by property (name)
			names = [ ob.name for ob in layer.objects ]
			name_idx = names.find( elm[1] );
			if name_idx >= 0:
				layer = layer.objects[ name_idx ]
				#layer = target;
				
		elif elm[0] == 2:
			#layer.objects ?		#find by property (name)
			names = [ ob.name for ob in layer.all_objects ]
			name_idx = names.find( elm[1] );
			if name_idx >= 0:
				layer = layer.objects[ name_idx ]
				#return target;
				
		#elif elm[0] == 4:
			#EXTREMELY EXPENSIVE
			#	So dont do this.
			
			
		
	return None;
"""

z2d_fakepreview_first_time = True;
z2d_fakepreview_prev_time = 0
z2d_fakepreview_fake_frame = 0
z2d_preview_links = [];

def injectFloatProp( target, pname, userfmin=0.0, userfmax = 1.0 ):
	target[ pname ] = 0.0
	
	from rna_prop_ui import (
		rna_idprop_ui_prop_get,
		rna_idprop_ui_prop_clear,
		rna_idprop_ui_prop_update,
		rna_idprop_ui_prop_default_set,
		rna_idprop_value_item_type,
	)
	
	#id_data
	#prop_type, is_array = rna_idprop_value_item_type( target[ pname ] )
	prop_ui = rna_idprop_ui_prop_get( target, pname )
	prop_ui["min"] = 0.0;
	prop_ui["max"] = 1.0;
	#prop_ui["subtype"] = 'COLOR';#('COLOR', "Linear Color", "Color in the linear space");
	rna_idprop_ui_prop_default_set( target, pname, target[ pname ] )
	

def injectColorProp( target, pname ):
	target[ pname ] = [ 0.0, 0.0, 0.0 ];
	
	from rna_prop_ui import (
		rna_idprop_ui_prop_get,
		rna_idprop_ui_prop_clear,
		rna_idprop_ui_prop_update,
		rna_idprop_ui_prop_default_set,
		rna_idprop_value_item_type,
	)
	
	#id_data
	#prop_type, is_array = rna_idprop_value_item_type( target[ pname ] )
	prop_ui = rna_idprop_ui_prop_get( target, pname )
	prop_ui["min"] = 0.0;
	prop_ui["max"] = 1.0;
	prop_ui["subtype"] = 'COLOR';#('COLOR', "Linear Color", "Color in the linear space");
	rna_idprop_ui_prop_default_set( target, pname, target[ pname ] )
	
	#target[ pname ].typecode = 'f'
	
	#??? other crap is inferred like min, max, subtype???
	#print( dir( target[ pname ] ) );
	#print( dir( target[ pname ][0] ) );
	
	#WM_OT_properties_edit -> wm.py 
	#	rna_vector_subtype_items 
	#	'COLOR'
	#call operator WM_OT_properties_edit on this target;
	#
	#MAYBE even call THIS operator:
	#	WM_OT_properties_add
	#		data_path
	#		
    #	rna_idprop_ui_create(item, prop, default=1.0)
	#
	#
    #self._init_subtype(prop_type, is_array, subtype)
	
	#Hoooowww to access and add in subtype???
	#target[ pname ].subtype = ('COLOR', "Linear Color", "Color in the linear space");
	
	""" 
	"Only ints floats and dicts are allowed in ID property arrays"
	target[ pname ] = bpy.props.FloatVectorProperty(
		name = pname,
		subtype = "COLOR",	#COLOR_GAMMA
		size = 3,
		min = 0.0,
		max = 1.0,
		default = (0.0,0.0,0.0)
	)
	""" 
	

@bpy.app.handlers.persistent
def z2d_fakepreview_timer_update():

	global z2d_preview_links;
	global z2d_fakepreview_first_time;
	global z2d_fakepreview_prev_time;
	global z2d_fakepreview_fake_frame;
	
	usefps = bpy.context.scene.render.fps
	
	not_during_anim = False;
	if bpy.context.screen:
		not_during_anim = bpy.context.screen.is_animation_playing;
		
	if ( len( z2d_preview_links ) > 0 ) and ( not not_during_anim ) :
	
		nowtime = time.clock();

		if z2d_fakepreview_first_time:
			z2d_fakepreview_first_time = False;
			z2d_fakepreview_prev_time = nowtime
			z2d_fakepreview_fake_frame = 0
		
		deltatime = (nowtime - z2d_fakepreview_prev_time);#.total_seconds()
		z2d_fakepreview_prev_time = nowtime;
		
		if deltatime <= 0:
			return 0.0;
			
		#Establish links ONCE, then ignore them...
		#unlike Godot, not sure about access paths to this data...
		validlinks = [];
		
		for link in z2d_preview_links:
		
			#Okay, so, this is a HUGE problem. If you mess with collections or names, this "breaks"
			dst = bpy.data.grease_pencils[ bpy.data.grease_pencils.find( link[0] ) ];
			src = bpy.data.grease_pencils[ bpy.data.grease_pencils.find( link[1] ) ];
			
			#Check if objects are STILL VALID
			valid = False;
			if dst:
				if src:
					
					valid = True;
			
			if valid:
				
				validlinks.append( link );
				
			else:
				print( "NOTE: Link invalidated: ", dst, src );
				
		z2d_preview_links = validlinks;
		
		for link in validlinks:
				
			dst = bpy.data.grease_pencils[ bpy.data.grease_pencils.find( link[0] ) ];
			src = bpy.data.grease_pencils[ bpy.data.grease_pencils.find( link[1] ) ];
			
			#Okay do the thing.
			if dst.get("z2d_preview_frame"):
				dst["z2d_preview_frame"] = dst["z2d_preview_frame"] + usefps*deltatime*dst["z2d_preview_speed"];
			else:
				dst["z2d_preview_frame_min"] = 1;
				dst["z2d_preview_frame_max"] = 250;
				dst["z2d_preview_speed"] = 1.0;
				dst["z2d_preview_frame"] = dst["z2d_preview_frame_min"];
				dst["z2d_preview_frameblock"] = [];
				injectColorProp( dst, "z2d_preview_colorize" );
				injectFloatProp( dst, "z2d_preview_colorize_factor", 0.0, 1.0 );
				
				#bpy.props.FloatVectorProperty
				dst["z2d_preview_thickness"] =  0;
				
			
			#Fix this later
			while dst["z2d_preview_frame"] >= dst["z2d_preview_frame_max"]:
			
				fdelta = ( dst["z2d_preview_frame_max"] - dst["z2d_preview_frame_min"] );
				if fdelta <= 0:
					dst["z2d_preview_frame"] = 0;
					break;
				
				dst["z2d_preview_frame"] = dst["z2d_preview_frame"] - fdelta;
				
				#dst["z2d_preview_frame"] -= fdelta * math.floor( 
				#	(dst["z2d_preview_frame"] - dst["z2d_preview_frame_min"])
				#	/ (dst["z2d_preview_frame_max"] - dst["z2d_preview_frame_min"])
				#);
			
			#Now we have the CURRENT FRAME to lookup:
			currframe = dst["z2d_preview_frame"];
			
			#Note this operation can be GREATLY improved in terms of performance.
			#We can remember WHICH frame in a layer we were looking at, etc...
			#And we can structure this differently to MINIMIZE editing needed.
			
			
			#Do we NEED to update this?
			#Check that first...
			
			layers_to_addupdate = {};#[];
			layer_id = 0;
			for layer in src.layers:
				if len( layer.frames ) > 0:#Must have frames to copy
					if not layer.hide:	#Must be visible
						#Is it MARKED for dont preview?
						
						#find CURRENT ACTIVE FRAME in src, and COPY IT OVER
						f_index = bisect_left( [ f.frame_number for f in layer.frames ], currframe )
						
						if False:
							f_index = 0;
							for f in layer.frames:
								if f.frame_number >= currframe:
									break;
								f_index += 1;
						f_index -= 1;
						if f_index < 0:
							f_index = 0;
							
						#we can speed this up using MEMORY potentially...
						#dst["z2d_preview_frameblock"].append( (layer.info, f_index) );
						
						if f_index >= len( layer.frames ):
							f_index = len(layer.frames) - 1;
						if f_index >= 0:
							layers_to_addupdate[ str(layer_id) ] = { 
								"name":layer.info
								,"id":layer_id
								,"frame":f_index 
							}
							#layers_to_addupdate.append( ( layer.info, layer_id, f_index ) );
				layer_id += 1;
			#Difference layers, adding and removing them is REALLY ANNOYING.
			#AKA, removing and adding keyframes is easier?
			
			#Make SURE we have something that changed ( dont update if we dont have to!!! )
			
			#Only remove changed (removed) layers, only remove and update changed frames (remove all frames in changed frame)
			
			needs_full_update = True;
			
			if dst.get( "z2d_lastcmpl" ):
				lastcmpl = dst["z2d_lastcmpl"];
				
				#?
				#layers_to_addupdate = [ ( layername, layerindex, frameindexinlayer )... ]
				#
				needs_full_update = False;
				
				for idxkey in lastcmpl:
					if idxkey in layers_to_addupdate:
					
						#Did ANYTHING change ?
						old = lastcmpl[ idxkey ]
						curr = layers_to_addupdate[ idxkey ]
						
						for okey in old:
							if curr[okey] == old[okey]:
								pass;
							else:
								needs_full_update = True;
								break;
						if needs_full_update:
							break;
					else:
						needs_full_update = True;
						break;
				
			if needs_full_update:
				
				for oldlayer in dst.layers:
					dst.layers.remove( oldlayer );
					
				for addme_index in layers_to_addupdate:
					addme = layers_to_addupdate[ addme_index ]
					
					if addme['id'] < len( src.layers ):
						
						layer = src.layers[ addme['id'] ];
						
						addframe = addme['frame']
						
						if addframe < len( layer.frames ):
							
							newlayer = dst.layers.new( addme['name'], set_active=False );
							
							newlayer.frames.copy( layer.frames[ addframe ] );
							
							#Hack to force frame to appear on 0 (may not work if frames go negative
							newlayer.frames[0].frame_number = 0;
							
							#newlayer.frames[0].frame_number = ?
							
							newlayer.opacity = layer.opacity;
							#newlayer.opacity = layer.opacity
							
							#newlayer.active_frame = newlayer.frames[0];
							
							#newlayer = dst.layers.new( layer.info, set_active=False );
					
							#newlayer.frames.copy( layer.frames[ f_index ] );
							
							#Apply adjusted Tint Color + Tint Factor, Stroke Thickness
							if dst["z2d_preview_colorize_factor"] > 0:
								#dst["z2d_preview_colorize"] = 
								newlayer.tint_color = dst["z2d_preview_colorize"]
								newlayer.tint_factor = dst["z2d_preview_colorize_factor"]
							#else:
							#	newlayer.tint_factor = 0.0;
								
							if dst["z2d_preview_thickness"] != 0:
								newlayer.line_change = dst["z2d_preview_thickness"]
							#else:
							#	newlayer.line_change = 0;
							
				#print( currframe, usefps, deltatime, layers_to_addupdate );
		
				dst.update_tag(); #trigger redraw???
				
				dst["z2d_lastcmpl"] = layers_to_addupdate;
			
	else:
		z2d_fakepreview_first_time = True;
	
	return 1/max(1,usefps);#0.0
	
#Find objects that have a "z2d_preview_grease_pencil" property on them...
#we do this slowly. (hm... not sure on that one...)
#

class Z2D_OT_fake_preview_operator(Operator):
	bl_idname = "object.z2d_fake_preview_operator"
	bl_label = "Z2DFakePreview"
	bl_description = "Z2DFakePreview" 
	bl_options = {'REGISTER'}
	
	@classmethod
	def poll(cls, context):
		return True;

	def execute(self, context):
		self.do_check( context );
		return {'FINISHED'}
		
	def invoke(self, context, event):
		self.do_check( context );
		return {'FINISHED'};

	def do_check(self,context):
		
		#Search ENTIRE SCENE and ALL COLLECTIONS for any Grease Pencil object that has:
		#
		identified_links = [];
		for child in context.scene.collection.all_objects:
			if child.type == 'GPENCIL':
				#??? are properties on the DATA or the OBJECT?
				#In this case, the DATA must be UNIQUE, not the OBJECT.
				objhasit = child.get('z2d_preview_grease_pencil', None);
				if objhasit:
					print( "#WARNING: z2d_preview_grease_pencil must be on the GPencil data, not the object" );
				
				if child.data:
					hasit = child.data.get('z2d_preview_grease_pencil', None)
					if hasit: #custom property
						#Look up the object name & data it is "previewing"
						
						targetcandidates = [];
						for child2 in context.scene.collection.all_objects:
							if child2.name == hasit:
								targetcandidates.append( child2 );
						
						if len( targetcandidates ) == 1:
						
							identified_links.append( ( child, targetcandidates[0], hasit ) );
						elif len( targetcandidates ) > 0:
							print ( "WARNING: more than one object in scene shares target name for preview object, fix this:" );
							print( targetcandidates );
						else:
							print( "WARNING: preview object could not find grease pencil object with name "+hasit );
							
						
		#Now that we have all POSSIBLE preview objects,
		#Link them to the timer update?
		if len( identified_links ) <= 0:
			
			print( "You must create a grease pencil object with a custom property on the grease pencil data of z2d_preview_grease_pencil." );
			#print( "Since we did not find any, we are doing this for you." );
			
			
			#Do we have a current grease pencil selected?
			
			targetob = None;
			if bpy.context.active_object:
				if bpy.context.active_object.type == 'GPENCIL':
					targetob = bpy.context.active_object
			
			if targetob:
			
				print( "Auto creating a PREVIEW grease pencil object" )
				
				#great? use its name.
				hasit = targetob.name
				
				gpdata = bpy.data.grease_pencils.new( "GPencil_PREVIEW" );
				gpdata['z2d_preview_grease_pencil'] = hasit;
				
				gpobj = bpy.data.objects.new( "PREVIEW", gpdata );
				#gpobj.link( gpdata );
				
				#Pick same collection as targetob ? huh... problematic
				
				context.scene.collection.objects.link( gpobj );	#huh
				
				identified_links.append( ( gpobj, targetob, hasit ) );
				
			else:
				print( "#ERROR No gpencil object selected to preview from. Select a gpencil object then try and preview it." );
			
		#Toggles it... huh. Not sure if this is GOOD or BAD.
		global z2d_preview_links;
		
		if len( z2d_preview_links ) > 0:
			z2d_preview_links = [];
		else:
				
			for linked in identified_links:
			
				obj = linked[0];
				target = linked[1];
				
				print( "Linking: ", obj, target, (obj.data.name, target.data.name)  );
				
				#obj.data.name
				#target.data.name
				
				#bpy.data.grease_pencils.find( 
				#bpy.data.grease_pencils[ ? ]
				
				#bpy.data.
				
				#Setup properties as DEFAULT here:
				if not obj.data.get( "z2d_preview_frame_min" ):
					obj.data["z2d_preview_frame_min"] = 1;
				if not obj.data.get( "z2d_preview_frame_max" ):
					obj.data["z2d_preview_frame_max"] = 250;
				if not obj.data.get( "z2d_preview_speed" ):
					obj.data["z2d_preview_speed"] = 1.0;
				
				obj.data["z2d_preview_frame"] = obj.data["z2d_preview_frame_min"];
				obj.data["z2d_preview_frameblock"] = [];
				
				if not obj.data.get( "z2d_preview_colorize" ):
					injectColorProp( obj.data, "z2d_preview_colorize" );
				if not obj.data.get( "z2d_preview_colorize_factor" ):
					injectFloatProp( obj.data, "z2d_preview_colorize_factor", 0.0, 1.0 );
				if not obj.data.get( "z2d_preview_thickness" ):
					obj.data["z2d_preview_thickness"] =  0;
				
				#z2d_preview_links.append( (obj, target) );
				z2d_preview_links.append( (obj.data.name, target.data.name) );
	
class Z2D_PT_fakepreview_context_z2d(Panel):
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'
	bl_context = "render"
	bl_label = "Z2D Preview"
	
	@classmethod
	def poll(cls, context):
		return context.scene != None

	def draw(self, context):
		layout = self.layout
		
		row = layout.row();
		row.operator('object.z2d_fake_preview_operator', text="Preview", icon="TRACKING_REFINE_BACKWARDS")
		