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

import math
import time
import importlib;
import array
import collections;
import random;

#New ideas:
#	Convert existing stuff INTO a graph representation.
#	(IE, merge and join strokes appropriately, create nodes SUCH THAT):
#		No line segments OR edges cross.
#		Graph is node, + strokedge (strokeedge is the actual PATH between nodes)
#	"Best match" should always work from OUTSIDE CONTOUR in.
#	Therefore, the OUTER contour AND its relative node positions need to match up as best as possible.
#	Likely to have MULTIPLE islands (especially with bullseye scenario)
#	Since paths CANNOT intersect, the simplified voronoi diagram of the nodes in use can be used to match.
#	and follow the outer contour of the valid voronoi triangles to match things.
#	This should let us match the outer contour and node points....
#
#

import numpy;		#THIS IS TRUE POWER!!!
#Okay numpy ACTUALLY has linear solvers built in (libblas)
#	Can we make a linear optomization problem via numpy?
#	AKA, Laplacian Deformation modifyer, and then just add points until the error is less than X?
#	(adding points to "nodes" or "high density areas" should be good.)
#	Its also interesting to think of a drawing as a signed distance field, where there is an interior (do a convex hull)
#	And each "step" increments the distance inward. this would yield a heightfield we want to distort...
#	we definitely want to use that "remove inherant transform" first thing, to attempt to reduce the delta's needed to be computed.
#		IE remove the translation, rotation and scale applied between two frames FIRST (known algorithm you made)
#		THEN make a distortion to minimize "error" ... hm...
#			(convert resultant unity data into nodes & strokes, nodes being near intersecting strokes.)
#			Add a deformable point for each node, and points along each stroke (hm... at x degree turns???)
#			Create laplacian weighting for vertices based on this new mesh
#				Which is, differential 2D geometry (distance from neighbor's weighted average)
#			Then solve so we can minimize the error to OTHER same kind of structure...
#				Error being sum squared error by distance between points?
#Can we make a animated representaion?
#	IE, add a point correspondence
#		Each point has a position, rotation and scale (x,y,rtheta,sx,sy)
#		this allows transformation interpolation between frames that CAN be adjusted once we have the linear.
#			(IE, each correspondence can be animated separately, along any path. Keyframe generation would not change.
#		Then we can compute binding weights as needed but...
#Okay...
#	each point should have a weighted sum of transformations applied to it.
#	each point's WEIGHTS should be the same throughout the transformation (assigned once)
#	the TRANSFORMS is what we are solving for. (defaults to identity)
#	p_animated[i] = sum( T[j] * w[i][j] * p_src[i] = ... , j )
#	least squares: min( sum( (|p_animated[i] - p_dst[i]|) ^2 ) )
#	Solve for all T[j]'s such that the error is minimized.
#	Well, since each point has a weighted sum of transforms, we can abuse this;
#	And pull out a different loop for each transform that goes through all the points it deforms...
#	error = sum( | sum( w[i][j] * T[j], j ) * p_src[i] - p_dst[i] |^2, i )
#	T[j] = [ dx, dy, rtheta, scale ]
#	
#	Many less unknowns than knowns (ideally)
#	If we just consider translation only (for now)
#	then its a linear weighted sum of vectors.
#	Let's call that vector list DELTAS[]
#	So now,
#		p_approx_dst[i] = sum( DELTA[j] * w[i][j], j ) + p_src[i]
#	Meaning,
#		error = sum( || p_approx_dst[i] - dst[i] ||, i )
#	Now how can we form this as a minimization problem.
#	each vertex has some list of weights.
#	If we FORCE each vertex to have the SAME LENGTH aka 1 weight for each element in DELTA...
#	Then we can reframe the equation in terms of the unknowns...
#	But the matrix gets crazy:  A * x = B
#		somematrix * soln = B
#	
#	[ 0*v[0].x + 0*v[1].x + 0.25*v[2].x + ...  ]*[ DELTA[i].x ] = [ dst[i].x ]
#	[   ]*[ DELTA[i].y ] = [ src[i].y ]
#
#	sum( weight[i][j]*vertex[i] + weight[i][j]*DELTA[j], j ) = dst[i]
#	
#	sum( weight[i][j]*vertex[i].x + weight[i][j]*DELTA[j].x, j ) = dst[i].x
#	sum( weight[i][j]*vertex[i].y + weight[i][j]*DELTA[j].y, j ) = dst[i].y
#
#	sum( weight[i][j]*DELTA[j].x, j ) = dst[i].x - sum( weight[i][j]*vertex[i].x, j )
#	sum( weight[i][j]*DELTA[j].y, j ) = dst[i].y - sum( weight[i][j]*vertex[i].y, j )
#
#	[ weight[0][0], weight[0][1], ... weight[0][j] ] * [ DELTA[0].x ] = dst[i].x - sum( weight[i][j]*vertex[i].x, j )
#	[ weight[0][0], weight[0][1], ... weight[0][j] ] * [ DELTA[0].y ] = dst[i].y - sum( weight[i][j]*vertex[i].y, j )
#		...
#
#	A = weight based on each vertex in each row.
#	x = solve for minimizing the error
#	b = vector of differences from destination to source.
#
#	This requires correspondences between vertices in source and destination to have been established!
#	And the error isnt established so that wont work...
#	IE the correspondence can change based on proximity.
#	
#	
#
#		... per each transform
#
#Okay. representations.
#	Convert drawing into:
#		paths + nodes
#	paths are formed from points n-close in distance AND tangency.
#	Nodes are where more than 1 path intersect or any sharp feature.
#	IE a path is only a smooth, nonintersecting path.
#	
#	Once you have this representation for BOTH images...
#	You can instantly walk the exterior path to produce a exterior path.
#	Blend this path using any of the "blend a closed loop" ideas you want.
#		<-> Multiple closed loops in a drawing can exist, so treat EACH separately.
#	
#	Typical process:
#	"0. Pre Process data 1. Defining correspondences 2. Computing a triangulation 3. Create the morph."
#

#
#Questions:
#
#	There IS a open source "flash" editor that DOES NOT HAVE shape tweens;
#	So. Can we fill that gap?
#	If we are GUARANTEED to have non-intersecting primitives;
#	And each primitive is a bezier curve made of only quadratic or linear segments;
#	Hm...
#
#	But in blender, we just have line segment paths;
#	And we can change their shape anytime....
#	so the notion of "path" becomes important...
#	Treat it like how fills in blender work?
#		AKA, blender "Rasterizes" first, 
#		Then we can know... if a pixel is "connected" or not to a path???
#		IE pixel belongs to N-strokes...
#	
#	There is a "convex hull" we can start with...
#	But... what about a "signed distance field" type hull?
#	IE, there are ALWAYS lines about...
#	But if we can represent those lines by a collection of signed distance field primitives...
#	
#	"features" that match == step 1....
#	IE we can REMOVE large sections of the drawing if we can match features.
#	What is a feature?
#
#	0. User defined feature points
#	1. any closed polygon and it's nearby "whiskers"
#	2. any group of "whiskers"
#	
#	prepare_frame():
#		coincidence_fills():
#			Link up fills with their original owner verts? fills are... very problematic in their representation :C
#			If we change STROKES the correspinding fill points should ALSO change. thats the idea.
#		simplify_strokes( type? cutdistnace? ? ):
#			Reduce complexity of strokes (stroke simplify)
#		cut_intersections():
#			Intersect all lines for all strokes first (CANNOT have intersection...)
#			CREATE new strokes that need to be intersected;
#			DELETE original strokes that had intersections
#		overlay_fills():
#			Overlap/cut ALL fills (FILLS cannot be stacked! must cut/merge...)
#			given all fill polygons; OVERLAY them:
#				Create voronoi triangulation for ALL fill polygon edges.
#					(Remove all interior verts based on top to bottom fills? hm... too complex?)
#				Assign color of triangles from TOP to bottom (via centroid)
#				Convert triangles into "fill" strokes (must "cut out" traversible fill regions)
#					start on edge triangle, add triangles to "polygon" that share same color.
#					trace outer contour (warning, possible to have "holes)
#					outer contour MAY have holes, so we must "cut those out" by breaking apart the polygon (vert cut bias)
#				add all new fill polygons
#				delete old fill polygons
#		2. Voronoi all points, keeping track of REAL edges used.
#			Note this REQUIRES no intersecting edges so, filter those out first...
#		3. Create NEW keyframe on NEW layer(s) ("FINAL_STROKE" and "FINAL_FILL") from data.
#		
#Question:
#	Can we just DIRECTLY export ink + fill like the clip.json data?
#	After all, its just triangles.
#	The only problem I can think of is the fills may be overlaid, so they have to be "flattened"
#		(and intersecting the fills may be LESS efficient)
#	Also simplification is needed!!!
#	BLEH
#	If we can have ANY representation; thats better than NONE.
#	Once we have the "polygons" + "strokes" well, "visually" identifying features becomes important.
#	HM....
#
#
#More questions:
#
#	"blend between two frames"
#
#		"Raster" first
#		"Features" next ( based on raster... has a maximum size of course (feature max radius) )
#		"edges" between features (features cut HOLES in drawing...)
#		both edges and features can collapse or grow out of the drawing. (relative to OTHER features in teh voronoi mesh...
#		
#		Simple case would be a:
#			Square, square translated and rotated
#			Should match accurately (But may have rotation wrong)
#		Next cases:
#			Square, TWO squares
#				Should create square from a single point.
#		Next case:
#			Bullseye, single square
#				Should shrink circles to points or to grow to fit outer circle?
#		ADd shape hints to each case to adjust it.
#
#
#		-> Convert frames into dframe representation
#			Start with "best match linear transform"
#				<-> use as DEFAULT linear transform, put drawing into -1..1 2D space FIRST.
#			Then, visual_salience_tree
#				"Render" features as line segments 1 pixel thick
#				TREE rendering from low resolution (16x16) and down to a max (512x512 is a lot)
#				Then a visualization of this "feature tree"
#					Should reveal areas of sparse detail and areas of complex detail...
#				Can we "match" the trees?
#			"feature" node... start on cusps and nodes, increase circle size until feature changes.
#			
#			the tree should now contain a representation of the drawing that SHOULD allow us to make polygons
#			(colors important? yes, style matching IS important but SHAPE is more important...)
#
#			cellular matching is a viable thing... (use feature tree "complex feature nodes" and try and match the two as best as possible via physics?)
#			
#			
#		-> Match features in dframes (applying user matches first)
#			User selected match is first and most absolute.
#			auto matching fills in remainder....
#
#		-> Compute linear deltas between dframe's
#			Note "tension" on each point...
#		
#

	

import bpy
import bpy_extras; #important
import mathutils
import bgl

from bpy.types import Operator

import tri_delaunay;
importlib.reload( tri_delaunay )

import gpu
import gpu_extras
#from gpu_extras.batch import batch_for_shader


#Some extra lifting

class icp_2d_proximitymetric:
	"""
		This is a specialized case. We dont want the NEAREST point, but rather that AVERAGE distance to nearby pointswithin a maximum radius.
		This provides a FAR smoother surface at the cost of absolute accuracy for convergence.
		However, if you want you can still use this structure to more efficiently get nearest points vs naaive solution.
		Will be less efficient in areas with dense clusters, this expects a fairly spread out distribution of points
		This will fallback to the "naaive" algorithm in the case of no points or cells found.
			IE you can improve this by adding other search algorithms in for that case (which is common!)
	"""
	def __init__( self, V, maxradius=None, radiuscuts=40 ):
		self.xmin = min( [ v[0] for v in V ] );
		self.ymin = min( [ v[1] for v in V ] );
		self.xmax = max( [ v[0] for v in V ] );
		self.ymax = max( [ v[1] for v in V ] );
		self.V = [ ( v[0], v[1] ) for v in V ];
		self.r = 1.0;
		#maxradius => becomes a function of the maximum bounds. usually 1/20th ?
		if maxradius == None:
			self.r = max( ( self.xmax - self.xmin ), ( self.ymax - self.ymin ) )/(1.0*radiuscuts)
		else:
			self.r = maxradius;
			
		self.grid = {};
		self.recell();
			
	def recell( self ):
		
		self.grid = {};
		for vi in range( len( self.V ) ):
			
			v = self.V[ vi ]
			pxmin = v[0] - self.r;
			pxmax = v[0] + self.r;
			pymin = v[1] - self.r;
			pymax = v[1] + self.r;
			
			cxmin = math.floor(pxmin/self.r);
			cxmax = math.floor(pxmax/self.r);
			cymin = math.floor(pymin/self.r);
			cymax = math.floor(pymax/self.r);
			
			cy = cymin;
			while cy <= cymax:
				cx = cxmin;
				while cx <= cxmax:
					key = ( cx, cy );
					if key in self.grid:
						self.grid[ key ].append( vi );
					else:
						self.grid[ key ] = [ vi ]
					cx += 1;
				cy += 1;
		
	def gradient( self, p, err=(0,0), zeroout=0 ):
		#"length" of gradient normalized vector should MEAN something.
		#specifically, magnitude == DISTANCE to NEAREST point.
		#direction == GRAIDENT ESTIM
		
		#1. determine CELL p is in
		key = ( math.floor( p[0]/self.r ), math.floor( p[1]/self.r ) )
		if key in self.grid:
			vilist = self.grid[ key ];
			if len( vilist ) > 0:
				vpoints = [ self.V[vi] for vi in vilist ];
				
				#Problem! Hm...
				vd = [ ( (v[0]-p[0]),(v[1]-p[1]) )  for v in vpoints ];
				vmd = [ ( v[0], v[1], (v[0]*v[0] + v[1]*v[1]) ) for v in vd ];
				lenbefore = len( vmd )
				vmd = [ v for v in filter( lambda x:x[2] > zeroout, vmd ) ]; #length CANNOT be zero; but if it is, we quit?
				lenafter = len( vmd )
				if lenafter == lenbefore:
					if len( vmd ) > 0:
						#weighted sum of angles?
						
						vmd = [ ( v[0], v[1], math.sqrt( v[2] ) ) for v in vmd ];
						vmd = [ ( v[0], v[1], v[2], ( self.r - v[2] )/self.r ) for v in vmd ]; #?? unsure about this.
						
						vmd = [ v for v in filter( lambda x:x[3] >= 0, vmd ) ]; #RADIUS filter.
						if len( vmd ) > 0:

							vmagsum = sum( [ v[2] for v in vmd ] )
							vweightsum = sum( [ v[3] for v in vmd ] )
							#vweightsum = sum( [ 1 for v in vmd ] )
							if vweightsum > 0:

								gsumx = sum( [ v[3]*v[0]/v[2] for v in vmd ] )*( 1/vweightsum );#len(vmd) )
								gsumy = sum( [ v[3]*v[1]/v[2] for v in vmd ] )*( 1/vweightsum );#len(vmd) )
								
								gsumm = math.sqrt( gsumx*gsumx + gsumy*gsumy )
								if gsumm > 0:
									gsumx /= gsumm;
									gsumy /= gsumm;
									
								#"length" of gradient normalized vector should MEAN something.
								#specifically, magnitude == DISTANCE to NEAREST point.
								#direction == GRAIDENT ESTIMATE
								mindistance = min( [ v[2] for v in vmd ] );
								gsumx *= mindistance
								gsumy *= mindistance

								#vwsum = sum( [ v[3] for v in vmd ] )
								#gsumx = sum( [ v[3]*v[0]/v[2] for v in vmd ] )/vwsum#len(vmd) )
								#gsumy = sum( [ v[3]*v[1]/v[2] for v in vmd ] )/vwsum#len(vmd) )

								return ( gsumx, gsumy );
						
		#Need a fallback for OFF CELL Entries... (hm...)
		#As in... nearest? gradient requires n nearest at LEAST? ... no? Hm...
		inear = self.nearest( p );
		v = self.V[ inear ];
		return ( v[0] - p[0], v[1]-p[1] )
		#self.nearest_direct( p );  #Ugly! can improve this. (cached nearest cell list?)
		#return (0,0);
				
		return err;
	
	def metric( self, p, err=0.0, f=10 ):
		#Since the gradient tells us this info, easy enough? Hm... 
		gvec = self.gradient( p );
		gvecmagsq = ( gvec[0]*gvec[0] + gvec[1]*gvec[1] )/(self.r*self.r)
		return math.exp( -gvecmagsq*gvecmagsq*f )
		
		#Improvements to this would consider "connectivity" as part of the weighting.
		#if NEIGHBORS are matching well, that increases their weight?
		
		#1. determine CELL p is in
		key = ( math.floor( p[0]/self.r ), math.floor( p[1]/self.r ) )
		if key in self.grid:
			vilist = self.grid[ key ];
			if len( vilist ) > 0:
				
				
				def linblend(v, ffac=10):
					#wehn v == p, returns 1/(1+0)
					#when |v-p| == r/2, returns 1/(1+ffac/2)
					#when |v-p| == r, returns 1/(1+ffac)
					#Increase f to reduce effect of farther points.
					#return 1 / (1+ffac*((v[0] - p[0])*(v[0] - p[0]) + (v[1] - p[1])*(v[1] - p[1]))/(self.r*self.r) )
					return 1 / (1+ffac*math.sqrt((v[0] - p[0])*(v[0] - p[0]) + (v[1] - p[1])*(v[1] - p[1]))/self.r )
					
				#Gauss blend?
				
				#dsqs = [ ( 1.0 - ((v[0] - p[0])*(v[0] - p[0]) + (v[1] - p[1])*(v[1] - p[1]))/( self.r*self.r ) , v ) for v in [ self.V[vi] for vi in vilist ] ]
				dsqs = [ ( linblend(v,f) , v ) for v in [ self.V[vi] for vi in vilist ] ]
				dsqs = [ v for v in filter( lambda x:x[0] >= 0, dsqs ) ];
				
				if len( dsqs ) > 0:
					#dsqs.sort( key=lambda x: x[0] );

					#dsqs represent something... 0.0 is the WORST, 1.0 is the BEST.

					avgdist = sum( [ v[0] for v in dsqs ] )/len(dsqs);  #HM! averaging SQUARED errors...

					return avgdist;
					#if avgdist <= self.r:
					#	return ( avgdist / self.r );

					#mindist = math.sqrt( mindist );
					#if mindist <= self.r:
					#	return ( mindist / self.r );
					#else:
					#	return err;
		return err;
	
	def nearest_direct( self, p ):
		#Naaive method is EXTREMELY SLOW so thats fine for now until its working
		distsqs = [ ( (v[0]-p[0])*(v[0]-p[0]) + (v[1]-p[1])*(v[1]-p[1]) ) for v in self.V ];
		#pairs = zip( distsqs, [ i for i in range( len( self.V ) ) ] );
		minpair = min( zip( distsqs, [ i for i in range( len( self.V ) ) ] ), key=lambda x:x[0] );
		return minpair[1];
	
	def nearest( self, p ):
		#1. determine CELL p is in
		key = ( math.floor( p[0]/self.r ), math.floor( p[1]/self.r ) )
		if key in self.grid:
			vilist = self.grid[ key ];
			mindist = math.inf;#2*self.r*self.r;
			mindisti = -1;
			for vi in vilist:
				v = self.V[ vi ]
				dx = v[0] - p[0];
				dy = v[1] - p[1];
				dsq = dx*dx + dy*dy
				if dsq < mindist:
					mindist = dsq;
					mindisti = vi;
			if mindisti >= 0:
				return mindisti;
		
		#Fallback for problem cells. potentially, it may be worth knowing some of this in advance (cells OUTSIDE the AABB have KNOWN "jump to" cells...)
		return self.nearest_direct( p );
	

def icp_2d_transform( V, T ):
	"""
		V = [ (x,y,..), ... ]
		T0 = [ Xx, Xy, Yx, Yy, x, y ] 2D affine transformation and offset
	"""
	return [
		(
			(v[0] * T[0]) + (v[1] * T[1]) + T[4]
			,(v[0] * T[2]) + (v[1] * T[3]) + T[5]
		)
		for v in V
	]

def icp_2d_transform_c( V, Ttrans, Trot, Tscale ):
	"""
		V = [ (x,y,..), ... ]
		T0 = [ Xx, Xy, Yx, Yy, x, y ] 2D affine transformation and offset
	"""
	cs = math.cos( Trot );
	ss = math.sin( Trot );
	return [
		(
			Tscale[0]*( (v[0] * cs) + (v[1] * ss) ) + Ttrans[0]
			,Tscale[1]*(  -(v[0] * ss) + (v[1] * cs) ) + Ttrans[1]
		)
		for v in V
	]


def icp_2d_average( V ):
	"""
		Returns average of all points in V (sum(x)/n, sum(y)/n)
		V = [ (x,y), ... ]
	"""
	n = 1.0*len( V );
	return (
		sum( [ v[0] for v in V ] )/n
		,sum( [ v[1] for v in V ] )/n
	)

def icp_2d_anglemean( V ):
	"""
		https://en.wikipedia.org/wiki/Mean_of_circular_quantities
		Given a list of angles [ 0.. 2*pi ] returns the MEAN of those angles.
		this is a angular mean / circular mean
	"""
	ss = sum( [ math.sin( a ) for a in V ] )/len(V);
	cs = sum( [ math.cos( a ) for a in V ] )/len(V);
	return math.atan2( ss, cs );

def icp_2d_tanaverage( V ):
	"""
		Returns average of 90 rotated version of all points in V (sum(x)/n, sum(y)/n)
		V = [ (x,y), ... ]
	"""
	n = 1.0*len( V );
	return (
		sum( [ -v[1] for v in V ] )/n
		,sum( [ v[0] for v in V ] )/n
	)

def icp_2d_radialaverage( V ):
	return (
		icp_2d_anglemean( [ math.atan2( v[1], v[0] ) for v in V ] )
		,sum( [ math.sqrt( v[0]*v[0] + v[1]*v[1] ) for v in V ] )/len( V )
	)

def icp_2d_diffv( V, p ):
	"""
		Returns list of sebtracting each point in v by p,  [ v[0]-p, v[1]-p,...,v[n]-p ]
		V = [ (x,y), ... ]
		p = (x,y)
	"""
	return [ ( v[0] - p[0], v[1] - p[1] ) for v in V ]

def icp_2d_transmul( A, B ):
	"""
		A = [ Xx, Xy, Yx, Yy, x, y ]
		B = [ Xx, Xy, Yx, Yy, x, y ]
		Returns combined transform, A * B
		
		Not sure if this is right:
		mA = [ aXx, aXy, ax ]
		     [ aYx, aYy, ay ]
			 [  0,  0, 1 ]
			 
		mB = [ bXx, bXy, bx ]
		     [ bYx, bYy, by ]
			 [  0,  0, 1 ]
			 
		[ aXx*bXx + aXy*bYx + 0, aXx*bXy + aXy*bYy + 0, aXx*bx + aXy*by + ax*1 ]
		[ aYx*bXx + aYy*bYx + 0, aYx*bXy + aYy*bYy + 0, aYx*bx + aYy*by + ay*1 ]
		[ 0, 0, 1 ]
	"""
	return [
		A[0]*B[0] + A[1]*B[2]
		, A[0]*B[1] + A[1]*B[3]
		, A[2]*B[0] + A[3]*B[2]
		, A[2]*B[1] + A[3]*B[3]
		, A[0]*B[4] + A[1]*B[5] + A[4]
		, A[2]*B[4] + A[3]*B[5] + A[5]
	];
	
	return A;

def icp_2d_outer_product_sum( X, Y ):
	"""
		Returns outer product sum of two vectors OF THE SAME LENGTH and CORRESPONDED
		So, this takes and sums up ALL the 2x2 outer product matrices.
		X = [ (x,y), ... ]
		Y = [ (x,y), ... ]
		
		Well, we really just want X * Y^T;
		#and since X and Y are both Nx2 column vectors...
		#Then...
		#c[i][j] = sum( X[i][k] * Y[i][k] )
	"""
	
	return (
		#( 
			sum( [ X[i][0]*Y[i][0] for i in range(len(X)) ] )
			, sum( [ X[i][0]*Y[i][1] for i in range(len(X)) ] )
		#), ( 
			,sum( [ X[i][1]*Y[i][0] for i in range(len(X)) ] )
			, sum( [ X[i][1]*Y[i][1] for i in range(len(X)) ] )
		#) 
	);
	
def icp_2d_svd_r( U, V ):
	"""
		U and V MUST be from icp_2d_svd, which FORCES their shape??
		U = ?
		V = ?
		Returns U * transpose( V );
		#Normal matrix multiplication algorithm? Uh...
		#do we KNOW something about this?
	"""
	#Since U and V are BOTH 2x2 matrices, this is easy enough:
	#
	#U[0][0] U[0][1]   V[0][0] V[0][1]   
	#U[1][0] U[1][1]   V[1][0] V[1][1]
	#
	return [
		U[0][0]*V[0][0] + U[0][1]*V[1][0]
		,U[0][0]*V[0][1] + U[0][1]*V[1][1]
		
		,U[1][0]*V[0][0] + U[1][1]*V[1][0]
		,U[1][0]*V[0][1] + U[1][1]*V[1][1]
	];
		
def icp_2d_svd22( M ):#const double a[4], double u[4], double s[2], double v[4]):
	#Martynas Sabaliauskas  https://scicomp.stackexchange.com/questions/8899/robust-algorithm-for-2-times-2-svd
	a = [ M[0], M[1], M[2], M[3] ];
	u = [0,0,0,0];
	s = [0,0,0,0];
	v = [0,0,0,0];
	s[0] = (
		math.sqrt( math.pow(a[0] - a[3], 2) + math.pow(a[1] + a[2], 2) ) 
		+ math.sqrt( math.pow(a[0] + a[3], 2) + math.pow(a[1] - a[2], 2))
	) / 2;
	s[1] = abs( s[0] - math.sqrt( math.pow(a[0] - a[3], 2) + math.pow(a[1] + a[2], 2)) );
	if s[0] > s[1]: #v[2] = (s[0] > s[1]) ?  : 0;
		v[2] = math.sin( ( math.atan2(2 * (a[0] * a[1] + a[2] * a[3]), a[0] * a[0] - a[1] * a[1] + a[2] * a[2] - a[3] * a[3])) / 2 )
	else:
		v[2] = 0
	v[0] = math.sqrt(1 - v[2] * v[2]);
	v[1] = -v[2];
	v[3] = v[0];
	if (s[0] != 0):
		u[0] = (a[0] * v[0] + a[1] * v[2]) / s[0];
		u[2] = (a[2] * v[0] + a[3] * v[2]) / s[0];
	else:
		u[0] = 1;
		u[2] = 0;
	if (s[1] != 0):
		u[1] = (a[0] * v[1] + a[1] * v[3]) / s[1]
		u[3] = (a[2] * v[1] + a[3] * v[3]) / s[1]
	else:
		u[1] = -u[2];
		u[3] = u[0];
		
	return u, s, v;
	

def icp_2d_svd( M ):
	"""
		M = [ [ x0 dot y0, x0 dot y1, ... ], ... ] ??? wait... not sure what these should be
		M in this case for usage is ALWAYS SQUARE.
		M should be formed from icp_2d_outer_product_sum (does it contain vectors or values??!?!?)
		This performs a singular value decomposition...
	"""
	u, s, v = icp_2d_svd22( M ); # Signs (for BOTH U and V) arent agreeing with numpy version; VALUES are almost exactly the same.
	
	U = [ [ -u[0], u[1] ],[ -u[2], u[3] ] ]; #Unsure about this hacky sign buisness. Possibly something else is wrong in the system??
	V = [ [ -v[0], v[1] ],[ -v[2], v[3] ] ];
	
	#reso = numpy.linalg.svd( [ [ M[0], M[1] ], [ M[2], M[3] ] ] ); #THIS gives same answer.	
	return U, [ s[0], s[3] ], V;

def icp_2d_polyRoot1( a0, a1 ):
	"""
	@brief Solve equation y = a0 + a1*x for y == 0 and ONLY real roots; unordered
	"""
	#If a1 is 0, there is no actual solution so we return a0
	if( a1 != 0 ):	#//#ERROR always dangerous when numerical precision is NOT used
		return [ -a0/a1 ];	#(y - a0)/a1
	else:
		return None;

def icp_2d_polyRoot2( a0, a1, a2 ):
	"""
	#@brief Solve equation y = a0 + a1*x + a2*x^2 for y == 0 and ONLY real roots; unordered
	"""
	#If a2 is 0, use 1 d solver
	if( a2 != 0 ):	##ERROR always dangerous when numerical precision is NOT used
		#a = a2 b = a1 c = a0
		#s0 = ( -a1 + sqrt( a1*a1 - 4 * a2 * a0 ) ) / (2 * a2)
		#s1 = ( -a1 - sqrt( a1*a1 - 4 * a2 * a0 ) ) / (2 * a2)
		#
		rootterm1 = a1*a1 - 4 * a2 * a0;
		if( rootterm1 >= 0 ):
			
			rootsq1 = math.sqrt( rootterm1 );
			a2div = ( 2 * a2 );
			return [
				( -a1 + rootsq1 ) / a2div
				,( -a1 - rootsq1 ) / a2div
			]
		else:
			#//Well we CAN solve for complex roots, and take the real part... but that might give erroneous values to the user?
			#//Hm. If you want REAL valued roots, these are useless.
			return None;
	else:
		return this.polyRoot1( a0, a1 );

def icp_2d_eigenSymmetric22VectorFromValue( A, root ):
	b = A[1];	#//b=x*y
	#c = 0;#A[2];	#//c=x*z
	d = A[3];	#//d=y*y
	#e = 0;#A[5];	#//e=y*z
	#f = 0;#A[8];	#//f=z*z
	return [
		root*( root - d )
		,b*(root)
	];
	
def icp_2d_eigenSymmetric22( A ):
	"""
		A is a 2x2 matrix, as a list of 4 floats
	"""
	
	Axx = A[0];
	Axy = A[1];
	Ayy = A[3];
	
	Axy2 = Axy*Axy;
	
	ceq1 = Axx*Ayy - Axy2;
	ceq2 = -( Axx + Ayy );
	ceq3 = 1;
	
	roots = icp_2d_polyRoot2( ceq1, ceq2, ceq3 );
	
	if len( roots ) == 2:
		v0 = icp_2d_eigenSymmetric22VectorFromValue( A, roots[0] );
		v1 = icp_2d_eigenSymmetric22VectorFromValue( A, roots[1] );
		return { 
			"eigenValues":roots
			, "eigenVectors":[v0,v1] 
		};
	elif len( roots ) == 1:
		v0 = icp_2d_eigenSymmetric22VectorFromValue( A, roots[0] );
		return { 
			"eigenValues":roots
			, "eigenVectors":[v0] 
		};
	else:
		return None;
	
def icp_2d_naturalbasis( V, com=None ):
	
	if com == None:
		com = (
			sum( [ v[0] for v in V ] )/len(V)
			,sum( [ v[1] for v in V ] )/len(V)
		)
	Rxx = sum( [ (v[0] - com[0])*(v[0] - com[0]) for v in V ] )
	Ryy = sum( [ (v[1] - com[1])*(v[1] - com[1]) for v in V ] )
	Rxy = sum( [ (v[0] - com[0])*(v[1] - com[1]) for v in V ] )
	#for v in V:
	#	rx = v[0] - com[0];
	#	ry = v[1] - com[1];
	#	m = 1;#m = mass/totalmass
	#	Rxx += m*( rx*rx );
	#	Ryy += m*( ry*ry );
	#	Rxy += m*( rx*ry );
		
	#Construct our symetric matrix from which we will extract eigenvalues
	A = [ 2*Rxx, 2*Rxy , 2*Rxy, 2*Ryy ];
	
	eigs = icp_2d_eigenSymmetric22( A ); #for a 2x2 this is solving a quadratic. easy. even a 4x4 can work (4th order polynomials do have solutions closed)
	
	roots = eigs["eigenValues"];
	vectors = eigs["eigenVectors"];
	
	#Greatest eigenvalue == MINIMUM moment of inertia axis.
	fwdaxis = (1,0);
	if roots == None:
		pass;#return ( (1,0), (0,1) );#DEFAULT basis vectors.
	elif len( roots ) == 2:
		fwdaxis = (1,0);
		if roots[0] < roots[1]:
			fwdaxis = vectors[1];
		elif roots[0] > roots[1]:
			fwdaxis = vectors[0];
		else:
			pass;#return ( (1,0), (0,1) ); #common pedantic case... perfectly radially symmetric
			
	elif len( roots ) == 1:
		fwdaxis = vectors[0];
	else:
		pass;#return ( (1,0), (0,1) );#DEFAULT basis vectors.
	
	#Warning: these axes can get VERY LARGE.
	
	#Large normalize is different; we avoid a multiplication first:
	fwdn = max( abs( fwdaxis[0] ), abs( fwdaxis[1] ) );
	fwdaxis = ( fwdaxis[0] / fwdn, fwdaxis[1] / fwdn );
	
	#Then we can do this (because squaring very large values is bad)
	fwdaxislen = math.sqrt( fwdaxis[0]*fwdaxis[0] + fwdaxis[1]*fwdaxis[1] );
	if fwdaxislen > 0:
		fwdaxis = ( fwdaxis[0]/fwdaxislen, fwdaxis[1]/fwdaxislen );
	else:
		fwdaxis = (1,0);
		
	#Well, we have a ESTIMATE of the axes. But they can be FLIPPED.
	#Eigenvalues tell us the correct "direction" usually, but might miss important details otherwise.
	#we probably want the natural direction to be toward the MOST projected mass... (sum squared x projected)
	
	return fwdaxis, ( -fwdaxis[1], fwdaxis[0] );

def icp_2d_naturalbasis_xweight( A, V ):
	
	dps = [ v[0]*A[0][0] + v[1]*A[0][1] for v in A ]
	dpsump = sum( [ v*v for v in filter( lambda x:x>0, dps ) ] )
	dpsumn = sum( [ v*v for v in filter( lambda x:x<0, dps ) ] )
	if dpsump > dpsumn:
		return A;
	else:
		return ( ( -A[0][0], -A[0][1] ) , ( A[0][1], -A[0][0] ) );
	
def icp_2d_best_sliding_match( A, B, istart=None, iend=None ):
	"""
		Given two lists of the same length; with values representing height
		Compute the index slide that minimizes the sum squared error.
	"""
	#def slide( V, off ):
	#	vlen = len(V);
	#	return [ V[ (i+off)%vlen ] for i in range( vlen ) ];
	
	Blen = len(B);
	
	if istart == None:
		istart = 0;
	if iend == None:
		iend = Blen-1;
		
	isamps = [ i+istart for i in range( (iend - istart) + 1 ) ];
	
	def ssq( A, B, off ):
		vlen = len(A);
		off += vlen; #Handle negatives.
		diffs = [ ( A[i] - B[(i+off)%vlen] ) for i in range( vlen ) ];
		return sum( [ v*v for v in diffs ] );
		
	minpair = min( [ ( ssq( A, B, i ), i ) for i in isamps ], key=lambda x:x[0] );
	
	return minpair[1];
	
def icp_2d_radians_to_unitrevolution( v ):
	rev = v/(2*math.pi);
	if rev < 0:
		rev += 1;
		if rev < 0:
			rev += 1;
	if rev >= 1:
		rev -= 1;
		if rev >= 1:
			rev -= 1;
	return rev;

def icp_2d_tri_area( vx, vh, bw, bmin, bmax ):
	"""
		Integral of triangle kernel centered at vx with height vh OVER bmin...bmax
		function of x is:
		0 when x < vx - bw
		vh*(x-vx-bw)/bw x < vx
		vh*( 1 - (x-vx-bw)/bw ) x < vx+bw
		0 else
	"""
	
	#Normalize inputs so 0 is the center of the unit triangle pulse.
	nbmin = (bmin - vx)/bw; #
	nbmax = (bmax - vx)/bw; #
	area = 0.0;
	
	if nbmin < 0:
		if nbmax < 0:
			#1 part: max( -1, nbmin )..min( 0, nbmax )
			x0 = max( -1, nbmin );
			x1 = min( 0, nbmax ); #h = (x+1)
			area += ((x1+1)-(x0+1))/2  #area under curve is 1/2 the rectangular area
		else:
			#2 parts. max( -1, nbmin )..0, 0..min( 1, nbmax ))
			x0 = max( -1, nbmin ); #h = (x+1)
			x1 = 0;
			area += ((1) - (x0+1))/2
			
			x0 = 0
			x1 = min( 1, nbmax );  #h = (1-x)
			area += ( (1-x0) - (1-x1) )/2
			
	elif nbmin < 1:
		#1 part: min( 1, nbmin )..min( 1, nbmax )
		x0 = min( 1, nbmin );
		x1 = min( 1, nbmax );  #h = (1-x)
		area += ( (1-x0) - (1-x1) )/2
	else:
		pass; #range does not overlap with any area
	
	return vh * bw * area ;
#print( icp_2d_tri_area( 0, 1, 1, -1, 0 ) ) #vx, vh, bw, bmin, bmax
#print( icp_2d_tri_area( 0, 1, 1, 0, 1 ) )
#print( icp_2d_tri_area( 0, 1, 1, -1, 1 ) )

def icp_2d_form_convolgram( A, n=360, bw=( 2.5/360 ) ): #Amust be ( 0..1, h ) values. default bandwidth is +- 2.5 degrees. bw cannot exceed 0.25...
	h = [ 0.0 for v in range( n ) ];
	bwn = n*bw
	nnum = 1.0*n;
	for v in A:
		vh = v[1]; #HEIGHT maximum...
		bv = v[0] * n;  #revolutions 0..1
		b0 = math.floor( bv - bwn );
		b1 = math.floor( bv + bwn );
		#Now, for THIS SAMPLE, add to h...
		#hm. "area of overlap" with a "bin"
		b = b0;
		while b <= b1:
			h[ (b+n)%n ] += icp_2d_tri_area( v[0], vh, bw, b/nnum, (b+1)/nnum );
			b += 1;
			
	#Averageize h? Other transforms?
	if False:
		maxh = max(h);#sum( h )/len(h); max(h);
		#h = [ v/avgh for v in h ];
		h = [ v/maxh for v in h ];
			
	return h;

def icp_2d_best_angular_match( X, Y, steps=5, bw=2.5 ): #
	"""
		Given TWO LISTS of 2D coordinates, both ALREADY centered at their means...
		Compute the best angular match to rotate Y to match X.
		steps should be < 10:
			36 * 2^steps == memory cost and speed cost
		bw is in degrees, defaults to 2.5
	"""
	#Samples are in REVOLUTIONS 0..1, magnitude => Hm.. radians bins? 
	rX = [ ( icp_2d_radians_to_unitrevolution( math.atan2( x[1], x[0] ) ) , math.sqrt( x[1]*x[1]+x[0]*x[0] ) ) for x in X ];
	rY = [ ( icp_2d_radians_to_unitrevolution( math.atan2( y[1], y[0] ) ) , math.sqrt( y[1]*y[1]+y[0]*y[0] ) ) for y in Y ];
	
	bw = bw/360.0;  #should we TWEAK bandwidth any? hm...
	nd = 36;#72; #5 degrees = 72*72 ... Hm.. 36*36 is better to start with?
	deltaangle = 0;
	nistart = 0;
	niend = nd-1;
	
	for iteration in range( steps ):#5 ): #36*2*2*2*2 = 576, so...

		cgX = icp_2d_form_convolgram( rX, n = nd, bw = bw ); #this is pretty expensive...
		cgY = icp_2d_form_convolgram( rY, n = nd, bw = bw );
		
		slideindex = icp_2d_best_sliding_match( cgX, cgY, istart=nistart, iend=niend );
		
		deltaangle = (slideindex/nd)*math.pi*2;
		
		nd *= 2;
		nistart = (slideindex*2) - 2; #Add some previous entries
		niend = (slideindex*2) + 2; #Add some next entries (remember this is INCLUSIVE)
		
		#Hm, in order to do this EFFICIENTLY...
		#Tricky. first, comparing two of these...
		#we DEFINITELY need to start with some kind of histogram representation...
		#Start with 10 degree bins... Then divide by 2 for each range? Hm... might want to divide more
		#
		#

		#convert samples to a triangular convolution histogram representation accurate to some number of degrees...
		#3600
		#3600
		
	
	return deltaangle;

def icp_2d_naturalbasis_adjust_scale( xprime_k, yprime_k, xT, xTheta ):
	
	#Scale should NOT be based on "all points" just "gradient" average.
	#Additionally, "scale" is... Hm...
	#Scale should be based on PROXIMITY / GRADIENT... not ALL points.
	
	rT = icp_2d_transmul( [ math.cos( -xTheta ), math.sin( -xTheta ), -math.sin( -xTheta ), math.cos( -xTheta ), 0,0 ], xT )
	
	Rxxmag = math.sqrt( rT[0]*rT[0] + rT[1]*rT[1] )
	Rxymag = math.sqrt( rT[2]*rT[2] + rT[3]*rT[3] )
	xprime_k_dotx = sum( [ abs( (v[0]*rT[0] + v[1]*rT[1] )/Rxxmag ) for v in xprime_k ] )/len(xprime_k); #already centered on mean
	xprime_k_doty = sum( [ abs( (v[0]*rT[2] + v[1]*rT[3] )/Rxymag ) for v in xprime_k ] )/len(xprime_k); #already centered on mean
	
	Ryxmag = math.sqrt( xT[0]*xT[0] + xT[1]*xT[1] )
	Ryymag = math.sqrt( xT[2]*xT[2] + xT[3]*xT[3] )
	yprime_k_dotx = sum( [ abs( (v[0]*xT[0] + v[1]*xT[1] )/Ryxmag ) for v in yprime_k ] )/len(yprime_k); #already centered on mean
	yprime_k_doty = sum( [ abs( (v[0]*xT[2] + v[1]*xT[3] )/Ryymag ) for v in yprime_k ] )/len(yprime_k); #already centered on mean
	
	xscale = xprime_k_dotx / yprime_k_dotx
	yscale = xprime_k_doty / yprime_k_doty
	
	return [ xscale*xT[0], xscale*xT[1], yscale*xT[2], yscale*xT[3], xT[4], xT[5] ]

		
def icp_2d_initial_T_direct_cens( xprime_k, yprime_k, x_m, y_m, xmb = None, anglebins = 5 ):
	
	#compute natural mass basis for each shape ( may want to WEIGHT points somehow... )
	if xmb == None:
		xmb = icp_2d_naturalbasis( xprime_k, com=(0,0) );
		xmb = icp_2d_naturalbasis_xweight( xmb, xprime_k )
	ymb = icp_2d_naturalbasis( yprime_k, com=(0,0) );
	ymb = icp_2d_naturalbasis_xweight( ymb, yprime_k )
	
	#What is the angle between these two natural basis vectors?
	delangle = -( math.acos( xmb[0][0]*ymb[0][0] + xmb[0][1]*ymb[0][1] ) );
	
	#Make sure to rotate the right way
	scheck = xmb[1][0]*ymb[1][0] + xmb[1][1]*ymb[1][1];
	if scheck < 0:
		delangle = -delangle;
		
	if True:
		
		#Sliding smooth kernel convolution:
		
		#Now we have the next thing:
		thetaest = icp_2d_best_angular_match( xprime_k, yprime_k, steps=10, bw = 1 ); #Given TWO LISTS of 2D coordinates, both ALREADY centered at their means...
	
		#dpslideangle = math.cos( testdelangle )*math.cos(delangle) + math.sin( testdelangle )*math.sin(delangle);
		dpslideangle = math.cos( thetaest )*math.cos(delangle) + math.sin( thetaest )*math.sin(delangle);
		if abs(dpslideangle) < math.cos( 80 ):#10*math.pi/180.0: #shoud be close to 1. However it needs an amount of tolerance.
			pass;
		else:
			print( "Angle corrected...", delangle, thetaest )
			delangle = thetaest;

	#Compute scaling factors based on the original and destination
	#xT = icp_2d_transmul( [], [ xmb[0][0], xmb[0][1], xmb[1][0], xmb[1][1], 0, 0 ] );
	#xT = icp_2d_transmul( [ math.cos(delangle), math.sin(delangle), -math.sin(delangle), math.cos(delangle), 0, 0], [ ymb[0][0], ymb[0][1], ymb[1][0], ymb[1][1], 0, 0 ] );
	xT = [ xmb[0][0], xmb[0][1], xmb[1][0], xmb[1][1], 0, 0 ]
	#Okay but, maybe "scale" is defineable in terms of gradient vectors too?
	#Maybe... but also maybe SCALE should not be INCLUDED here?
	#
	#Scale sould NOT be based on the average, but instead, something else (like nearest gradient projected)
	#
	
	Rxxmag = math.sqrt( xT[0]*xT[0] + xT[1]*xT[1] )
	Rxymag = math.sqrt( xT[2]*xT[2] + xT[3]*xT[3] )
	xprime_k_dotx = sum( [ abs( (v[0]*xT[0] + v[1]*xT[1] )/Rxxmag ) for v in xprime_k ] )/len(xprime_k); #already centered on mean
	xprime_k_doty = sum( [ abs( (v[0]*xT[2] + v[1]*xT[3] )/Rxymag ) for v in xprime_k ] )/len(xprime_k); #already centered on mean
	
	Ryxmag = math.sqrt( ymb[0][0]*ymb[0][0] + ymb[0][1]*ymb[0][1] )
	Ryymag = math.sqrt( ymb[1][0]*ymb[1][0] + ymb[1][1]*ymb[1][1] )
	yprime_k_dotx = sum( [ abs( (v[0]*ymb[0][0] + v[1]*ymb[0][1] )/Ryxmag ) for v in yprime_k ] )/len(yprime_k); #already centered on mean
	yprime_k_doty = sum( [ abs( (v[0]*ymb[1][0] + v[1]*ymb[1][1] )/Ryymag ) for v in yprime_k ] )/len(yprime_k); #already centered on mean
	
	#apply scaling (doesnt assume uniform scaling?)
	xscale = 1;#xprime_k_dotx / yprime_k_dotx;
	yscale = 1;#xprime_k_doty / yprime_k_doty;
	R = [
		xscale*math.cos( delangle )
		, xscale*math.sin( delangle )
		, -yscale*math.sin( delangle )
		, yscale*math.cos( delangle ) 
	];
	
	#print( xscale, yscale )

	y_m_R = icp_2d_transform( [ y_m ], [ R[0], R[1], R[2], R[3], 0, 0 ] )[0] #R * y_m
	
	p = ( x_m[0] - y_m_R[0], x_m[1] - y_m_R[1] );#icp_2d_diffv( [ x_m ], y_m_R )[0]; #x_m - R * y_m    #
	
	return [ R[0], R[1], R[2], R[3], p[0], p[1] ], delangle, xscale, yscale, x_m, y_m, (xprime_k_dotx,xprime_k_doty); #DELTA transformation, NOT absolute...

def icp_2d( X, Y, T0=None, epsilon=0.001, niters=100, maxradius=None, f=10 ):
	"""
		X = [ (x,y,..), ... ] TO / DST
		Y = [ (x,y,..), ... ] FROM / SRC
		T0 = [ Xx, Xy, Yx, Yy, x, y ] 2D affine transformation and offset
		Finds a transform that "best" matches Y to X
			Must include postion(x,y), rotation (theta), scale (sx, sy)
			Does not need to include Shear? Or is that a bonus?
	"""
	#if T0 == None:
	#	#T, Tangle, Txscale, Tyscale, Tx_m, Ty_m, Txsb = icp_2d_initial_T_direct_cens( xprime_k, yprime_k, x_m, y_m, xmb=xmb, anglebins=anglebins )
	#	#T = icp_2d_initial_T( X, Y, f=f, maxradius=maxradius );

	#else:
	#	T = T0[:];

	#Dissalow T0.
	x_m = icp_2d_average( X );
	y_m = icp_2d_average( Y );
	xprime_k = icp_2d_diffv( X, x_m ); #eliminate translation? no?
	yprime_k = icp_2d_diffv( Y, y_m );
	xmb = icp_2d_naturalbasis( X, com=x_m );#Again, X should NEVER change. we shouldn't UPDATE it with a selection!!!
	xprime_k_dotx = sum( [ abs( ((v[0])*xmb[0][0] + (v[1])*xmb[0][1] ) ) for v in xprime_k ] )/len(xprime_k);
	xprime_k_doty = sum( [ abs( ((v[0])*xmb[1][0] + (v[1])*xmb[1][1] ) ) for v in xprime_k ] )/len(xprime_k);
	Twhats = icp_2d_initial_T_direct_cens( xprime_k, yprime_k, x_m, y_m, xmb=xmb )
	T = Twhats[0];
	Tsc = icp_2d_naturalbasis_adjust_scale( xprime_k, yprime_k, Twhats[0], Twhats[1] )
	T = Tsc;
	
	#return T, [ ( T, 0 ) ]
	
	#Rxxmag = math.sqrt( rT[0]*rT[0] + rT[1]*rT[1] )
	#Rxymag = math.sqrt( rT[2]*rT[2] + rT[3]*rT[3] )
	#xprime_k_dotx = sum( [ abs( (v[0]*rT[0] + v[1]*rT[1] )/Rxxmag ) for v in xprime_k ] )/len(xprime_k); #already centered on mean
	#xprime_k_doty = sum( [ abs( (v[0]*rT[2] + v[1]*rT[3] )/Rxymag ) for v in xprime_k ] )/len(xprime_k); #already centered on mean
	
	if True:
		#Reimplement ICP but this time keep translation, rotation, scale SEPARATE. 
		#    Current transform is ( trans, rotangle, scale )
		#Algorithm:
		#    Compute closest points / gradients
		#    Recenter transformed data on current center
		#    Compute that SVD of the outer product of the matched vectors
		#    this gets rotation
		#    translation is always delta centers
		#    recompute scale
		#    assign new transform, compute error metric / change in transform
		#
		
		xnn = icp_2d_proximitymetric( X, maxradius=maxradius );
		
		Ttran = [ T[4], T[5] ];
		xxm = math.sqrt( T[0]*T[0] + T[1]*T[1] )
		yym = math.sqrt( T[2]*T[2] + T[3]*T[3] )
		Tscale = [ xxm, yym ];
		Trot = math.atan2( T[1]/xxm, T[0]/xxm );
		
		print( "O: ", Ttran, Trot, Tscale )
			
		Tprogress = []
		Tchangesq = 1;
		iteration = 0;
		while Tchangesq > epsilon and iteration < niters:#for iteration in range( 100 ):
		
			y_k = icp_2d_transform_c( Y, Ttran, Trot, Tscale );  #Transform original dataset by CURRENT composite transformation.
			
			y_m = icp_2d_average( y_k );#Get current centroid of transformed Y ? or do we use Ttran?
			
			#approximate gradient or nearest point?
			#g_k = [ ( (y[0]-y_m[0],y[1]-y_m[1]), xnn.gradient( y ) ) for y in y_k ]; #Get approximate gradient vectors (vector to nearest from y_k)
			
			x_nearest = [ X[ xnn.nearest( y ) ] for y in y_k ]
			g_k = [ 
					( (y_k[i][0]-y_m[0], y_k[i][1]-y_m[1])
					 , ( x_nearest[i][0] - y_k[i][0], x_nearest[i][1] - y_k[i][1] )
					) for i in range( len( x_nearest ) ) 
			]
			
			x_k = [ ( (y_k[i][0] + g_k[i][1][0]), (y_k[i][1] + g_k[i][1][1]) ) for i in range(len(y_k)) ]
			
			x_m = icp_2d_average( x_k );#Get TRUE centroid of dst

			nextTtran = [ x_m[0] - y_m[0] + Ttran[0], x_m[1] - y_m[1] + Ttran[1] ]; #Delta translation -> absolute translation
			
			xprime_k = icp_2d_diffv( x_k, x_m ); #position relative to current TRUE centroid
			yprime_k = icp_2d_diffv( y_k, y_m ); #position relative to current TRUE centroid
			
			#Rotation adjustment is hard to compute. (sum of torques from gradients)
			torquesum = sum( [ v[0][0]*v[1][1] - v[0][1]*v[1][0] for v in g_k ] )/len( g_k ) # r cross F, F = gradient U x V = Ux*Vy-Uy*Vx
			lensum = sum( [ math.sqrt( v[0][0]*v[0][0] + v[0][1]*v[0][1] ) for v in g_k ] )/len( g_k );#Sum of ARM lengths
			lentsum = sum( [ math.sqrt( v[1][0]*v[1][0] + v[1][1]*v[1][1] ) for v in g_k ] )/len( g_k );#Sum of TORQUE lengths
			torqueangle = math.asin( torquesum / (lensum + lentsum) ); #Possibly unstable
			
			nextTrot = Trot - torqueangle;# - torquesum / lensum;#math.asin( torquesum / lensum );
			
			#Scale is computed last.
			#What is "scale"... since we have an ORIGINAL scale from xmb...

			xprime_k_dotx = sum( [ abs( ((v[0]-x_m[0])*xmb[0][0] + (v[1]-x_m[1])*xmb[0][1] ) ) for v in x_k ] )/len(x_k);
			xprime_k_doty = sum( [ abs( ((v[0]-x_m[0])*xmb[1][0] + (v[1]-x_m[1])*xmb[1][1] ) ) for v in x_k ] )/len(x_k);
			
			cs = math.cos(nextTrot)
			ss = math.sin(nextTrot)
			xsv = [ abs( (v[1][0]*cs + v[1][1]*ss ) ) for v in g_k ]
			ysv = [ abs( ( -v[1][0]*ss + v[1][1]*cs ) ) for v in g_k ]
			#Alternate scale computations:
			#yprime_k_dotx = Tscale[0] + ( sum( xsv )/len(xsv) + max(xsv) )/2; #already centered on mean
			#yprime_k_doty = Tscale[1] + ( sum( ysv )/len(ysv) + max(ysv) )/2; #already centered on mean
			yprime_k_dotx = Tscale[0] + sum( xsv )/(2*len(xsv)); #already centered on mean
			yprime_k_doty = Tscale[1] + sum( ysv )/(2*len(ysv)); #already centered on mean
			
			xscale = yprime_k_dotx; #xprime_k_dotx / yprime_k_dotx
			yscale = yprime_k_doty; #xprime_k_doty / yprime_k_doty
			#xscale = yprime_k_dotx / xprime_k_dotx
			#yscale = yprime_k_doty / xprime_k_doty
			
			#print( (xscale, yscale), xprime_k_dotx, xprime_k_doty, yprime_k_dotx, yprime_k_doty,  )

			nextTscale = Tscale;#[ xscale, yscale ]; #Tscale
	
			# from the g_k ?
			
			#y_k2 = icp_2d_transform_c( Y, nextTtran, nextTrot, Tscale );
			#y_m2 = icp_2d_average( y_k2 );#Get current centroid of transformed Y
			#yprime_k2 = icp_2d_diffv( y_k, y_m ); #position relative to current TRUE centroid
			#ydps2 = [ ( abs( v[0]*cs + v[1]*ss ), abs( -ss*v[0] + v[1]*cs ) ) for v in yprime_k ];
			#ydps2 = [ ( abs( v[1][0]*cs + v[1][1]*ss ), abs( -ss*v[1][0] + v[1][1]*cs ) ) for v in g_k ];
			#axxsum = [ (1.0 + abs( v[1][0]*cs + v[1][1]*ss )) for v in g_k ]
			#ayysum = [ (1.0 + abs( -ss*v[1][0] + v[1][1]*cs )) for v in g_k ]
			
			#delsx = sum( axxsum )/len( axxsum ) #sum of gradient projected onto x axis
			#delsy = sum( ayysum )/len( ayysum ) #sum of gradient projected onto y axis
			
			#avgsx = sum( [ v[0] for v in ydps2 ] )/( len(ydps2) )
			#avgsy = sum( [ v[1] for v in ydps2 ] )/( len(ydps2) )
			
			#nextTscale = [
			#	(max( axxsum ) + delsx)/2
			#	,(max( ayysum ) + delsy)/2
			#]
			
			#print( nextTtran, nextTrot, nextTscale, torqueangle )
	
			#Compute rotation from GRADIENTS
			#torquesum = sum( [ v[0][0]*v[1][0] + v[0][1]*v[1][1] for v in g_k ] )/len(g_k) # r cross 
			#lensum = sum( [ math.sqrt( v[0][0]*v[0][0] + v[0][1]*v[0][1] ) for v in g_k ] )/len(g_k);#Sum of ARM lengths
			#lensum = sum( [ math.sqrt( v[1][0]*v[1][0] + v[1][1]*v[1][1] ) for v in g_k ] );#Sum of TORQUE lengths
			#netangle = torquesum / lensum;#math.asin( torquesum / lensum );
			#print( netangle )
			#nextTrot = netangle;
			
			#Compute rotation matrix? (this is garbage)
			#tprod = icp_2d_outer_product_sum( yprime_k, xprime_k );
			#U, S, V = icp_2d_svd( tprod );
			#R = icp_2d_svd_r( U, V );
			#nextTrot = Trot + math.atan2( R[0], R[1] ); #delta rotation -> absolute rotation
			
			#Tscale
			#y_k = icp_2d_transform_c( Y, Ttran, Trot, Tscale );  #Transform original dataset by CURRENT composite transformation.
			
			Tnext = [
				nextTscale[0]*math.cos(nextTrot)
				,nextTscale[0]*math.sin(nextTrot)
				,-nextTscale[1]*math.sin(nextTrot)
				,nextTscale[1]*math.cos(nextTrot)
				,nextTtran[0]  #Arent these scaled and rotated by something?
				,nextTtran[1]
			]
			
			Ttran = nextTtran 
			Trot = nextTrot
			Tscale = nextTscale
			
			Tchangesq = sum( [ (Tnext[i] - T[i])*(Tnext[i] - T[i]) for i in range( len( T) ) ] );#||(Tnext - T)||
			
			Tprogress.append( (T[:], Tchangesq) )
		
			T = Tnext
			iteration += 1;
			
		Tprogress.append( (T[:], 0) )
			
		return T, Tprogress;

	
#Hm, lets take a look.
def pfeatures_getvoronoigroups( Glist ):
	gpointinft = [];
	gpoints = [];
	plookupdict = {};
	for group in Glist:
		center = Glist[ group ][1]
		radius = center[3];
		plookupdict[ ( center[0], center[1] ) ] = len(gpointinft);# group;
		gpointinft.append( ( center, radius, group ) );
		gpoints.append( center );
		
	xmin = min( [ v[0] for v in gpoints ] )
	xmax = max( [ v[0] for v in gpoints ] )
	ymin = min( [ v[1] for v in gpoints ] )
	ymax = max( [ v[1] for v in gpoints ] )
	xdelta = xmax - xmin;
	ydelta = ymax - ymin;
		
	dt = tri_delaunay.triangulate( gpoints );  #well this sucks. whats the point of constrained, if it doesn't preserve indexing?!?!
	#vertex ordering does NOT change in this case? Hm... ( just points ? )
	
	#V is feature list, Vsl is site list, Vtx is tris context
	edges = {};#map edge (min, max) -> dict of triangle indexes -> starting triangle vertex index
	triss = {};#map triangle -> dict of edges -> starting triangle vertex index
	verts = {};#map vertex -> dict of edges -> bit for feature edge??
	
	def getbestvertex( x, y ):
		coordpair = (x,y);#((math.floor(x*8192)/4096.0), (math.floor(y*8192)/4096.0))
		if coordpair in plookupdict:
			return plookupdict[ coordpair ];
		else:
			#Okay... we know something about "infinite" vertices here.
			#So long as they ARE infinite vertices, I'm OK with that.
			if (x < xmin-xdelta) or (x > xmax+xdelta) or (y < ymin-ydelta) or (y > ymax+ydelta):	#Not sure where this comes from...
				pass;#print( "INF VERTEX: ", coordpair );pass;	#was a INFINITE vertex...
			else:
				pass;#print( "BAD VERTEX: ", coordpair );
		return None;
		
	usetriangles = []
	for tris in dt.triangles:
		trio = (
			getbestvertex( tris.vertices[0].x, tris.vertices[0].y )
			, getbestvertex( tris.vertices[1].x, tris.vertices[1].y )
			, getbestvertex( tris.vertices[2].x, tris.vertices[2].y )
		)
		if trio[0] != None and trio[1] != None and trio[2] != None:
			usetriangles.append( trio );
		else:
			pass;#print( "BAD TRIS: ", trio ); #"infinite" triangle may contain edges? Hm...
	
	trisi = 0;
	for tris in usetriangles:
		e0 = ( min(tris[0], tris[1]), max(tris[0], tris[1]) )
		e1 = ( min(tris[1], tris[2]), max(tris[1], tris[2]) )
		e2 = ( min(tris[0], tris[2]), max(tris[0], tris[2]) )
		
		if not e0 in edges:
			edges[ e0 ] = {};
		if not e1 in edges:
			edges[ e1 ] = {};
		if not e2 in edges:
			edges[ e2 ] = {};
		edges[ e0 ][ trisi ] = tris[0];
		edges[ e1 ][ trisi ] = tris[1];
		edges[ e2 ][ trisi ] = tris[2];
		
		triss[ trisi ] = {};
		triss[ trisi ][e0] = tris[0];
		triss[ trisi ][e1] = tris[1];
		triss[ trisi ][e2] = tris[2];
		
		if not tris[0] in verts:
			verts[ tris[0] ] = {};
		if not tris[1] in verts:
			verts[ tris[1] ] = {};
		if not tris[2] in verts:
			verts[ tris[2] ] = {};
		
		if not e0 in verts[ tris[0] ]:
			verts[ tris[0] ][ e0 ] = False;
		if not e1 in verts[ tris[1] ]:
			verts[ tris[1] ][ e1 ] = False;
		if not e2 in verts[ tris[2] ]:
			verts[ tris[2] ][ e2 ] = False;
		
		trisi += 1;
		
	return edges, triss, verts, plookupdict, gpointinft
	
	


def generalized_affine_match( seta, setb, targetdelta = 0.0001, maxiters=10 ):
	#
	#Kabsch algorithm => finds rotation matrix
	#
	#github.com/charnley/rmsd  ??
	#
	#ITerative Closest Point algorithm ??
	#
	#https://github.com/ClayFlannigan/icp/blob/master/icp.py
	#
	#https://stackoverflow.com/questions/20120384/iterative-closest-point-icp-implementation-on-python
	#
	#Intensity Assisted ICP ? (makes more sense, BIGGER features have higher INTENSITY)
	#
	#Veclocity Updating ICP VICP ?
	#	<- requires a "FindClosestPoint()
	#	<- requires a SVD algorithm (ok)
	#
	#Okay, there is a LOT MORE to this ICP than the academic nonsense.
	#Lots of "robustness" issues.
	#ALL WE WANT is a GOOD rotate, scale and translate match between two keyframes,
	#	Which we can USE DURING INTERPOLATION,
	#	and then minimize the amount of blended stroke data points.
	#	I have a suspicious that shape tweens do this on the front end potentianlly.
	#
	
	#
	#	"Fitting affine and orthogonal transformations between two sets of points"
	#
	#	pi = A points  [ (x,y,...) ]
	#	qi = B points  [ (x,y,...) ]
	#
	#	Objective is to find Affine matrix A,
	#	and translation vector t
	#	SUCH THAT pi ~= A*qi + t  ( A[i] = A*B[i] + t )
	#
	#	Possible objective is:
	#		Minimize  sum(  || pi - A*qi - t ||^2  )
	#	
	#	? Unsure if IMPLICIT ordering of points counts as "matched" points;
	#		<-wagering that it DOES, because a individual pi, qi is MATCHED UP as a pair.
	#		so, you would need to determine the pi / qi for THAT TRANSFORM, since they aren't matched.
	#	Regardless,
	#
	#	t = 1/m * sum( pi - A * qi );  #(m is the length of the sum or count)
	#	sum( transpose( Ejk * qi ) * A * qi ) = sum( transpose( Ejk*qi ) * ( pi - t ) )
	#	
	#	Given some starting A, set w = 0*
	#	Calculate the w'th:
	#		t[w+1] = 1/m * sum( pi - A[w] * qi );
	#	Calculate:
	#		sum( transpose( Ejk * qi ) * A[w+1] * qi ) = sum( transpose( Ejk*qi ) * ( pi - t[w+1] ) )
	#
	#	?? what is the convergence criteria???
	#	
	#
	
	#the real problem:
	#seta and setb should be SOME LIST of [ ( x,y, ... ), ... ] points.
	#
	#The original algorithm seemed to ASSUME correspondence between points (as if we already KNEW what pairs were important)
	#
	Afmatrix = [1,0,0,1,0,0];# [ax, ay, bx, by, x, y]
	#Note the INITIAL transformation is a BIAS. so, watch out for that.

	m = 1.0*len( seta );	#Huh! we are matching seta TO setb.
	
	#we need a EFFICIENT search structure for "nearest point in b"...
	#Generate p,q list matching (simply find CLOSEST POINT)
	bxmin = min( [ v[0] for v in setb ] );
	bxmax = max( [ v[0] for v in setb ] );
	bymin = min( [ v[1] for v in setb ] );
	bymax = max( [ v[1] for v in setb ] );
	bsize = max( bymax-bymin, bxmax-bxmin );
	
	bcount = 32;
	
	#PADDING just in case
	bxmin -= bsize*0.5/bcount;
	bymin -= bsize*0.5/bcount;
	bxmax += bsize*0.5/bcount;
	bymax += bsize*0.5/bcount;
	bsize = max( bymax-bymin, bxmax-bxmin );
	
	bcellsize = bsize/bcount;
	bgrid = [ [ [] for x in range(32) ] for y in range( 32 ) ]; # [y][x] = []
	for vi in range( len( setb ) ):
		#add to grid?
		v = setb[ vi ]
		xi = math.floor( bcount*((v[0] - bxmin) / bsize) )
		yi = math.floor( bcount*((v[1] - bymin) / bsize) )
		
		#From THIS CELL, what are the NEAREST elements (must be NEAR TO THE CELL, so that ANY POINT in the cell is guaranteed to check JUST the neighbors in THIS cell. pvis)
		#Expensive, but we gotta start somewhere.
		#
		#
		
		
		bgrid[ yi ][ xi ].append( vi );
	
	def nearest_q_index( p ):
		#increasing ring search (only have to find ONE)
		xi = math.floor( bcount*((p[0] - bxmin) / bsize) )
		yi = math.floor( bcount*((p[1] - bymin) / bsize) )
		#xi, yi can EASILY be outside of the grid. thats fine...
		#Search for POSSIBLE nearby vectors...
		return 0;
		
		pairsinradius = [ (xi, yi) ];
		
		sr = 1;
		
		six = -sr;
		siy = -sr;
		pairsinradius = []; #Hm!
		while siy <= sr:
			pairsinradius.append( ( xi + six, yi + siy ) );
			siy += 1;
			
		siy = sr;
		while six <= sr:
			pairsinradius.append( ( xi + six, yi + siy ) );
			six += 1;
			
		six = sr;
		while siy >= -sr:
			pairsinradius.append( ( xi + six, yi + siy ) );
			siy -= 1;
			
		siy = -sr;
		while six >= sr:
			pairsinradius.append( ( xi + six, yi + siy ) );
			six -= 1;
			
		#Done with radius; CHECK PAIRS now.... (not accurate, ignores circle issues...
		#pairsinradius
		
		sr += 1;
		
		return -1;
		
	def matmul( A, p ):
		return [ 
			A[4] + p[0]*A[0] + p[1]*A[1]
			, A[5] + p[0]*A[2] + p[1]*A[3] 
		];
		
	def matmul_notrans( A, p ):
		return [ 
			p[0]*A[0] + p[1]*A[1]
			, p[0]*A[2] + p[1]*A[3] 
		]
	
	def matmul_nomul( A, p ):
		return [ p[0] + A[4], p[1] + A[5] ];
		
	def psubtract( a, b ):
		return [ a[0] - b[0], a[1] - b[1] ];
		
	for niteration in range( maxiters ):
	
		#Generate seta -> setb correspondence indicies
		corrinds = [ nearest_q_index( p ) for p in seta ];
		
		#Current matrix
		A = Afmatrix[:];
			
		#Compute all the p_i - A * q_i values
		
		trandelts = [ psubtract( seta[i], matmul_notrans( A, setb[ corrinds[ i ] ] ) ) for i in range( len( seta ) ) ];	
		t_w1 = [ sum( [ v[0]/m for v in trandelts ] ), sum( [ v[1]/m for v in trandelts ] ) ];
		#Solves dS/dt ?
		
		#the NEXT problem is dS/dajk...
		#Not sure what the hell this is;
		#Looks like there is some kind of giant matrix being formed and multiplied, then inverted...
		#
	
		stepsubs = [ psubtract( seta[i], t_w1 ) for i in range( len( seta ) ) ]; #sum( ( pi - t[w+1] ) ) ???
		
		#stepsubs is a list of vector ERRORS.
		#sooo... how to extract affine from THAT?
		#	
		
		#Now what?
		#Want to SOLVE FOR A_w1...
		#Well we have EVERYTHING HERE except A_w1
		#sum( transpose( Ejk * qi ) * A_w1 * qi ) = sum( transpose( Ejk*qi ) * ( pi - t_w1 ) )
		#Soooo... there is a SUM on both SIDES...
		#	can we even... what???
		#	transpose( Ejk * qi ) => LIST where qi == 1.
		#	Meaning...
		#		A(w+1)*qi = 1 *( pi - t_w1 );  ?? for EACH i ?
		#	So we have a SVD type overdetermined system... A has 4 variables ONLY.
		#	Hm...
		#	The heck is this garbage.
		#	
		#	WEll, the sum on the RIGHT hand side should be calculatable.
		#
		
		A_w1 = [
			1  #how to compute THIS?
			,0
			,0
			,1
			,t_w1[0]
			,t_w1[1]
		]
		
		Afmatrix = A_w1[:]
		
		#Ejk in 2D can only be [ 1,0,0,0 ], [0,1,0,0], [0,0,1,0], and [0,0,0,1]
		#So.  Matrix * vector == vector; then vector times vector == matrix???
		#
		
		#A[w+1] * qi = stepsubs
	
		#sum( transpose( Ejk * qi ) * A[w+1] * qi ) = sum( transpose( Ejk*qi ) * ( pi - t[w+1] ) )
		
		#	Given some starting A, set w = 0*
		#	Calculate the w'th:
		#		t[w+1] = 1/m * sum( pi - A[w] * qi );
		#	Calculate:
		#		Ejk is a SELECTION matrix, it only has a single 1 where it is j,k...
		#		sum( transpose( Ejk * qi ) * A[w+1] * qi ) = sum( transpose( Ejk*qi ) * ( pi - t[w+1] ) )
		#
		#	?? what is the convergence criteria???
		
	
	return Afmatrix


def compute_natural_direction( L ):
	"""
		Given a list of line segment DELTAS [ dx, dy ]
		Return the natural direction (dx, dy) of the bundle, the largest gap angle in radians, and the cut angle in radians
	"""
	def angle_rad(x,y):
		return math.atan2( y, x );
	
	angles = [];
	for seg in L:
		hx = seg[0]/2.0;
		hy = seg[1]/2.0;
		
		angles.append( ( angle_rad(-hx, -hy), 1, (-hx,-hy) ) );
		angles.append( ( angle_rad(hx, hy), 1, (hx,hy) ) ); 
		
	angles.sort( key=lambda x:x[0] );
	
	#Now, FIND the LARGEST GAP in this list (because of its SYMMETRY, we don't have to care about WRAPPING...
	pa = angles[0][0];
	i = 1;
	largesti = 0;
	largestgap = 0;
	largestgapv = (pa,pa);
	while i < len( angles ):
		a = angles[i][0];
		w = angles[i][1];
		da = (a - pa)
		if da > largestgap:
			largestgap = da;
			largestgapv = ( pa, a );
			largesti = max( 0, i-1 );
		pa = a;
		i += 1;
	
	#Excellent! Now we have the LARGEST GAP index.
	cutangle = (largestgapv[0] + largestgapv[1])/2.0;
	
	#Compute the CUT vector
	cx = math.cos( cutangle );
	cy = math.sin( cutangle );
	
	#Now the interesting part. we have WEIGHTS for EACH VECTOR.
	#Since we DID NOT want to pick a sidedness/bias, and instead opted for NATURAL DIRECTION,
	#Now we have to SPLIT the dataset by our cut vector, and then take the WEIGHTED vector averages of BOTH SIDES.
	#Then, we can take their angles, flip one by 180, and average to get the TRUE natural angle, weighted.
	#This will prevent short, tiny segments from biasing the results.
	#It will also HEAVILY bias longer deltas, which makes SENSE in a drawing context (larger gap == FASTER stroke, == more INTENT)
	lbinangles = [];
	rbinangles = [];
	i = 1;
	while i < len( angles ):
		dir = angles[i][2];
		if dir[0]*cx + dir[1]*cy < 0:
			lbinangles.append( i )
		else:
			rbinangles.append( i )
		i += 1;
	
	#Great! REcalculate WEIGHTED vector sum:
	lbinsum = [0,0];
	for i in lbinangles:
		lbinsum[0] += angles[i][2][0];
		lbinsum[1] += angles[i][2][1];
		
	rbinsum = [0,0];
	for i in rbinangles:
		rbinsum[0] += angles[i][2][0];
		rbinsum[1] += angles[i][2][1];
		
	#Do the angle calculation.
	langle = angle_rad(lbinsum[0], lbinsum[1]);
	rangle = angle_rad(rbinsum[0], rbinsum[1]);
	
	if langle < rangle:
		langle += math.pi;
	else:
		rangle += math.pi;
		
	avgbestangle = (rangle + langle)/2.0;
	
	#dx = math.cos( cutangle + math.pi/2 ); #Hmmm... is it ALWAYS +?
	#dy = math.sin( cutangle + math.pi/2 );
	
	dx = math.cos( avgbestangle );
	dy = math.sin( avgbestangle );

	return (dx, dy), largestgap, cutangle;

def bresenham(x0, y0, x1, y1):
	"""Yield integer coordinates on the line from (x0, y0) to (x1, y1).

	Input coordinates should be integers.

	The result will contain both the start and the end point.
	"""
	dx = x1 - x0
	dy = y1 - y0

	xsign = 1 if dx > 0 else -1
	ysign = 1 if dy > 0 else -1

	dx = abs(dx)
	dy = abs(dy)

	if dx > dy:
		xx, xy, yx, yy = xsign, 0, 0, ysign
	else:
		dx, dy = dy, dx
		xx, xy, yx, yy = 0, ysign, xsign, 0

	D = 2*dy - dx
	y = 0

	for x in range(dx + 1):
		yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
		if D >= 0:
			y += 1
			D -= 2*dx
		D += 2*dy


def compute_line_center( P ):

	ppx = P[0];
	ppy = P[1];
	ppz = P[2];
	
	sumlengths = 0;
	sumx = 0.0;
	sumy = 0.0;
	sumz = 0.0;
	
	maxpointi = len( P );
	pointi = 1;
	while pointi < maxpointi:
		
		p = P[ pointi ];
		dx = p[0] - ppx;
		dy = p[1] - ppy;
		dz = p[2] - ppz;
		dm = math.sqrt( dx*dx + dy*dy + dz*dz );
		
		if dm > 0 :
			sumlengths += dm;
			
			sumx += (p[0] + ppx)/(2.0*dm);
			sumy += (p[1] + ppy)/(2.0*dm);
			sumz += (p[2] + ppz)/(2.0*dm);
		
		ppx = p[0];
		ppy = p[1];
		ppz = p[2];
		
		pointi += 1;
	
	sumx *= sumlengths
	sumy *= sumlengths
	sumz *= sumlengths
	
	return [sumx, sumy, sumz], [];
	
def temp_compute_best_center_point( L ):

	pcenter = [0,0,0]
	Lcount = 1.0*len( L )
	for p in L:
		pcenter[0] += p[0]/Lcount
		pcenter[1] += p[1]/Lcount
		pcenter[2] += p[2]/Lcount
	return pcenter

def temp_compute_best_fit_plane( L ):
	
	#cheating:
	#https://blenderartists.org/t/calculating-a-best-fit-plane-to-vertices-solved/444182
	#and its not the best but I just need one that works. I'll get mine later.
	
	pcenter = temp_compute_best_center_point( L );
	
	#Note that, compute_line_center is better for LINE SEGMENT data, as opposed to POINT data.
	
	Sxx = 0.0
	Sxy = 0.0
	Sxz = 0.0
	Syz = 0.0
	Syy = 0.0
	Szz = 0.0
	#Now that we have the CENTER computed...
	for p in L:
		rx = p[0] - pcenter[0]
		ry = p[1] - pcenter[1]
		rz = p[2] - pcenter[2]
		#sum squared and products???
		Sxx += rx*rx;
		Sxy += rx*ry;
		Sxz += rx*rz;
		Syz += ry*rz;
		Syy += ry*ry;
		Szz += rz*rz;
	
	Scovariancematrix = mathutils.Matrix()#[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
	Scovariancematrix.Identity(3);
	
	Scovariancematrix[0][0] = Sxx
	Scovariancematrix[0][1] = Sxy
	Scovariancematrix[0][2] = Sxz
	
	Scovariancematrix[1][0] = Sxy
	Scovariancematrix[1][1] = Syy
	Scovariancematrix[1][2] = Syz
	
	Scovariancematrix[2][0] = Sxz
	Scovariancematrix[2][1] = Syz
	Scovariancematrix[2][2] = Szz
	
	# calculating the normal to the plane
	matrix_perfect = True;
	try:
		Scovariancematrix.invert()	#does not HAVE an inverse...
		matrix_perfect = False;
	except ValueError as e :
		pass;
	
	if not matrix_perfect:
		itermax = 500
		iter = 0
		vec = mathutils.Vector((1.0, 1.0, 1.0))
		vec2 = (vec @ Scovariancematrix)/(vec @ Scovariancematrix).length	#vec2 = (vec*Scovariancematrix)/(vec*Scovariancematrix).length
		#vec2 = (Scovariancematrix*vec)/(Scovariancematrix*vec).length
		while vec != vec2 and iter < itermax:
			iter+=1
			vec = vec2
			vec2 = (vec @ Scovariancematrix)/(vec @ Scovariancematrix).length	#vec2 = (vec*Scovariancematrix)/(vec*Scovariancematrix).length
			#vec2 = (Scovariancematrix*vec)/(Scovariancematrix*vec).length
		normal = vec2
		
		return normal, pcenter
	else:
	
		#this happens if ALL of the components along a SINGLE AXIS are exactly zero.
		#So, find that axis! easy.
		usenormal = [0,0,1]
		if Sxx < Syy:
			if Sxx < Szz:
				usenormal = [1,0,0]#axis x is 0
			else:
				pass;#axis z is 0
		elif Syy < Szz:
			usenormal = [0,1,0]#axis y is 0
		else:
			pass;#axis z is 0
	
		return usenormal, pcenter
	
def select_with_relevance( L, anglethresh = 45.0, distweight = 0.05, maxlength=math.inf, sharpcut=60.0 ):
	#
	#Resample stroke with even distance, paying attention to "sharp" points with large enough distance ooooor
	#
	#Lancoz filter? concavity check???
	#
	#



	#Okay FIRST we look for "high distance divergance" cuts.
	#THEN we look for "sharp" points (defined by sharpcut)
	#THEN with the remaining individual cuts, we work toward the center and do the angle sume select method...
	#
	R = []
	
	if True:
			
		imax = len( L );
		i = 0;
		ppx = L[i][0]
		ppy = L[i][1]
		pEp = L[-1];
		pdist = 0;
		pdistpflag = 0;
		pdistpratio = 0;
		
		R.append( (ppx, ppy, i) );
		i += 1;
		
		while i < imax:
			px = L[i][0];
			py = L[i][1];
			
			dx = (px - ppx);
			dy = (py - ppy);
			dist = dx*dx + dy*dy;
			dratio = 0;
			if pdist > 0:
				dratio = (dist / pdist)-1.0;
			
			if dratio < -0.125:
				if pdistpratio > 0: #was PREVIOUSLY increasing
					#divergence POINT
					R.append( (px, py, i) );
				pdistpflag = -1;#is now DECREASING
			elif dratio > 0.125:
				if pdistpratio < 0: #was PREVIOUSLY decreasing
					#divergence POINT
					R.append( (px, py, i) );
				pdistpflag = 1;#is now INCREASING
			
			pdistpratio = dratio;
				
			ppx = px;
			ppy = py;
			pdist = dist;
			i += 1;
			
		if R[-1][2] < imax-1:
			R.append( (L[-1][0], L[-1][1], imax-1) );
			
	#Okay, now we have R broken into DISTANCE divergence points...
	
	#
		
	return R
	
def regression_select_filter_velocity_dp( L, anglethresh = 30.0, distweight = 0.05, maxlength=math.inf ):
	#this one isnt very good either.
	#we REALLY care about "high interest" points, which is, where the DISTANCE diverges (it was decreasing now its increasing)
	#Finding THOSE points is a good start.
	#THEN you can do the "angular" filtering bit. (high distance or angle divergance points are "features" pretty much always...)
	#
	
	
	#Okay this one is a bit tricky...
	#Effectively, we want to run through our "velocity filter" first...
	#That should smooth things out well.
	#Then we go through and detect areas of large angle change per point?
	#Subdivide areas along the curve like that first
	#then for each subsection, 
	#	follow the curve with angle incrementing (absolute angle incrementing)
	#	Add a point every theta degrees and reset angle following?
	#	Do this from BOTH directions until the indexes cross, and meet halfway on the crossed indexes?
	#
	#

	RB = [];
	RE = [];
	
	radthresh = 0.01745329251994329576923690768489 * anglethresh; #pi/180
	
	#pB0 = L[0];
	#pE0 = L[-1];
	pBimax = len( L );
	pBi = 1;
	pEi = -2;
	pBp = L[0];
	pEp = L[-1];
	pBdp = None;#mathutils.Vector( (0,0,0) );
	pEdp = None;#mathutils.Vector( (0,0,0) );
	aBsum = 0;
	aEsum = 0;
	aBdsum = 0;
	aEdsum = 0;
	
	while pBi < pBimax:
	
		pBn = L[pBi]
		pEn = L[pEi]
		
		pBd = mathutils.Vector( ( pBn[0] - pBp[0], pBn[1] - pBp[1], 0.0) );
		pEd = mathutils.Vector( ( pEn[0] - pEp[0], pEn[1] - pEp[1], 0.0) );
		
		#HEY! this angle needs to be SIGNED dangit!!.. does it???
		#NO but it needs to remember the TOTAL difference in angle..
		#Problematic!
		def getanglebetween( A, B, xi=0, yi=1 ):
		
			dpE = (A[xi]*B[xi] + A[yi]*B[yi]);
			magsumEdp = math.sqrt(A[xi]*A[xi] + A[yi]*A[yi]);
			magsumEd = math.sqrt(B[xi]*B[xi] + B[yi]*B[yi]);
			magmul = magsumEdp*magsumEd;
			if magmul > 0:
				acterm = dpE/magmul
			else:
				acterm = 0;
			if acterm > 1:
				return 0;
			elif acterm < -1:
				return math.pi;
			else:
				return math.acos( acterm );
			
		
		#angle + distance weighting BETWEEN these vectors...
		delBa = 0;
		if pBdp == None:
			pBdp = pBd;
		else:
			#try:
			#	delBa = pBd.angle( pBdp );
			#except:
			#	delBa = 0;#math.pi;
			delBa = getanglebetween(pBd,pBdp);
			
		delEa = 0;
		if pEdp == None:
			pEdp = pEd;
		else:
			#try:
			#	delEa = pEd.angle( pEdp );
			#except:
			#	delEa = 0;#math.pi;
			delEa = getanglebetween(pEd,pEdp);
		
		#aBdsum += pBd.length / distweight;
		#aEdsum += pEd.length / distweight;
		aBdsum += pBd.length;
		aEdsum += pEd.length;
					
		#aBsum += abs( delBa );
		#aEsum += abs( delEa );
		aBsum = max( abs( delBa ), aBsum);
		aEsum = max( abs( delEa ), aEsum);
		
		addpointB = False;
		addpointE = False;
		
		#Its a function where aBdsum matters LESS as aBsum INCREASES...
		#	1.0 - min( 0, (aBsum - radthresh) )
		
		#if ( aBsum > radthresh and aBdsum >= 1.0 ) or ( aBsum > radthresh*2 ) or ( (aBdsum*distweight) > maxlength ):# and aBdsum >= ( 1.0 - 10.0*min( 0.0, (aBsum - radthresh) ) ):
		if ( aBsum > radthresh and aBdsum >= distweight ) or ( (aBdsum) > maxlength ):# and aBdsum >= ( 1.0 - 10.0*min( 0.0, (aBsum - radthresh) ) ):
			addpointB = True;
			#aBsum = 0.0;
			aBsum = 0;#abs( delBa );
			aBdsum = 0.0;
			pBdp = pBd;#None
			
		#if ( aEsum > radthresh and aEdsum >= 1.0 ) or ( aEsum > radthresh*2 ) or ( (aEdsum*distweight) > maxlength ):# and aEdsum >= ( 1.0 - 10.0*min( 0.0, (aEsum - radthresh) ) ):
		if ( aEsum > radthresh and aEdsum >= distweight ) or ( (aEdsum) > maxlength ):# and aEdsum >= ( 1.0 - 10.0*min( 0.0, (aEsum - radthresh) ) ):
			addpointE = True;
			#aEsum = 0.0;
			aEsum = 0;#abs( delEa );
			aEdsum = 0.0;
			pEdp = pEd;#None;
			
		if pBimax + pEi < pBi:
			#we have met in the middle. Do we add this point? (compare the two directions)
			
			if addpointB or addpointE or ( ( (aEsum + aBsum) > radthresh and (aEdsum+aBdsum) >= 1.0 ) or ( (aEdsum+aBdsum) > radthresh*2 ) ):# and aBdsum >= ( 1.0 - 10.0*min( 0.0, (aBsum - radthresh) ) ):
				RB.append( (pBn[0], pBn[1], pBi) );
			
			break;
			
		else:
			if addpointB:
				RB.append( (pBn[0], pBn[1], pBi) );
			
			if addpointE:
				RE.insert( 0, (pEn[0], pEn[1], pBimax + pEi) );
			
		
		#if abs( delBa ) > 0:
		#	pBdp = pBd
			
		#if abs( delEa ) > 0:
		#	pEdp = pEd
		
		pBp = pBn;
		pEp = pEn;
	
		pBi += 1;
		pEi -= 1;
	
	RB.extend( RE );
	
	#Second pass:
	#LARGE deltas in features need to be cut up by distweight
	#However, to do this CONSISTENTLY is a bit impossibleness.
	#Same with the FORWARD ONLY angular divisions...
	#Hm...
	
	
	
	return RB;
	
	
def allocate_python_numeric_array( size=32, type='d', default = 0 ):
	return array.array( type, [ default for v in range( size ) ] )
	
class TempStroke:
	"""
		Represents a single stroke created from some interface, in this case blender stroke data is converted into a 6 dimensional vector.
		Should work for any x,y,z, pressure, strength stroke data.
	
	"""

	def __init__(self, points, original_index):
		self.original_index = original_index
		self.bounds = None;
		self.original_points = [] 
		self.points = [] 
		self.load( points );
	
	def load( self, Spoints ):
		self.original_points = [];
		self.points = [];
		if len( Spoints ) > 0:
			pointi = 0;
			pointimax = len( Spoints );
			while pointi < pointimax:
				p = Spoints[ pointi ];
				self.original_points.append( ( p.co[0], p.co[1], p.co[2], p.pressure, p.strength, pointi ) );
				self.points.append( ( p.co[0], p.co[1], p.co[2], p.pressure, p.strength, pointi ) );
				pointi += 1;
				

class TempFrame:
	"""
		Class meant to represent a ordered set of strokes.
		
		Handles detecting and computing features and anything about a "frame" of data.
	"""
	def __init__(self, framesource = None, framesourceindex = None ):
		self.strokes = [];
		self.bounds = None;
		self.original_index = framesourceindex;
		self.frame_number = -1;
		if framesource != None:
			self.load( framesource )
		
		self.center = []
		self.center_bounds = [0,0,0]
		self.center_normal = [ 0, 0, 1 ]
		self.center_left = [ -1, 0, 0 ]
		self.center_up = [ 0, 1, 0 ]
		
		self.center2d_bounds = [ 0, 0, 0 ]
		self.center2d = [0,0,0];
		self.center2d_rotation = 0;
		self.center2d_unitbound = 1;
		
	def load( self, A ):
		self.frame_number = A.frame_number;
		self.strokes = [];
		strokei = 0;
		strokeimax = len( A.strokes );
		while strokei < strokeimax:
			self.strokes.append( TempStroke( A.strokes[ strokei ].points, strokei ) );
			strokei += 1;
			
	def compute_center_projection( self ):
	
		#Requires center. (3D center that is)
		allpoints = [];
		#strokecenters = [];
		for stroke in self.strokes:
		#	strokecenters.append( compute_line_center( stroke.points ) );
			for point in stroke.points:
				allpoints.append( point );
				
		#strokecenters ?
		
		#Get the best fit plane normals to this data
		normal, pcenter = temp_compute_best_fit_plane( allpoints );
		
		self.center = mathutils.Vector( ( pcenter[0],pcenter[1],pcenter[2] ) )
		self.center_normal = mathutils.Vector( (normal[0], normal[1], normal[2]) )
		
		#this is NOT ok for a "left" by default.
		allpointsavg = [0,0,0];
		for stroke in self.strokes:
			
			#Just a point average. AKA, the "center" so it deforms weirdly.
			#for point in stroke.points:
			#	dv = ( (point[0] - self.center[0]), (point[1] - self.center[1]), (point[2] - self.center[2]) )
			#	allpointsavg[0] += dv[0]
			#	allpointsavg[1] += dv[1]
			#	allpointsavg[2] += dv[2]
			
			#weighted average of delta line segments is more stable, but its still weird.
			pointimax = len( stroke.points )
			pp = stroke.points[ 0 ]
			pointi = 1;
			while pointi < pointimax:
				p = stroke.points[ pointi ]
				
				dv = ( p[0] - pp[0], p[1] - pp[1], p[2] - pp[2] )
				allpointsavg[0] += dv[0];
				allpointsavg[1] += dv[1];
				allpointsavg[2] += dv[2];
				
				pp = p;#( p[0], p[1], p[2] );
				pointi += 1;
				
		#This is where we need the MEDIAN slope algorithm...
		#Or something else? bounding convex hull in 2D?
			
		#Compute an "up" vector... determine by looking at bounding box LARGEST axis?
		#svA = A.get_bounds_size_vector();
		tanvec = mathutils.Vector( ( allpointsavg[0], allpointsavg[1], allpointsavg[2] ) ).normalized()
		
		sidedir = self.center_normal.cross( tanvec ).normalized();#default to BOUNDS axis...
		newup = sidedir.cross( self.center_normal ).normalized();
		
		#Basis is now; perp: self.center_normal, up: newup. left:sidedir
		
		#print( self.center, tanvec, sidedir, newup, self.center_normal );
		
		self.center_left = sidedir
		self.center_up = newup
		
	
	def apply_center_projection( self ):
		
		for stroke in self.strokes:
			pointi = 0;
			pointimax = len( stroke.points );
			while pointi < pointimax:
				
				p = stroke.points[ pointi ]
				
				pv = mathutils.Vector( ( p[0], p[1], p[2] ) ) - self.center
				
				#Remove parts of pv ALONG normal
				pv = pv - ( pv.dot( self.center_normal ) )*self.center_normal
				
				#Project into the 2D plane defined by 
				stroke.points[ pointi ] = ( pv.dot( self.center_left ), pv.dot( self.center_up ), 0, p[3], p[4], p[5] )
				
				pointi += 1;
				
	def compute_2d_linecenter( self ):
		
		sumlengths = 0.0;
		sumx = 0.0;
		sumy = 0.0;
		
		sumslopexx = 0.0;
		sumslopexy = 0.0;
		
		sumslopeyx = 0.0;
		sumslopeyy = 0.0;
		
		
		for stroke in self.strokes:
			
			pointimax = len( stroke.points );
			
			if pointimax > 0 :
				
				pointi = 0;
				ppx = stroke.points[pointi][0];
				ppy = stroke.points[pointi][1];
				
				pointi = 1;				
				while pointi < pointimax:
					
					p = stroke.points[ pointi ];
					dx = p[0] - ppx;
					dy = p[1] - ppy;
					dm = math.sqrt( dx*dx + dy*dy );
					
					if dm > 0 :
						sumlengths += dm;#(1.0/dm);
						
					pointi += 1;
						
		for stroke in self.strokes:
			
			pointimax = len( stroke.points );
			
			if pointimax > 0 :
				
				pointi = 0;
				ppx = stroke.points[pointi][0];
				ppy = stroke.points[pointi][1];
				
				pointi = 1;				
				while pointi < pointimax:
					
					p = stroke.points[ pointi ];
					dx = p[0] - ppx;
					dy = p[1] - ppy;
					dm = math.sqrt( dx*dx + dy*dy );
					
					if dm > 0 :
						#sumlengths += dm;#(1.0/dm);
						
						sumx += (p[0] + ppx)*(dm/sumlengths)/2.0;
						sumy += (p[1] + ppy)*(dm/sumlengths)/2.0
						
						if dx < 0:
							sumslopexx += -dx/sumlengths;
							sumslopexy += -dy/sumlengths;
						else:
							sumslopexx += dx/sumlengths;
							sumslopexy += dy/sumlengths;
						
						if dy < 0:
							sumslopeyx += -dx/sumlengths;
							sumslopeyy += -dy/sumlengths;
						else:
							sumslopeyx += dx/sumlengths;
							sumslopeyy += dy/sumlengths;
						
					ppx = p[0];
					ppy = p[1];
					
					pointi += 1;
		
		
		return [sumx, sumy], [sumslopexx, sumslopexy], [sumslopeyx, sumslopeyy];
		
	def get_points_2D( self ):
		R = [];
		Rmap = [];
		strokei = 0;
		for stroke in self.strokes:
			pointi = 0;
			for point in stroke.points:
				R.append( ( point[0], point[1] ) );
				Rmap.append( ( strokei, pointi ) )
				pointi += 1;
			strokei += 1;
		return R, Rmap
		
	def reproject_2d_points( self, center, xaxis ):
	
		#normalize x axis
		#xaxislen = math.sqrt( xaxis[0]*xaxis[0] + xaxis[1]*xaxis[1] );
		#Xx = xaxis[0]/xaxislen;
		#Xy = xaxis[1]/xaxislen;
	
		#Make points relative to center
		vcenter = mathutils.Vector( ( center[0], center[1], 0 ) );
		vxaxis = mathutils.Vector( ( xaxis[0],  xaxis[1], 0 ) );
		vxaxis.normalize();
		vyaxis = mathutils.Vector( ( -vxaxis[1], vxaxis[0], 0 ) );
		vyaxis.normalize();
		
		#Compute MAXIMUM DISTANCE from center
		pmmax = 0.0;
		for stroke in self.strokes:
			
			pointi = 0;
			pointimax = len( stroke.points );
			while pointi < pointimax:
				
				p = stroke.points[ pointi ]
				
				pv = mathutils.Vector( ( p[0], p[1], 0.0 ) ) - vcenter;
				
				rx = pv.dot( vxaxis )
				ry = pv.dot( vyaxis )
				
				pm = math.sqrt( rx*rx + ry*ry );
				if pm > pmmax:
					pmmax = pm;
				
				#Project relative to the 2D axis
				stroke.points[ pointi ] = ( rx, ry, 0.0, p[3], p[4], p[5] )
				
				pointi += 1;
				
		#Scale to -1..1 ??
		if pmmax > 0:
			pmscale = 1.0/pmmax;
			for stroke in self.strokes:
				pointi = 0;
				pointimax = len( stroke.points );
				while pointi < pointimax:
					p = stroke.points[ pointi ]
					stroke.points[ pointi ] = ( pmscale*p[0], pmscale*p[1], 0.0, p[3], p[4], p[5] )
					pointi += 1;
		
		return vcenter, vxaxis, vyaxis, pmmax;
		
	def compute_2d_bounds( self ):
	
		#Determine maximum extents FROM a geocenter:
		self.center2d_bounds = [0,0,0,0,0,0];
		for stroke in self.strokes:
			for point in stroke.points:
				dx = point[0] - self.center2d[0]; #Point is IN projected plane space.
				dy = point[1] - self.center2d[1];
				dz = point[2] - self.center2d[2];
				self.center2d_bounds[0] = min( self.center2d_bounds[0], dx )
				self.center2d_bounds[1] = min( self.center2d_bounds[1], dy )
				self.center2d_bounds[2] = min( self.center2d_bounds[2], dz )
				self.center2d_bounds[3] = max( self.center2d_bounds[3], dx )
				self.center2d_bounds[4] = max( self.center2d_bounds[4], dy )
				self.center2d_bounds[5] = max( self.center2d_bounds[5], dz )
	
				
	def detect_2d_features( self, anglethresh=45, distance = 0.05, maxlength=0.25 ):
	
		#GREATLY improved feature detection would include the following:
		#Compute total path distance
		#resample evenly in distance as best as possible, but always including source points?
		#Computing stuff along the curve (eigenvalues? aka wideness/sharpness)
		#huh -> see Polygon Morphing.pdf
		#
		#
	
		#For each stroke, add a feature for the beginning and end of the stroke.
		#Filter stroke (we filter on "velocity" IE points nearby are filtered MORE than points farther away...)
		#
		features = [];
		
		features_per_stroke = [];
		
		strokeimax = len( self.strokes );
		strokei = 0;
		while strokei < strokeimax:
			originalstroke = self.strokes[ strokei ];  
			
			FS = [];
			
			#Add beginning and end of stroke (always)
			if len( originalstroke.points ) > 0:
				FS.append( len( features ) );
				features.append( ( originalstroke.points[0][0], originalstroke.points[0][1], strokei, 0 ) )
				
			#Create filtered stroke? with SAME # of points? (essential)
			#
			strokepoints = originalstroke.points;
			
			#Look for feature points to care about
			sra = regression_select_filter_velocity_dp( strokepoints, anglethresh=anglethresh, distweight = distance, maxlength = maxlength );
			
			#sra = select_with_relevance( strokepoints, anglethresh=anglethresh, distweight = distance, maxlength = maxlength, sharpcut=60.0 );
			
			#Hm. need to break apart TOO long of a feature if they exist...
			#	Use maxlength value?
			
			srai = 0;
			while srai < len( sra ):
				spridex = sra[ srai ][2];
				FS.append( len( features ) );
				features.append( ( strokepoints[spridex][0], strokepoints[spridex][1], strokei, spridex ) )
				srai += 1;
				
			if len( originalstroke.points ) > 0:
				FS.append( len( features ) );
				features.append( ( originalstroke.points[-1][0], originalstroke.points[-1][1], strokei, len( originalstroke.points ) -1 ) )
			
			#FS needs to be sorted...
			
			features_per_stroke.append( FS ); #JUST the feature indices.
			
			strokei += 1;
		
		#Probably needs intersections added too...

		return features, features_per_stroke
		
	def clean_and_nodalize( self, quantize_factor = 100 ):
		pass;
		
		#Maybe its more efficient to "Bezierify" each stroke first.
		#THEN quantize, THEN compute intersections (because we can do that with quad beziers)
		#Hm...
	
		#Step 1: Convert ALL inputs into integer coordinates.
		#	v = math.floor( v * quantize_factor );
		
		#Step 2: Shift integer coordinates to be + only
		
		#Step 3: Remove all duplicate points (this is easy enough)
		
		#Step 4: rasterize quantized points seta[i] - nham (so we know WHAT LINES cross WHAT CELLS)
		
		#Step 5: this allows us to create NEW edges based on ANY pixel that has same / overlapped coordinates.
		
		#Step 6: Construct all nodes and edges needed.
		
		#Now we have completely cleaned the input. It is guaranteed NON overlapping, it has NODES and EDGES.
		
		#This information THEN FORMS closed polygons (sometimes)
		#	Close each POLY and mark it as such.
		
		#Now we have nodes, edges, and closed polygons formed from edges.
		
		#
		
def bisect_left(a, x, lo=0, hi=None):
	"""Return the index where to insert item x in list a, assuming a is sorted.
	The return value i is such that all e in a[:i] have e < x, and all e in
	a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will
	insert just before the leftmost x already there.
	Optional args lo (default 0) and hi (default len(a)) bound the
	slice of a to be searched.
	"""
	if hi is None:
		hi = len(a)
	while lo < hi:
		mid = (lo+hi)//2
		if x < a[mid]: hi = mid
		else: lo = mid+1
	return lo
	
def remap_feature_points( V, VS, q=4096.0 ):
	"""
		input must be -1..1 inclusive, V is a list of feature (x,y,...) and VS is a list of lists of indices into V (stroke features)
		output returns the remapped feature and stroke feature set (so there are NO duplicates...)
	"""
	srange = q;
	smin = -1;
	
	if True:
		minx = V[0][0];
		maxx = V[0][0];
		miny = V[0][1];
		maxy = V[0][1];
		for v in V:
			minx = min( v[0], minx );
			maxx = min( v[0], maxx );
			miny = min( v[1], miny );
			maxy = min( v[1], maxy );
			
		xrange = maxx - minx;
		yrange = maxy - miny;
		srange = q / max( 1.0, max( xrange, yrange ) );
		smin = min( -1.0, min( minx, miny ) );
		
	feats = [];
	uniquepoints = {};
	voriginaltofeatindex = {};
	featstoindex={};
	for vindex in range(len(V)):
		v = V[ vindex ]
		xq = math.floor( (v[0] - smin)*srange - q )/q;
		yq = math.floor( (v[1] - smin)*srange - q )/q;
		pair = ( xq, yq );
		if pair in uniquepoints:
			uniquepoints[ pair ].append( vindex );
			voriginaltofeatindex[ vindex ] = featstoindex[ pair ]
		else:
			uniquepoints[ pair ] = [ vindex ]
			voriginaltofeatindex[ vindex ] = len(feats)
			featstoindex[ pair ] = len( feats );
			feats.append( pair );
			
	strokefeats = [];
	for s in VS:
		msl = []
		for fi in s:
			msl.append( voriginaltofeatindex[ fi ] );	#wait... this doesnt make much sense really...
		strokefeats.append( msl );
			
	return feats, strokefeats, ( voriginaltofeatindex, uniquepoints, featstoindex )

	
def x_voronoi_compute_voronoi_rep( V, VS ):

	usetriangles = [];
	segments = [];
	if True:
		
		for stroke in VS:
			for ip in range(len(stroke)):
				if ip != 0:
					#segments.append( ( stroke[ip-1], stroke[ip] ) );
					segments.append( ( min( stroke[ip-1], stroke[ip] ), max( stroke[ip-1], stroke[ip] ) ) );
					
		#for v
		#quantverts = [ ((math.floor(v[0]*8192.0)/4096.0), (math.floor(v[1]*8192.0)/4096.0), 0) for v in V ]
		#quantverts = [ (v[0], v[1]) for v in V ]
		
		dt = tri_delaunay.triangulate( V, segments );  #well this sucks. whats the point of constrained, if it doesn't preserve indexing?!?!
		
		#dt.triangles
		#dt.vertices
		#dt.external
		
		#Did our verts get resorted?? Hm... should be since they represent.
		
		plookupdict = {}
		for pointi in range(len(V)):
			coordpair = ( V[pointi][0], V[pointi][1] );
			if coordpair in plookupdict:
				plookupdict[ coordpair ].append( pointi );
				print( "WARNING: invalid input." )
			else:
				plookupdict[ coordpair ] = [ pointi ]
		
		def getbestvertex( x, y ):
			coordpair = (x,y);#((math.floor(x*8192)/4096.0), (math.floor(y*8192)/4096.0))
			if coordpair in plookupdict:
				poptions = plookupdict[ coordpair ];
				if len( poptions ) > 1:
					print( "DUPLIVERT: " );
				return plookupdict[ coordpair ][0];
			else:
				#Okay... we know something about "infinite" vertices here.
				#So long as they ARE infinite vertices, I'm OK with that.
				if abs(coordpair[0]) > 20 or abs(coordpair[1]) > 20:	#Not sure where this comes from...
					pass;#print( "INF VERTEX: ", coordpair );pass;	#was a INFINITE vertex...
				else:
					print( "BAD VERTEX: ", coordpair );
			return None;
		
		for tris in dt.triangles:
			trio = ( 
				getbestvertex( tris.vertices[0].x, tris.vertices[0].y )
				, getbestvertex( tris.vertices[1].x, tris.vertices[1].y )
				, getbestvertex( tris.vertices[2].x, tris.vertices[2].y )
			)
			if trio[0] != None and trio[1] != None and trio[2] != None:
				usetriangles.append( trio );
			else:
				pass;#print( "BAD TRIS: ", trio ); #"infinite" triangle may contain edges? Hm...
		
			#logging.debug( str(len(dt.vertices)) + " vertices")
			#logging.debug( str(len(dt.triangles)) + " triangles")
		
	#V is feature list, Vsl is site list, Vtx is tris context
	edges = {};#map edge (min, max) -> dict of triangle indexes -> starting triangle vertex index
	triss = {};#map triangle -> dict of edges -> starting triangle vertex index
	verts = {};#map vertex -> dict of edges -> bit for feature edge??
	
	featureedges = {}
	for s in segments:
		featureedges[ s ] = {}	#known feature edge -> dict of triangles it touches?
	
	trisi = 0;
	for tris in usetriangles:
		e0 = ( min(tris[0], tris[1]), max(tris[0], tris[1]) )
		e1 = ( min(tris[1], tris[2]), max(tris[1], tris[2]) )
		e2 = ( min(tris[0], tris[2]), max(tris[0], tris[2]) )
		
		if not e0 in edges:
			edges[ e0 ] = {};
		if not e1 in edges:
			edges[ e1 ] = {};
		if not e2 in edges:
			edges[ e2 ] = {};
		edges[ e0 ][ trisi ] = tris[0];
		edges[ e1 ][ trisi ] = tris[1];
		edges[ e2 ][ trisi ] = tris[2];
		
		triss[ trisi ] = {};
		triss[ trisi ][e0] = tris[0];
		triss[ trisi ][e1] = tris[1];
		triss[ trisi ][e2] = tris[2];
		
		if not tris[0] in verts:
			verts[ tris[0] ] = {};
		if not tris[1] in verts:
			verts[ tris[1] ] = {};
		if not tris[2] in verts:
			verts[ tris[2] ] = {};
		
		if not e0 in verts[ tris[0] ]:
			verts[ tris[0] ][ e0 ] = False;
		if not e1 in verts[ tris[1] ]:
			verts[ tris[1] ][ e1 ] = False;
		if not e2 in verts[ tris[2] ]:
			verts[ tris[2] ][ e2 ] = False;
		
		if e0 in featureedges:
			featureedges[ e0 ][ trisi ] = tris[0];
			verts[ e0[0] ][ e0 ] = True
			verts[ e0[1] ][ e0 ] = True
		if e1 in featureedges:
			featureedges[ e1 ][ trisi ] = tris[1];
			verts[ e1[0] ][ e1 ] = True
			verts[ e1[1] ][ e1 ] = True
		if e2 in featureedges:
			featureedges[ e2 ][ trisi ] = tris[2];
			verts[ e2[0] ][ e2 ] = True
			verts[ e2[1] ][ e2 ] = True
		
		trisi += 1;
		
	#for fe in featureedges:
	#	verts
	
	return edges, triss, verts, featureedges #, Vsl, Vtc
	
	
def _xconcave_hull_from( points, vvmap=None, startingpointidx=0, maxdistance=0.1 ):
	"""
	2 Find the k-nearest points to the current point.
	3 From the k-nearest points, select the one which corresponds to the largest right-hand turn from the previous angle. Here we will use the concept of bearing and start with an angle of 270 degrees (due West).
	4 Check if by adding the new point to the growing line string, it does not intersect itself. If it does, select another point from the k-nearest or restart with a larger value of k.
	5 Make the new point the current point and remove it from the list.
	6 After k iterations add the first point back to the list.
	7 Loop to number 2.
	"""
	R = [];
	R.append( startingpointidx );
	
	curri = startingpointidx;
	
	maxdsq = maxdistance*maxdistance;
	
	#Eh screw it.
	def _vert_dist_sq( A, B ):
		return A[0]*A[0] + B[0]*B[0]
		
	def _closest_points_within_sq( i, maxdistancesq, distancemap={}, alreadyvisited={} ):
		nearest_index = 0;
		newdistmap = {};
		for othervertid in vvmap[ i ]:
			if not i in alreadyvisited:
				newdistmap[ i ] = _vert_dist_sq( points[ i ], points[ othervertid ] );
		
		for vk in newdistmap:
			distancemap[ vk ] = newdistmap[ vk ];
			if not vk in alreadyvisited:
				alreadyvisited[ vk ] = True;
				if distancemap[ vk ] < maxdistancesq:
					distancemap, alreadyvisited = _closest_points_within_sq( vk, maxdistancesq, distancemap, alreadyvisited );
		return distancemap, alreadyvisited;
	
	
	dmap, already = _closest_points_within_sq( curri, maxdsq );#"distance to neighbor points"
	#connect to closest angle?
	
	print( dmap, already );
	
	return {}
	
def x_voronoi_classitris( feats, strokefeats, edges, triss, vverts, vfeatureedges ):

	featedges = {};
	for strokeidx in range( len( strokefeats ) ):
		stroke = strokefeats[strokeidx]
		iprev = None;
		ipoint = 0;
		for featidx in range( len( stroke ) ):
			feati = stroke[ featidx ];
			if iprev != None:
				featedges[ ( min( feati, iprev ), max( feati, iprev ) ) ] = ( strokeidx, featidx )
			iprev = feati;

	R = {}
	for trisidx in triss:
		trisclass = 0;
		for edge in triss[ trisidx ]:
			edgekey = ( min(  edge[0],  edge[1] ), max(  edge[0],  edge[1] ) )
			if edgekey in featedges:
				trisclass += 1;
		R[trisidx] = trisclass;
	return R;
		
	
def x_voronoi_make_alpha_shape_tree( feats, strokefeats, edges, triss, vverts, vfeatureedges ):

	#Extract CONVEX hull from voronoi representation?
	#Then remove all triangles connected 1 deep? (remove strip)
	#Repeat? lol.
	#	this WILL guarantee a "level curve" representation which is kinda neat.
	#	so you'd just get "edge loops" that sort of represent the "information height" ?
	#	AND you are guaranteed for them to not overlap...
	#	Plus, as a consistent representation, we get a tree like structure to represent it?
	#


	#Alright, this can't work as implemented (Because it does NOT respect holes, thanks to intersection issues.)
	
	#So, what do we do here.
	
	#1, we can easily get the OUTER hull edges...
	#But then...
	#
	#edges[ edge_key ][ tris_idx ] = vertex index
	#triss[ tris_idx ][ edge_key ] = triangle index
	#vverts[ tris_idx ][ edge_key ] = boolean for feature edge...
	#
	#edges = {};#map edge (min, max) -> dict of triangle indexes -> starting triangle vertex index
	#triss = {};#map triangle -> dict of edges -> starting triangle vertex index
	#vverts = {};#map vertex -> dict of edges -> bit for feature edge??
	
	#find any extrema point, start there.
	#Then we can find a "containment angle" to rotate to (with a distance clamp?)
	
	#Find extrema points
	xminidx = 0;
	xmaxidx = 0;
	yminidx = 0;
	ymaxidx = 0;
	for fi in range( len( feats ) ):
		if feats[fi][0] < feats[xminidx][0]:
			xminidx = fi;
		if feats[fi][0] > feats[xmaxidx][0]:
			xmaxidx = fi;
		if feats[fi][1] < feats[yminidx][1]:
			yminidx = fi;
		if feats[fi][1] > feats[ymaxidx][1]:
			ymaxidx = fi;
	
	#Pick a point from LARGEST extrema:
	extremadx = feats[xmaxidx][0] - feats[xminidx][0];
	extremady = feats[ymaxidx][1] - feats[yminidx][1];
	
	startingpoint = 0;
	if abs( extremadx ) > abs( extremady ):
		startingpoint = xminidx
	else:
		startingpoint = yminidx
	
	vvmap = {};
	for vertkey in vverts:
		if not vertkey in vvmap:
			vvmap[ vertkey ] = {};
		for edgekey in vverts[ vertkey ]:
			vvmap[ vertkey ][ edgekey[0] ] = edgekey[0];
			vvmap[ vertkey ][ edgekey[1] ] = edgekey[1];
	
	#Now we can begin:
	hull = _xconcave_hull_from( feats, vvmap, startingpoint, maxdistance=0.25 );
	
	#Make pairs of line segments from the hull?
	
	return hull;
	
	
	#def generate_debug_triangles_on_frame( C, VE, VF, VT, T, zlevel=0, rescale=(1,1,0,0) ):
	#generate_debug_triangles_on_frame( C, edges, Afeatures, triss, triss, zlevel=-1, rescale=Arescale );
		
	#1. classify ALL edges:
	#	0 == FIXED, actually part of a feature edge
	#	1+ == DEPTH from OUTSIDE of shape (edges with 1 are the alpha 1 shape, 2 is alpha 2 shape, etc...
	#
	featedges = {};
	for strokeidx in range( len( strokefeats ) ):
		stroke = strokefeats[strokeidx]
		iprev = None;
		ipoint = 0;
		for featidx in range( len( stroke ) ):
			feati = stroke[ featidx ];
			if iprev != None:
				featedges[ ( min( feati, iprev ), max( feati, iprev ) ) ] = ( strokeidx, featidx )
			iprev = feati;
	
	trisedges = {};
	triangle_to_real_edges = {};
	edge_to_triangles = {};
	for trisidx in range( len( triss ) ):
		tris = triss[ trisidx ];
	
		nrealedges = 0;
		edgeidx = 0;
		for edge in triss[ trisidx ]:
			edgekey = ( min(  edge[0],  edge[1] ), max(  edge[0],  edge[1] ) )
			if edgekey in featedges:
				nrealedges += 1;
			else:
				trisedges[ edgekey ] = ( trisidx, edgeidx );
				
			if edgekey in edge_to_triangles:
				pass;
			else:
				edge_to_triangles[ edgekey ] = {}
			edge_to_triangles[ edgekey ][ trisidx ] = edgeidx;
				
			edgeidx += 1;
		triangle_to_real_edges[ trisidx ] = nrealedges;
		
	triangle_to_connections = {}; #Maps triangle idx -> # of shared edges with other triangles
	current_edge_triangles = {};
	trisidx = 0;
	for trisidx in range( len( triss ) ):
		tris = triss[ trisidx ];
	
		otherkeycount = 0;
				
		for edge in triss[ trisidx ]:
			edgekey = ( min(  edge[0],  edge[1] ), max(  edge[0],  edge[1] ) )
				
			for otherkey in edge_to_triangles[ edgekey ]:
				if otherkey != trisidx:
					otherkeycount += 1;
		
		if otherkeycount == 2:
			current_edge_triangles[ trisidx ] = otherkeycount;
		triangle_to_connections[ trisidx ] = otherkeycount
			
	#
	triangles_left = {};
	for trisidx in triangle_to_connections:
		triangles_left[ trisidx ] = triangle_to_connections[ trisidx ];
	
	#
	#Okay, now for the interesting parts:
	unchecked_edge_triangles = {};
	for trisidx in current_edge_triangles:
		unchecked_edge_triangles[ trisidx ] = trisidx;
		
	def _find_outer_edge_key( trisidx ):
		
		outer_edge_key = None;
		
		for edge in triss[ trisidx ]:
			edgekey = ( min(  edge[0],  edge[1] ), max(  edge[0],  edge[1] ) )
			
			otherkeycount = 0;
			for otherkey in edge_to_triangles[ edgekey ]:
				otherkeycount += 1;
				
			if otherkeycount == 1:
				for otherkey in edge_to_triangles[ edgekey ]:
					if otherkey == trisidx:
						outer_edge_key = edgekey;
						break;
						
		return outer_edge_key;
		
	#	Iterate around the current_edge_triangles (keep note!)
	#
	#
	real_outer_edges = {};
	
	for i in range( 5 ):# True:
	
		next_unchecked_edge_triangles = {};
		defeinitely_remove_these = {};
		
		for trisidx in unchecked_edge_triangles:
		
			#Now, WHICH edge is it?
			outer_edge_key = _find_outer_edge_key( trisidx );
			
			if outer_edge_key != None:
			
				#We have an outer edge!
				#HOWEVER, it connects to TWO TRIANGLES.
				#If that outer edge is NOT a featedges
				if outer_edge_key in featedges:
					real_outer_edges[ outer_edge_key ] = outer_edge_key;  #MUST be considered.
					pass;
				else:
					#Hold it. Do we have ANY other feature edges? we should not traverse INTO feature edges.
				
				
					#Well, THIS "outer" triangle should be removed, and the two it is connected to BECOME outer edges.
					
					#Oh? we can REMOVE this triangle and add the TWO it is connected to, to our iteration?
					#And we have to re-update the edge data...
					#since we REMOVED a triangle and all that...
					
					othertriangles = {};
					for edge in triss[ trisidx ]:
						edgekey = ( min(  edge[0],  edge[1] ), max(  edge[0],  edge[1] ) )
						
						if edgekey in featedges:
							real_outer_edges[ edgekey ] = edgekey;  #MUST be considered.
							pass;
						else:
							
							if trisidx in edge_to_triangles[ edgekey ]:
								del edge_to_triangles[ edgekey ][ trisidx ];
							for trikey in edge_to_triangles[ edgekey ]:
								
								othertriangles[ trikey ] = trikey
								
					for nexttrikey in othertriangles:
						if nexttrikey != trisidx:
							#nextouter_edge_key = _find_outer_edge_key( nexttrikey );
							#if nextouter_edge_key != None:
								next_unchecked_edge_triangles[ nexttrikey ] = nexttrikey;
								
					defeinitely_remove_these[ trisidx ] = trisidx;
			else:
				#Doesnt have any outer edges??? so ignore it?? shouldnt happen...
				#Must be a INTERIOR triangle... Hm..
				pass;
				
		for killthese in defeinitely_remove_these:
			if killthese in next_unchecked_edge_triangles:
				del next_unchecked_edge_triangles[ killthese ];
			
		unchecked_edge_triangles = next_unchecked_edge_triangles;
		
	"""
	for trisidx in current_edge_triangles:
	
		#Now, WHICH edge is it?
		outer_edge_key = None;
		
		for edge in triss[ trisidx ]:
			edgekey = ( min(  edge[0],  edge[1] ), max(  edge[0],  edge[1] ) )
			
			otherkeycount = 0;
			for otherkey in edge_to_triangles[ edgekey ]:
				otherkeycount += 1;
				
			if otherkeycount == 1:
				for otherkey in edge_to_triangles[ edgekey ]:
					if otherkey == trisidx:
						outer_edge_key = edgekey;
						break;
						
		if outer_edge_key != None:
			real_outer_edges[ outer_edge_key ] = trisidx;
	"""
	
	#Okay, but, a outer triangle COULD be one that can be removed...
	#	Remove it, if it's outer edge does NOT contain a REAL edge...
	#
		
	#Now we have TRIANGLE and FEATURE edges.
	
	#Now we have to look for TRIANGLES that have only 2 shared triangle edges => classify.
	return real_outer_edges;
	
def x_voronoi_convex_ring_tree( feats, strokefeats, edges, triss, vverts, vfeatureedges ):

	#Extract CONVEX hull from voronoi representation?
	#Then remove all triangles connected 1 deep? (remove strip)
	#Repeat? lol.
	#	this WILL guarantee a "level curve" representation which is kinda neat.
	#	so you'd just get "edge loops" that sort of represent the "information height" ?
	#	AND you are guaranteed for them to not overlap...
	#	Plus, as a consistent representation, we get a tree like structure to represent it?
	#
	R = [];
	
	"""
	featedges = {};
	for strokeidx in range( len( strokefeats ) ):
		stroke = strokefeats[strokeidx]
		iprev = None;
		ipoint = 0;
		for featidx in range( len( stroke ) ):
			feati = stroke[ featidx ];
			if iprev != None:
				featedges[ ( min( feati, iprev ), max( feati, iprev ) ) ] = ( strokeidx, featidx )
			iprev = feati;
	"""
	
	edge_to_triangles = {};#we MODIFY this as we remove triangles...
	for tkey in triss:
		for edge in triss[ tkey ]:
			edgekey = ( min(  edge[0],  edge[1] ), max(  edge[0],  edge[1] ) )
			if not edgekey in edge_to_triangles:
				edge_to_triangles[ edgekey ] = {};
			edge_to_triangles[ edgekey ][ tkey ] = tkey;
			
	vert_to_triangles = {};
	for tkey in triss:
		for edge in triss[ tkey ]:
			edgekey = ( min(  edge[0],  edge[1] ), max(  edge[0],  edge[1] ) )
			if not edgekey[0] in vert_to_triangles:
				vert_to_triangles[ edgekey[0] ] = {};
			if not edgekey[1] in vert_to_triangles:
				vert_to_triangles[ edgekey[1] ] = {};
			vert_to_triangles[ edgekey[0] ][tkey] = tkey
			vert_to_triangles[ edgekey[1] ][tkey] = tkey

	def _find_outer_edge_key( trisidx ):
	
		outer_edge_key = None;
		
		for edge in triss[ trisidx ]:
			edgekey = ( min(  edge[0],  edge[1] ), max(  edge[0],  edge[1] ) )
			
			otherkeycount = 0;
			for otherkey in edge_to_triangles[ edgekey ]:
				otherkeycount += 1;
				
			if otherkeycount == 1:
				for otherkey in edge_to_triangles[ edgekey ]:
					if otherkey == trisidx:
						outer_edge_key = edgekey;
						break;
						
		return outer_edge_key;
		
	"""
	def _deletetris( trisidx ):
		del remaining_tris[ trisidx ];
		for edgekey in triss[ trisidx ]:
			if edgekey in edge_to_triangles:
				if trisidx in edge_to_triangles[ edgekey ]:
					del edge_to_triangles[ edgekey ][ trisidx ];
	"""
		
	remaining_tris = {};
	for tkey in triss:
		remaining_tris[ tkey ] = tkey;
		
	while True:#for iasdf in range( 20 ):	#remaining_tris
	
		Rring = [];
		
		killtriss = {};
		#edge_triangles = {};
		
		for tkey in remaining_tris:
			hasouteredge = _find_outer_edge_key( tkey );
			if hasouteredge == None:
				pass; #inner triangle!
			else:
				#edge_triangles[ tkey ] = hasouteredge #OUTER triangle
			
				killtriss[ tkey ] = tkey;
				
				#for edgekey in triss[ tkey ]:
				#	if edgekey in edge_to_triangles:
				#		for trissy in edge_to_triangles[ edgekey ]:
				#			killtriss[ trissy ] = tkey;
							
				#WAIT! delete all TRIANGLES that have ANY VERTEX shared by this OUTER edge
				#	That should do it.
				for triskey2 in vert_to_triangles[ hasouteredge[0] ]:
					killtriss[ triskey2 ] = triskey2;
				
				for triskey2 in vert_to_triangles[ hasouteredge[1] ]:
					killtriss[ triskey2 ] = triskey2;
			
				#if edge_triangles[ tkey ] in featedges:
				Rring.append( hasouteredge );
			
		"""for tkey in edge_triangles:
			
			killtriss[ tkey ] = tkey;
						
			for edgekey in triss[ tkey ]:
				if edgekey in edge_to_triangles:
					for trissy in edge_to_triangles[ edgekey ]:
						killtriss[ trissy ] = tkey;
		
			#if edge_triangles[ tkey ] in featedges:
			Rring.append( edge_triangles[ tkey ] );
		"""
		
		R.append( Rring );
		
		for trisidx in killtriss:
			if trisidx in remaining_tris:
				del remaining_tris[ trisidx ];
			for edge in triss[ trisidx ]:
				edgekey = ( min(  edge[0],  edge[1] ), max(  edge[0],  edge[1] ) )
				if edgekey in edge_to_triangles:
					if trisidx in edge_to_triangles[ edgekey ]:
						del edge_to_triangles[ edgekey ][ trisidx ];
			
		done_iterating = True;
		for t in remaining_tris:
			done_iterating = False;
			break;
		if done_iterating:
			break;
			
	return R;#List of [ (i0,i1), ... ] line segments?

def x_voronoi_skeleton( feats, strokefeats, edges, triss, vverts, vfeatureedges ):
	"""
		Given all the triangles, checks if:
			No edges are feature edges -> becomes node (node) and connect halfedge points
			1 edge is a feature edge -> Connect line between other edgers
			2 edges are feature edges -> connect line to centroid (node)
			3 edges are feature edges -> create centered node (node), no connections
		
	"""
	
	R = [];
	
	featedges = {};
	for strokeidx in range( len( strokefeats ) ):
		stroke = strokefeats[strokeidx]
		iprev = None;
		ipoint = 0;
		for featidx in range( len( stroke ) ):
			feati = stroke[ featidx ];
			if iprev != None:
				featedges[ ( min( feati, iprev ), max( feati, iprev ) ) ] = ( strokeidx, featidx )
			iprev = feati;
			
	edgekey_to_points = {};
	for edge in edges:
		edgekey = ( min(  edge[0],  edge[1] ), max(  edge[0],  edge[1] ) );
		
		if edgekey in featedges:
			pass;
		else:
			x0 = feats[ edge[0] ][0] 
			y0 = feats[ edge[0] ][1]
			x1 = feats[ edge[1] ][0] 
			y1 = feats[ edge[1] ][1]
			edgekey_to_points[ edgekey ] = ( (x0+x1)/2.0, (y0+y1)/2.0 );
		
	triangle_to_node = {};
	for trisidx in triss:
		featidxs = {};
		fedgeks = {};
		nsedges = 0;
		for edge in triss[ trisidx ]:
			edgekey = ( min(  edge[0],  edge[1] ), max(  edge[0],  edge[1] ) );
			if edgekey in edgekey_to_points:
				fedgeks[ edgekey ] = edgekey
				nsedges += 1;
			featidxs[ edgekey[0] ] = edgekey[0]
			featidxs[ edgekey[1] ] = edgekey[1]
		xsum = 0.0;
		ysum = 0.0;
		for fid in featidxs:
			xsum += feats[ fid ][0];
			ysum += feats[ fid ][1];
		xsum /= 3.0;
		ysum /= 3.0;
		
		for edgekey in fedgeks:
			R.append( [ ( xsum, ysum ), edgekey_to_points[ edgekey ] ] )
			
			
	return R;

def compute_polo_rep( A, AF ):

	# assume CENTER is at 0,0, and data is WITHIN -1,1
	Arolo = [];
	for fp in A:
		revolus = math.atan2( fp[1], fp[0] ) / 6.283185307179586476925286766559;# /(2*pi)
		if revolus < 0:
			revolus += 1.0;
		distamus = math.sqrt( fp[0]*fp[0] + fp[1]*fp[1] ) / 1.4142135623730950488016887242097; #sqrt(2)
		Arolo.append( ( revolus, distamus ) )
	return Arolo;
	
def generate_debug_dots_on_frame( SC, FL, radius = 0.05, type = 0, zvalue = 0, xoffset = 0, rescale=(1,1,0,0) ):

	if type == 1:
		radius *= 1.0/1.4142135623730950488016887242097; #1/sqrt(2)
	else:
		pass;
		
	for fs in FL:
		me = SC.strokes.new()
		
		f = (
			fs[0]*rescale[0] + rescale[2]
			,fs[1]*rescale[1] + rescale[3]
		)
		
		pointlist = []
		if type == 1:
			pointlist = [
				( f[0]-radius, f[1]-radius )
				,( f[0]+radius, f[1]-radius )
				,( f[0]+radius, f[1]+radius  )
				,( f[0]-radius, f[1]+radius )
				,( f[0]-radius, f[1]-radius )
			]
		elif type == 3:
			pointlist = [
				( f[0], f[1]-radius )
				,( f[0], f[1]+radius )
			]
		else:
			pointlist = [
				( f[0]-radius, f[1] )
				,( f[0], f[1]-radius )
				,( f[0]+radius, f[1]  )
				,( f[0], f[1]+radius )
				,( f[0]-radius, f[1] )
			]
		
		for apoint in pointlist:
			npi = len( me.points )
			me.points.add( count=1, pressure=1, strength=1 )
			me.points[ npi ].co = ( xoffset+apoint[0], apoint[1], zvalue )
		
		#SC.strokes.close( me );

def generate_debug_linestrip_on_frame( SC, pointlist, zvalue = 0, rescale=(1,1,0,0) ):
	if len(pointlist) > 0:
		me = SC.strokes.new()
		me.points.add( count=len(pointlist), pressure=1, strength=1 )
		npi = 0;
		for apoint in pointlist:
			#npi = len( me.points )
			me.points[ npi ].co = ( apoint[0]*rescale[0] + rescale[2], apoint[1]*rescale[1] + rescale[3], zvalue )
			#me.points[ npi ].pressure = 1;
			#me.points[ npi ].select = False;
			npi += 1;
			
		#SC.strokes.close( me );
		
def generate_debug_linestrips_on_frame( SC, striplist, zvalue = 0, rescale=(1,1,0,0) ):
	for F in striplist:
		generate_debug_linestrip_on_frame( SC, F, zvalue, rescale=rescale );
					
def generate_debug_lines_on_frame( SC, V, VLs, zvalue = 0, rescale=(1,1,0,0) ):
	for VL in VLs:
		me = SC.strokes.new()
		me.points.add( count=len(VL), pressure=1, strength=1 )
		npi = 0;
		for vi in VL:
			me.points[ npi ].co = ( V[vi][0]*rescale[0] + rescale[2], V[vi][1]*rescale[1] + rescale[3], zvalue )
			npi += 1;
		
def generate_debug_feature_strokes_on_frame( C, F, SF, RM, zvalue=0, rescale=(1,1,0,0) ):
	for flist in SF:
		poynts = [];
		for fidx in flist:
			fpn = (
				F[ fidx ][0]*rescale[0] + rescale[2]
				,F[ fidx ][1]*rescale[1] + rescale[3]
			)
			poynts.append( fpn );
		generate_debug_linestrip_on_frame( C, poynts, zvalue=zvalue )
		
def generate_debug_triangles_on_frame( C, VE, VF, VT, T, zlevel=0, rescale=(1,1,0,0), shrink=1.0 ):
	TE = []
	for trisi in T:
		tcenterx = 0
		tcentery = 0
		for edge in VT[ trisi ]:
			p0=VF[ edge[0] ]
			p1=VF[ edge[1] ]
			tcenterx += p0[0] + p1[0];
			tcentery += p0[1] + p1[1];
		tcenterx /= 6;
		tcentery /= 6;
		for edge in VT[ trisi ]:
			sp0=VF[ edge[0] ]
			sp1=VF[ edge[1] ]
			p0 = ( shrink*(sp0[0] - tcenterx), shrink*(sp0[1] - tcentery) )
			p1 = ( shrink*(sp1[0] - tcenterx), shrink*(sp1[1] - tcentery) )
			me = C.strokes.new()
			me.points.add( count=2, pressure=1, strength=1 )
			me.points[ 0 ].co = ( (tcenterx+p0[0])*rescale[0]+rescale[2], (tcentery+p0[1])*rescale[1]+rescale[3], zlevel )
			me.points[ 1 ].co = ( (tcenterx+p1[0])*rescale[0]+rescale[2], (tcentery+p1[1])*rescale[1]+rescale[3], zlevel )
	
				
def drawnodeverts( C, VV, V, VFE, zlevel=0.0, xoffset = 0 ):
	dots1 = []
	dots2 = [];
	dots3 = [];
	
	for vi in VV:
		vcount = 0;
		for edge in VV[ vi ]:
			if VV[vi][edge]:
				vcount += 1;
		
		if vcount == 0:
			pass;
		elif vcount == 1:  #2...
			dots1.append( V[ vi ] )
		elif vcount == 2:  #2...
			dots2.append( V[ vi ] )
		else:  #2...
			dots3.append( V[ vi ] )

	generate_debug_dots_on_frame( C, dots1, radius=0.04, type = 0, zvalue = zlevel );
	generate_debug_dots_on_frame( C, dots2, radius=0.01, type = 1, zvalue = zlevel );
	generate_debug_dots_on_frame( C, dots3, radius=0.02, type = 0, zvalue = zlevel );

def perform_fblend( gpl, Aindex, Bindex, curridx ):

	A = TempFrame( gpl.frames[ Aindex ], Aindex );
	B = TempFrame( gpl.frames[ Bindex ], Bindex );
	
	#First off, reduce frame to something we can deal with...
	A.compute_center_projection();
	A.center_left = [ 1, 0, 0 ]  #vIEW VECTOR???
	A.center_up = [ 0, 1, 0 ]
	A.apply_center_projection();
	
	B.compute_center_projection();
	B.center_left = [ 1, 0, 0 ]
	B.center_up = [ 0, 1, 0 ]
	B.apply_center_projection();
	
	A.compute_2d_bounds();
	B.compute_2d_bounds();
	
	AnewCenter= [ 
		(A.center2d_bounds[0] + A.center2d_bounds[3])/2.0
		,(A.center2d_bounds[1] + A.center2d_bounds[4])/2.0
		,(A.center2d_bounds[2] + A.center2d_bounds[5])/2.0
	];
	BnewCenter= [ 
		(B.center2d_bounds[0] + B.center2d_bounds[3])/2.0
		,(B.center2d_bounds[1] + B.center2d_bounds[4])/2.0
		,(B.center2d_bounds[2] + B.center2d_bounds[5])/2.0
	];
	
	Acenter2d, Ax2d, Ay2d, Ascale2d = A.reproject_2d_points( AnewCenter, [1,0] );	#	xslope, yslope
	A.compute_2d_bounds();
	
	Bcenter2d, Bx2d, By2d, Bscale2d = B.reproject_2d_points( BnewCenter, [1,0] );	#	xslope, yslope
	B.compute_2d_bounds();
	
	#Probably need to clean up what we have now.
	A.clean_and_nodalize( 100 );	#Basically, NO LINES may cross. This appropriately cuts strokes. And forms "nodes" where those cuts join.
	B.clean_and_nodalize( 100 );	#	Might as well quantize it all at this point too... making sure no vertices are doubled?
	
	Arawfeatures, Arawstrokefeatures = A.detect_2d_features( anglethresh=15, distance=0.001, maxlength=0.25 );	#30 - 45 is good. unfortunately, a LARGE simple curve should be broken into multiple features...
	Brawfeatures, Brawstrokefeatures = B.detect_2d_features( anglethresh=15, distance=0.001, maxlength=0.25 );	#

	Afeatures, Astrokefeatures, Arawmaps = remap_feature_points( Arawfeatures, Arawstrokefeatures, q = 4096 )	#Remap features so there are NO DUPLICATES: (duplicate "features" are a NODE")
	Bfeatures, Bstrokefeatures, Brawmaps = remap_feature_points( Brawfeatures, Brawstrokefeatures, q = 4096 )
	
	Arescale = ( Ascale2d, Ascale2d, (A.center[0]+Acenter2d[0]), (A.center[1]+Acenter2d[1]) )
	Brescale = ( Bscale2d, Bscale2d, (B.center[0]+Bcenter2d[0]), (B.center[1]+Bcenter2d[1]) )
	
	C = gpl.frames.new( curridx, active=False );
	
	#Rasterize into pixels
	#	Find nearest pixel on target
	#	For each pixel, attempt to find a candidate pixel to move to
	#	"nearest pixel within" is fine...
	#
	
	#generate_debug_feature_strokes_on_frame( C, Afeatures, Astrokefeatures, Arawmaps, zvalue = -0.1, rescale=Arescale );
	#generate_debug_feature_strokes_on_frame( C, Bfeatures, Bstrokefeatures, Brawmaps, zvalue = 0.1, rescale=Brescale );

	
	#Okay, is there an aggressive version?
	#"matching" a feature line...
	#Well? features are like BLAST processing in a way.
	#we have SECTION of "DNA" that we want to match.
	#but its a bit more continuous; hm.
	#We can quantize the angle into some discrete set of "angles" but "distance" is still NOT evenly sampled like with genomes...
	#So we are looking for a SECTION of the target where our SOURCE matches...
	#And even if it does match, it has a dependancy that adjusts how it matches to others...
	#Hm...
	#
	#ANY kind of matching should work.
	#I dont care how...
	#
	#the "control point" generation might need some work, but I bet there is a natural version of control points to work with.
	#
	#If you can make control points, then you must match the control points from A to B.
	#if you CAN match those control points,
	#Then you can likely do a grid distortion or maybe just a linear triangular distortion of some kind...
	#Or, a "feature" whcih is a directed line segment can be used?
	#	This is definitely superior for what you are trying to do...
	#	How do we match up features?
	#
	#well, from the SOURCE,
	#
	#Find blocks of the image that are similar...
	#	? well we have a strokes -> features thing, but that ignores possible visual connections and uses the raw data itself.
	#	Probably need to clean that up...
	#	
	#Radial representation lets you shuffle the X, and then scale the Y to attempt to match...
	#The shuffling can be extreme in X along the radials... and we should be able to get a good initial guess...
	#Hm.
	#
	#Well, blender DOES have a "Interpolate" that only works IFF:
	#	Strokes are IN ORDER and SAME DIRECTION
	#So, maybe instead of trying to do it all,
	#You can start with "attempt to re-order strokes" so they "match up" better?
	#
	#Not the worst idea... But, we also know if we RE-order strokes, it BREAKS all previous morphs...
	#So thats something.
	#We'll have to add our "order list sensitive" stroke morph capability...
	#
	
	
	
	#Apolo = compute_polo_rep( Afeatures, Astrokefeatures );
	#Bpolo = compute_polo_rep( Bfeatures, Bstrokefeatures );
	
	#Shuffle Bpolo left and right to reduce SSE?
	#Hm...
	
	#Avoro = x_voronoi_compute_voronoi_rep( Afeatures, Astrokefeatures );
	#Bvoro = x_voronoi_compute_voronoi_rep( Bfeatures, Bstrokefeatures );
	
	#with the voronoi representation, we need  to classify triangles in some way... huh. problems.
	#Largest triangles are blended first? Bubble tree? Hm...
	#	Bubble tree allows for deformation relative to bubbles but... seem strange.
	#	Might have to break it up into layers of bubble sizes... Number of bubbles arent important as much as their overall coverage???
	#	Bubbles can SHRINK or GROW?
	#
	
	
	#( edges, triss, verts, featureedges );
	
	#With the voronoi diagram of FEATURES,
	#we should be able to compute "skeletal points" that is, find all triangles that have NO real edges on them, and construct a centroid.
	#Then, we can connnect any neighboring triangle centers as part of the skeleton itself.
	#This gives us a graph problem, we should be able to take advantage of for deformations (relative to the skeleton representation)
	#	That is, if we can get TWO skeletons, and we know the relative weighting to skeleton points, 
	#	That can give an estimate of the blended pose...
	#Note this idea will create a new cloud of points with intrinsic connections and properties;
	#
	#
	
	
	#Excellent, with the voronoi diagram triangles, we can do a LOT:
	#Such as, compute interior skeleton estimates (connect NON edge edges that are very long first, create centered nodes in triangles made out of entirely NON connected edges...
	#Aringtree = x_voronoi_convex_ring_tree( Afeatures, Astrokefeatures, Avoro[0], Avoro[1], Avoro[2], Avoro[3] );
	#Bringtree = x_voronoi_convex_ring_tree( Bfeatures, Bstrokefeatures, Bvoro[0], Bvoro[1], Bvoro[2], Bvoro[3] );

	#Askelly = x_voronoi_skeleton( Afeatures, Astrokefeatures, Avoro[0], Avoro[1], Avoro[2], Avoro[3] );
	#Bskelly = x_voronoi_skeleton( Bfeatures, Bstrokefeatures, Bvoro[0], Bvoro[1], Bvoro[2], Bvoro[3] );
	
	#Excellent, with the voronoi diagram triangles, we can do a LOT:
	#Such as, compute interior skeleton estimates (connect NON edge edges that are very long first, create centered nodes in triangles made out of entirely NON connected edges...

	#Well, get the convex hull FIRST.
	#Determine what edges are on it (pairwise)
	#Follow pairs and add if we can add a triangle?
	#
	
	#"Alpha shape" is a thing.
	#Well, a delanuy triangulation is important to do exactly that (simply removes exterior triangle features...)
	#sooo I guess we are stuck with that concept no matter what we do.
	#
	#Maybe its not that bad?
	#
	#Triangulate, Classify OUTER edges of concave hull (via alpha threshold)
	#Then, iterate INSIDE the shape for "connected to hull" edges..
	#And repeat?
	#	...
	#This should automatically form some kind of nested tree structure of concave outer shapes....
	#
	#Maybe we just do ONE layer and do some kinda shape estimate?
	#
	#Blend the exterior first?
	#
	#
	#
	
	
	
	#Current stroke features MAY intersect in their line segments.
	#Hm.
	#Well, what if we just use these segments (that is, two GUARANTEED connected features) and:
	#for all NON terminated segments:
	#    find nearest points to INTERSECT, and adjust "slide" position to best match ANGLE ?
	#		?It wouldn't be that hard to fit a quadratic polynomical to the points on this feature.
	#		But what would that get us? a way to match on a curve? eh.
	#
	#Does a quadtree help? No because its cutting and proximity is kinda weird for a quadtree. (umbrella tree would be OK)
	#
	
	
	
	#Alright, even though we HAVE stroke features;
	#SOME of these feature edges can OVERLAP or be TOO CLOSE.
	#Visually, this means each "segment" (connection of any two features) should have a thickness/radius;
	#Since this means every segment is actually a "dual capsule" because the two end radii could differ,
	#We have some way to combine them together...
	#If we take THIS data, and rasterize it,
	#Then we can do some sort of "video" comparison...
	#
	#Since there are a lot less line segments this should be fine.
	#
	#One thing to think about as well is,
	#Since this is such a constrained problem,
	#We should be able to voronoi both sides, and then the 3D voronoi is only connecting the two.
	#But really that just means something like "point to point" nonsense...
	#
	#Somehow we have to put ALL these line segments in a SEPERABLE space...
	#Well, the "angle" is a good start (0..90 since it goes EITHER DIRECTION)
	#Then we have lists of "angle" + "length" segments...
	#and a rough general location of where they are...
	#So if we cut up space into a space hash...
	#
	#Per stroke:
	#	Compute starting angle
	#	Then delta angles and lengths
	#	this is now a "feature stroke"
	#
	#Note that we are trying to find ANY place on the target where feature stroke could FIT.
	#It may fit CROSS feature stokes as well...
	#
	#Hm. Problematic.
	#
	#that linear_assignment_problem is still helpful.
	#
	#ConCAVE hulling question?
	#If we construct a concave hull of what we have drawn, then we can START with that as the base for deformation...
	#Meaning, if we can match the outer concave hulls, then we can adjust and repeat maybe? Or at least have starting points and work from the outside in...
	#This makes the most sense, "shadows" frame to frame should match?
	#but so should feature cluster positions!
	#
	#
	#
	
	#C.is_edited = True; did nothing.
	
	if False: 
	
		

		Arescale = ( Ascale2d, Ascale2d, (A.center[0]+Acenter2d[0]), (A.center[1]+Acenter2d[1]) )
		Brescale = ( Bscale2d, Bscale2d, (B.center[0]+Bcenter2d[0]), (B.center[1]+Bcenter2d[1]) )
		
		#Worst case:
		#each stroke WITH a "node" endpoint (at least one) is attempted to be matched...
		#but its the WHOLE NET that matters.
		#Better to make a close net, and match exterior path first, and remove/repeat to interior?!?!
		#
		
		#Draw feature POINTS
		#generate_debug_dots_on_frame( C, Afeatures, radius=0.002, type = 0, zvalue = -1, rescale=Arescale );
		#generate_debug_dots_on_frame( C, Bfeatures, radius=0.002, type = 1, zvalue = 1, rescale=Brescale );
	
		#Draw ONLY connected features (hm...)
		#we know how to connect Astrokefeatures, 
		#    but what about when those are combined in some way? Hm...
		#    and overlapping feature lines should CREATE nodes???
		#
		generate_debug_feature_strokes_on_frame( C, Afeatures, Astrokefeatures, Arawmaps, zvalue = -1, rescale=Arescale );
		generate_debug_feature_strokes_on_frame( C, Bfeatures, Bstrokefeatures, Brawmaps, zvalue = 1, rescale=Brescale );

		#generate_debug_linestrips_on_frame( C, [ Apolo ], zvalue = -1, rescale=(1,1,2,0) );#(Arescale[0],Arescale[1],Arescale[2],Arescale[3]) );
		#generate_debug_linestrips_on_frame( C, [ Bpolo ], zvalue = 1, rescale=(1,1,2,0) );#(Arescale[0],Arescale[1],Arescale[2],Arescale[3]) );
		
		#"wrapped" lines are a bit weirder to handle... Hm...
		#generate_debug_lines_on_frame( C, Apolo, Astrokefeatures, zvalue = -1, rescale=(1,1,0,0) );
		#generate_debug_lines_on_frame( C, Bpolo, Bstrokefeatures, zvalue = 1, rescale=(1,1,0,0) );
		
		#generate_debug_dots_on_frame( C, Apolo, radius=0.0025, type = 0, zvalue = -1, rescale=(1,1,0,0) );
		#generate_debug_dots_on_frame( C, Bpolo, radius=0.0025, type = 1, zvalue = 1, rescale=(1,1,0,0) );
		
		#Draws ALL triangles except the infinite ones:
		
		"""
		Atrisclass = x_voronoi_classitris( Afeatures, Astrokefeatures, Avoro[0], Avoro[1], Avoro[2], Avoro[3] );
		Atlist = {}
		for trisk in Atrisclass:
			if Atrisclass[ trisk ] == 2: Atlist[trisk]=trisk;
		generate_debug_triangles_on_frame( C, Avoro[0], Afeatures, Avoro[1], Atlist, zlevel=-1, rescale=Arescale, shrink=0.95 );
		
		Btrisclass = x_voronoi_classitris( Bfeatures, Bstrokefeatures, Bvoro[0], Bvoro[1], Bvoro[2], Bvoro[3] );
		Btlist = {}
		for trisk in Btrisclass:
			if Btrisclass[ trisk ] == 2: Btlist[trisk]=trisk;
		generate_debug_triangles_on_frame( C, Bvoro[0], Bfeatures, Bvoro[1], Btlist, zlevel=1, rescale=Brescale, shrink=0.95 );
		"""
		
		#generate_debug_lines_on_frame( C, Afeatures, Aalphat, zvalue = -1, rescale=Arescale );
		#generate_debug_lines_on_frame( C, Bfeatures, Balphat, zvalue = 1, rescale=Brescale );
		
		#generate_debug_linestrips_on_frame( C, Askelly, zvalue = -1, rescale=Arescale );
		#generate_debug_linestrips_on_frame( C, Bskelly, zvalue = 1, rescale=Brescale );
		
		#ringdepthmax = len( Aringtree );
		#for ringi in range( len( Aringtree ) ):
		#	ring = Aringtree[ ringi ];
		#	print( "A", len(ring) );
		#	generate_debug_lines_on_frame( C, Afeatures, ring, zvalue = -1 - 1.0*ringi/ringdepthmax, rescale=Arescale );
			
		#ringdepthmax = len( Bringtree );
		#for ringi in range( len( Bringtree ) ):
		#	ring = Bringtree[ ringi ];
		#	print( "B", len(ring) );
		#	generate_debug_lines_on_frame( C, Bfeatures, ring, zvalue = 1 + 1.0*ringi/ringdepthmax, rescale=Brescale );

		#re-assign colors? Hm...
		 
	#C.is_edited = False; did nothing.
	
	return C;
	
def perform_fblend2( gpl, Aindex, Bindex, curridx ):

	A = TempFrame( gpl.frames[ Aindex ], Aindex );
	B = TempFrame( gpl.frames[ Bindex ], Bindex );
	C = gpl.frames.new( curridx, active=False );
	
	#Specialized methodology:
	
	#All strokes need to be cut into paths. All paths begin and end at a node. Each node has a type.
	#	Require a maximum node radius for each node.
	#	Some nodes are too close together.This will be problematic and form a node cluster...
	#
	#Maybe that construction itself is a good thing. Start at a point, increase radius by including the next point until we accidentally include other points.
	#	IE, a straight line would be made of many of these inclusion spheres (this technique would work in N-D)
	#	But it lets us represent parts of a stroke by the stuff AROUND it...
	#	If you had a tail straight out, and it curled to the back, the radius of the spheres WOULDNT change much because they collide with the tail itself...
	#	Huh. might be simple enough to take a look at.
	#		But it does NOT respect artist mistakes (drawing multiple strokes so they connect) so look at that too...
	#One problem is the construction IS biased by stroke usage...
	#	So that sucks.
	#Maybe do that "rasterize" thing and use the centroids on that raster grid instead of the actual data?
	#Is it easier to do this computation via the raster method???
	#	Hm...
	#Raster method DOES have the advantage of linking and ignoring artist mistakes.
	#
	#Regardless, once we HAVE the spheres, then we can check if any of the spheres overlap (natural linkage)
	#This should form sphere islands / the topology of the drawing itself based on the lines used.
	#
	#
	
	#Then, the only representation we have IS a graph of paths.
	#Finding where a path maps onto the next frame is key. And we can't really assume much, but, 
	
	#Convert all line segments ( vi0, vi1, si, aabb, length, normalized direction, normalized tangent ) into a hash (so we can quickly find them.) (ignore 0 length segments)

	#For each line segment
	#	Compute curvature (follow segments along direction until angle difference is greater than theta or distance is greater than threshold)
	#	From middle of line segment, compute medial ball (move ball center and increase radius by distance until we hit any other line segment)
	#	Can we also do a Bresenham ray trace algorithm? (find other line segment "across" from this one in BOTH directions...)
	#
	
	
	return C;

def bscreen_cast_to_strokes( frame, context, screenpoint ):
	
	#Construct SCREEN RAY -> Cast to point (ray -> point distance?)
	raydirection = bpy_extras.view3d_utils.region_2d_to_vector_3d( context.region, context.region_data, screenpoint );
	raycenter = bpy_extras.view3d_utils.region_2d_to_location_3d( context.region, context.region_data, screenpoint, [0.0,0.0,0.0] );	#What? distance from? (+ is OUT of screen?)
	
	d_best = None;
	d_best_blend = None;
	d_closest = math.inf;#64*64	#hm
	for strokei in range( len( frame.strokes ) ):
		stroke = frame.strokes[ strokei ];
		for pointi in range( len( stroke.points ) ):
			point = stroke.points[ pointi ];
			delvec = [ point.co[i] - raycenter[i] for i in [0,1,2] ];
			dp = delvec[0]*delvec[0] + delvec[1]*delvec[1] + delvec[2]*delvec[2]
			if dp < d_closest:
				d_closest = dp;
				d_best = [ strokei, pointi ];
	
	return d_best, d_closest;
	
	
class Z2D_OT_fblend_direct(Operator):
	bl_idname = "object.z2d_fblend_direct"
	bl_label = "Frame Blend Direct"
	bl_description = "Frame Blend Direct" 
	bl_options = {'REGISTER'}
	
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
			
		gpd = context.gpencil_data	#https://docs.blender.org/api/blender2.8/bpy.types.GreasePencil.html#bpy.types.GreasePencil
		gpl = gpd.layers.active	#https://docs.blender.org/api/blender2.8/bpy.types.GreasePencilLayers.html#bpy.types.GreasePencilLayers
		if gpl:	#https://docs.blender.org/api/blender2.8/bpy.types.GPencilLayer.html#bpy.types.GPencilLayer
		
			allframeids = [ f.frame_number for f in gpl.frames ];	#Get all frame numbers for this layer
			#allframeids.sort();		#Sorted, just in case.
			currframeid = context.scene.frame_current;	#Get the CURRENT frame we are on (dont replace frames that exist.)
			if not currframeid in allframeids:
			
				nextframeaidx = bisect_left( allframeids, currframeid );#because thats how the algorithm works... index just BEFORE so...
				prevframeaidx = nextframeaidx - 1;
				
				if nextframeaidx < len( allframeids ) and prevframeaidx < len( allframeids ) and prevframeaidx >= 0 and nextframeaidx > 0:
					
					newframe = perform_fblend( gpl, prevframeaidx, nextframeaidx, currframeid )
					"""
					if newframe.id_data:
						newframe.id_data.update_tag();
						
					for s in newframe.strokes:
						if s.points.id_data:
							s.points.id_data.update_tag();
						if s.id_data:
							s.id_data.update_tag();
							
					if gpl.frames.id_data:
						gpl.frames.id_data.update_tag();
							
					#if bpy.context.gpencil_data_owner:
					#	bpy.context.gpencil_data_owner.update_tag();	#??? did nothing.
					gpd.update_tag();
					#gpl.update_tag();
					#newframe.update_tag();
					"""
					
				else:
					print( "Must have prior and next keyframe at frame "+str(currframeid) )
			else:
				print( "Cannot interpolate on a keyframe" )
		else:
			print( "Must have a greasepencil layer active with keyframes" )#settings = context.tool_settings.gpencil_sculpt

		#
		
		return {'FINISHED'};
	
	
# Define a nested property.

#What kind of structure do we need?
#1. we DO remember a list of "stroke sections" that we cut for blending ( strokeid, begini, endi )
#	Float, Int, FloatVector, IntProperty, IntVectorProperty, PointerProperty? 

#Registers as z2d_blend_prop (CANNOT DO THIS because frames are a bpy_struct)
#class Z2D_fblend_Frame_Property( bpy.types.PropertyGroup ):
#	factor: bpy.props.FloatProperty( name="factor", min=0.0, max=1.0, default=0.0, options={'ANIMATABLE'} ) #? PROPORTIONAL ? 
#	sections: bpy.props.IntVectorProperty( name="sections", size=6, options={'HIDDEN'} ) #( strokeid (?), begini, endi, strokeid, begini, endi )
#	node_points: bpy.props.FloatVectorProperty( name="node_points", size=4, options={'HIDDEN'} ) #Position of nodes on A, B (ax, ay, bx, by)
	
#registered as collection of z2d_blend_pairs  Z2D_fblend_Gpencil_Layer_Frame_Properties
class Z2D_fblend_Gpencil_Layer_Frame_Properties( bpy.types.PropertyGroup ):
	#target: bpy.props.PointerProperty( type=bpy.types.GPencilFrame, options={'HIDDEN'} ); #poll?
	target_est_index: bpy.props.IntProperty( options={'HIDDEN'} );#Frames can MOVE! they are just in a collection!!! Augh!
	target_est_frame_number: bpy.props.IntProperty( options={'HIDDEN'} );
	target_est_next_index: bpy.props.IntProperty( options={'HIDDEN'} );
	target_est_next_frame_number: bpy.props.IntProperty( options={'HIDDEN'} );#We really need to maintain this.. named frames???
	target_est_stroke_count: bpy.props.IntProperty( options={'HIDDEN'} ); #You can edit frames, but... hm.
	target_est_next_stroke_count: bpy.props.IntProperty( options={'HIDDEN'} ); #
	
	factor: bpy.props.FloatProperty( name="factor", min=0.0, max=1.0, default=0.0, options={'ANIMATABLE'} ) #? PROPORTIONAL ? 
	sections: bpy.props.IntVectorProperty( name="sections", size=6, options={'HIDDEN'} ) #( strokeid (?), begini, endi, strokeid, begini, endi )
	node_points: bpy.props.FloatVectorProperty( name="node_points", size=4, options={'HIDDEN'} ) #Position of nodes on A, B (ax, ay, bx, by)
	
class Z2D_fblend_Gpencil_Layer_Properties( bpy.types.PropertyGroup ):
	#Constructing a TARGET LAYER is a problem.
	#IE, how do we KNOW what layer this was referring to??
	#Name + index test?
	target_info: bpy.props.StringProperty( options={'HIDDEN'} ); #layer name == layer.info
	target_est_index: bpy.props.IntProperty( options={'HIDDEN'} );#Layers can MOVE! they are just in a collection!!! Augh!
	
	#target: bpy.props.PointerProperty( type=bpy.types.GPencilLayer, options={'HIDDEN'} ); #poll?
	frames: bpy.props.CollectionProperty( type=Z2D_fblend_Gpencil_Layer_Frame_Properties, options={'HIDDEN'} )

#Registers as z2d_blend_props
class Z2D_fblend_Gpencil_Properties( bpy.types.PropertyGroup ):
	layers: bpy.props.CollectionProperty( type=Z2D_fblend_Gpencil_Layer_Properties, options={'HIDDEN'} )
	#per layer (pointer property?)
		#per frame (pointer property?)
			#factor, sections, node_points, ?

class OBJECT_OP_Z2D_fblend_Interpolate(bpy.types.Operator):

	bl_idname =	"gpencil.z2d_fblend_interpolate"
	bl_label = "Similar to Interpolate, but attempts to match PATH LENGTH from the FIRST keyframe (lowest frame index) to the NEXT keyframe. Data is stored on the low frame key as a interpolatable frame, and you can reassign what stroke indexes map to what NEXT stroke indexes. Checks and shows the result."
	bl_description = "Z2D Interpolate"
	bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}  # enable undo for the operator.

	def blend_lists( self, A, B, bfactor, ranges ):
		"""
			A and B must be same length, and each element must be same length.
			
			make this a generator later please... (more efficient by a LOT)
		"""
		R = [];
		a_i = 0;
		a_i_max = len( A );
		finv = 1.0 - bfactor;
		while a_i < a_i_max:
			R.append( [ ((A[a_i][e]*finv) + (B[a_i][e]*bfactor)) for e in ranges ] );
			a_i += 1;
		return R;
		
	def screen_cast_to_strokes( self, context, screenpoint, frameindex = 0 ):
		
		#Construct SCREEN RAY -> Cast to point (ray -> point distance?)
		raydirection = bpy_extras.view3d_utils.region_2d_to_vector_3d( context.region, context.region_data, screenpoint );
		raycenter = bpy_extras.view3d_utils.region_2d_to_location_3d( context.region, context.region_data, screenpoint, [0.0,0.0,0.0] );	#What? distance from? (+ is OUT of screen?)
		
		d_best = None;
		d_best_blend = None;
		
		"""
		#Convert "view" to "world" ... ? Huh...
		
		d_closest = 64*64	#hm
		d_closest_blend = 64*64	#hm
		
		if len( self.frame_strokes ) > 0:
		
			framestrokei = 0;
			frameastrokes = self.frame_strokes[ frameindex ]
			framestrokei_max = len( frameastrokes );
			while framestrokei < framestrokei_max:
			
				strokei = frameastrokes[ framestrokei ];
				
				#Warning: strokelist might be a DICTIONARY...
			
				s_index = 0;
				for s in self.strokes_mem[ strokei ]:
				
					delvec = [ s[i] - raycenter[i] for i in [0,1,2] ];
					#dp = raydirection[0]*delvec[0] + raydirection[1]*delvec[1] + raydirection[2]*delvec[2]
					dp = delvec[0]*delvec[0] + delvec[1]*delvec[1] + delvec[2]*delvec[2]
					
					if dp < d_closest:
						d_closest = dp;
						d_best = [ framestrokei, strokei, s_index ]
									
					#
					s_index += 1;
				
				#Repeat algorithm for self.strokes_mem_blend{} which is a DICT not a MAP
				if strokei in self.strokes_mem_blend:
						
					s_index = 0;
					for s in self.strokes_mem_blend[ strokei ]:
					
						delvec = [ s[i] - raycenter[i] for i in [0,1,2] ];
						#dp = raydirection[0]*delvec[0] + raydirection[1]*delvec[1] + raydirection[2]*delvec[2]
						dp = delvec[0]*delvec[0] + delvec[1]*delvec[1] + delvec[2]*delvec[2]
						
						if dp < d_closest_blend:
							d_closest_blend = dp;
							d_best_blend = [ framestrokei, strokei, s_index ]
										
						#
						s_index += 1;
							
				framestrokei += 1;
				
		#print( raycenter, raydirection, d_best );
		"""
		
		return d_best, d_best_blend;
		
	
	def coord_to_screenpx( self, context, pts ):
		func = bpy_extras.view3d_utils.location_3d_to_region_2d
		return [ func(context.region, context.region_data, coord) for coord in pts ];
	
	def draw_line_3d( self, color, start, end, width=1):
		bgl.glLineWidth(width)
		bgl.glColor4f(*color)
		bgl.glBegin(bgl.GL_LINES)
		bgl.glVertex3f(*start)
		bgl.glVertex3f(*end)
			
	#new draw line
	def bgl_draw_line( self, v1, v2, rgba):
	
		if self._bgl_draw_lineshader:
			pass
		else:
			self._bgl_draw_lineshader = gpu.shader.from_builtin('2D_UNIFORM_COLOR') if not bpy.app.background else None
		shader = self._bgl_draw_lineshader

		coords = [(v1[0], v1[1]), (v2[0], v2[1])]
		batch = gpu_extras.batch.batch_for_shader(shader, 'LINES', {"pos": coords})

		# noinspection PyBroadException
		try:
			if v1 is not None and v2 is not None:
				shader.bind()
				shader.uniform_float("color", rgba)
				batch.draw(shader)
		except:
			pass
			
	def bgl_draw_lines( self, vL, rgba):
	
		if self._bgl_draw_lineshader:
			pass
		else:
			self._bgl_draw_lineshader = gpu.shader.from_builtin('2D_UNIFORM_COLOR') if not bpy.app.background else None
		shader = self._bgl_draw_lineshader

		coords = [ (v[0], v[1]) for v in vL ];#[(v1[0], v1[1]), (v2[0], v2[1])]
		batch = gpu_extras.batch.batch_for_shader(shader, 'LINES', {"pos": coords})

		# noinspection PyBroadException
		try:
			shader.bind()
			shader.uniform_float("color", rgba)
			batch.draw(shader)
		except:
			pass
			
	def _bgl_prepare_thicklines( self, lines, thick=1 ):
		#lines must be len() = 2*n 
		coords = [];
		i = 0;
		while i < len( lines ):
			v0 = lines[0];
			v1 = lines[1];
			x0 = v0[0]
			y0 = v0[1]
			x1 = v1[0]
			y1 = v1[1]
			dx = x1 - x0;
			dy = y1 - y0;
			dm = math.sqrt( dx*dx + dy*dy );
			dx /= dm;
			dy /= dm;
			tx = -dy*thick;
			ty = dx*thick;
			coords.extend( [
				(x0+tx,y0+ty)
				,(x1+tx,y1+ty)
				,(x1-tx,y1-ty)
				,(x0+tx,y0+ty)
				,(x1-tx,y1-ty)
				,(x0-tx,y0-ty)
			] );
			i += 2;
		return coords;
	
	def bgl_draw_thicklines( self, coords, rgba, thick=1 ):
	
		if self._bgl_draw_lineshader:
			pass
		else:
			self._bgl_draw_lineshader = gpu.shader.from_builtin('2D_UNIFORM_COLOR') if not bpy.app.background else None
		shader = self._bgl_draw_lineshader
		
		#coords = self._bgl_prepare_thicklines( [ (x0,y0), (x1,y1) ], thick=thick );
		
		#“‘POINTS’, ‘LINES’, ‘TRIS’ or ‘LINES_ADJ’”.
		batch = gpu_extras.batch.batch_for_shader(shader, 'TRIS', {"pos": coords})

		# noinspection PyBroadException
		try:
			shader.bind()
			shader.uniform_float("color", rgba)
			batch.draw(shader)
		except:
			pass
			
	def bgl_draw_thickline( self, x0, y0, x1, y1, rgba, thick=1 ):
	
		self.bgl_draw_thicklines( [ (x0, y0), (x1,y1) ], rgba=rgba, thick=thick );
			
	def bgl_draw_linestrip( self, vL, rgba):
	
		if self._bgl_draw_lineshader:
			pass
		else:
			self._bgl_draw_lineshader = gpu.shader.from_builtin('2D_UNIFORM_COLOR') if not bpy.app.background else None
		shader = self._bgl_draw_lineshader

		coords = [ (v[0], v[1]) for v in vL ];#[(v1[0], v1[1]), (v2[0], v2[1])]
		batch = gpu_extras.batch.batch_for_shader(shader, 'LINE_STRIP', {"pos": coords})

		# noinspection PyBroadException
		try:
			shader.bind()
			shader.uniform_float("color", rgba)
			batch.draw(shader)
		except:
			pass
		

	def draw_typo_2d( self, pos, text):
		font_id = 0	 # XXX, need to find out how best to get this.
		# draw some text
		blf.position(font_id, pos[0], pos[1], 0)
		blf.size(font_id, 20, 72)
		blf.draw(font_id, text)

	def draw_callback_3d( self, context):
		pass;

		
	def compute_length_normalized_stroke( self, A ):
	
		if len( A ) > 0:
			
			line_total_length = [];

			asum = 0.0;
			pp = A[0];
			for p in A:
				v = math.sqrt( (p[0] - pp[0])*(p[0] - pp[0]) + (p[1] - pp[1])*(p[1] - pp[1]) + (p[2] - pp[2])*(p[2] - pp[2]) )
				line_total_length.append( asum );
				asum += v;
				pp = p;
			line_total_length.append( asum );
			
			if asum > 0:
				line_total_length_normalized = [ v/asum for v in line_total_length ];
			else:
				line_total_length_normalized = [ 0 for v in line_total_length ];
			
			return line_total_length_normalized, asum;
			
		return [], 0;
		
	def factor_interp( self, L, Lfactors, Lindex, targetfactor, rangeL ):
		"""
			rangeL = [ 0,1,2 ] for L being [x,y,z] ...
			Lfactors is a list of floats, cdf 0..1 inclusive range
			
		"""
		Lindexnext = min( Lindex + 1, len( Lfactors )-1 );
		delfac = targetfactor - Lfactors[ Lindex ];
		delden = Lfactors[ Lindexnext ] - Lfactors[ Lindex ];
		interp_factor = 0;
		if delden > 0:
			interp_factor = delfac / delden;
			
		interp_factor_inv = 1.0 - interp_factor;
		
		#print( "F:", Lfactors[ Lindex ], targetfactor, Lfactors[ Lindexnext ] ,  interp_factor  );
		
		return [ ((L[ Lindex ][ e ] * interp_factor_inv) + (L[ Lindexnext ][ e ] * interp_factor)) for e in rangeL ]
		
	def blend_lists( self, A, B, bfactor, ranges ):
		"""
			A and B must be same length, and each element must be same length.
			
			make this a generator later please... (more efficient by a LOT)
		"""
		R = [];
		a_i = 0;
		a_i_max = len( A );
		finv = 1.0 - bfactor;
		while a_i < a_i_max:
			R.append( [ ((A[a_i][e]*finv) + (B[a_i][e]*bfactor)) for e in ranges ] );
			a_i += 1;
		return R;
	
	def stroke_cumulant_blend( self, A, B, Afactors, Bfactors, rangeV ):
		"""
			Given arrays A, B of [a,b,...] values,
			Return an array of length( A ) such that the output positions match B,
			Given by the A pathlength factors and B pathlength factors
			
			Requires:
				len( A ) = len( Afactors )
				len( B ) = len( Bfactors )
				All elements in Afactors to be float 0..1 inclusive, sorted from least to greatest (it's a discrete CDF)
				All elements in Bfactors to be float 0..1 inclusive, sorted from least to greatest (it's a discrete CDF)
				Every element in A is same length and at least 1 element
				Every element in B is same length and at least 1 element
				Every element in A and B are same length and at least 1 element
		"""
		
		a_index = 0;
		b_index = 0;
		a_index_max = len( A );
		b_index_max = len( B );
		R = [];
		
		while a_index < a_index_max:	#For each point in A
		
			a_factor = Afactors[ a_index ]
		
			while Bfactors[ b_index ] < a_factor:	#Ensure Bfactors[ b_index ] is LESS OR EQUAL to Afactors[ a_index ]
				b_index += 1;
				if b_index >= b_index_max:
					b_index = b_index_max - 1;
					if b_index < 0:
						b_index = 0;
					break;
			while Bfactors[ b_index ] > a_factor:	#Possible to overshoot. Not allowed.
				b_index -= 1;
				if b_index < 0:
					b_index = 0;
					break;
				
			np = self.factor_interp( B, Bfactors, b_index, a_factor, rangeV );
			
			R.append( np );
		
			a_index += 1;
		
		return R;
	
	def draw_callback_2d( self, context):
	
		bfactor = self.current_blend_factor;
		
		if True:
		
			#Moving into USER DEFINED mode.
			#IE, click and drag == CREATE link.
			#Links have a radius, and if you click on a EXISTING link, it will move...
			#Right click cancels...
			#Links MUST SNAP to a VERTEX. NO exceptions. The strokes will THEN be cut and joined as a "user blend node"
			#Every link has a ID + name (unique) in probably x64 characters...
			#

			#self.cached_featurePointsB_screen = self.coord_to_screenpx( context, self.cached_featurePointsB )
			if self.drag_point_begin != None and self.drag_point_end != None:
				mystroke = [ self.drag_point_begin[0], self.drag_point_end[0] ];
				scords = self.coord_to_screenpx( context, mystroke );
				
				self.bgl_draw_thickline( scords[0][0], scords[0][1], scords[1][0], scords[1][1], [1, math.sin( self.deltatime() )/4 + 0.25 ,1,0.5], thick=2 );
				
			#Draw ALL OTHER lines? Hm.. (but not thick?)
			#"THICK" line is the SELECTED one? hm...
			
			#"Arrow" triangle? helps to see direction? Can we color vertices?? (shadeR???)
			
			#Draw current and next frame? Hm...
			
			#Draw/update current blend in another THREAD? (on change, trigger blend update?)
			#	Huh! Then we draw it at different blend values (Use a shader to do the blending (later,easy))
			
			if self.frames:
				if len( self.frames ) > 0:
				
					#z2d_blend_pairs  z2d_blend_props -> ???
				
					"""
					
					FP = self.frames[ 0 ];
					
					#Hm... ADD IT if it doesnt exist? ... but it SHOULD exist...
					
					print( FP ) #pointer property??
					print( FP.z2d_blend_prop )
					print( FP.z2d_blend_prop[0] )
					
					
					print( dir( FP.z2d_blend_prop[0] ) )
					
					#tags = context.preferences.addons[__name__].preferences.principled_tags
					#normal_abbr = tags.normal.split(' ')

					FP.z2d_blend_prop.factor;
					FP.z2d_blend_prop.sections;
					FP.z2d_blend_prop.node_points;
					
					dolines = [];
					for np in FP.z2d_blend_prop.node_points:
						dolines.append( ( np[0], np[1] ) );
						dolines.append( ( np[2], np[3] ) );
					
					self.bgl_draw_thicklines( self.coord_to_screenpx( context, self._bgl_prepare_thicklines( self, dolines, thick=1 ) ), [0,1,0,1] );
					
					"""
					
		if False: #Show detected pixel feature approximations (where are things we need to track?)
		
			def tmpdrawfeats( X, symbol=0 ):
				boxcoords = [];
				for fk in X:
					rfeat = X[fk]
					nparam = rfeat[0];#some sort of "value"
					b = rfeat[1]
					if symbol == 1:
						boxcoords.append( (b[0], b[1], b[2]) );
						boxcoords.append( (b[3], b[4], b[5]) );
						boxcoords.append( (b[3], b[1], b[2]) );
						boxcoords.append( (b[0], b[4], b[5]) );
						boxcoords.append( (b[0], b[4], b[2]) );
						boxcoords.append( (b[3], b[1], b[5]) );
						boxcoords.append( (b[0], b[1], b[5]) );
						boxcoords.append( (b[3], b[4], b[2]) );
					else:
						boxcoords.append( (b[0], b[1], b[2]) );
						boxcoords.append( (b[3], b[1], b[2]) );
						
						boxcoords.append( (b[3], b[1], b[2]) );
						boxcoords.append( (b[3], b[4], b[2]) );
						
						boxcoords.append( (b[3], b[4], b[2]) );
						boxcoords.append( (b[0], b[4], b[2]) );
						
						boxcoords.append( (b[0], b[4], b[2]) );
						boxcoords.append( (b[0], b[1], b[2]) );
				return boxcoords;
						
			def tmpdrawfeatures( Y, allpoints=False, symbol=0 ):
				#map_by_groups[ groupid ][0][ nk ] = retfeats[ nk ];
				boxcoords = [];
				for fgroup in Y:
				
					c = Y[fgroup][1];
					r = c[3];
					boxcoords.append( (c[0]-r, c[1], c[2]) );
					boxcoords.append( (c[0], c[1]-r, c[2]) );
					
					boxcoords.append( (c[0], c[1]-r, c[2]) );
					boxcoords.append( (c[0]+r, c[1], c[2]) );
					
					boxcoords.append( (c[0]+r, c[1], c[2]) );
					boxcoords.append( (c[0], c[1]+r, c[2]) );
					
					boxcoords.append( (c[0], c[1]+r, c[2]) );
					boxcoords.append( (c[0]-r, c[1], c[2]) );
				
					if allpoints:
						X = Y[fgroup][0];
						boxcoords.extend( tmpdrawfeats( X, symbol=symbol ) )
					
				return boxcoords;
				
			featview = True;
			featdetail = False;
			
			if self.cached_featuresA == None:
				if self.show_featuresA != None:
					self.cached_featuresA = tmpdrawfeatures( self.show_featuresA, allpoints= featdetail )
					self.cached_featuresA_screen = self.coord_to_screenpx( context, self.cached_featuresA )
			if self.cached_featuresB == None:
				if self.show_featuresB != None:
					self.cached_featuresB = tmpdrawfeatures( self.show_featuresB, allpoints= featdetail, symbol=1 )
					self.cached_featuresB_screen = self.coord_to_screenpx( context, self.cached_featuresB )
			if self.cached_featurePointsA == None:
				if self.show_featuresApixels != None:
					self.cached_featurePointsA = tmpdrawfeats( self.show_featuresApixels )
					self.cached_featurePointsA_screen = self.coord_to_screenpx( context, self.cached_featurePointsA )
			if self.cached_featurePointsB == None:
				if self.show_featuresBpixels != None:
					self.cached_featurePointsB = tmpdrawfeats( self.show_featuresBpixels, symbol=1 )
					self.cached_featurePointsB_screen = self.coord_to_screenpx( context, self.cached_featurePointsB )
				
			#Cached DATA is one thing, but if the SCREEN is NOT CHANGING we shouldn't re-convert these to screen coords...
			#for now, lets just cache them...
			if self.cached_featuresA_screen != None:
				self.bgl_draw_lines( self.cached_featuresA_screen, [1,0.5,0,1] );	
			if self.cached_featuresB_screen != None:
				self.bgl_draw_lines( self.cached_featuresB_screen, [0,0.5,1,1] );	
			
			if not featview:
				self.bgl_draw_lines( self.cached_featurePointsA_screen, [0.5,0.25,0,1] );
				self.bgl_draw_lines( self.cached_featurePointsB_screen, [0,0.25,0.5,1] );
				
			if False:
				
				#self.astralfeatures_A; == [ dist_ratio, angle, weightratio, groupkeyA ] 
				#self.astralfeatures_B; == [ dist_ratio, angle, weightratio, groupkeyA ] 
				#self.astralMatch; == [ Akey, Bkey, value ]
				
				matchlines = [];
				for akey in self.springMatches:  #
					bkey = self.springMatches[ akey ];
					f0 = self.show_featuresA[ akey ];
					f1 = self.show_featuresB[ bkey ];
					c0 = f0[1];
					c1 = f1[1];
					matchlines.append( c0[:3] );
					matchlines.append( c1[:3] );
					
				if len( matchlines ) > 0 :
					self.bgl_draw_lines( self.coord_to_screenpx( context, matchlines ), [0.0,0.0,1,1] );
					
				springlines = [];
				for elm in self.springMassess:
					p0 = elm[0];
					f0 = self.show_featuresA[ elm[1] ];
					f1 = self.show_featuresB[ elm[2] ];
					c0 = f0[1];
					c1 = f1[1];
					springlines.append( c0[:3] );
					springlines.append( [ p0[0], p0[1], 0 ] );
					springlines.append( [ p0[0], p0[1], 0 ] );
					springlines.append( c1[:3] );
					
				if len( springlines ) > 0:
					self.bgl_draw_lines( self.coord_to_screenpx( context, springlines ), [0.0,0.0,1,1] );
				
			if False:
				
				#self.astralfeatures_A; == [ dist_ratio, angle, weightratio, groupkeyA ] 
				#self.astralfeatures_B; == [ dist_ratio, angle, weightratio, groupkeyA ] 
				#self.astralMatch; == [ Akey, Bkey, value ]
				
				matchlines = [];
				agreedmatches = self.astralMatch[0]
				for akey in agreedmatches:  #
					amatch = agreedmatches[akey]
					g0 = self.astralfeatures_A[ amatch[0] ];
					g1 = self.astralfeatures_B[ amatch[1] ];
					f0 = self.show_featuresA[ g0[0] ];	#DIFFERENT now...
					f1 = self.show_featuresB[ g1[0] ];
					c0 = f0[1];
					c1 = f1[1];
					matchlines.append( c0[:3] );
					matchlines.append( c1[:3] );
					
				self.bgl_draw_lines( self.coord_to_screenpx( context, matchlines ), [0.0,0.0,1,1] );
					

			if False:
				
				def drawvoronoidsda( X, color=[0,0,0,1], offset=(0,0,0) ):
					#return edges, triss, verts, plookupdict, gpointinft
					rawverts = []
					verts = X[2];
					edges = X[0];
					realpoints = X[4];
					plookupi = X[3]
					for edge in edges:
						v0 = realpoints[ edge[0] ][0];
						v1 = realpoints[ edge[1] ][0];
						v0 = ( v0[0]+offset[0], v0[1]+offset[1], v0[2]+offset[2] )
						v1 = ( v1[0]+offset[0], v1[1]+offset[1], v1[2]+offset[2] )
						rawverts.append( v0 );
						rawverts.append( v1 );
					self.bgl_draw_lines( self.coord_to_screenpx( context, rawverts ), color );
				
				#for edgy in self.show_featuresAvoronoi[1]
				
				#self.bgl_draw_lines( self.cached_featuresA_screen, [1,0.5,0,1] );	
				drawvoronoidsda( self.show_featuresAvoronoi, color=[0,0.5,0,1], offset=(0,0,0) )
				drawvoronoidsda( self.show_featuresBvoronoi, color=[1,0.5,1,1], offset=(0,0,0) )
				
				#Note we should also have the BLENDED one?
			
			
		if False:
			
			for match in self.sf_matchups:
			
				if not match in self.sf_match_cache:
				
					fA = self.frame_features[0][ match[0] ];
					fB = self.frame_features[1][ match[1] ];
					
					A = [ [ v.co[0],v.co[1],v.co[2] ] for v in fA[0] ]
					B = [ [ v.co[0],v.co[1],v.co[2] ] for v in fB[0] ]
					
					Anorm, Anormlen = self.compute_length_normalized_stroke( A )
					Bnorm, Bnormlen= self.compute_length_normalized_stroke( B )#NORMALIZE strokes?
					
					Anorm = Anorm[:-1]
					Bnorm = Bnorm[:-1]
					
					if match[2]:
						Bnorm = [ 1.0 - Bnorm[ len( Bnorm ) - x - 1 ] for x in range( len( Bnorm ) ) ]; #Flip list. AND flip values???
						B = [ B[ len( B ) - x - 1 ] for x in range( len( B ) ) ]#Flip B points too. (normalization is invertible!)
					
					#print( len( A), len(B), len(Anorm), len(Bnorm ) )
					cblend = self.stroke_cumulant_blend( A, B, Anorm, Bnorm, [0,1,2] );#,2] );
					
					self.sf_match_cache[ match ] = ( A, cblend );
							
				bmatch = self.sf_match_cache[ match ];
				
				if len( bmatch[0] ) > 1:
								
					gotstroke = self.blend_lists( bmatch[0], bmatch[1], bfactor, [0,1,2] );
					
					screencords = self.coord_to_screenpx( context, gotstroke );
					
					self.bgl_draw_linestrip( screencords, [0.2, 1, 0.2, 1] );	
				else:
					pass;
				"""
				self.sf_match_cache[ match ] = ?
				
				#Great! we have just completed the FIRST ROUND of BASIC matchups!
				
				#
				#AS A RESULT, we can NOW cache these subfeatures into the system, from which we can then construct a blend.
				#
				#And, as per normal, we NORMALIZE the features by length and blend them THAT way...
				#
				#
				self.sf_match_cache = {};
				
				fA = self.frame_features[0][ match[0] ];
				fB = self.frame_features[1][ match[1] ];
				doflip = match[2];
				
				screencords = self.coord_to_screenpx( context, [ v.co for v in fA[0] ] );
				
				self.bgl_draw_linestrip( screencords, [0.2,1,0.2,1] );
				
				screencords = self.coord_to_screenpx( context, [ v.co for v in fB[0] ] );
				
				self.bgl_draw_linestrip( screencords, [0.2,0.2,1.0,1] );
				"""
			
		if False:
			
			if len( self.frame_strokes ) > 0:
			
				#self.strokes_mappings[ framei ] = deezestrokes;
				
				for idx in range( len( self.strokes_mappings[ 0 ] ) ):
				
					usestrokepair = False;	#Nice!
				
					strokeidx = self.strokes_mappings[ 0 ][ idx ]
					
					#Mapping of stroke INDEXES -> Also with some extra stuff about bounding boxes or maybe something more stable???
					Aidx = strokeidx;
					
					Bidx = self.strokes_mappings[1][ idx % len( self.strokes_mappings[ 1 ] ) ];
					flipB = 0;
					
					if Aidx in self.strokes_paired_mappings:
						#Now it gets complicated.
						#stroke Aidx is MAPPED to another subsection or list of strokes...
						#Hm!
						spmap = self.strokes_paired_mappings[ Aidx ];
						Bidx = spmap[0]
						flipB = spmap[2]
						usestrokepair = True;
						
					if usestrokepair:
						
						#Well, if we assume something about stroke A mapping to stroke B in any sense...
						#	IE, the SAME strokes are there...
						#
						
						A = self.strokes_mem[ Aidx ];
						
						#Construct a blendmap between the strokes maybe? Hmmmm...
						#	How we do this? index -> (i0, i1, dt) ? easy enough.
						#
						#Store and updated IF IT CHANGES.
						blendkey = (Aidx, Bidx, flipB);
						cblend = None;
						if blendkey in self.strokes_mem_blend:
							cblend = self.strokes_mem_blend[ blendkey ];
						else:
							B = self.strokes_mem[ Bidx ];
							
							Anorm = self.strokes_mem_normalized[ Aidx ]
							Bnorm = self.strokes_mem_normalized[ Bidx ]
							
							if flipB > 0:
								Bnorm = [ 1.0 - Bnorm[ len( Bnorm ) - x - 1 ] for x in range( len( Bnorm ) ) ]; #Flip list. AND flip values???
								B = [ B[ len( B ) - x - 1 ] for x in range( len( B ) ) ]#Flip B points too. (normalization is invertible!)
						
							cblend = self.stroke_cumulant_blend( A, B, Anorm, Bnorm, [0,1,2] );#,2] );
						
							self.strokes_mem_blend[ blendkey ] = cblend
						
						bfactor = self.current_blend_factor;
						
						gotstroke = self.blend_lists( A, cblend, bfactor, [0,1,2] );
						
						screencords = self.coord_to_screenpx( context, gotstroke );
						
						
						#Construct the blend from strokeA to strokeB given the relative cumulant factor arrays (path length, normalized)
						#resmo = self.stroke_angulant_blend( strokeA, strokeB, strokeAcum, strokeBcum, [0,1,2], {} );
						#self.strokes_mem_blend[ linkpair[0] ] = resmo;	#Blend self.strokes_mem -> result!
						
						
						#
						#blendmapping is kinda the weird part.
						#A Stroke in A can map to MORE than one stroke in B.
						#In other words, there are PARTS of the stroke that should map to OTHER parts...
						#
						
						#Hm...
						if len( screencords ) > 0:
							#prev = screencords[0]
							#for p in screencords:	#Not efficient, but does let us interpolate...
							
								#Color should MEAN SOMETHING.
							
							#	self.bgl_draw_linestrip( prev, p, [1,0,0,0] );
							#	prev = p;
							self.bgl_draw_linestrip( screencords, [1,0,0,0] );
					
		if False:
			
			#
			#Hm data driven...
			#
			bgl.glDisable(bgl.GL_DEPTH_TEST)
			bgl.glEnable(bgl.GL_BLEND)
			bgl.glLineWidth( 1.0 )
			
			def get_lettercode( i ):
				if i < 26:
					return chr( ord('A') + i%26 )
				else:
					return chr( ord('A') + math.floor(i/26)%26 ) + chr( ord('A') + (i)%26 );
				
			if len( self.frames ) > 0:
				
				currlayer = bpy.context.active_gpencil_layer
				currframe = self.frames[ 0 ];
				
				#
				#Wait... could we have done this with the other gpencil data? seems like it...
				#	Try that again, now that you removed the stupid '@classmethod' calls...
				#	We would WAY rather store data IN each gpencilframe...
				#
				cuts = [];#context.gpencil_data.greaser_data.get_cuts( currlayer, currframe );
				
				#CANNOT WORK: cuts = currframe.greaser_data.get_cuts();
			
				#
				#Okay... Now we have to map frame id <--> data for that frame... Ugly!
				#
				
				#cuts = fgd.get_cuts();
				cpi = 0;
				for cut in cuts:
				
					#cut.stroke_index
					#cut.point_index
					#cut.next_stroke_index
					#cut.next_point_index
					
					#CAREFUL! Might have been broken with editing...
					p0 = self.strokes_mem[ cut.stroke_index ][ cut.point_index ];
					p1 = self.strokes_mem_blend[ cut.next_stroke_index ][ cut.next_point_index ];
					
					pp = self.coord_to_screenpx( context, [ p0, p1 ] );
					verts = [ (v[0], v[1], 0) for v in pp ];
					
					bgl.glBegin(bgl.GL_LINES)
					bgl.glColor4f( 1.0, 0.5, 0.0, 0.5 )
					bgl.glVertex3f( *verts[0] );
					bgl.glColor4f( 0.0, 1.0, 1.0, 0.5 )
					bgl.glVertex3f( *verts[1] );
					bgl.glEnd()
						
					tex = get_lettercode( cpi );
						
					bgl.glColor4f( 1.0, 0.5, 0.0, 0.5 )
					self.draw_typo_2d(verts[0], tex)
					bgl.glColor4f( 0.0, 1.0, 1.0, 0.5 )
					self.draw_typo_2d(verts[1], tex)
					
					cpi += 1;
					
						
				bgl.glLineWidth(1)
				bgl.glDisable(bgl.GL_BLEND)
				bgl.glEnable(bgl.GL_DEPTH_TEST)
				bgl.glColor4f(0.0, 0.0, 0.0, 1.0)
				
	def applyframe( self, gpl, currentframeid ): #layer,  context.scene.frame_current;
	
		bfactor = self.current_blend_factor;
						
		strokepack = [];
		
		if True:
			
			if len( self.frame_strokes ) > 0:
			
				#self.strokes_mappings[ framei ] = deezestrokes;
				
				for idx in range( len( self.strokes_mappings[ 0 ] ) ):
				
					usestrokepair = False;	#Nice!
				
					strokeidx = self.strokes_mappings[ 0 ][ idx ]
					
					#Mapping of stroke INDEXES -> Also with some extra stuff about bounding boxes or maybe something more stable???
					Aidx = strokeidx;
					
					Bidx = self.strokes_mappings[1][ idx % len( self.strokes_mappings[ 1 ] ) ];
					flipB = 0;
					
					if Aidx in self.strokes_paired_mappings:
						#Now it gets complicated.
						#stroke Aidx is MAPPED to another subsection or list of strokes...
						#Hm!
						spmap = self.strokes_paired_mappings[ Aidx ];
						Bidx = spmap[0]
						flipB = spmap[2]
						usestrokepair = True;
						
					if usestrokepair:
						
						#Well, if we assume something about stroke A mapping to stroke B in any sense...
						#	IE, the SAME strokes are there...
						#
						
						A = self.strokes_mem[ Aidx ];
						
						#Construct a blendmap between the strokes maybe? Hmmmm...
						#	How we do this? index -> (i0, i1, dt) ? easy enough.
						#
						#Store and updated IF IT CHANGES.
						blendkey = (Aidx, Bidx, flipB);
						cblend = None;
						if blendkey in self.strokes_mem_blend:
							cblend = self.strokes_mem_blend[ blendkey ];
						else:
							B = self.strokes_mem[ Bidx ];
							
							Anorm = self.strokes_mem_normalized[ Aidx ]
							Bnorm = self.strokes_mem_normalized[ Bidx ]
							
							if flipB > 0:
								Bnorm = [ 1.0 - Bnorm[ len( Bnorm ) - x - 1 ] for x in range( len( Bnorm ) ) ]; #Flip list. AND flip values???
								B = [ B[ len( B ) - x - 1 ] for x in range( len( B ) ) ]#Flip B points too. (normalization is invertible!)
						
							cblend = self.stroke_cumulant_blend( A, B, Anorm, Bnorm, [0,1,2] );#,2] );
						
							self.strokes_mem_blend[ blendkey ] = cblend
						
						bfactor = self.current_blend_factor;
						
						gotstroke = self.blend_lists( A, cblend, bfactor, [0,1,2] );
						
						#More to it than that...
						strokepack.append( ( gotstroke, self.strokes_mem_sources[ Aidx ], self.strokes_mem_sources[Bidx] ) );
						
		if len( strokepack ) > 0:

			#gpd = context.gpencil_data	#https://docs.blender.org/api/blender2.8/bpy.types.GreasePencil.html#bpy.types.GreasePencil
			#gpl = gpd.layers.active	#https://docs.blender.org/api/blender2.8/bpy.types.GreasePencilLayers.html#bpy.types.GreasePencilLayers
			#if gpl:	#https://docs.blender.org/api/blender2.8/bpy.types.GPencilLayer.html#bpy.types.GPencilLayer
			if True:
			
				allframeids = [ f.frame_number for f in gpl.frames ];	#Get all frame numbers for this layer
			
				currframeid = currentframeid;#context.scene.frame_current;	#Get the CURRENT frame we are on (dont replace frames that exist.)
				if not currframeid in allframeids:
			
					C = gpl.frames.new( currframeid );
					
					for strokei in range( len( strokepack ) ):
						spt = strokepack[strokei];
						stroke = spt[0]
							
						Asrcframe = self.frames[ spt[1][0] ];
						Bsrcframe = self.frames[ spt[2][0] ];
						
						Asrc = Asrcframe.strokes[ spt[1][1] ];
						Bsrc = Bsrcframe.strokes[ spt[2][1] ];
						
						me = C.strokes.new()
						
						me.display_mode = Asrc.display_mode
						me.draw_cyclic = Asrc.draw_cyclic
						me.end_cap_mode = Asrc.end_cap_mode
						me.gradient_factor = Asrc.gradient_factor
						me.gradient_shape = Asrc.gradient_shape
						me.line_width = Asrc.line_width
						me.material_index = Asrc.material_index;
						
						me.points.add( len( stroke ) );
						npi = 0;
						for pointi in range( len( stroke ) ):
							point = stroke[ pointi ]
							
							Asrcpoint = Asrc.points[ pointi ]
							#print( dir( Asrcpoint ) );
							#Asrcpointprops = [ i for i in Asrcpoint.__class__.__dict__.keys() if i[:1] != '_' ]
							#print( Asrcpointprops );
							#for propname in Asrcpointprops:
							#	if not callable( getattr( Asrcpoint, propname ) ):
							#		me.points[ propname ] = Asrcpoint[ propname ];
									
							me.points[ npi ].pressure =Asrcpoint.pressure;
							me.points[ npi ].select = Asrcpoint.pressure;
							me.points[ npi ].strength =Asrcpoint.strength;
							me.points[ npi ].uv_factor =Asrcpoint.uv_factor;
							#me.points[ npi ].uv_rotation =Asrcpoint.uv_rotation;
							
							me.points[ npi ].co = ( point[0], point[1], point[2] )
							
							#me.points[ npi ].? = ?
							
							npi += 1;
							
				else:
					print("Cannot overwrite current frame");
			else:
				print("No grease pencil layer");
					
	def deltatime( self ):
		return (time.process_time() - self.original_starttime)
				
	def _begin( self, context ):
	
		self.mouse_dx = 0;
		self.mouse_dy = 0;
		self.mouse_count = 0;
		
		self.original_starttime = time.process_time() 
		
		self.strokes = [];
		self.strokes_mem = [];
		self.strokes_mem_bounds = [];	#bounding box per total stroke
		self.strokes_mem_normalized = [];
		self.frame_strokes = [];
		self.frames = [];
		self.strokes_mem_sources = {};
		self.strokes_mappings = {};
		self.strokes_paired_mappings = {};
		
		self.strokes_mem_blend = {}; #Hmmmm... KEYED pairs!

		self.current_blend_factor = 0.5;
		self.current_x_sinerev = 0.0;
		self.drag_point_begin = None;
		self.drag_point_end = None;
		self.left_drag = False;
		self.left_release = False;
		
		
		self._bgl_draw_lineshader = None;

		self.cached_featuresA = None;
		self.cached_featuresB = None;
		self.cached_featurePointsA = None;
		self.cached_featurePointsB = None;
					
		self.show_featuresA = None
		self.show_featuresB = None
		self.show_featuresApixels = None;
		self.show_featuresBpixels = None;
		
		layer = context.active_gpencil_layer;
		
		if len( layer.frames ) <= 0:
			return;
			
		#Find the index in layer.frames that the current frame is on:
		framei = 0;
		for frame in layer.frames:
			if frame == layer.active_frame:
				break;
			framei += 1;
		framei_next = framei + 1;
		if framei_next >= len( layer.frames ):
			framei_next = framei;
			
		if False: #Wrap around to 0? Hm...
			if framei_next == framei:
				framei_next = 0;
			
		#In this case we use 2 frames to interpolate between
		self.frame_indexes = [ framei, framei_next ]
		
		self.frames = [ layer.frames[ framei ], layer.frames[ framei_next ] ]
		
		#ONLY https://docs.blender.org/api/current/bpy.types.GreasePencil.html#bpy.types.GreasePencil
		
		#So->
		#	Layer by name (rough, but OK)
		#		Keyframe PAIR by... what?
		#		
		
		#Okay. Can we do this in a DIFFERENT way?
		#	Animation data???
		#		Hm... SEems tied to frame index. How to update this I have no idea...
		#
		#animation_data
		#	.action
		#	.drivers
		#	.nla_tracks
		#		[i].name
		#		[i].strips
		#			.action
		#			.fcurves
		#			.strips
		#			
		#	-> Abusing the timeline animation keyframes maybe???
		#
		#Action has an ID; thats important? Hm.
		#	<- Maybe we need to CREATE some kind of channel FOR the interpolations???
		#		uh... "driver" ?
		
		#this may need something else...
		#	Typically developers of add-ons use Empty types and attach custom properties to them.
		#Okay fine but that isnt really enough...
		#Keyframing on custom properties?
		#	object.keyframe_insert(data_path='["prop"]')
		#
		#Even if we store properties "per keyframe"
		#	Without some kind of information IN that keyframe to deal with...
		#	And I dont want to "inject special strokes" just to store data on them!! gods...
		#
		#Whatever we do, it MUST be exposed.
		#Can we create a new kind of data and attach it as GPENCIL? per layer though??? hm...
		#	-> int property, pooled frame blends? (have to UPDATE them?)
		#Can animate a int property, so, name matches maybe?
		#	This is good and accessible.
		#But where do we STORE the data for a frame blend???
		#	<-> ??? baaahhhhhh, thats a lotta values that dont need to be animated.
		#		Hrm... ultimately a "completed blend" is 2 float arrays of SAME LENGTH to interpolate positions from initial to final.
		#		And what else?
		#
		#Animateable layername_bidx -> int of pooled blend? (only exists if needed, always "nearest / floor" interpolated)
		#	Pooled properties by INTEGER, mapping to blend frames.
		#Hm... then at least the "blend index" is exposed to the interface ins some way...
		#Also means its possible to correct issues in the UI;
		#	But its pretty painful. Bummer.
		#	Odd we can mess with markers in the UI. Hm.
		#ANYTHING THAT WORKS is important here.
		#	<- user editing driving blending.
		#	-> MUST be saved to file; MUST be user correctable.
		#	-> MUST be adaptive to changes, updates, add/remove frames...
		#		Is there a listener for "animation changes" ???
		#			That would help...
		#
		
		if True:  #get pair index
		
			#Make sure layer property exists 
			layer_prop_name = layer.info+'_z2df';
			
			if layer_prop_name in context.gpencil_data:
				pass;
			else:
				context.gpencil_data[ layer_prop_name ] = -1;	#bpy.props.IntProperty( options={'ANIMATABLE'} );
			
			#we want to ANIMATE a property.
			#So, that PROPERY must have an animation data...
			#But where is it stored?
			#
			
			#print( context.gpencil_data.keyframe_insert )
			#context.gpencil_data[ layer_prop_name ].keyframe_insert( data_path=layer_prop_name, frame=self.frames[0].frame_number );
			
			
			#Make sure we have animation_data?
			if context.gpencil_data.animation_data:
				pass;
			else:
				print( "anim data is None" );
				context.gpencil_data.animation_data_create();
				
			try:
				
				#Make sure we have an action? nla_tracks?? uh???
				print( context.gpencil_data.animation_data.nla_tracks );
				
				#HERE NEXT how do we store data per pair of keyframes if we can't uniquely identify keyframes? Hm...
				
				#context.gpencil_data.layers[]
				
					#nla_tracks???
					#context.gpencil_data.animation_data.nla_tracks
					#print( context.gpencil_data.animation_data.nla_tracks );
					
					#context.gpencil_data.animation_data = 
				#layers["ink"].opacity
					
				#print( context.gpencil_data.animation_data, dir ( context.gpencil_data.animation_data ) )
				
				#Get curve
				hasanim = context.gpencil_data.animation_data.action.fcurves.find( layer_prop_name );
				if hasanim:
					pass;
				else:
					context.gpencil_data.animation_data.action.keyframe_insert( data_path="layers["+'ERROR'+"]."+layer_prop_name, frame=self.frames[0].frame_number );
					hasanim = context.gpencil_data.animation_data.action.fcurves.find( layer_prop_name );
					hasanim.extrapolation = 'CONSTANT'
				
				#Get KEYFRAME or ADD if it doesnt exist... (inefficient, use binary search later?)
				useindex = None;
				for kp in hasanim.keyframe_points:
					if kp.co[0] == self.frames[0].frame_number:
						useindex = kp.co[1];
						break;
				
				ppairs = context.gpencil_data.z2d_blend_pairs;
				
				if useindex == None or useindex < 0:
					#Must create a new index for this frame pair... (hm, int vector?)
					useindex = len( ppairs );
					ppairs.add()
					
					context.gpencil_data[ layer_prop_name ] = useindex
					context.gpencil_data.animation_data.action.keyframe_insert( data_path=layer_prop_name, frame=self.frames[0].frame_number );
					hasanim = context.gpencil_data.animation_data.action.fcurves.find( layer_prop_name );
					hasanim.extrapolation = 'CONSTANT'
				
				
				keypair = ppairs[ useindex ]
				
				print( keypair );
			except Exception as e:
				print( e )
			
		if False:
			
			#Add layer property lookup if it doesnt exist:
			foundit = None;
			for p in context.gpencil_data.z2d_blend_props:
				if p.target_info == layer.info:
					foundit	= p;
			if foundit == None:
				huff = context.gpencil_data.z2d_blend_props.add(); #Always has name, rna_type?
				huff.target_info = layer.info;
				
				huff.name = layer.info;
				huff.target_est_index = 0;  #
				
				#Given a layer, get its index?
				for flayer in context.gpencil_data.layers:
					if flayer == layer:
						#print( "found it", huff.target_est_index, flayer.info, layer.info );
						break;
					huff.target_est_index += 1;
				foundit = huff;
				
				#print( dir( huff ) );
				#print( dir( context.gpencil_data.z2d_blend_props ) );
				#print( len( context.gpencil_data.z2d_blend_props ) );
				
				#Okay but... where is 0 memory address for as_pointer() ?
				#	Hah! lol.
				
				#as_pointer() -> okay but... when does it change? Hm... Is it even the same on file loading??!?!
				#as_pointer() is a C memory location, it changes on loading and such...
				#Useful hiuristic during same program operation though
				#"pointer" only stays consistent DURING program existence.
				#	So frame index is important too...
				#	How do we maintain consistency here?
				#	Its not safe to do pointer math so...
					
				#/* Pointers copied from GPENCIL_ViewLayerData. */
				#struct BLI_memblock *gp_object_pool;
				#struct BLI_memblock *gp_layer_pool;
				#struct BLI_memblock *gp_vfx_pool;
				#struct BLI_memblock *gp_material_pool;
				#struct BLI_memblock *gp_light_pool;
				#struct BLI_memblock *gp_maskbit_pool;
				#
				#BLI_memblock <- this means addressess CANNOT be reliabley used.
				#    without some seriously hacky shit
				#		(reverse engineer points to blocks? huh...)
				#		(block pointers will NOT be in order soooooo)
				#
				#https://developer.blender.org/diffusion/B/browse/master/source/blender/blenlib/BLI_memblock.h
				#
					
				#Check for frame keyframe pair...
				#	How???
					
				#Okay, we have foundit ( context.gpencil_data.z2d_blend_props )  z2d_blend_pairs
				#	How do we add in the frame?
				
				#print( self.frames[0].path_from_id( "frame_number" ) ) GPencilFrame does not support path creation.
				
				#hok = self.frames[0].driver_add();
				#print( hok );
				
				print( "FP0:", self.frames[0].as_pointer() ) #212988872
				print( "FP1:", self.frames[1].as_pointer() ) #212988088
				
				
				#Okay we added the layer.
				#Now what about the frames? Hm... problems ensue.
				#	<- common to add/remove frames.
				#	<- syncrhonizing could be impossible without "names" in frames
					
				#Check if we have this layer...
				#Add it if not?
				#Otherwise, do we have these frames?
				#
			
		if True:
		
			fA = self.frames[0];
			fB = self.frames[1];
			
			#Step 1: process frames
			#	Reduce frames to 2D (hm... abuse 3d coordinate as "layer index" ?
			#	compute "paths" and "nodes"
			#		must select detail radius? (generally 1/200th of the joined bounds)
			#		must select adjoining angle (generally 30-60 degrees)
			#	Special tags for certain nodes
			#		cusp (sharp feature along a known path)
			#		whisker (end of a stroke that is attached to another node)
			#	Premise is, convert current inked lines into CLOSED AREAS.
			#	If we do this, then we trace the EXTERIOR PATH as the polygon to interpolate.
			#	Blend the two exterior polygons.
			#	Then, remove exterior polygon areas
			#	Blend this polygon path.
			#	Repeat until no more areas remain.
			#	This means the OUTSIDE of each isolated polygon island will be known simple to interpolate.
			#	the INSIDE could be weird (a single path rather than an area)
			#
			#	Want to convert to a intermediate representaion first.
			#	of paths and nodes. nodes can be "intersection", "blob", "cusp", "whiskerend" and are where smooth paths intersect
			#		paths should be a SMOOTH curve of any path from node to node.
			#		This means a path can have any number of inflection points so long as they are smooth.
			#	In this way, we can construct a graph that represents nodes and paths, and shuld be able to trace exterior paths.
			#	many paths can connect two nodes. How can we judge exteriorness?
			#	Well, node types I guess.
			#	Add cusp and inflection nodes along a path then.
			#
			#	Hm. if incoming and outgoing edges are sorted CCW from (1,0) (bias direction +x)
			#	per node that is.
			#	That still doesnt let us trace the exterior.
			#	
			#	Perhaps "find problematics" splits apart the obvious path parts from "potential issues" via radius.
			#	in this case...
			#
			#
			#Make a hash grid for each point so we can do efficient proximity testing and find "node points"
			#	
			#
			
			A = TempFrame( self.frames[0], 0 );
			B = TempFrame( self.frames[1], 1 );
			
			#First off, reduce frame to something we can deal with...
			A.compute_center_projection();
			A.center_left = [ 1, 0, 0 ]  #VIEW VECTOR???
			A.center_up = [ 0, 1, 0 ]
			A.apply_center_projection();
			
			B.compute_center_projection();
			B.center_left = [ 1, 0, 0 ]
			B.center_up = [ 0, 1, 0 ]
			B.apply_center_projection();
			
			A.compute_2d_bounds();
			B.compute_2d_bounds();
			
			AnewCenter= [ 
				(A.center2d_bounds[0] + A.center2d_bounds[3])/2.0
				,(A.center2d_bounds[1] + A.center2d_bounds[4])/2.0
				,(A.center2d_bounds[2] + A.center2d_bounds[5])/2.0
			];
			BnewCenter= [ 
				(B.center2d_bounds[0] + B.center2d_bounds[3])/2.0
				,(B.center2d_bounds[1] + B.center2d_bounds[4])/2.0
				,(B.center2d_bounds[2] + B.center2d_bounds[5])/2.0
			];
			
			Acenter2d, Ax2d, Ay2d, Ascale2d = A.reproject_2d_points( AnewCenter, [1,0] );	#	xslope, yslope
			A.compute_2d_bounds();
			
			Bcenter2d, Bx2d, By2d, Bscale2d = B.reproject_2d_points( BnewCenter, [1,0] );	#	xslope, yslope
			B.compute_2d_bounds();
				
			Arescale = ( Ascale2d, Ascale2d, (A.center[0]+Acenter2d[0]), (A.center[1]+Acenter2d[1]) )
			Brescale = ( Bscale2d, Bscale2d, (B.center[0]+Bcenter2d[0]), (B.center[1]+Bcenter2d[1]) )
			
			
			#self.frame_indexes = [ framei, framei_next ]
			if self.frames[0].frame_number != context.scene.frame_current:
			
				C = context.active_gpencil_layer.frames.new( context.scene.frame_current, active=False );
				print( "made frame" );
				
				
				

			print( "ok, done ", self.frames[0].frame_number, " ", context.scene.frame_current );
			
			#CAN WE ADD SOMETHING TO A FRAME????
			#	WE know we can add something to a GPencil ITSELF. Layer properties maybe???
			#
			#it SOUNDS like we can add a property to a frame...
			#But... what kinds of properties can we add?
			#	TECHNICALLY we dont want to ANIMATE said property.
			#	Hm.
			#

		args = ( self, context )
		self._handle_3d = bpy.types.SpaceView3D.draw_handler_add( OBJECT_OP_Z2D_fblend_Interpolate.draw_callback_3d, args, 'WINDOW', 'POST_VIEW' )
		self._handle_2d = bpy.types.SpaceView3D.draw_handler_add( OBJECT_OP_Z2D_fblend_Interpolate.draw_callback_2d, args, 'WINDOW', 'POST_PIXEL' )
	
	
	
	def _finished( self, context ):
	
		if self._handle_3d != None:
			bpy.types.SpaceView3D.draw_handler_remove(self._handle_3d, 'WINDOW')
			self._handle_3d = None;
			
		if self._handle_2d != None:
			bpy.types.SpaceView3D.draw_handler_remove(self._handle_2d, 'WINDOW')
			self._handle_2d = None;
		
	def _apply_manual_cut( self, context ):
	
		prefA = self.drag_point_begin[0];
		prefB = self.drag_point_end[1];
		
		if prefA != None and prefB != None:
			pass;#context.gpencil_data.greaser_data.add_cut( bpy.context.active_gpencil_layer, self.frames[0], prefA, prefB );
		else:
			pass;
			
		self.drag_point_begin = None;
		self.drag_point_end = None;
			
		pass;

	#def __init__(self):
	#	print("Start")
	#	self._handle_3d = None;
	#	self._handle_2d = None;
		

	#def __del__(self):
	#	print("End")
	
	@classmethod
	def poll(cls, context):
	
		#In this case, must have some set of grease pencil layers selected...
	
		#Polled is called when you mouse over this thing
		return True;#GreaserAnimatorUIValidContext( context );

	def modal(self, context, event):
		if event.type == 'MOUSEMOVE':  # Update
			if self.mouse_count == 0:
				self.mouse_dx = 0;
				self.mouse_dy = 0;
			else:
				self.mouse_dx = event.mouse_region_x - self.mouse_x;
				self.mouse_dy = event.mouse_region_y - self.mouse_y;
			self.mouse_x = event.mouse_region_x;
			self.mouse_y = event.mouse_region_y;
			self.mouse_count += 1;
			
			#MODE UPDATE:
			if self.left_drag:
				
				fbest, fcloseness = bscreen_cast_to_strokes( self.frames[1], context, [ self.mouse_x, self.mouse_y ] );
				
				atpoint = self.frames[1].strokes[ fbest[0] ].points[ fbest[1] ].co;
				
				self.drag_point_end = [ atpoint, 1, fbest ];
				
			self.execute(context)
			
		elif event.type == 'LEFTMOUSE':	 # Confirm/Apply
		
			if event.value == 'PRESS':
			
				self.left_drag = True;
				
				#Check CURRENT "nodes" list
				#	If we are within radius of one, we are MOVING that node.
				#Otherwise,
				#	Create new node at this starting point (note that it CAN be empty, indicating a null->create transition)
				
				fbest, fcloseness = bscreen_cast_to_strokes( self.frames[0], context, [ self.mouse_x, self.mouse_y ] );
				
				atpoint = self.frames[0].strokes[ fbest[0] ].points[ fbest[1] ].co;
				
				self.drag_point_begin = [ atpoint, 0, fbest ];
				
			else:
			
				if event.value == 'RELEASE':
				
					print( "RELEASE: ", event.value );
					#OFFICIAL release value...
					#self._apply_manual_cut( context );
					
				else:
					pass;
					
				self.left_drag = False;
				self.left_release = True;
				
			self.execute(context)
			
		elif event.type in ('RIGHTMOUSE', 'ESC'):  # Cancel/Ignore
			self._finished( context );
			context.area.tag_redraw()
			return {'FINISHED'}	#'CANCELLED'}
			
		elif event.type in ('WHEELUPMOUSE', 'ESC'):  # Zoom in?
			pass;
		elif event.type in ('WHEELDOWNMOUSE', 'ESC'):  # Zoom out?
			pass;
		elif event.type in ('MIDDLEMOUSE', 'ESC'):  # Pan view
			pass;#if event.value == 'PRESS':
		
		elif event.type in ('RET' ):  # APPLY delta frame...
		
			if event.value == 'PRESS':
			
				#def applyframe( self, gpl, currentframeid ): #layer,  context.scene.frame_current;
	
				self.applyframe( context.active_gpencil_layer, context.scene.frame_current );
				
				self._finished( context );
				context.area.tag_redraw()
			
				return {'FINISHED'}	#'CANCELLED'}
		
		elif event.type in ('SPACE' ):  # APPLY delta frame...
		
			if event.value == 'PRESS':
			
				#Oh boy.
				#	FRAME FILL:
				#	
						
				layer = context.active_gpencil_layer;
				#self.frame_indexes = [ framei, framei_next ];
				
				#print( self.frame_indexes );
				
				actualframes = [ layer.frames[ self.frame_indexes[0] ].frame_number, layer.frames[ self.frame_indexes[1] ].frame_number ];
				
				framei = actualframes[0]+1;
				
				#if actualframes[0] == actualframes[2]:
				#	pass;
				
				while framei < actualframes[1]:
				
					#context.scene.frame_current = framei
					
					self.current_blend_factor = 1.0*(framei - actualframes[0])/(actualframes[1] - actualframes[0])
					
					self.applyframe( layer, framei );
					
					print( framei );
				
					framei += 1;
				
				self._finished( context );
				context.area.tag_redraw()
			
				return {'FINISHED'}	#'CANCELLED'}
		else:
			print( event.type, event );

		return {'RUNNING_MODAL'}
		
	def invoke(self, context, event):
	
		#Uh. do this DIRECTLY? So we can "Fill" ?
	
		if context.area.type == 'VIEW_3D':
		
			"""
			segs = [
				( 3, 0 )
				,(2.5,1)
				,(-2.5, -1)
				,(2,0.5)
			];
			print( "T",compute_natural_direction( segs ) );
			print( "T",compute_natural_direction( [ (1, 0) ] ) );
			print( "T",compute_natural_direction( [ (1000, 0) ] ) );
			print( "T",compute_natural_direction( [ (1, 1) ] ) );
			print( "T",compute_natural_direction( [ (-1, 1) ] ) );
			print( "T",compute_natural_direction( [ (1, -1) ] ) );
			print( "T",compute_natural_direction( [ (-1, -1) ] ) );
			print( "T 0, 45", compute_natural_direction( [ (1, 1), (1, 1) ] ) );
			print( "T 0, 45", compute_natural_direction( [ (1000, 1), (1000, 1) ] ) );
			print( "T 90", compute_natural_direction( [ (1, 1), (1, -1) ] ) );
			print( "T 135", compute_natural_direction( [ (1, 1), (-1, 0) ] ) );
			"""
		
		
			self._begin( context );
			
			#Other important things! 
			#We would like circles to drag about? hm... no, lines with circles so they DONT occlude geometry...
			#The data should be saved IN the gpencil frame ITSELF, so it knows frame -> frame what the blend data is. (reorder requires redoing)
			#
			self.execute( context )
			
			context.window_manager.modal_handler_add( self )
			
			return {'RUNNING_MODAL'}
		else:
			self.report({'WARNING'}, "View3D not found, cannot run operator")
			return {'CANCELLED'}
			
	def execute( self, context ):
	
		#Update data... Huh...
		#Screen size??? Bah!
		
		self.current_x_sinerev += self.mouse_dx/960.0 ;
		
		#Triangle wave instead of sin wave?
		def triwave( v ):
			f = ( v - math.floor(v) )
			if f < 0.5:
				return f*2;	#0==0, 0.5 == 1
			else:
				return 2*(1-f); #0.5= .5 1 = 0
				
		self.current_blend_factor = triwave( self.current_x_sinerev );
		if self.current_blend_factor < 0:
			self.current_blend_factor = 0;
		elif self.current_blend_factor > 1:
			self.current_blend_factor = 1;
		
		context.area.tag_redraw()
		
		return {'FINISHED'};	#'RUNNING_MODAL'	'CANCELLED'	'PASS_THROUGH'
	