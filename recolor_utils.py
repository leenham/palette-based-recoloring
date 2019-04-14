import math
import numpy as np

def rgb2lab4arr(arr):
	result = np.zeros(arr.shape)
	if arr.shape[1]==3:
		for i in range(arr.shape[0]):
			result[i] = rgb2lab(arr[i])
	return result

def lab2rgb4arr(arr):
	result = np.zeros(arr.shape)
	if arr.shape[1]==3:
		for i in range(arr.shape[0]):
			result[i] = lab2rgb(arr[i])
	return result

def rgb2lab_2(rgb):
	RGB = rgb.copy()
	#for i in range(3):
	#	v = rgb[i]/255
	#	if v>0.04045 :
	#		v = math.pow((v+0.055)/1.055,2.4)
	#	else:
	#		v /= 12.92
	#	RGB[i] = 100*v

	X = RGB[0]*0.412453 + RGB[1]*0.357580 + RGB[2]*0.180423;
	Y = RGB[0]*0.212671 + RGB[1]*0.715160 + RGB[2]*0.072169;
	Z = RGB[0]*0.019334 + RGB[1]*0.119193 + RGB[2]*0.950227;
		
	X = X/(255.0*0.950456)
	Y = Y/(255.0*1.0)
	Z = Z/(255.0*1.088754)
	#XYZ = [X/0.950456,Y/1,Z/1.088754]
	#XYZ = XYZ/255
	#for i in range(3):
	#	v =XYZ[i]
	#	if v>0.008856 :
	#		v = math.pow(v,1/3)
	#	else:
	#		v *= 7.787037
	#		v += 16/116
	#	XYZ[i] = v
	[L,a,b] = [0,0,0]
	fx,fy,fz = 0,0,0
	if 	Y>0.008856 :
		fy = math.pow(Y,1/3)		
	else:
		fy = 7.787*Y + 16/116
	L = 116.0*fy -16.0
	#L = max(0,L)
	if X>0.008856 :
		fx = math.pow(X,1/3)
	else:
		fx = 7.787*X+16/116

	if Z>0.008856 :
		fz = math.pow(Z,1/3)
	else:
		fz = 7.787*Z+16/116
	a = 500.0*(fx-fy)
	b = 200.0*(fy-fz)
	return np.array([L,a,b])

def lab2rgb_2(lab):
	L = lab[0]
	a = lab[1]
	b = lab[2]
	d = 6/29
	fy = (L+16)/116
	fx = fy+a/500
	fz = fy-b/200

	Y = (fy-16/116)*3*d*d
	if fy>d :
		Y = fy*fy*fy;
	X = (fx-16/116)*3*d*d
	if fx>d :
		X = fx*fx*fx;
	Z = (fz-16/116)*3*d*d
	if fz>d :
		Z= fz*fz*fz;
	X*= 0.950456*255.0
	Y*= 1.0*255.0
	Z*= 1.088754*255.0
	R = 3.240479*X - 1.537150*Y - 0.498535*Z
	G = (-0.969256)*X + 1.875992*Y + 0.041556 * Z
	B = 0.055648*X + (-0.204043)*Y + 1.0570311 * Z
	#RGB = [R,G,B]
	#for i in range(3):
	#	v = RGB[i]/100
	#	if v> 0.0405/12.92 :
	#		v = math.pow(v,1/2.4)
	#		v *= 1.055
	#		v -= 0.055
	#	else:
	#		v *= 12.92
	#	RGB[i] = round(v*255)
	return np.array([R,G,B])

def rgb2lab(rgb):
	R,G,B = rgb/255
	if R>0.04045 :
		R = math.pow((R+0.055)/1.055,2.4)
	else:
		R = R/12.92
	if G>0.04045 :
		G = math.pow((G+0.055)/1.055,2.4)
	else:
		G = G/12.92
	if B>0.04045 :
		B = math.pow((B+0.055)/1.055,2.4)
	else:
		B = B/12.92

	R = R*100
	G = G*100
	B = B*100
	X = R * 0.4124 + G * 0.3576 + B * 0.1805
	Y = R * 0.2126 + G * 0.7152 + B * 0.0722
	Z = R * 0.0193 + G * 0.1192 + B * 0.9505
	X = X/95.047
	Y = Y/100
	Z = Z/108.883

	if X>0.008856:
		X = math.pow(X,1/3)
	else:
		X = (X*7.787)+16/116
	if Y>0.008856:
		Y = math.pow(Y,1/3)
	else:
		Y = (Y*7.787)+16/116
	if Z>0.008856:
		Z = math.pow(Z,1/3)
	else:
		Z = (Z*7.787)+16/116
	L = (116*Y)-16
	A = 500 * (X-Y)
	B = 200 * (Y-Z)
	return np.array([L,A,B])

def lab2rgb(lab):
	Y = (lab[0] + 16)/116
	X =  lab[1]/500 + Y
	Z = Y - lab[2]/200 

	if Y>0.206893034422 :
		Y = math.pow(Y,3)
	else:
		Y = (Y-16/116)/7.787

	if X>0.206893034422 :
		X = math.pow(X,3)
	else:
		X = (X-16/116)/7.787

	if Z>0.206893034422 :
		Z = math.pow(Z,3)
	else:
		Z = (Z-16/116)/7.787

	X = 95.047*X /100
	Y = 100*Y /100
	Z = 108.883*Z/100
	R = X *  3.2406 + Y * -1.5372 + Z * (-0.4986)
	G = X * (-0.9689) + Y *  1.8758 + Z *  0.0415
	B = X *  0.0557 + Y * -0.2040 + Z *  1.0570

	if R > 0.0031308:
		R = 1.055 * math.pow(R,1 / 2.4) - 0.055
	else:
		R = 12.92 * R;
	if G > 0.0031308:
		G = 1.055 * math.pow(G,1 / 2.4) - 0.055
	else:                    
		G = 12.92 * G;
	if B > 0.0031308:
		B = 1.055 * math.pow(B,1 / 2.4) - 0.055
	else:
		B = 12.92 * B;

	R = R * 255;
	G = G * 255;
	B = B * 255;

	return np.array([R,G,B])


def distance2(c1,c2):
	#return labDistance2(c1,c2)
	return np.sum(np.square(c1-c2))

def distance(c1,c2):
	return math.sqrt(distance2(c1,c2))

def labDistance2(col1,col2):
	l1,a1,b1 = col1
	l2,a2,b2 = col2
	K1 = 0.045
	K2 = 0.015
	del_L = l1-l2
	c1 = math.sqrt(a1*a1+b1*b1)
	c2 = math.sqrt(a2*a2+b2*b2)
	c_ab = c1-c2
	h_ab = (a1-a2)*(a1-a2)+(b1-b2)*(b1-b2) - c_ab*c_ab
	return del_L*del_L + c_ab *c_ab /(1+K1*c1)/(1+K1*c1) + h_ab / (1+K2*c1)/(1+K2*c1)
	#return np.sum(np.square(c1-c2))

def labDistance(c1,c2):
	return math.sqrt(labDistance2(c1,c2))

def getSigma(centers):
	'''
		compute params sigma: mean distance between all pairs of original palette color
	'''
	dist_sum = 0
	size = centers.shape[0]
	for i in range(size):
		for j in range(i+1,size):
			dist_sum += distance(centers[i],centers[j])
	return dist_sum/(size*(size-1)/2)

def getLambda(equation_group,solution):
	return np.matmul(solution,np.linalg.inv(equation_group))
def updateLab(self, src, dst):
	#both src and dst are lab color	
	return dst

def isLegalRGB(col_rgb):
	return (col_rgb[0]>=0 and col_rgb[0]<256) and (col_rgb[1]>=0 and col_rgb[1]<256) and (col_rgb[2]>=0 and col_rgb[2]<256)

def isLegalLab(col_lab):
	return isLegalRGB(lab2rgb(col_lab))

def labBoundary(src,dst):
	mid = (src+dst)/2
	#print("Boundary:src={} dst={}".format(src,dst))
	if distance2(src,dst)<0.0001 :
		return mid
	if not isLegalLab(mid):
		return labBoundary(src,mid)
	else:
		return labBoundary(mid,dst)

def labIntersect(src,dst):
	if not isLegalLab(dst):
		return labBoundary(src,dst)
	else:
		return labIntersect(dst,dst+dst-src)
