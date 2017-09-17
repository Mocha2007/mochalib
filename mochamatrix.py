from random import random
from math import cos,sin

def rmatrix(rows,columns):
	newmatrix=[]
	for row in range(rows):
		newrow=[]
		for column in range(columns):
			newrow+=[random()]
		newmatrix+=[newrow]
	return newmatrix

def matrixadd(m1,m2):
	if [len(m1),len(m1[0])]!=[len(m2),len(m2[0])]:raise Exception('The matrices must be the same size!\n'+str(len(m1))+'x'+str(len(m1[0]))+', '+str(len(m2))+'x'+str(len(m2[0])))
	for row in range(len(m1)):
		for column in range(len(m1[row])):
			m2[row][column]+=m1[row][column]
	return m2
	
def matrixmul(m1,m2):
	if len(m1[0])!=len(m2):raise Exception('The number of columns in the first matrix must equal the number of rows in the second!\n'+str(len(m1))+'x'+str(len(m1[0]))+', '+str(len(m2))+'x'+str(len(m2[0])))
	new=[]
	for row in range(len(m1)):
		newrow=[]
		for column in range(len(m1[row])):
			if column==0:#cells dont exist yet, so i have to make them
				for value in m2[0]:
					newrow+=[m1[row][0]*value]
			else:#now i can just add
				for value in range(len(m2)):
					newrow[value]+=m1[row][column]*m2[column][value]
		new+=[newrow]
	return new
	
def matrixscalar(matrix,c):
	newmatrix=[]
	for row in matrix:
		newrow=[]
		for value in row:
			newrow+=[value*c]
		newmatrix+=[newrow]
	return newmatrix

def matrixsub(m1,m2):
	return matrixadd(m1,matrixscalar(m2,-1))
	
def transpose(matrix):
	newmatrix=[]
	for row in range(max(len(matrix),len(matrix[0]))):
		newrow=[]
		for column in range(max(len(matrix),len(matrix[0]))):
			try:
				newrow+=[matrix[column][row]]
			except IndexError:
				pass
		newmatrix+=[newrow] if newrow!=[] else []
	return newmatrix

def identity(size):
	if size%1!=0 or size<1:raise ValueError('The size must be a natural number!')
	matrix=[]
	for row in range(size):
		newrow=[]
		for column in range(size):
			newrow+=[1 if row==column else 0]
		matrix+=[newrow]
	return matrix

def zero(size):
	if size%1!=0 or size<1:raise ValueError('The size must be a natural number!')
	matrix=[]
	for row in range(size):
		newrow=[]
		for column in range(size):
			newrow+=[0]
		matrix+=[newrow]
	return matrix
	
def det(matrix):
	if len(matrix[0])!=len(matrix):raise Exception('The matrix must be square!\n'+str(len(matrix))+'x'+str(len(matrix[0])))
	if len(matrix)==1:return matrix[0][0]
	if len(matrix)==2:return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
	s=0
	for row in range(len(matrix)):
		constant=matrix[row][0]
		#now time to find the smaller matrix
		newmatrix=[]
		for row2 in range(len(matrix)):
			newrow=[]
			for column2 in range(len(matrix[row])):
				if row2!=row and column2!=0:
					newrow+=[matrix[row2][column2]]
			newmatrix+=[newrow] if len(newrow)>0 else []
		value=constant*det(newmatrix)*(-1)**(row)
		#print(constant,'* det(',newmatrix,') *',(-1)**(row))
		s+=value
	return s

def adj(matrix):
	if len(matrix[0])!=len(matrix):raise Exception('The matrix must be square!\n'+str(len(matrix))+'x'+str(len(matrix[0])))
	matrix=transpose(matrix)
	adjmatrix=[]
	for row in range(len(matrix)):
		adjrow=[]
		for column in range(len(matrix[0])):
			#now time to find the det of the smaller matrix
			newmatrix=[]
			for row2 in range(len(matrix)):
				newrow=[]
				for column2 in range(len(matrix[row])):
					if row2!=row and column2!=column:
						newrow+=[matrix[row2][column2]]
				newmatrix+=[newrow] if len(newrow)>0 else []
			value=det(newmatrix)*(-1)**(row+column)
			adjrow+=[value]
		adjmatrix+=[adjrow]
	return adjmatrix

def inverse(matrix):
	return matrixscalar(adj(matrix),1/det(matrix))
	
def matrixdiv(m1,m2):
	if len(m1[0])!=len(m2):raise Exception('The number of columns in the first matrix must equal the number of rows in the second!\n'+str(len(m1))+'x'+str(len(m1[0]))+', '+str(len(m2))+'x'+str(len(m2[0])))
	if len(m2[0])!=len(m2):raise Exception('The divisor must be a square matrix!\n'+str(len(m2))+'x'+str(len(m2[0])))
	if det(m2)==0:raise ZeroDivisionError('The determinant of the divisor must be nonzero!')
	return matrixmul(m1,inverse(m2))

def matrixexp(matrix,exp):
	nm=[]
	for r in matrix:
		nr=[]
		for c in r:
			nr+=[c]
		nm+=[nr]
	for i in range(exp-1):
		nm=matrixmul(nm,matrix)
	return nm
	
def augmatrixsolve(matrix,augment):
	return matrixdiv(transpose(augment),transpose(matrix))

def roundm(matrix,places):
	nm=[]
	for row in matrix:
		nr=[]
		for value in row:
			if abs(int(value)-value)<10**-places:nr+=[int(value)]
			else:nr+=[round(value,places)]
		nm+=[nr]
	return nm

def infpower(matrix):
	if len(matrix[0])!=len(matrix):raise Exception('The matrix must be square!\n'+str(len(matrix))+'x'+str(len(matrix[0])))
	for row in matrix:
		pass
		#if abs(sum(row))>1:raise ValueError("The absolute value of each row's sum must be equal to or less than one!")
	o=matrixmul(matrix,matrix)
	while o!=roundm(matrixmul(o,matrix),10):
		o=roundm(matrixmul(o,matrix),10)
	return o

def disp(matrix):
	for row in matrix:
		print(row)

def companion(*coefficients):
	if coefficients[0]!=1:
		c=coefficients[0]
		for i in range(len(coefficients)):
			coefficients[i]=coefficients[i]/c
	coefficients=coefficients[::-1][:-1]#reverse list then remove last value
	nm=[]
	for r in range(len(coefficients)):
		nr=[]
		for c in range(len(coefficients)):
			if r==c+1:nr+=[1]#the diagonalish ones
			elif c+1!=len(coefficients):nr+=[0]#the filler zeroes
			else:nr+=[-coefficients[r]]#must be the coefficients
		nm+=[nr]
	return nm

def trace(matrix):
	if len(matrix[0])!=len(matrix):raise Exception('The matrix must be square!\n'+str(len(matrix))+'x'+str(len(matrix[0])))
	trace=0
	for i in range(len(matrix)):
		trace+=matrix[i][i]
	return trace

def smallmatrixsqrt(matrix):
	if len(matrix[0])!=len(matrix) or len(matrix) not in [1,2]:raise Exception('The matrix must be 1x1 or 2x2!\n'+str(len(matrix))+'x'+str(len(matrix[0])))
	if len(matrix)==1:return [[matrix[0][0]**.5]]
	#fixes no response for zero matrix
	if matrix==zero(2):return (zero(2))
	tau=trace(matrix)
	d=det(matrix)
	s=d**.5
	t=(tau+2*s)**.5
	skip12=False
	skip34=False
	try:
		root1=matrixscalar(matrixadd(matrix,matrixscalar(identity(2),s)),1/t)#+s+t
		root2=matrixscalar(matrixadd(matrix,matrixscalar(identity(2),s)),-1/t)#+s-t
	except ZeroDivisionError:
		skip12=True
	t=(tau-2*s)**.5
	try:
		root3=matrixscalar(matrixadd(matrix,matrixscalar(identity(2),-s)),1/t)#-s+t
		root4=matrixscalar(matrixadd(matrix,matrixscalar(identity(2),-s)),-1/t)#-s-t
	except ZeroDivisionError:
		skip34=True
	if skip12 and skip34:return None
	if skip12:return root3,root4
	if skip34:return root1,root2
	return root1,root2,root3,root4

def flipy(matrix):
	nm=[]
	for r in matrix:
		nm+=[r[::-1]]
	return nm

def flipx(matrix):
	return matrix[::-1]
	
def rot(matrix,quarterturns):
	if quarterturns%1!=0:raise ValueError('The number of quarter-turns must be an integer!\n'+str(quarterturns))
	if quarterturns%4==0:return matrix
	if quarterturns%4!=1:return rot(rot(matrix,quarterturns-1),1)
	return flipy(transpose(matrix))

def rotationmatrix(theta):
	return [[cos(theta),-sin(theta)],[sin(theta),cos(theta)]]

def disp2(matrix):
	for r in matrix:
		rs=''
		for c in r:
			rs+='\u2588' if c==1 else ' '
		print(rs)

def clean(matrix):
	#clean up unnecessary negatives and floats
	for i in range(len(matrix)):
		for j in range(len(matrix[i])):
			if matrix[i][j]%1==0:matrix[i][j]=int(matrix[i][j])
	return matrix

def echelon(matrix):
	for i in range(len(matrix)):
		#step 1: see if the 'corner' is nonzero. if not, swap shit/etc
		found=1
		if matrix[i][i]==0:
			found=0
			#check to see if another row has it
			for j in range(i+1,len(matrix)):
				if matrix[j][i]!=0:#if so, switch the rows!
					oldi=matrix[i][:]
					oldj=matrix[j][:]
					matrix[i]=oldj
					matrix[j]=oldi#verified to work
					found=1
					break
		if found:
			c=matrix[i][i]
			#div errything in that row so the left is 1
			for j in range(len(matrix[i])):
				matrix[i][j]=matrix[i][j]/c
			#check to see if this column in the lower rows all start with zeroes. if not, make it so!
			for j in range(i+1,len(matrix)):
				if matrix[j][i]!=0:
					c=matrix[j][i]
					for k in range(len(matrix[j])):
						matrix[j][k]-=c*matrix[i][k]
	#move all zero rows to bottom
	for i in range(len(matrix)):
		allzeroes=1
		for j in range(len(matrix)):
			if matrix[i][j]!=0:
				allzeroes=0
				break
		if allzeroes:
			matrix.append(matrix.pop(i))
	#verify the leading terms are one AND all things below them are zeroes (a temporary bugfix)
	for i in range(len(matrix)):
		if matrix[i]!=[0]*len(matrix):#skip all zero rows
			leading=-1
			for j in range(len(matrix[0])):
				if matrix[i][j]!=0:
					leading=j
					break
			#verify it is a ONE
			c=matrix[i][leading]
			for j in range(len(matrix[0])):
				matrix[i][j]=matrix[i][j]/c
			#check rows below leading
			for j in range(i+1,len(matrix)):
				if matrix[i]!=[0]*len(matrix):#skip all zero rows
					#find leading term
					for k in range(len(matrix[0])):
						if matrix[j][k]!=0:
							newlead=k
							break
					#if leading term below, then sub all terms by c (since we just verified the leading above is a one)
					if newlead<=leading and matrix[j][newlead]!=0:#the and is there cause python is retarded
						for k in range(len(matrix[0])):
							matrix[j][k]-=matrix[i][k]
				else:break
	return clean(matrix)

def rre(matrix):
	matrix=echelon(matrix)
	#check each row and check the column with the 1st 1
	for i in range(len(matrix)):
		#check each column for the nonzero
		firstnonzero=-1
		for j in range(len(matrix[0])):
			if matrix[i][j]!=0:
				firstnonzero=j
				break
		#check each row above for other nonzeros
		for j in range(i):
			if matrix[j][firstnonzero]!=0:
				#make it zero!!!
				c=matrix[j][firstnonzero]
				#for each row element, subtract c*that one
				for k in range(len(matrix[0])):
					matrix[j][k]-=c*matrix[i][k]
	return clean(matrix)
	

alphabet='abcdefghijklmnopqrstuvwxyz'
lm={
	'a':[[1,1,1,1,1],[1,0,0,0,1],[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1]],
	'b':[[1,1,1,1,0],[1,0,0,0,1],[1,1,1,1,0],[1,0,0,0,1],[1,1,1,1,0]],
	'c':[[1,1,1,1,1],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,1]],
	'd':[[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,0]],
	'e':[[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,1]],
	'f':[[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,1],[1,0,0,0,0],[1,0,0,0,0]],
	'g':[[1,1,1,1,1],[1,0,0,0,0],[1,0,1,1,1],[1,0,0,0,1],[1,1,1,1,1]],
	'h':[[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1]],
	'i':[[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],
	'j':[[0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,1],[1,1,1,1,1]],
	'k':[[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1]],
	'l':[[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,1]],
	'm':[[1,0,0,0,1],[1,1,0,1,1],[1,0,1,0,1],[1,0,0,0,1],[1,0,0,0,1]],
	'n':[[1,0,0,0,1],[1,1,0,0,1],[1,0,1,0,1],[1,0,0,1,1],[1,0,0,0,1]],
	'o':[[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1]],
	'p':[[1,1,1,1,1],[1,0,0,0,1],[1,1,1,1,1],[1,0,0,0,0],[1,0,0,0,0]],
	'q':[[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,1,0,1],[1,1,1,1,1]],
	'r':[[1,1,1,1,1],[1,0,0,0,1],[1,1,1,1,1],[1,0,0,1,0],[1,0,0,0,1]],
	's':[[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,1],[0,0,0,0,1],[1,1,1,1,1]],
	't':[[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],
	'u':[[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1]],
	'v':[[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0]],
	'w':[[1,0,0,0,1],[1,0,0,0,1],[1,0,1,0,1],[1,1,0,1,1],[1,0,0,0,1]],
	'x':[[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1]],
	'y':[[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],
	'z':[[1,1,1,1,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,1,1,1,1]],
	' ':zero(5)
}

long=[[0.0, 0.01818181818181818, 0.0, 0.0, 0.0, 0.0, 0.03636363636363636, 0.05454545454545454, 0.0, 0.01818181818181818, 0.23636363636363636, 0.07272727272727272, 0.21818181818181817, 0.0, 0.2545454545454545, 0.01818181818181818, 0.01818181818181818, 0.0, 0.01818181818181818, 0.0, 0.03636363636363636, 0.0], [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0], [0.06666666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.24, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.04, 0.2, 0.0, 0.0, 0.12, 0.28], [0.0, 0.0, 0.0, 0.0, 0.0, 0.30434782608695654, 0.0, 0.0, 0.0, 0.0, 0.2608695652173913, 0.043478260869565216, 0.2608695652173913, 0.0, 0.0, 0.0, 0.043478260869565216, 0.0, 0.0, 0.043478260869565216, 0.043478260869565216, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0], [0.425, 0.0, 0.0, 0.0, 0.075, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.05, 0.0, 0.025, 0.25, 0.0, 0.0, 0.05, 0.075], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.42857142857142855, 0.0, 0.14285714285714285, 0.0, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0, 0.2857142857142857], [0.0625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0625, 0.125, 0.0, 0.0625, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.1875, 0.0, 0.0, 0.0, 0.25], [0.07407407407407407, 0.0, 0.0, 0.14814814814814814, 0.25925925925925924, 0.18518518518518517, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3333333333333333], [0.0, 0.0, 0.0, 0.0, 0.05263157894736842, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05263157894736842, 0.05263157894736842, 0.0, 0.05263157894736842, 0.7894736842105263, 0.0], [0.06666666666666667, 0.0, 0.28888888888888886, 0.24444444444444444, 0.022222222222222223, 0.0, 0.15555555555555556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044444444444444446, 0.044444444444444446, 0.0, 0.022222222222222223, 0.06666666666666667, 0.044444444444444446], [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.06666666666666667, 0.0, 0.0, 0.2, 0.06666666666666667, 0.0, 0.0, 0.0, 0.0, 0.43333333333333335, 0.0, 0.0, 0.03333333333333333, 0.03333333333333333, 0.0, 0.0, 0.0, 0.03333333333333333, 0.0, 0.0, 0.0, 0.13333333333333333], [0.5, 0.0, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0], [0.05263157894736842, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5526315789473685, 0.05263157894736842, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05263157894736842, 0.0, 0.0, 0.02631578947368421, 0.2631578947368421], [0.14285714285714285, 0.0, 0.0, 0.07142857142857142, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.21428571428571427, 0.14285714285714285, 0.21428571428571427, 0.0, 0.10714285714285714, 0.07142857142857142, 0.0, 0.03571428571428571, 0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0], [0.13793103448275862, 0.0, 0.034482758620689655, 0.0, 0.20689655172413793, 0.0, 0.034482758620689655, 0.0, 0.06896551724137931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034482758620689655, 0.10344827586206896, 0.0, 0.0, 0.0, 0.3793103448275862], [0.1320754716981132, 0.1509433962264151, 0.0, 0.018867924528301886, 0.018867924528301886, 0.0, 0.0, 0.0, 0.018867924528301886, 0.03773584905660377, 0.0, 0.18867924528301888, 0.1509433962264151, 0.018867924528301886, 0.1320754716981132, 0.03773584905660377, 0.018867924528301886, 0.0, 0.018867924528301886, 0.018867924528301886, 0.018867924528301886, 0.018867924528301886]]
