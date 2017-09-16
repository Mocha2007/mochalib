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
	matrix=[]
	for row in range(size):
		newrow=[]
		for column in range(size):
			newrow+=[1 if row==column else 0]
		if newrow!=[]:
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
	if det(m2)==0:raise Exception('The determinant of the divisor must be nonzero!')
	return matrixmul(m1,inverse(m2))
	
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
	o=matrixmul(matrix,matrix)
	while o!=roundm(matrixmul(o,matrix),10):
		o=roundm(matrixmul(o,matrix),10)
	return o

def disp(matrix):
	for row in matrix:
		print(row)