def matrixadd(m1,m2):
	for row in range(len(m1)):
		for column in range(len(m1[row])):
			m2[row][column]+=m1[row][column]
	return m2
	
def matrixmul(m1,m2):
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
	if len(matrix[0])!=len(matrix):raise Exception('The matrix must be square!\n'+str(len(matrix))+'x'+str(len(matrix[0]))+'\n'+str(matrix))
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
	if det(matrix)==0:raise Exception('The determinant of the matrix must be nonzero!')
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
	if len(m1[0])!=len(m2):raise Exception('The number of columns in the first matrix must equal the number of rows in the second!')
	if len(m2[0])!=len(m2):raise Exception('The divisor must be a square matrix!')
	if det(m2)==0:raise Exception('The determinant of the divisor must be nonzero!')
	return matrixmul(m1,inverse(m2))
	
def augmatrixsolve(matrix,augment):
	return matrixdiv(transpose(augment),transpose(matrix))

hw2a1=[[0,-1,6],[1,1,-7],[3,2,-1]]
hw2a2=[[5],[0],[2]]
