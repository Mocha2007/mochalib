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
	for row in matrix:
		for value in row:
			value=value*c
	return matrix
	
def transpose(matrix):
	newmatrix=[]
	for row in range(len(matrix)):
		newrow=[]
		for column in range(len(matrix[row])):
			newrow+=[matrix[column][row]]
		newmatrix+=[newrow]
	return newmatrix

def identity(size):
	matrix=[]
	for row in range(size):
		newrow=[]
		for column in range(size):
			newrow+=[1 if row==column else 0]
		matrix+=[newrow]
	return matrix