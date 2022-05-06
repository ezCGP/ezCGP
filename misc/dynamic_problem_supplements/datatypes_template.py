'''
Generic lists of datatypes and how they can be mutated.
The intention is that they can then be inherited into a more customized and specific datatype.
'''

class MyInt():
	def __init__(self, value=None):
		if value is not None:
			self.value = int(value)
		else:
			self.mutate()

	def mutate(self):
		pass



class MyFloat():
	def __init__(self, value=None):
		if value is not None:
			self.value = float(value)
		else:
			self.mutate()

	def mutate(self):
		pass



class MyString():
	def __init__(self, value=None):
		if value is not None:
			self.value = str(value)
		else:
			self.mutate()

	def mutate(self):
		pass
