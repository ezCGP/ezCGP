import numpy as np

# maybe consider doing this for symbolic regression data

class MyData(np.ndarray):
	'''
	I got this after using an __init__ method without even doing a np.ndarray.__init__
	 * TypeError: Required argument 'shape' (pos 1) not found
	Stackoverflow says to use __new__ instead of __init__
	'''
	'''
	def __init__(self, data=[]):
		print("first")
		self.data = np.array(data)
		self.name = "bob"
		self.count = 11
		#super().__init__(shape=self.data.shape)
	'''

	# this is what was recommended
	def __new__(cls, data):
		ting = np.asarray(data).view(cls)
		#import pdb; pdb.set_trace()
		ting.name = "bob"
		ting.count = 11
		return ting


class MyData2(np.ndarray):
	def __new__(self, data):
		self.data = np.array(data)
		self.name = "joe"
		return self


class MyData3(np.ndarray):
	def __new__(cls, data):
		instance = np.asarray(data).view(cls)
		instance.mydata = np.array(data) # CAREFUL, np.ndarray has a .data attribute already
		instance.name = "elmo"
		return instance

ting = MyData(data=[1,2,3]) #this worked but it's weird
ting2 = MyData2(data=[1,2,3]) #this worked BUT isinstance(ting2, np.ndarray) returns False
ting3 = MyData3(data=[1,2,3]) #this worked




# what about if we want arg data types like int
class MyArgInt(int):
	def __init__(self, value=None):
		if value is None:
			self.value = np.random.randint(0,10)
		else:
			self.value = value

	def mutate(self):
		self.value = np.random.randint(0,10)


arg0 = MyArgInt()
print(arg0, type(arg0)) #NOTE printing arg0 it tries to use the int.__str__() or int.__repr__()
assert(isinstance(arg0,int)), "arg0 not an int before"
print("before:", arg0.value)
arg0.mutate()
print("after:", arg0.value)
assert(isinstance(arg0,int)), "arg0 not an int after"




# and floats?
class MyArgFloat(float):
	def __init__(self, value=None):
		if value is None:
			self.value = np.random.random()*10
		else:
			self.value = value

	def mutate(self):
		self.value = np.random.random()*10

	def __str__(self):
		#print(arg1) runs arg1.__str__()
		return "hi!"+str(self.value)

	def __repr__(self):
		return str(self.value)

arg1 = MyArgFloat()
print(arg1, type(arg1))
assert(isinstance(arg1, float)), "arg1 not an float before"
print("before:", arg1.value)
arg1.mutate()
print("after:", arg1.value)
assert(isinstance(arg1, float)), "arg1 not an float after"






class MyArgFloat1(float):
	def __new__(cls, value=None):
		if value is None:
			value = np.random.random()*10
		instance = super().__new__(MyArgFloat1, value)
		instance.value = value
		return instance

	def mutate(self):
		value = np.random.random()
		#self = super().__new__(MyArgFloat1, value)
		#self.value = value
		self = MyArgFloat1(value)
		return self


arg2 = MyArgFloat1()
print(arg2, type(arg2))
assert(isinstance(arg2, float)), "arg2 not a float before"
print("before:", arg2, arg2.value)
arg2 = arg2.mutate()
print("after", arg2, arg2.value) # only worked when we return something in mutate()...not what we want though






class A():
	def __init__(self, ting):
		self.ting = ting

class B(A):
	def __init__(self):
		self.other = "poop"
		super().__init__(44)

this = A(5)
that = B()