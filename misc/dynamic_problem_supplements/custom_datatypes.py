class SomeName(int):
	def __new__(cls, value):
		return super(SomeName, cls).__new__(cls, value)

class TryFloat(float):
	def __new__(cls, value):
		return super(TryFloat, cls).__new__(cls, value)

class HowAboutStr(str):
	def __new__(cls, value):
		return super(HowAboutStr, cls).__new__(cls, value)

class AnotherOne(int):
	def __new__(cls, value):
		return super(AnotherOne, cls).__new__(cls, value)

