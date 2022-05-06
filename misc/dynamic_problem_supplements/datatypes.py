'''
introduce basic datatypes that can later be made into customized types
...gonna think of these as more as data rather than args/hyperparams that get mutated

Not sure how to dynamically make class names on the fly. Going the less ideal way of
actually writing a separate python script...but I don't like it.
'''

class MyInt(int):
    def __new__(cls, value):
        return super(MyInt, cls).__new__(cls, value)


'''
# other datatypes:
bool
list
dict
tuple
long
set
'''
fake_json = {'datatypes': {'SomeName': int,
                           'AnotherOne': int,
                           'HowAboutStr': str,
                           'TryFloat': float}}


def write_new_class(filehandle, name, dtype):
    filehandle.write("class %s(%s):\n" % (name, dtype))
    filehandle.write("\tdef __new__(cls, value):\n")
    filehandle.write("\t\treturn super(%s, cls).__new__(cls, value)\n\n" % (name))


with open("custom_datatypes.py", "w") as f:
    for class_name, dtype in fake_json['datatypes'].items():
        # don't now how to get str repr of dataclass...doing if statement for now
        if dtype==int:
            dtype_str = "int"
        elif dtype==float:
            dtype_str = "float"
        elif dtype==str:
            dtype_str = "str"
        else:
            print("can't process %s dtype yet" % dtype)
            continue
        write_new_class(f, class_name, dtype_str)




import custom_datatypes
ting = custom_datatypes.AnotherOne(2)
ting_float = custom_datatypes.TryFloat(2)
ting_str = custom_datatypes.HowAboutStr(2)
print(isinstance(ting, custom_datatypes.AnotherOne))





# okay now what happens if i want to write custom methods
'''
but do we really need to do any of this? if they want custom classes
is one thing but do they really want to restrict if one type of int
can be added to another type of int?
Surely we only need to strongly type this at the level of int vs float vs str
rather than at the level of their own custom datatypes.
'''

operator_dict = {}

def my_add(a,b):
    return a+b

operator_dict[my_add] = {'inputs': [int, int],
                         'args': [],
                         'output': int}


