'''
start scoping out how to write the mdf class file
'''




def write_imports(filehandle):
    modules = ['os',
               'sys',
               #('numpy','np'),
               'custom_datatypes']

    for module in modules:
        if isinstance(module, str):
            filehandle.write("import %s\n" % module)
        elif isinstance(module, tuple):
            assert(len(module)==2), "Given tuple %s has unexpected shape" % module
            module_name, alias = module
            filehandle.write("import %s as %s\n" % (module_name, alias))
        else:
            print("unexpected datatype %s" % module)
            continue

    filehandle.write("\n") # trailing for gap 


def start_class(filehandle, *arg_list):
    attributes = {'this': 3}
    filehandle.write("class My_MDF():\n")
    arg_str = "self"
    for arg in arg_list:
        arg_str += ", %s" % arg
    filehandle.write("\tdef __init__(%s):\n" % arg_str)

    for key, value in attributes.items():
        filehandle.write("\t\tself.%s = %s\n" % (key, value))

    for arg in arg_list:
        filehandle.write("\t\tself.%s = %s\n" % (arg, arg))

    # not great python syntax but technically nothing wrong with always writing pass,
    # and it'll come in handy if arg_list and attributes are empty
    filehandle.write("\t\tpass\n")
    filehandle.write("\n") # gap



def write_method(filehandle, method_str):
    # assume method_str already has all the \n and stuff in it
    if isinstance(method_str, str):
        # wait, this may be a problem if method_str doesn't have the right indents on each line. TODO. be cautious.
        filehandle.write("%s\n" % method_str) # add trailing new line for gaps between methods

    elif isinstance(method_str, list):
        # maybe it'll be easier if it's just a list of str to write so that we can handle the index here and not earlier
        for line in method_str:
            filehandle.write("\t%s\n" % line)
        filehandle.write("\n") # trailing new line for gap between methods

    else:
        print("can't handle this dtype (%s) yet in write_method" % method_str)
        return



# MAIN
filename = "my_mdf.py"
fake_method = ["def poop(self,a,b):",
               "\tprint('Look! Poop! %s -> %s') % (a,b)"]
with open(filename, "w") as f:
    write_imports(f)
    start_class(f, 'that')
    write_method(f, fake_method)

print("Done")
