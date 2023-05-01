# import uuid
# filename = str(uuid.uuid4())
# # print(filename)

# import os

# # folder path
# dir_path = r'Tensorflow\\workspace\\images\\train'

# # list file and directories
# res = os.listdir(dir_path)
# count = 0
# for i in res:
#     file_extension = i.split(".")
#     old_file = os.path.join(dir_path,i)
#     new_file = os.path.join(dir_path,filename+'.'+file_extension[1])  
#     os.rename(old_file, new_file) 
#     count = count + 1
#     if(count % 2) == 0:
#         filename = str(uuid.uuid4())


# ------------------------------------------------------------------------------------------------
# rename XML's annotation

import os
import xml.etree.ElementTree as ET

for file in os.listdir('xml'):
    file_name = file
    full_file = os.path.abspath(os.path.join('xml',file_name))

    # Passing the path of the
    # xml document to enable the
    # parsing process
    tree = ET.parse(full_file)
    print(tree)
    
    # getting the parent tag of
    # the xml document
    root = tree.getroot()
    names = tree.findall('./filename')
    filename = file.split('.')

    for n in names:
        originalExt = n.text.split('.')
        n.text = filename[0] +'.'+ originalExt[1]
        print(n.text)


    tree.write('renameXML/'+file_name)