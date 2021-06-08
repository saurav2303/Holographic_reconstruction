import os
# Function to rename multiple files

i = 1
path="F:/hologram/"
for filename in os.listdir(path):
  my_dest ="HOLO_" + str(i).zfill(3) + ".jpg"
  my_source =path + filename
  my_dest =path + my_dest
  # rename() function will
  # rename all the files
  os.rename(my_source, my_dest)
  i += 1
  
  #it is necessary to rename whole image file name as os.listdir() take care of alphanumeric order
