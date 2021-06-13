
***** geekforgeeks demo *******

from bs4 import BeautifulSoup 
# Reading the data inside the xml 
# file to a variable under the name 
# data 
with open('dict.xml', 'r') as f: 
	data = f.read() 

# Passing the stored data inside 
# the beautifulsoup parser, storing 
# the returned object 
Bs_data = BeautifulSoup(data, "xml") 

# Finding all instances of tag 
# `unique` 
b_unique = Bs_data.find_all('unique') 

print(b_unique) 

# Using find() to extract attributes 
# of the first instance of the tag 
b_name = Bs_data.find('child', {'name':'Frank'}) 

print(b_name) 

# Extracting the data stored in a 
# specific attribute of the 
# `child` tag 
value = b_name.get('test') 

print(value) 

*********************************************************************

*************** coding scripted.xml ********************************

import xml.etree.ElementTree as ET  

tree = ET.parse('blank_.xml')      *** we are going to ADD ( x_position , y_position ) of stage ***
root = tree.getroot()
no_name = root.find('no_name')


for j in range (1) :
    for i in range (41):

        b = ET.SubElement(no_name, 'Point_row{}{}'.format(j+1,str(i).zfill(5)))
        b.set('runtype','NDSetupMultipointListItem')
        bChecked = ET.SubElement(b, 'bChecked') 
        strName = ET.SubElement(b, 'strName') 
        dXPosition = ET.SubElement(b, 'dXPosition')
        dYPosition = ET.SubElement(b, 'dYPosition')
        dZPosition = ET.SubElement(b, 'dZPosition')
        dPFSOffset = ET.SubElement(b, 'dPFSOffset')
        baUserData = ET.SubElement(b, 'baUserData')

        bChecked.set('runtype', 'bool') 
        bChecked.set('value', 'true') 
        strName.set('runtype', 'CLxStringW') 
        strName.set('value', '') 
        dXPosition.set('runtype', 'double') 
        if (j==0 and i==0):
          dXPosition.set('value', '{}'.format(10000)) 
          dYPosition.set('runtype', 'double') 
          dYPosition.set('value', '{}'.format(-11000)) 
          dZPosition.set('runtype', 'double') 
        elif (j==0):
          dXPosition.set('value', '{}'.format(10000-(674*i)+67*i)) 
          dYPosition.set('runtype', 'double') 
          dYPosition.set('value', '{}'.format(-11000)) 
          dZPosition.set('runtype', 'double')
        else :
          dXPosition.set('value', '{}'.format(10000-(674*i)+67*i)) 
          dYPosition.set('runtype', 'double') 
          dYPosition.set('value', '{}'.format(-11000+(670*j)-67*i))
          dZPosition.set('runtype', 'double')
        dZPosition.set('value', '2730.860000000000127') 
        dPFSOffset.set('runtype', 'double') 
        dPFSOffset.set('value', '-1.000000000000000') 
        baUserData.set('runtype', 'CLxByteArray') 
        baUserData.set('value', '') 


print (ET.tostring(root))
with open("blank.xml", "wb") as f:    ************** now we will have our blank xml file updated ************
    f.write(ET.tostring(root)) 

