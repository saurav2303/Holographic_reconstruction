p=[]
cropped=[]
i=0
path = "F:/image_directory"
for file in os.listdir(path):  
    img=cv2.imread(file)
    p.append(img)
    

    for j in range(4) :
        cropped.append(p[i][500:1000,500*j :500*(j+1)])
        imageio.imwrite('cropped_{}.jpg'.format(str(j+801+(i*4)).zfill(4)) ,cropped[j+(i*4)])
    i+=1
    
    #this is going to pick four 500X500 size image from each image in directory
