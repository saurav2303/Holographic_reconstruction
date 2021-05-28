from ij import IJ
from ij.io import FileSaver
from ij.io import OpenDialog
from ij import WindowManager
import time
import timeit


"""od= OpenDialog("choose file",None)
filename = od.getFileName()
directory = od.getDirectory()
path = od.getPath()"""
time_start = time.clock()
for i in range (144,1280,16):
	  if i==144:
		  imp = IJ.openImage("C:/Users/Saurav/Desktop/Raw_imgs/10_beads/stitch_gap16/croppedPNG_144.jpg");
		  IJ.run(imp,"32-bit","");
		  imp.show();
		  imp = IJ.openImage("C:/Users/Saurav/Desktop/Raw_imgs/10_beads/stitch_gap16/croppedPNG_160.jpg");
		  IJ.run(imp, "32-bit", "");
		  imp.show();
		  IJ.run(imp, "2D Stitching", "first_image=croppedPNG_144.jpg use_channel_for_first=[Red, Green and Blue] second_image=croppedPNG_160.jpg use_channel_for_second=[Red, Green and Blue] use_windowing how_many_peaks=5 create_merged_image fusion_method=[Linear Blending] fusion_alpha=1.50 fused_image=380 compute_overlap x=0 y=0");
		  imp.show();
		  p=IJ.getImage();
		  IJ.saveAs(p, "png", "C:/Users/Saurav/Desktop/Raw_imgs/10_beads/stitch_gap16/160.png");
	  else :
		  imp = IJ.openImage("C:/Users/Saurav/Desktop/Raw_imgs/10_beads/stitch_gap16/{}.png".format(i));
		  IJ.run(imp, "32-bit", "");
		  imp.show();
		  imp = IJ.openImage("C:/Users/Saurav/Desktop/Raw_imgs/10_beads/stitch_gap16/croppedPNG_{}.jpg".format(i+16));
		  IJ.run(imp, "32-bit", "");
		  imp.show();
		  IJ.run(imp, "2D Stitching", "first_image={}.png use_channel_for_first=[Red, Green and Blue] second_image=croppedPNG_{}.jpg use_channel_for_second=[Red, Green and Blue] use_windowing how_many_peaks=5 create_merged_image fusion_method=[Linear Blending] fusion_alpha=1.50 fused_image={} compute_overlap x=0 y=0".format(i,i+16,i+16));
		  imp.show();
		  p=IJ.getImage();
		  IJ.saveAs(p, "png", "C:/Users/Saurav/Desktop/Raw_imgs/10_beads/stitch_gap16/{}.png".format(i+16));



#run your code
time_elapsed = (time.clock() - time_start)
print("total computation time",time_elapsed)
