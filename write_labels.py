import numpy as np
import os 

for filename in os.listdir("/media/asad/adas_cv_2/caffe/labels/labels"):
    filename=filename.split(".")[:-1]
    filename="".join(filename)+".png"
    image_path=os.path.join("/media/asad/adas_cv_2/caffe/football",filename)
    assert os.path.exists(image_path)
    with open("/media/asad/adas_cv_2/caffe/train.txt","a") as f:
        f.writelines(image_path)
        f.write("\n")
