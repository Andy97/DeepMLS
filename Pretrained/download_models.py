import wget
import zipfile
import os

if __name__ == '__main__':
  print("Downloading Pretrained Models...")
  if(not os.path.exists("d7_1p_pretrained.zip")):
    print("Downloading d7_1p_pretrained model...")
    wget.download("http://home.ustc.edu.cn/~freelin/DeepMLS/Pretrained_Models/d7_1p_pretrained.zip", 'd7_1p_pretrained.zip')
    print("\n")
  if(not os.path.exists("d6_1p_pretrained.zip")):
    print("Downloading d6_1p_pretrained model...")
    wget.download("http://home.ustc.edu.cn/~freelin/DeepMLS/Pretrained_Models/d6_1p_pretrained.zip", 'd6_1p_pretrained.zip')
    print("\n")
  
  if(not os.path.exists("d7_1p_pretrained")):
    with zipfile.ZipFile("d7_1p_pretrained.zip", 'r') as zip_ref:
      zip_ref.extractall("")
  
  if(not os.path.exists("d6_1p_pretrained")):
    with zipfile.ZipFile("d6_1p_pretrained.zip", 'r') as zip_ref:
      zip_ref.extractall("")