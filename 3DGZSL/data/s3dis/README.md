This is the folder for the S3DIS dataset, the overall original file structure is as follow:
Stanford3dDataset_v1.2_Aligned_Version/
  Area_1/xxxx/Annotations + xxxx.txt
  Area_2/xxxx/Annotations + xxxx.txt
  Area_3/xxxx/Annotations + xxxx.txt
  Area_4/xxxx/Annotations + xxxx.txt
  Area_5/xxxx/Annotations + xxxx.txt
  Area_6/xxxx/Annotations + xxxx.txt
  
Notice: there is an extra character in the Stanford3dDataset_v1.2_Aligned_Version data in Area_5/hallway_6. Please fix it manually. "Area_5 -> hallway_6 -> Annotations -> ceiling_1.txt -> line 180389 -> del NUL"

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
After proprecessing, the folder 'processed_data' can be generated in this folder.
