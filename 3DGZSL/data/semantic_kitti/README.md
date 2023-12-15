This is the folder for the SemanticKITTI dataset: 
First, download the original dataset from the http://semantic-kitti.org/dataset.html#download, including: 
1. KITTI Odometry Benchmark Velodyne point clouds (80 GB)
2. KITTI Odometry Benchmark calibration data (1 MB)
3. Download SemanticKITTI label data (179 MB)

Decompress the files, and reformat the overall file structure as follow: 
- sequences
  - 00
    velodyne (000000.bin, ..., xxxxxx.bin)
    labels (000000.label, ..., xxxxxx.label)
    calib.txt
    times.txt
    poses.txt
  - 01
    xxx
  - 02
    xxx
  ... 
  - 
    xxx
- semantic-kitti.yaml
- semantic-kitti-all.yaml
   
  


