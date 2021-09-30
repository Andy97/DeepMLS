# Script for tfrecords generation

Before using this script, please first compile [main.cpp](https://github.com/Andy97/DeepMLS/tree/master/vdb_tsdf) and get an executable file named "vdb_tsdf".

The executable will be used to generate groudtruth sdf values and gradients as supervision for training, which will be used in script. ([here](https://github.com/Andy97/DeepMLS/blob/master/scripts/generate_tf_records.py#L79))

For each Shapenet model, there will be **2 input files** to this script:

#### **1.Watertight mesh file**：the watertight version of each original shapenet model, generated using scripts from occupancy networks
We assume that each mesh file has already been normalized to [-0.95, 0.95]^3 (which equals to [-1,1]^3 with 5% padding).

This input is declared [here](https://github.com/Andy97/DeepMLS/blob/master/scripts/generate_tf_records.py#L53).

#### **2.Sampled points file**: sampled 100k points with normal from input #1
This input is the groundtruth points sampled from surface, we will draw 3k points from these points plus gaussian noise as network input.

Please also note that since these points are sampled from input #1, all these points should also be bounded by bounding box [-0.95, 0.95]^3.

This input is declared [here](https://github.com/Andy97/DeepMLS/blob/master/scripts/generate_tf_records.py#L42).
