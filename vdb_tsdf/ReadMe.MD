## Groundtruth SDF Data Generation
Given a watertight mesh, we use [OpenVDB](https://www.openvdb.org/) library to generate the tsdf field.  
As clarified in paper,  more sdf samples near the surface will be chosen.  
For details, please refer to the source code.

### Dependencies
##### 1. [OpenVDB](https://www.openvdb.org/)
##### 2. [OpenMesh](https://www.openmesh.org/)
