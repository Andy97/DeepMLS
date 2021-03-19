#pragma warning (disable: 4146)
#define _USE_MATH_DEFINES
#include <cmath>
#define NOMINMAX
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/GridOperators.h>

#include "OpenMesh\Core\IO\MeshIO.hh"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"

#include <iostream>
inline bool debug_output(const char *string)
{
	std::cout << "=======================================================\n";
	std::cout << "Assertion failed:" << string << "!!!\n";
	std::cout << "=======================================================\n";
	system("pause");
	return false;
}

#define myassert(expression) (bool)(       \
            (!!(expression)) ||            \
            (debug_output(#expression)) \
        )

// Select mesh type (TriMesh) and kernel (ArrayKernel)
// and define my personal mesh type (MyMesh)
struct DPTraits : public OpenMesh::DefaultTraits
{
	typedef OpenMesh::Vec3d Point; // use double-values points/normals
	typedef OpenMesh::Vec3d Normal;
};
typedef OpenMesh::TriMesh_ArrayKernelT<DPTraits>  TriMesh;

void fetch_sdf(const char *model_filename, const char *sdf_filename, int grid_resolution = 256)
{
	auto mesh = std::make_shared<TriMesh>();
	if (!OpenMesh::IO::read_mesh(*mesh, model_filename))
	{
		std::cerr << "Error: cannot read mesh file " << model_filename << std::endl;
		return;
	}
	std::cout << "Finished reading mesh.\n";

	double bbox[6] = { 1,1,1,-1,-1,-1 };
	std::vector<openvdb::Vec3f> verts;
	std::vector<openvdb::Vec3I> faces;
	for (auto vtx = mesh->vertices_begin(); vtx != mesh->vertices_end(); vtx++)
	{
		auto& pt = mesh->point(*vtx);
		verts.push_back(openvdb::Vec3f(pt[0], pt[1], pt[2]));

		//for (int i = 0; i < 3; i++)
		//{
		//	bbox[i] = std::max(pt[i], bbox[i]);
		//	bbox[i + 3] = std::min(pt[i], bbox[i + 3]);
		//}
	}
	for (auto face = mesh->faces_begin(); face != mesh->faces_end(); face++)
	{
		openvdb::Vec3I face_idx;
		auto fvhandle = mesh->fv_begin(*face);
		for (int vitr = 0; vitr < 3; vitr++, fvhandle++)
		{
			face_idx[vitr] = fvhandle->idx();
		}
		faces.push_back(face_idx);
	}

	double diag_len = (bbox[0] - bbox[3])*(bbox[0] - bbox[3]) + (bbox[1] - bbox[4])*(bbox[1] - bbox[4])
		+ (bbox[2] - bbox[5])*(bbox[2] - bbox[5]);
	diag_len = std::sqrt(diag_len);
	double center[3] = { bbox[0] + bbox[3], bbox[1] + bbox[4], bbox[2] + bbox[5] };
	for (int i = 0; i < 3; i++)
	{
		center[i] /= 2.0;
	}
	double voxel_size = 2.0 / grid_resolution;

	printf("Model center: %f,%f,%f, diagonal length: %f, voxel size: %f\n", center[0], center[1], center[2],
		diag_len, voxel_size);

	auto grid = openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(
		*openvdb::math::Transform::createLinearTransform(voxel_size),
		verts, faces, std::vector<openvdb::Vec4I>(), grid_resolution / 4.0, grid_resolution / 4.0
		);
	openvdb::tools::signedFloodFill(grid->tree());

	std::vector<std::vector<std::vector<double>>> sdf_grids;
	std::vector<std::vector<std::vector<openvdb::math::Vec3s>>> sdf_grids_grad;
	sdf_grids.resize(grid_resolution + 1);
	sdf_grids_grad.resize(grid_resolution + 1);
	for (int i = 0; i < sdf_grids.size(); i++)
	{
		sdf_grids[i].resize(grid_resolution + 1);
		sdf_grids_grad[i].resize(grid_resolution + 1);
		for (int j = 0; j < sdf_grids.size(); j++)
		{
			sdf_grids[i][j].resize(grid_resolution + 1);
			sdf_grids_grad[i][j].resize(grid_resolution + 1);
			std::fill(sdf_grids[i][j].begin(), sdf_grids[i][j].end(), -10000);
			std::fill(sdf_grids_grad[i][j].begin(), sdf_grids_grad[i][j].end(), openvdb::math::Vec3s(-10000, -10000, -10000));
		}
	}
	auto grad_field = openvdb::tools::gradient<openvdb::FloatGrid>(*grid);
	//openvdb::tools::signedFloodFill(grad_field->tree());
	myassert(grid_resolution % 2 == 0);
	int half_grid_resolution = grid_resolution / 2;

	int filled_value_count = 0;
	int maxx, maxy, maxz;
	int minx, miny, minz;
	maxx = maxy = maxz = -1;
	minx = miny = minz = 1;
	
	//extract sdf samples with fabs(sdf) < sdf_truncation
	const double sdf_truncation = 2.0 / 16;
	std::vector<OpenMesh::Vec3i> sampled_points;
	sampled_points.clear();
	
	//iterate the active voxels of grid
	// Iterate over all active values but don't allow them to be changed.
	for (openvdb::FloatGrid::ValueOnCIter iter = grid->cbeginValueOn(); iter.test(); ++iter) {
		const float& value = *iter;
		if (iter.isVoxelValue()) {
			//if (value <= 0)
			//	std::cout << iter.getCoord() << " value: " << value << std::endl;
			int x, y, z;
			x = iter.getCoord().x();
			y = iter.getCoord().y();
			z = iter.getCoord().z();

			maxx = std::max(x, maxx);
			maxy = std::max(y, maxy);
			maxz = std::max(z, maxz);

			minx = std::min(x, minx);
			miny = std::min(y, miny);
			minz = std::min(z, minz);

			x += half_grid_resolution;
			y += half_grid_resolution;
			z += half_grid_resolution;

			if (x >= 0 && y >= 0 && z >= 0 && x <= grid_resolution && y <= grid_resolution && z <= grid_resolution)
			{
				filled_value_count++;
				sdf_grids[x][y][z] = value;
				if (fabs(value) < sdf_truncation)
				{
					//aggressive generate more samples near surface
					if(fabs(value) < sdf_truncation / 4)
						sampled_points.push_back(OpenMesh::Vec3i(x, y, z));
					else if(x % 4 == 0 && y % 4 == 0 && z % 4 == 0)
						sampled_points.push_back(OpenMesh::Vec3i(x, y, z));
					else if(fabs(value) < sdf_truncation / 2 && x % 2 == 0 && y % 2 == 0 && z % 2 == 0)
						sampled_points.push_back(OpenMesh::Vec3i(x, y, z));
				}
			}
		}
		else
		{
			std::cout << "Not voxel node sdf value" << std::endl;
			//int junk;
			//std::cin >> junk;
		}
	}
	std::cout << maxx << "," << maxy << "," << maxz << "\n";
	std::cout << minx << "," << miny << "," << minz << "\n";
	std::cout << sampled_points.size() << " points with |sdf| < " << sdf_truncation << "\n";
	
	int filled_grad_count = 0;
	for (openvdb::Vec3SGrid::ValueOnCIter iter = grad_field->cbeginValueOn(); iter.test(); ++iter) {
		const openvdb::math::Vec3s& value = *iter;
		if (iter.isVoxelValue()) {
			//if (value <= 0)
			//	std::cout << iter.getCoord() << " value: " << value << std::endl;
			int x, y, z;
			x = iter.getCoord().x();
			y = iter.getCoord().y();
			z = iter.getCoord().z();

			maxx = std::max(x, maxx);
			maxy = std::max(y, maxy);
			maxz = std::max(z, maxz);

			minx = std::min(x, minx);
			miny = std::min(y, miny);
			minz = std::min(z, minz);

			x += half_grid_resolution;
			y += half_grid_resolution;
			z += half_grid_resolution;

			if (x >= 0 && y >= 0 && z >= 0 && x <= grid_resolution && y <= grid_resolution && z <= grid_resolution)
			{
				filled_grad_count++;
				sdf_grids_grad[x][y][z] = value;
			}
		}
		else
		{
			//std::cout << "Not voxel node sdf grad" << std::endl;
			//int junk;
			//std::cin >> junk;
		}
	}

	auto valid_grid_coeff = [](int x, int y, int z, int max)
	{
		if (x < 0 || y < 0 || z < 0)
			return false;
		if (x > max || y > max || z > max)
			return false;
		return true;
	};
	
	int coordx, coordy, coordz;
	FILE *wf = fopen(sdf_filename, "wb");
	std::vector<float> buffered_data;
	buffered_data.reserve(sampled_points.size() * 7);
	for (int i = 0; i < sampled_points.size(); i++)
	{
		OpenMesh::Vec3i point = sampled_points[i];
		coordx = point[0];
		coordy = point[1];
		coordz = point[2];
		myassert(fabs(sdf_grids[coordx][coordy][coordz]) < sdf_truncation);
		if (sdf_grids_grad[coordx][coordy][coordz].x() < -1000)
		{
			//openvdb does not generate gradient we want
			float gradx, grady, gradz;
			//center difference
			//x
			if (valid_grid_coeff(coordx - 1, coordy, coordz, grid_resolution))
			{
				myassert(sdf_grids[coordx - 1][coordy][coordz] > -1000);
				if (valid_grid_coeff(coordx + 1, coordy, coordz, grid_resolution))
				{
					myassert(sdf_grids[coordx + 1][coordy][coordz] > -1000);
					//center difference
					gradx = (sdf_grids[coordx + 1][coordy][coordz] - sdf_grids[coordx - 1][coordy][coordz]) / 2.0;
				}
				else
					gradx = sdf_grids[coordx][coordy][coordz] - sdf_grids[coordx - 1][coordy][coordz];
			}
			else
			{
				gradx = sdf_grids[coordx + 1][coordy][coordz] - sdf_grids[coordx][coordy][coordz];
			}

			//y
			if (valid_grid_coeff(coordx, coordy - 1, coordz, grid_resolution))
			{
				myassert(sdf_grids[coordx][coordy - 1][coordz] > -1000);
				if (valid_grid_coeff(coordx, coordy + 1, coordz, grid_resolution))
				{
					myassert(sdf_grids[coordx][coordy + 1][coordz] > -1000);
					//center difference
					grady = (sdf_grids[coordx][coordy + 1][coordz] - sdf_grids[coordx][coordy - 1][coordz]) / 2.0;
				}
				else
					grady = sdf_grids[coordx][coordy][coordz] - sdf_grids[coordx][coordy - 1][coordz];
			}
			else
			{
				grady = sdf_grids[coordx][coordy + 1][coordz] - sdf_grids[coordx][coordy][coordz];
			}

			//z
			if (valid_grid_coeff(coordx, coordy, coordz - 1, grid_resolution))
			{
				myassert(sdf_grids[coordx][coordy][coordz - 1] > -1000);
				if (valid_grid_coeff(coordx, coordy, coordz + 1, grid_resolution))
				{
					myassert(sdf_grids[coordx][coordy][coordz + 1] > -1000);
					//center difference
					gradz = (sdf_grids[coordx][coordy][coordz + 1] - sdf_grids[coordx][coordy][coordz - 1]) / 2.0;
				}
				else
					gradz = sdf_grids[coordx][coordy][coordz] - sdf_grids[coordx][coordy][coordz - 1];
			}
			else
			{
				gradz = sdf_grids[coordx][coordy][coordz + 1] - sdf_grids[coordx][coordy][coordz];
			}

			sdf_grids_grad[coordx][coordy][coordz] = openvdb::math::Vec3s(gradx, grady, gradz);
			sdf_grids_grad[coordx][coordy][coordz].normalize();
		}
		myassert(sdf_grids_grad[coordx][coordy][coordz].x() > -1000);
		sdf_grids_grad[coordx][coordy][coordz].normalize();
		buffered_data.push_back(2.0*coordx / grid_resolution - 1);
		buffered_data.push_back(2.0*coordy / grid_resolution - 1);
		buffered_data.push_back(2.0*coordz / grid_resolution - 1);
		buffered_data.push_back(sdf_grids[coordx][coordy][coordz]);
		buffered_data.push_back(sdf_grids_grad[coordx][coordy][coordz].x());
		buffered_data.push_back(sdf_grids_grad[coordx][coordy][coordz].y());
		buffered_data.push_back(sdf_grids_grad[coordx][coordy][coordz].z());
	}
	fwrite(&(buffered_data[0]), sizeof(float), buffered_data.size(), wf);
	fclose(wf);
	
	printf("done\n");	
}

int main(int argc, char *argv[])
{
	if (argc != 3 && argc != 4)
	{
		printf("Usage: vdb_tsdf model_obj_filename output_sdf_filename [resolution=256]\n");
		return 0;
	}
	int grid_resolution = 256;
	
	if (argc >= 4)
	{
		sscanf(argv[3], "%d", &grid_resolution);
		printf("grid resolution set to %d\n", grid_resolution);
	}
	
	openvdb::initialize();
	fetch_sdf(argv[1], argv[2], grid_resolution);
	std::cout << "Finished converting mesh." << std::endl;
	return 0;
}