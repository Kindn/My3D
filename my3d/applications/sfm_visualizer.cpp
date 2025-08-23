/*
 * filename: sfm_visualizer.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Auto-sparse-reconstruction pipeline
 */

#include <iostream>

#include "utils/io.h"
#include "base/image/OpenCVImageReader.h"
#include "base/image/image_io.h"
#include "base/reconstruction/Visualizer.h"
#include "feature/OpenCVORBExtractor.h"
#include "feature/OpenCVMatcher.h"
#include "feature/OpenCVBatchFeatureExtractor.h"
#include "feature/OpenCVBatchFeatureMatcher.h"
#include "feature/SIFTExtractor.h"
#include "feature/BrutalForceMatcher.h"
#include "sfm/SceneBuilder.h"
#include "sfm/IncrementalSFM.h"

#include "simple_json/json.hpp"

using namespace my3d; 

bool loadVizData(const std::string &data_dir, 
                 base::VizData &viz_data) {
    if (!boost::filesystem::exists(data_dir)) {
        std::cerr << "[ERROR] " << data_dir << ": No such directory. " << std::endl; 
        return false; 
    }

    viz_data.cameras.clear(); 
    viz_data.point_cloud.clear(); 

    const std::string path_camera_poses = (boost::filesystem::path(data_dir) / "MY3D_Camera_Poses.txt").string(); 
    const std::string path_points_3D = (boost::filesystem::path(data_dir) / "MY3D_Points_3D.txt").string(); 

    std::string line; 

    std::ifstream ifs_camera_poses(path_camera_poses);
    if (!ifs_camera_poses.is_open()) {
        std::cerr << "[ERROR] Could not open file " << path_camera_poses << std::endl; 
        return false; 
    }
    while (std::getline(ifs_camera_poses, line))
    {
        size_t id; 
        Eigen::Vector3d pos; 
        Eigen::Quaterniond rot;
        std::stringstream line_ss(line);  
        line_ss >> id >> pos.x() >> pos.y() >> pos.z() >> 
                            rot.w() >> rot.x() >> rot.y() >> rot.z(); 
        const Eigen::Matrix3d rot_mat = rot.toRotationMatrix(); 

        base::VizData::VizCamera camera; 
        camera.id = id; 
        camera.color = cv::Vec3b(0, 0, 255); 
        camera.R = (cv::Mat_<float>(3, 3) << rot_mat(0, 0), rot_mat(0, 1), rot_mat(0, 2), 
                                              rot_mat(1, 0), rot_mat(1, 1), rot_mat(1, 2), 
                                              rot_mat(2, 0), rot_mat(2, 1), rot_mat(2, 2)); 
        camera.t = (cv::Mat_<float>(3, 1) << pos.x(), pos.y(), pos.z()); 
        viz_data.cameras.push_back(camera); 
        std::cout << "\rLoaded " << viz_data.cameras.size() << " camera(s). "; 
    }
    std::cout << std::endl; 
    

    std::ifstream ifs_points_3D(path_points_3D); 
    if (!ifs_points_3D.is_open()) {
        std::cerr << "[ERROR] Could not open file " << path_points_3D << std::endl; 
        return false; 
    }
    size_t point_3D_id = 0; 
    while (std::getline(ifs_points_3D, line))
    {
        base::VizData::VizPoint3D point_3D; 
        int b, g, r;
        point_3D.id = point_3D_id; 
        std::stringstream line_ss(line);  
        line_ss >> point_3D.position.x >> point_3D.position.y >> point_3D.position.z >> 
                         r >> g >> b; 
        point_3D.color[0] = (uchar)b; 
        point_3D.color[1] = (uchar)g; 
        point_3D.color[2] = (uchar)r; 
        viz_data.point_cloud.push_back(point_3D); 
        std::cout << "\rLoaded " << viz_data.point_cloud.size() << " point(s). "; 
    }
    std::cout << std::endl; 

    ifs_camera_poses.close(); 
    ifs_points_3D.close(); 

    return true; 
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cout << "Usage: " << argv[0] << " /path/to/dataset" << std::endl; 
        return -1; 
    }
    const std::string data_dir = argv[1]; 
    
    base::VizData viz_data; 
    if (!loadVizData(data_dir, viz_data)) {
        std::cerr << "[ERROR] Failed to load data. " << std::endl; 
    }

    const size_t num_cameras = viz_data.cameras.size(); 
    const size_t num_points_3D = viz_data.point_cloud.size(); 
    std::cout << "[INFO] num_cameras: " << num_cameras << ", num_points: " << num_points_3D << std::endl; 

    cv::viz::Viz3d window(data_dir); 
    window.setBackgroundColor(cv::viz::Color::black()); 
    window.showWidget("World Frame", cv::viz::WCoordinateSystem()); 

    while (!window.wasStopped()) {
        /* Show cameras */ 
        char camera_name[1000]; 
        for (size_t i = 0; i < viz_data.cameras.size(); ++i) {
            const cv::Vec2d fov(1.0f, 1.0f); 
            sprintf(camera_name, "camera_%ld", i); 
            const cv::Affine3f pose(viz_data.cameras[i].R, viz_data.cameras[i].t); 
            const cv::viz::WCameraPosition w_camera_postition{fov, 0.1, cv::viz::Color(viz_data.cameras[i].color[0], 
                                                                                       viz_data.cameras[i].color[1], 
                                                                                       viz_data.cameras[i].color[2])}; 
            window.showWidget(camera_name, w_camera_postition); 
            window.setWidgetPose(camera_name, pose); 
        }

        /* Show point cloud */
        const std::string point_cloud_name = "point_cloud"; 
        std::vector<cv::Point3f> points; 
        std::vector<cv::Vec3b> point_colors; 
        for (const auto &viz_point_3D : viz_data.point_cloud) {
            points.push_back(viz_point_3D.position); 
            point_colors.push_back(viz_point_3D.color); 
        }
        if (!points.empty() && !point_colors.empty()) {
            const cv::viz::WCloud w_cloud(points, point_colors); 
            window.showWidget(point_cloud_name, w_cloud); 
        }

        window.spinOnce(1, true);
    }

    window.close();

    return 0; 
}