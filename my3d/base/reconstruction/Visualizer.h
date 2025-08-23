/*
 * filename: Visualizer.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _MY3D_BASE_RECONSTRUCTION_VISUALIZER_H_
#define _MY3D_BASE_RECONSTRUCTION_VISUALIZER_H_

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include "base/reconstruction/Reconstruction.h" 
#include "utils/threading.h" 

namespace my3d {
namespace base {

struct VizData {
    struct VizCamera {
        size_t id; 
        cv::Mat R; 
        cv::Mat t; 
        cv::Vec3b color; 
    }; 

    struct VizPoint3D {
        size_t id; 
        cv::Point3f position; 
        cv::Vec3b color; 
    }; 

    std::vector<VizPoint3D> point_cloud; 
    std::vector<VizCamera> cameras; 

    void clear() {
        point_cloud.clear(); 
        cameras.clear(); 
    }
}; 

/**
 * @brief An OpenCV-based visualizer 
*/
class Visualizer {
public: 
    cv::Vec3b color_camera = cv::Vec3b{50, 50, 255}; 
    cv::Vec3b color_new_camera = cv::Vec3b{50, 255, 50};
    cv::Vec3b color_ref_camera_0 = cv::Vec3b{255, 50, 50}; 
    cv::Vec3b color_ref_camera_1 = cv::Vec3b{255, 255, 50}; 
    cv::Vec3b color_modified_point = cv::Vec3b{100, 180, 180}; 
    cv::Vec3b color_new_point = cv::Vec3b{50, 255, 50}; 

    Visualizer(const std::shared_ptr<Reconstruction> &reconstruction, 
               const std::string &window_name): 
    reconstruction_{reconstruction} {
        window_ = cv::viz::Viz3d(window_name); 
    }

    void run(); 

    void threadVisualization(); 

    void join(); 

    void close(); 

private: 
    void getVizData(); 

    cv::Vec3b getCameraColor(const size_t &camera_id) const ; 

    cv::Vec3b getPoint3DColor(const size_t &point_id) const; 

private: 
    cv::viz::Viz3d window_;     
    std::thread thread_viz_; 
    std::unique_ptr<util::ThreadGuard> thread_guard_viz_; 
    VizData viz_data_; 
    std::shared_ptr<Reconstruction> reconstruction_; 
    size_t num_current_cameras_{0}; 
    size_t num_current_point_cloud_{0}; 
}; 

}
}

#endif // _MY3D_BASE_RECONSTRUCTION_VISUALIZER_H_
