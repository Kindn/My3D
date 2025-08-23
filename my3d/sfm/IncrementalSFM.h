/*
 * filename: IncrementalSFM.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _MY3D_SFM_INCREMENTAL_SFM_H_
#define _MY3D_SFM_INCREMENTAL_SFM_H_

#include "base/reconstruction/BundleAdjustment.h"
#include "base/reconstruction/Scene.h"
#include "base/reconstruction/Track.h"
#include "base/reconstruction/Reconstruction.h"
#include "base/reconstruction/Visualizer.h"
#include "base/pose.h"
#include "base/camera/projection.h"
#include "base/triangulation.h"
#include "estimators/AP3PPoseEstimator.h"
#include "estimators/EPnPPoseEstimator.h"
#include "estimators/LORANSAC.h"
#include "estimators/TriangulationEstimator.h"

namespace my3d {
namespace sfm {

/**
 * @brief An incremental structure-from-motion mapper. 
*/
class IncrementalSFM {
public: 
    enum NextBestViewSelectionMethod {
        MAX_VISIBLE_TRACK_NUM = 0
    };

    /**
     * @brief Used for sorting edges 
    */
    struct EdgeInfo {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        size_t edge_id;
        size_t num_triangulated_points;
        double median_triangulated_angle;
        size_t num_connected_nodes_1;
        size_t num_connected_nodes_2;
        size_t num_correspondences; 
        long int model_type;
    };

    /**
     * @brief Used for sorting images 
    */
    struct ImageInfo {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        size_t image_id; 
        size_t num_visible_tracks; 
        std::unordered_map<size_t, size_t> corrs_3D2D; 
    }; 

    struct GlobalBundleAdjustmentConfig {
        bool print_report{true}; 
        base::BundleAdjustment::Config ba_config; 
    }; 

    struct Config {
        /* Mininum two-view-geometry inlier number for an initial image pair */
        int min_num_init_corrs{100}; 
        /* Minimum number of 3D points after a successful registration */
        size_t min_num_points{30}; 
        /* Only the observations with reprojections not greater than max_track_reproj_error (in pixels) 
           should be included in a track */
        double max_track_reproj_error{3.0}; 
        /* Minimum length of a track */
        size_t min_track_length{3UL}; 
        /* Maximum reprojection error for absolute pose estimation */
        double max_abs_pose_reproj_error{12.0}; 
        /* Minimum inlier ratio of absolute pose estimation in next image registration */
        double min_abs_pose_inlier_ratio{0.2};
        /* Minimum inlier number of absolute pose estimation in next image registration */
        size_t min_abs_pose_inlier_num{15}; 
        /* Epsilon for AP3P */
        double abs_pose_epsilon{1e-5}; 
        /* Method for next best view selection */
        NextBestViewSelectionMethod nbv_method{NextBestViewSelectionMethod::MAX_VISIBLE_TRACK_NUM};
        /* Global bundle adjustment configuration */
        GlobalBundleAdjustmentConfig gba_config;
        /* Maximum reprojection error in pixels to merge tracks */
        double max_merge_reproj_error{3.0}; 
        /* Minimum focal length ratio of non-bogus focal length */ 
        double min_focal_length_ratio{0.0}; 
        /* Maximum focal length ratio of non-bogus focal length */
        double max_focal_length_ratio{std::numeric_limits<double>::infinity()}; 
        /* Maximum transitivity to search correspondences */
        size_t max_transitivity{5}; 
        /* Minimum triangulation angle (in degrees) between 2 observations */
        double min_tri_angle_deg{1.0}; 
        /* Minimum triangulation angle (in degrees) for initial image pair */
        double min_tri_angle_deg_init{1.0}; 
        /* Minimum number of observations to triangulate a track */
        size_t min_tri_num_obs{3}; 
        /* Minimum inlier ratio of triangulation */
        double min_tri_inlier_ratio{0.35}; 
        /* Maximum inlier error of robust triangulation */ 
        double max_tri_inlier_angular_error_deg{2.0}; 
        /* Mininum two-view-geometry triangulation rate of an initial pair */
        double min_init_two_view_geo_tri_rate{0.2}; 
        /* Whether use recursive triangulation */
        bool use_recursive_triangulation{true}; 
        /* Global BA interval */
        size_t global_ba_interval{5}; 
    };

    struct Report {
        bool success;
        EigenUMap<size_t, base::Scene::CameraInfo> camera_infos;
        EigenUMap<size_t, base::Track> tracks;
        double reprojection_rmse;
    };  

    struct BundleAdjustmentReport {
        int num_iter{-1};
        double final_cost{-1};
        double final_squared_reprojection_error{-1};
        bool success{false};
        int ret_val;
        std::string error_info{""};
        size_t num_cameras{0};
        size_t num_points_3D{0};
        std::vector<size_t> camera_ids; 
        std::vector<size_t> track_ids;
        size_t num_observations{0};
        size_t num_merged_observations{0}; 
        size_t num_completed_observations{0};  

        void print() const {
            std::cout << "----------Bundle Adjustment Report----------" << std::endl; 
            std::cout << "state: " << (success ? "success" : "failed") << std::endl; 
            std::cout << "return value: " << ret_val << std::endl; 
            if (!success) {
                std::cout << "error information: " << error_info << std::endl; 
            }
            std::cout << "number of cameras: " << num_cameras << std::endl; 
            std::cout << "number of 3D points: " << num_points_3D << std::endl; 
            std::cout << "number of observations: " << num_observations << std::endl; 
            std::cout << "number of merged observations: " << num_merged_observations << std::endl; 
            std::cout << "number of completed observations: " << num_completed_observations << std::endl; 
        }
    }; 

    // typedef estimator::RANSAC<estimator::AP3PPoseEstimator> AbsolutePoseEstimator; 
    typedef estimator::LORANSAC<estimator::AP3PPoseEstimator, estimator::EPnPPoseEstimator> AbsolutePoseEstimator; 

public: 
    IncrementalSFM() = delete; 
    explicit IncrementalSFM(const Config &config); 
    ~IncrementalSFM();

    double computeInitialPairScore(const EdgeInfo &edge_info) const; 
    
    void config(const Config &config); 

    /**
     * @brief Reset the mapper. Always call this function before reconstruction
    */
    void reset();

    /**
     * @brief Prepare for a new reconstruction
    */
    void beginReconstruction(const std::shared_ptr<base::Reconstruction> reconstruction); 

    /**
     * @brief 
    */
    void endReconstruction(Report &report); 
 
    std::shared_ptr<base::Reconstruction> getReconstruction() const {
        return reconstruction_; 
    }

    /** 
     * @brief Compute initial tracks (without triangulation) according to the scene graph
    */
    void computeInitialTracks(); 

    /**
     * @brief The full pipeline of incremental SfM
    */
    Report reconstruct(const std::shared_ptr<base::Reconstruction> reconstruction);

    /**
     * @brief Find the initial image pair. Should be called after beginReconstruction()
     * 
     * @param inital_pair[out]     the image_ids/node_ids of the initial image pair
     * @param initial_edge_id[out] the edge_id of the initial image pair
    */
    bool findInitialImagePair(std::pair<size_t, size_t> &initial_pair, 
                              size_t &initial_edge_id); 

    /**
     * @brief Find candidates initial image pair. Should be called after beginReconstruction()
     * 
     * @param inital_pairs[out]     the image_ids/node_ids of the initial image pair
     * @param initial_edge_ids[out] the edge_id of the initial image pair
    */
    bool findInitialImagePairCandidates(std::vector<std::pair<size_t, size_t>> &initial_pairs, 
                                        std::vector<size_t> &initial_edge_ids); 

    /**
     * @brief Register the initial image pair 
     *  (1) Estimate the two-view geometry \n
     *  (2) Triangulate observed feature points \n
     *  (3) Create initial tracks
    */
    bool registerInitialImagePair(const std::pair<size_t, size_t> &pair, 
                                  size_t initial_edge_id); 

    

    /**
     * @brief Find next image
     * 
     * @param next_image_id[out]          the ID of found image
     * @param correspondences_3D2D[out]   (track_id, point_2D_idx)
    */
    bool findNextImage(size_t &next_image_id, 
                       std::unordered_map<size_t, size_t> &correspondences_3D2D);

    /**
     * @brief Find next image with maximum number of visible tracks
    */
    bool findNextImagesWithMaxVisibleTracks(std::vector<size_t> &next_image_ids, 
                                 std::vector<std::unordered_map<size_t, size_t>> &correspondences_3D2Ds); 
    
    /**
     * @brief Find candidates for next image
    */
    bool findNextImageCandidates(std::vector<size_t> &next_image_ids, 
                                 std::vector<std::unordered_map<size_t, size_t>> &correspondences_3D2Ds); 

    /**
     * @brief Register next image
     * 
     * @retval    whether the image is successfully registered
    */
    bool registerNextImage(const size_t &image_id, 
                           const std::unordered_map<size_t, size_t> &correspondences_3D2D); 

    /**
     * @brief Refine absolute pose 
    */
    BundleAdjustmentReport refineAbsolutePose(const size_t &image_id); 

    /**
     * @brief Global bundle adjustment for all registered images and existing tracks
    */
    BundleAdjustmentReport solveGlobalBundleAdjustment(const bool &should_update = false); 

    /**
     * @brief Local bundle adjustment for the last registered image, its visible tracks, and covisible 
     * images  
    */
    BundleAdjustmentReport solveLocalBundleAdjustment(const size_t &image_id, 
                                                      const bool &should_update = false); 

    /**
     * @brief Update parameters of cameras and coordinates of tracks according to the result of BA 
    */
    void updateStructureAndMotionFromBA(const base::BundleAdjustment &ba_solver); 

    /**
     * @brief Try to merge a track with any of its corresponding points
     * 
     * @param track_id the ID of the track to be merged 
     * @retval number of merged observation(s) (track element(s))
    */
    size_t mergeTrack(const size_t &track_id); 

    /**
     * @brief Try to merge multiple tracks with any of its corresponding points
     * 
     * @param track_ids   the list of track IDs to be merged 
     * @retval number of merged observation(s) (track element(s))
    */
    size_t mergeTracks(const std::vector<size_t> &track_ids); 

    /**
     * @brief Try to create new tracks from the given image. Note that the image should be registered 
     * and its pose should be known
     * 
     * @retval number of newly triangulated observations 
    */
    size_t triangulateImage(const size_t &image_id); 

    /**
     * @brief Triangulte a given track. The given track may be splitted into several new tracks according 
     * to the triangulation inliers 
     * 
     * @retval number of triangulated observations
    */
    size_t triangulateTrack(const size_t &track_id);

    /**
     * @brief Robustly triangulate a list of observations (represented by track elements). Please make sure 
     * that there is not an observation's correspongding camera has bogus parameters or undetermined pose  
     */ 
    estimator::RobustTriangulationEstimator::Report 
    triangulateObservations(const typename estimator::RANSACConfig &tri_config, 
                            const base::TrackElementList &elements); 

    /**
     * @brief Try to recursively add observations that may have failed to be triangulated before 
     * due to inaccurate poses to a track
     * 
     * @retval number of completed observations  
    */
    size_t completeTrack(const size_t &track_id); 

    /**
     * @brief Try to complete multiple tracks 
    */
    size_t completeTracks(const std::vector<size_t> &track_ids); 

    /**
     * @brief Complete triangulations for image. Tries to create new tracks for not 
     * yet triangulated observations and tries to complete existing tracks. Returns 
     * the number of completed observations.
    */
    size_t completeImage(const size_t &image_id); 

    /**
     * @brief Filter tracks with too large reprojection errors 
     * 
     * @retval number of filtered observations
    */
    size_t filterTracks(const std::vector<size_t> &track_ids, 
                        const bool &init = false); 

    /**
     * @brief Filter tracks with large reprojection error 
     * 
     * @retval number of filtered observations
    */
    size_t filterTracksWithLargeReprojectionError(const double &max_reproj_error, 
                                                  const std::vector<size_t> &track_ids, 
                                                  const bool &init = false); 

    /**
     * @brief Filter tracks with small triangulation angle 
     * 
     * @retval number of filtered observations
    */
    size_t filterTracksWithSmallTriangulationAngle(const double &min_tri_angle_rad, 
                                                   const std::vector<size_t> &track_ids, 
                                                   const bool &init = false); 
    
    /**
     * @brief Filtered observations with negative depth
    */
    size_t filterObservationsWithNegativeDepth(const bool &init = false); 

    /**
     * @brief Filter all the existing tracks 
    */
    size_t filterAllTracks(const bool &init = false); 

    // /**
    //  * @brief Filter images 
    // */
    // size_t filterImages(const std::vector<size_t> &image_ids); 

    // /**
    //  * @brief Filter all the registered images 
    // */
    // size_t filterAllImages(); 

    /**
     * @brief Check if the image with given ID is registered
    */
    inline bool isImageRegistered(const size_t &image_id) const {
        return reconstruction_->isImageRegistered(image_id); 
    }

    /**
     * @brief Check if the image with given ID is filtered
    */
    inline bool isImageFiltered(const size_t &image_id) const {
        return reconstruction_->isImageFiltered(image_id); 
    }

    /**
     * @brief Check if the track with given ID exists
    */
    inline bool existsTrack(const size_t &track_id) const {
        return reconstruction_->existsTrack(track_id);
    }

    /**
     * @brief Check if the image with given ID exists
    */
    inline bool existsImage(const size_t &image_id) const {
        return reconstruction_->existsImage(image_id); 
    } 

protected: 
    Config config_;

    // base::Scene *scene_;
    // /* (track_id, track). Tracks are 3D points that are already been reconstructed */
    // std::unordered_map<size_t, base::Track*> tracks_;
    // std::unordered_set<size_t> registered_image_ids_;
    // std::unordered_set<size_t> filtered_image_ids_;
    // size_t ref_image_id_0_;
    // size_t ref_image_id_1_;

    // std::pair<size_t, size_t> initial_image_pair_;
    // size_t initial_edge_id_;

    std::shared_ptr<base::Reconstruction> reconstruction_; 


};

}
}

#endif // _MY3D_SFM_INCREMENTAL_SFM_H_
