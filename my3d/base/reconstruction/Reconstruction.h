/*
 * filename: Reconstruction.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _MY3D_BASE_RECONSTRUCTION_RECONSTRUCTION_H_
#define _MY3D_BASE_RECONSTRUCTION_RECONSTRUCTION_H_

#include <tuple>

#include "base/reconstruction/Scene.h"

namespace my3d {
namespace base {

class Reconstruction {
public: 
    Reconstruction(base::Scene *scene): 
    scene_{scene} {
        assert(scene_ != nullptr); 
    }

public: 

    void reset(base::Scene *scene); 

    inline void reset() {
        reset(scene_); 
    }

    inline void clearTracks() {
        tracks_.clear(); 
    }

    inline void clearRegisteredImageIDs() {
        registered_image_ids_.clear(); 
    }

    inline void clearFilteredImageIDs() {
        filtered_image_ids_.clear(); 
    }

    inline base::Scene *getScene() {
        return scene_; 
    } 

    inline const base::Scene *getScene() const {
        return scene_; 
    }

    inline std::unordered_map<size_t, base::Track*> getTracks() const {
        return tracks_; 
    }

    inline base::Track *getTrack(const size_t &track_id) {
        if (tracks_.count(track_id) > 0) {
            return tracks_.at(track_id); 
        } else {
            return nullptr; 
        }
    }

    inline const base::Track *getTrack(const size_t &track_id) const {
        if (tracks_.count(track_id) > 0) {
            return tracks_.at(track_id); 
        } else {
            return nullptr; 
        }
    } 

    inline std::unordered_set<size_t> getRegisteredImageIDs() const {
        return registered_image_ids_; 
    }

    inline size_t getNumRegisteredImages() const {
        return registered_image_ids_.size(); 
    }

    inline std::unordered_set<size_t> getFilteredImageIDs() const {
        return filtered_image_ids_; 
    }

    inline size_t getNumFilteredImages() const {
        return filtered_image_ids_.size(); 
    }

    inline size_t getRefImageID0() const {
        return ref_image_id_0_; 
    }

    /**
     * @brief Note: The existence and registration of the id is guaranteed by the user
    */
    inline void setRefImageID0(const size_t ref_img_id_0) {
        ref_image_id_0_ = ref_img_id_0; 
    }

    inline size_t getRefImageID1() const {
        return ref_image_id_1_; 
    }

    /**
     * @brief Note: The existence and registration of the id is guaranteed by the user
    */
    inline void setRefImageID1(const size_t ref_img_id_1) {
        ref_image_id_1_ = ref_img_id_1; 
    }

    inline std::pair<size_t, size_t> getInitialImagePair() const {
        return initial_image_pair_; 
    }

    /**
     * @brief Note: The existence and registration of the ids is guaranteed by the user
    */
    inline void setInitialImagePair(const std::pair<size_t, size_t> &initial_image_pair) {
        initial_image_pair_ = initial_image_pair; 
    }

    inline size_t getInitialEdgeID() const {
        return initial_edge_id_; 
    }

    /**
     * @brief Note: The existence and registration of the id is guaranteed by the user
    */
    inline void setInitialEdgeID(const size_t &intial_edge_id) {
        initial_edge_id_ = initial_edge_id_; 
    }

    bool registerInitialImagePair(const std::pair<size_t, size_t> &initial_pair, 
                                  size_t initial_edge_id); 

    bool registerImage(const size_t &image_id); 

    void deregisterImage(const size_t &image_id); 

    /**
     * @brief Add an observation. This function will do nothing if images_[image_id] or 
     * tracks_.at(track_id) does not exist.
    */
    void addObservation(size_t track_id, size_t image_id, size_t observation_idx);

    /**
     * @brief Remove an observation. This function will do nothing if images_[image_id] or 
     * tracks_.at(track_id) does not exist.
    */
    void removeObservation(size_t track_id, size_t image_id, size_t observation_idx); 

    /**
     * @brief Add a track
     * 
     * @retval the ID of the added track 
    */
    size_t addTrack(const base::TrackElementList &elements, 
                    const Eigen::Vector3d &position, 
                    bool is_triangulated = true); 

    /**
     * @brief Remove a track. This function will do nothing if the track with given id 
     * does not exist 
    */
    void removeTrack(const size_t &track_id); 

    /**
     * @brief Deregister an camera/image/node and mark it as filtered 
     * This function will do nothing if images_[image_id] or tracks_.at(track_id) does not exist.
    */
    void removeImage(const size_t &image_id); 

    void normalize(const double &extent = 10.0, 
                   const double &p0 = 0.1, 
                   const double &p1 = 0.9, 
                   const bool &use_images = true); 
    
    std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> 
    computeBoundsAndCentroid(const double &p0, const double &p1, 
                             const bool use_images) const; 

    /**
     * @brief Transform the registered images and triangulated tracks 
    */
    void transform(const double &scale, 
                   const Eigen::Quaterniond &rotation, 
                   const Eigen::Vector3d &translation); 

    inline bool hasTrack(const size_t &image_id, 
                         const size_t &observation_idx, 
                         size_t *track_id) const {
        return scene_->hasTrack(image_id, observation_idx, track_id); 
    }

    inline bool isImageRegistered(const size_t &image_id) const {
        return registered_image_ids_.count(image_id) != 0; 
    }

    /**
     * @brief Check if the image with given ID is filtered
    */
    inline bool isImageFiltered(const size_t &image_id) const {
        return filtered_image_ids_.count(image_id) != 0; 
    }

    /**
     * @brief Check if the track with given ID exists
    */
    inline bool existsTrack(const size_t &track_id) const {
        return tracks_.find(track_id) != tracks_.end();
    }

    /**
     * @brief Check if the image with given ID exists
    */
    inline bool existsImage(const size_t &image_id) const {
        return scene_->existsNode(image_id); 
    } 

    inline void addModifiedTrackID(const size_t &track_id) {
        modified_track_ids_.insert(track_id); 
    }

    inline void clearModifiedTrackIDs() {
        modified_track_ids_.clear(); 
    }

    inline void addNewTrackID(const size_t &track_id) {
        new_track_ids_.insert(track_id); 
    }

    inline void clearNewTrackIDs() {
        new_track_ids_.clear(); 
    }

    inline void addNewImageID(const size_t &image_id) {
        new_image_ids_.insert(image_id); 
    }

    inline void clearNewImageIDs() {
        new_image_ids_.clear(); 
    }

    inline bool isTrackModified(const size_t &track_id) const {
        return modified_track_ids_.count(track_id) > 0; 
    } 

    inline bool isTrackNew(const size_t &track_id) const {
        return new_track_ids_.count(track_id) > 0; 
    }

    inline bool isImageNew(const size_t &image_id) const {
        return new_image_ids_.count(image_id) > 0; 
    }

private: 
    base::Scene *scene_;
    /* (track_id, track). Tracks are 3D points that are already been reconstructed */
    std::unordered_map<size_t, base::Track*> tracks_;
    std::unordered_set<size_t> registered_image_ids_;
    std::unordered_set<size_t> filtered_image_ids_;
    size_t ref_image_id_0_;
    size_t ref_image_id_1_;

    std::pair<size_t, size_t> initial_image_pair_;
    size_t initial_edge_id_; 

    std::unordered_set<size_t> modified_track_ids_; 
    std::unordered_set<size_t> new_track_ids_; 
    std::unordered_set<size_t> new_image_ids_; 

    size_t num_added_tracks_{0}; 
}; 

}
}

#endif // _MY3D_BASE_RECONSTRUCTION_RECONSTRUCTION_H_
