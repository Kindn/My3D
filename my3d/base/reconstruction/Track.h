/*
 * filename: Track.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _MY3D_BASE_RECONSTRUCTION_TRACK_H_
#define _MY3D_BASE_RECONSTRUCTION_TRACK_H_

#include <iostream>
#include <vector>

#include "utils/eigen_types.h"

namespace my3d {
namespace base {

struct TrackElement {
    TrackElement(size_t _image_id, size_t _feature_idx): 
    image_id(_image_id), feature_idx(_feature_idx) {

    }

    /* The ID of image in which the track element is observed. */
    size_t image_id;
    /* The index of the feature in image_id */
    size_t feature_idx;
};

typedef std::vector<TrackElement> TrackElementList;

class Track {
public:
    Track() {}
    ~Track() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Add an element to the track. The uniqueness of the image ID 
     * needs to be guaranteed by the user.
    */
    void addElement(const TrackElement &element);

    /**
     * @brief Add an element to the track. The uniqueness of the image ID 
     * needs to be guaranteed by the user.
    */
    void addElement(const size_t &image_id, const size_t &feature_idx);

    /**
     * @brief Add multiple elements
    */
    void addElements(const TrackElementList &elements); 

    /**
     * @brief Remove the element with given image_id.
    */
    void removeElement(const size_t &image_id, const size_t &feature_idx);

    /**
     * @brief Check if the track contains an image
    */
    bool hasImage(const size_t &image_id) const; 

    inline size_t length() const {
        return elements_.size(); 
    }

    inline bool isTriangulated() const {
        return is_triangulated_;
    }

    inline void setTriangulated(const bool &is_triangulated) {
        is_triangulated_ = is_triangulated; 
    }

    inline void set3DPosition(const Eigen::Vector3d &pos_3d) {
        pos_3d_ = pos_3d; 
    }

    inline Eigen::Vector3d get3DPosition() const {
        return pos_3d_; 
    }

    inline void setColor(const uint8_t color[3]) {
        color_[0] = color[0];
        color_[1] = color[1];
        color_[2] = color[2];
    }

    inline void getColor(uint8_t *color) const {
        color[0] = color_[0];
        color[1] = color_[1];
        color[2] = color_[2];
    }

    inline void setId(const size_t &id) {
        id_ = id;
    }

    inline size_t getId() const {
        return id_;
    }

    inline TrackElementList &getElements() {
        return elements_; 
    }

    inline const TrackElementList &getElements() const {
        return elements_; 
    }

    inline void setElements(const TrackElementList &elements) {
        elements_ = elements; 
    }

    inline bool existsElement(const size_t &image_id, const size_t &feature_idx) const {
        for (const auto &element : elements_) {
            if (element.image_id == image_id && 
                element.feature_idx == feature_idx) {
                return true; 
            }
        }
        return false; 
    } 

    inline bool existsImage(const size_t &image_id) const {
        for (const auto &element : elements_) {
            if (element.image_id == image_id) {
                return true; 
            }
        }
        return false;
    }

private:
    /* ID */
    size_t id_;
    /* 3D position of the track. */
    Eigen::Vector3d pos_3d_;
    /* Color (RGB) of the track. */
    uint8_t color_[3] = {255};
    /* Elements of the track. */
    TrackElementList elements_;
    /* Whether the track is triangulated. */
    bool is_triangulated_{false};
};

}
}

#endif // _MY3D_BASE_RECONSTRUCTION_TRACK_H_
