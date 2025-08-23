/*
 * filename: Reconstruction.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "base/reconstruction/Reconstruction.h"

namespace my3d {
namespace base {

void Reconstruction::reset(base::Scene *scene) {
    for (auto &track : tracks_) {
        if (track.second != nullptr) {
            delete track.second;
            track.second = nullptr;
        }
    }
    clearTracks();
    clearRegisteredImageIDs();
    clearFilteredImageIDs(); 
    clearModifiedTrackIDs(); 
    clearNewTrackIDs(); 
    clearNewImageIDs(); 
    num_added_tracks_ = 0; 
}

bool Reconstruction::registerInitialImagePair(const std::pair<size_t, size_t> &initial_pair, 
                                             size_t initial_edge_id) {
    if (scene_->existsNode(initial_pair.first) && 
        scene_->existsNode(initial_pair.second) && 
        scene_->existsEdge(initial_edge_id)) {
        base::Scene::Edge* const initial_edge = scene_->getEdge(initial_edge_id);
        if (initial_edge->node_ids[0] == initial_pair.first && initial_edge->node_ids[1] == initial_pair.second) {
            registered_image_ids_.clear(); 
            registered_image_ids_.insert(initial_pair.first);
            registered_image_ids_.insert(initial_pair.second);
            initial_image_pair_ = initial_pair;
            initial_edge_id_ = initial_edge_id;
        } else {
            return false; 
        }
    } else {
        return false; 
    }

    return true; 
}

bool Reconstruction::registerImage(const size_t &image_id) {
    if (existsImage(image_id)) {
        registered_image_ids_.insert(image_id);
    } else {
        return false; 
    }

    return true; 
}

void Reconstruction::deregisterImage(const size_t &image_id) {
    if (!existsImage(image_id) || 
        !isImageRegistered(image_id) || 
        isImageFiltered(image_id)) {
        return; 
    }
    
    std::unordered_set<size_t> tracks_to_be_removed;
    for (const auto &track : tracks_) {
        for (const auto &elem : track.second->getElements()) {
            if (elem.image_id == image_id) {
                removeObservation(track.first, image_id, elem.feature_idx); 
            }
        } 
        if (track.second->length() < 2) {
            tracks_to_be_removed.insert(track.first);
        }
    }
    for (const size_t &track_id : tracks_to_be_removed) {
        removeTrack(track_id);
    }

    registered_image_ids_.erase(image_id); 
} 

void Reconstruction::addObservation(size_t track_id, size_t image_id, size_t observation_idx) {
    if (existsTrack(track_id) && existsImage(image_id)) {
        assert(!tracks_.at(track_id)->existsImage(image_id));
        assert(!scene_->hasTrack(image_id, observation_idx, nullptr)); 
        tracks_.at(track_id)->addElement(image_id, observation_idx); 
        scene_->addObservation(track_id, image_id, observation_idx);
    }
}

void Reconstruction::removeObservation(size_t track_id, size_t image_id, size_t observation_idx) {
    if (existsTrack(track_id) && existsImage(image_id)) {
        tracks_.at(track_id)->removeElement(image_id, observation_idx); 
        scene_->removeObservation(track_id, image_id); 
    }
}

size_t Reconstruction::addTrack(const base::TrackElementList &elements, 
                                const Eigen::Vector3d &position, 
                                bool is_triangulated) {
    base::Track *track = new base::Track;
    assert(track != nullptr); 
    const size_t track_id = num_added_tracks_++;
    track->setId(track_id);
    track->addElements(elements);
    track->setTriangulated(is_triangulated);
    track->set3DPosition(position);
    scene_->addTrack(*track);
    tracks_.insert(std::make_pair(track_id, track));

    return track_id; 
}

void Reconstruction::removeTrack(const size_t &track_id) {
    if (!existsTrack(track_id)) {
        return; 
    }

    assert(scene_ != nullptr); 
    assert(tracks_.at(track_id) != nullptr); 
    scene_->removeTrack(*(tracks_.at(track_id)));
    delete tracks_.at(track_id); 
    tracks_.erase(track_id);
}

void Reconstruction::removeImage(const size_t &image_id) {
    if (!existsImage(image_id)) {
        return; 
    }

    filtered_image_ids_.insert(image_id); 
    if (isImageRegistered(image_id)) {
        registered_image_ids_.erase(image_id); 
    }
}

void Reconstruction::normalize(const double &extent, 
                               const double &p0, 
                               const double &p1, 
                               const bool &use_images) {
    assert(extent > 0); 

    if ((use_images && registered_image_ids_.size() < 2) || 
        (!use_images && tracks_.size() < 2)) {
        return; 
    } 

    auto bound = computeBoundsAndCentroid(p0, p1, use_images); 

    const double old_extent = (std::get<1>(bound) - std::get<0>(bound)).norm(); 
    const double scale = 
        (old_extent < std::numeric_limits<double>::epsilon()) ? 1.0 : (extent / old_extent); 

    transform(scale, Eigen::Quaterniond::Identity(), -scale * std::get<2>(bound)); 
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> 
Reconstruction::computeBoundsAndCentroid(const double &p0, const double &p1, 
                                         const bool use_images) const {
    assert(p0 > 0); 
    assert(p0 < 1); 
    assert(p1 > 0); 
    assert(p1 < 1); 
    assert(p0 < p1); 

    const size_t num_elems = 
        use_images ? registered_image_ids_.size() : tracks_.size(); 
    if (num_elems == 0) {
        return std::make_tuple(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 
                               Eigen::Vector3d::Zero()); 
    }

    std::vector<double> coords_x(num_elems); 
    std::vector<double> coords_y(num_elems); 
    std::vector<double> coords_z(num_elems); 
    if (use_images) {
        size_t idx = 0; 
        for (const size_t &image_id : registered_image_ids_) {
            const Eigen::Vector3d position = scene_->getNodeTranslation(image_id); 
            coords_x[idx] = position.x(); 
            coords_y[idx] = position.y(); 
            coords_z[idx] = position.z(); 
            ++idx; 
        }
    } else {
        size_t idx = 0; 
        for (const auto &track : tracks_) {
            const Eigen::Vector3d position = track.second->get3DPosition(); 
            coords_x[idx] = position.x(); 
            coords_y[idx] = position.y(); 
            coords_z[idx] = position.z(); 
            ++idx; 
        }
    }

    std::sort(coords_x.begin(), coords_x.end()); 
    std::sort(coords_y.begin(), coords_y.end()); 
    std::sort(coords_z.begin(), coords_z.end()); 

    const size_t P0 = 
        static_cast<size_t>((coords_x.size() > 3) ? p0 * (coords_x.size() - 1) : 0); 
    const size_t P1 = 
        static_cast<size_t>((coords_x.size() > 3) ? p1 * (coords_x.size() - 1) : coords_x.size() - 1); 
    
    const Eigen::Vector3d bbox_min(coords_x[P0], coords_y[P0], coords_z[P0]); 
    const Eigen::Vector3d bbox_max(coords_x[P1], coords_y[P1], coords_z[P1]); 

    Eigen::Vector3d centroid(0, 0, 0); 
    for (size_t i = P0; i <= P1; ++i) {
        centroid.x() += coords_x[i]; 
        centroid.y() += coords_y[i]; 
        centroid.z() += coords_z[i]; 
    }
    centroid /= static_cast<double>(P1 - P0 + 1); 

    return std::make_tuple(bbox_min, bbox_max, centroid); 
}

void Reconstruction::transform(const double &scale, 
                               const Eigen::Quaterniond &rotation, 
                               const Eigen::Vector3d &translation) {
    for (const size_t &image_id : registered_image_ids_) {
        const Eigen::Vector3d old_pos = scene_->getNodeTranslation(image_id); 
        const Eigen::Quaterniond old_rot = scene_->getNodeRotation(image_id); 
        const Eigen::Vector3d pos = 
            scale * (rotation.normalized().toRotationMatrix() * old_pos) + translation; 
        const Eigen::Quaternion rot = (rotation.normalized() * old_rot).normalized(); 
        scene_->getNode(image_id)->setPose(rot, pos); 
    }
    for (const auto &track : tracks_) {
        const Eigen::Vector3d old_pos = track.second->get3DPosition(); 
        const Eigen::Vector3d pos = 
            scale * (rotation.normalized().toRotationMatrix() * old_pos) + translation; 
        track.second->set3DPosition(pos); 
    }
}

}
}
