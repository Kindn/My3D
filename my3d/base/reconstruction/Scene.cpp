/*
 * filename: Scene.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "base/reconstruction/Scene.h"

namespace my3d {
namespace base {

Scene::CameraInfo::CameraInfo(): 
rotation{Eigen::Quaterniond::Identity()}, 
translation{Eigen::Vector3d::Zero()} {

}

Scene::TwoViewGeometryInfo::TwoViewGeometryInfo(): 
fundamental_matrix{Eigen::Matrix3d::Identity()}, 
homography{Eigen::Matrix3d::Identity()}, 
essential_matrix{Eigen::Matrix3d::Identity()}, 
R{Eigen::Matrix3d::Identity()}, 
t{Eigen::Vector3d::Zero()} {

}

Scene::Node::Node(size_t _id, int _width, int _height, 
                  const EigenVec<Point2D> &_points_2D, 
                  const std::unordered_set<size_t> &_connected_node_ids): 
id{_id}, width{_width}, height{_height}, 
points_2D{_points_2D}, connected_node_ids{_connected_node_ids} {

}

Eigen::Matrix3x4d Scene::Node::getProjectionMatrix() const {
    const Eigen::Matrix3d rotation = base::quaternion2RotationMatrix(getRotation()); 
    const Eigen::Vector3d translation = getTranslation(); 
    return getCameraMatrix() * base::composePose(rotation.inverse(), -rotation.inverse() * translation); 
}

double Scene::Node::computeReprojectionError(const Eigen::Vector3d &point_3D, 
                                             const size_t &point_2D_idx) const {
    if (point_2D_idx >= points_2D.size()) {
        return std::numeric_limits<double>::infinity(); 
    }
    const Eigen::Vector2d point_2D = points_2D[point_2D_idx].position; 
    // const Eigen::Matrix3x4d proj_matrix = getProjectionMatrix(); 
    // const double error1 =  base::computeReprojectionError(point_3D, point_2D, proj_matrix); 
    const base::CameraBase *camera = 
        new base::RadialPinHoleCamera(getCameraMatrix(), camera_info.k1, camera_info.k2); 
    const Eigen::Matrix3d rotation = base::quaternion2RotationMatrix(getRotation()); 
    const Eigen::Vector3d translation = getTranslation(); 
    const Eigen::Matrix3x4d pose = base::composePose(rotation.inverse(), -rotation.inverse() * translation); 
    const double error = base::computeReprojectionError(point_3D, point_2D, pose, camera); 
    delete camera; 
    // std::cout << "(" << error1 << ", " << error2 << std::endl; 
    return error; 
}

bool Scene::Node::hasBogusCameraParams(const double &min_focal_length_ratio, 
                                       const double &max_focal_length_ratio) const {
    const size_t size = std::max(width, height); 
    const double fx_ratio = camera_info.fx / size; 
    const double fy_ratio = camera_info.fy / size; 
    bool has_bogus_focal_lengths = fx_ratio < min_focal_length_ratio || fx_ratio > max_focal_length_ratio || 
                                    fy_ratio < min_focal_length_ratio || fy_ratio > max_focal_length_ratio; 
    bool has_bogus_principal_points = camera_info.cx < 0.0 || camera_info.cx > width || 
                                      camera_info.cy < 0.0 || camera_info.cy > height; 
            
    return has_bogus_focal_lengths || has_bogus_principal_points;
}

Scene::Scene() {

}

Scene::~Scene() {
    clear();
}

void Scene::clear() {
    for (auto &edge : edges_) {
        if (edge.second != nullptr) {
            delete edge.second;
            edge.second = nullptr;
        }
    }
    edges_.clear();
    for (auto &node : nodes_) {
        if (node.second != nullptr) {
            delete node.second;
            node.second = nullptr;
        }
    }
    nodes_.clear();
}

bool Scene::addNode(const size_t &id, const size_t &width, const size_t &height, 
                    const EigenVec<Point2D> &points_2D, 
                    const std::string &name, 
                    const CameraInfo &camera_info) {
    if (existsNode(id)) {
        return false;
    }

    Node *node = new Node(id, width, height, points_2D, std::unordered_set<size_t>());
    node->camera_info = camera_info;
    node->name = name; 
    nodes_.insert(std::make_pair(id, node));

    return true;
}

bool Scene::addEdge(const size_t &id, 
                    const std::array<size_t, 2> &node_ids, 
                    const std::vector<std::pair<size_t, size_t>> &correspondences, 
                    const Edge::SceneEdgeModelType &model_type, 
                    const TwoViewGeometryInfo &two_view_geom_info) {
    assert(node_ids[0] != node_ids[1]); 

    if (existsEdge(id)) {
        return false;
    }
    if (!existsNode(node_ids[0]) || !existsNode(node_ids[1])) {
        return false;
    }

    Node *node1 = nodes_.at(node_ids[0]);
    Node *node2 = nodes_.at(node_ids[1]);
    assert(node1->id == node_ids[0]);
    assert(node2->id == node_ids[1]);
    
    if (std::find(node1->connected_node_ids.begin(), node1->connected_node_ids.end(), node_ids[1]) != 
        node1->connected_node_ids.end()) {
        return false;
    }

    node1->connected_node_ids.insert(node_ids[1]);
    node2->connected_node_ids.insert(node_ids[0]);
    
    Edge *edge = new Edge(id, node_ids, correspondences);
    edge->model_type = model_type; 
    edge->two_view_geom_info = two_view_geom_info;
    edges_.insert(std::make_pair(id, edge));
    node1->connected_edges.insert(std::make_pair(node_ids[1], edge));
    node2->connected_edges.insert(std::make_pair(node_ids[0], edge));

    return true;
}

void Scene::removeNode(const size_t &id) {
    auto it = nodes_.find(id);
    if (it != nodes_.end()) {
        delete it->second;
        nodes_.erase(it);
    }
}

void Scene::removeNodeWithEdges(const size_t &id) {
    auto it = nodes_.find(id);
    if (it == nodes_.end()) {
        return;
    } else {
        assert(it->second->id == id);
        // Remove related connection information in its connected nodes
        std::unordered_set<size_t> connected_node_ids = it->second->connected_node_ids;
        for (auto &connected_node_id : connected_node_ids) {
            Node *connected_node = nodes_.at(connected_node_id);
            for (auto it_id = connected_node->connected_node_ids.begin(); 
                 it_id != connected_node->connected_node_ids.end(); ++it_id) {
                if (*(it_id) == id) {
                    connected_node->connected_node_ids.erase(it_id);
                    break;
                }
            }
            connected_node->connected_edges.erase(id);
        }
        // Remove connected edges from this scene
        std::unordered_map<size_t, Edge*> connected_edges = it->second->connected_edges;
        for (auto &connected_edge : connected_edges) {
            size_t connected_edge_id = connected_edge.second->id;
            delete connected_edge.second;
            edges_.erase(connected_edge_id);
        }
        // Finally remove the node.
        delete it->second;
        nodes_.erase(it);
    }
}

void Scene::removeEdge(const size_t &id) {
    auto it = edges_.find(id);
    if (it == edges_.end()) {
        return;
    } else {
        assert(it->second->id == id);
        Node *node1 = nodes_.at(it->second->node_ids[0]);
        Node *node2 = nodes_.at(it->second->node_ids[1]);
        
        for (auto it_connected_node_ids = node1->connected_node_ids.begin(); 
             it_connected_node_ids != node1->connected_node_ids.end(); 
             ++it_connected_node_ids) {
            if ((*it_connected_node_ids) == node2->id) {
                node1->connected_node_ids.erase(it_connected_node_ids);
                break;
            }
        }
        for (auto it_connected_node_ids = node2->connected_node_ids.begin(); 
             it_connected_node_ids != node2->connected_node_ids.end(); 
             ++it_connected_node_ids) {
            if ((*it_connected_node_ids) == node1->id) {
                node2->connected_node_ids.erase(it_connected_node_ids);
                break;
            }
        }
        node1->connected_edges.erase(node2->id);
        node2->connected_edges.erase(node1->id);

        delete it->second;
        edges_.erase(it);
    }
}

void Scene::addObservation(const size_t &track_id, 
                           const size_t &node_id, 
                           const size_t &feature_idx) {
    if (!existsNode(node_id)) {
        return;
    }

    base::Scene::Node* const node = nodes_.at(node_id);
    if (feature_idx >= node->points_2D.size()) {
        return;
    }
    node->observations[track_id] = feature_idx;
}

void Scene::removeObservation(const size_t &track_id, 
                              const size_t &node_id) {
    if (!existsNode(node_id)) {
        return;
    }

    base::Scene::Node* const node = nodes_.at(node_id);
    std::unordered_map<size_t, size_t> &observations = node->observations;
    if (observations.find(track_id) != observations.end()) {
        observations.erase(track_id);
    }
}

void Scene::addTrack(const base::Track &track) {
    const base::TrackElementList &elements = track.getElements();
    const size_t track_id = track.getId();
    for (auto &element : elements) {
        addObservation(track_id, element.image_id, element.feature_idx);
    }
}

void Scene::removeTrack(const base::Track &track) {
    const base::TrackElementList &elements = track.getElements();
    const size_t track_id = track.getId();
    for (auto &element : elements) {
        const size_t node_id = element.image_id;
        removeObservation(track_id, node_id);
    }
}

bool Scene::checkTrackDirectVisibilityInNode(const base::Track &track, 
                                             const size_t &node_id, 
                                             size_t &point_2D_idx) const {
    /* If the node does not exist, return false */
    if (!existsNode(node_id)) {
        return false;
    }

    for (auto &element : track.getElements()) {
        /* If the node is already included by the track, return true */
        if (element.image_id == node_id) {
            point_2D_idx = element.feature_idx;
            return true;
        }

        /* Check whether the 2D points in the given track have correspondences with the 
           given node */
        if (!existsNode(element.image_id)) {
            continue;
        } else {
            const base::Scene::Node* const node_in_track = nodes_.at(element.image_id);
            // Check the edges connected to the node in track
            for (auto &connected_edge : node_in_track->connected_edges) {
                if (!existsEdge(connected_edge.first)) {
                    continue;
                } 
                const base::Scene::Edge* const edge = edges_.at(connected_edge.first);
                // For edges that are connected to the given node
                if (edge->node_ids[0] == element.image_id && edge->node_ids[1] == node_id) {
                    for (auto &corr : edge->correspondences) {
                        if (corr.first == element.feature_idx) {
                            point_2D_idx = corr.second; 
                            return true;
                        }
                    }
                } else if (edge->node_ids[1] == element.image_id && edge->node_ids[0] == node_id) {
                    for (auto &corr : edge->correspondences) {
                        if (corr.second == element.feature_idx) {
                            point_2D_idx = corr.first;
                            return true;
                        }
                    }
                }
            }
        }
    }

    return false;
}

std::vector<std::pair<size_t, size_t>> 
Scene::findDirectCorrespondences(const size_t &node_id, 
                                 const size_t &feature_idx) const {
    std::vector<std::pair<size_t, size_t>> correspondences; 

    if (!existsNode(node_id)) {
        return correspondences; 
    }

    const Node* const node = nodes_.at(node_id); 
    const size_t num_features = node->points_2D.size(); 
    if (feature_idx >= num_features) {
        return correspondences; 
    }

    for (auto &edge : node->connected_edges) {
        assert(edge.second->node_ids[0] == node_id || 
               edge.second->node_ids[1] == node_id);
        if (edge.second->node_ids[0] == node_id) {
            for (auto &corr : edge.second->correspondences) {
                if (corr.first == feature_idx) {
                    correspondences.push_back(std::make_pair(edge.second->node_ids[1], corr.second));
                }
            }
        } else if (edge.second->node_ids[1] == node_id) {
            for (auto &corr : edge.second->correspondences) {
                if (corr.second == feature_idx) {
                    correspondences.push_back(std::make_pair(edge.second->node_ids[0], corr.first));
                }
            }
        }
    }

    return correspondences; 
}

std::vector<std::pair<size_t, size_t>> 
Scene::findTransitiveCorrespondences(const size_t &node_id, 
                                     const size_t &feature_idx, 
                                     const size_t &max_transitivity) const {
    std::vector<std::pair<size_t, size_t>> correspondences; 

    if (max_transitivity == 0) {
        return correspondences; 
    } else if (max_transitivity == 1) {
        return findDirectCorrespondences(node_id, feature_idx); 
    }

    if (!existsNode(node_id)) {
        return correspondences; 
    }

    const Node* const node = nodes_.at(node_id); 
    const size_t num_features = node->points_2D.size(); 
    if (feature_idx >= num_features) {
        return correspondences; 
    }

    correspondences.emplace_back(node_id, feature_idx);
    // Collected correspondences represented by (image_id, feature_indices) 
    std::unordered_map<size_t, std::unordered_set<size_t>> image_corrs; 
    image_corrs[node_id].insert(feature_idx);
    size_t last_level_begin = 0; 
    size_t last_level_end = 1; 
    for (size_t level = 0; level < max_transitivity; ++level) {
        for (size_t i = last_level_begin; i < last_level_end; ++i) {
            const std::pair<size_t, size_t> &ref_corr = correspondences[i]; 
            const size_t &ref_image_id = ref_corr.first; 
            const size_t &ref_feature_idx = ref_corr.second; 
            const std::vector<std::pair<size_t, size_t>> current_corrs = 
                findDirectCorrespondences(ref_image_id, ref_feature_idx); 
            for (const auto &current_corr : current_corrs) {
                const size_t &current_corr_image_id = current_corr.first; 
                const size_t &current_corr_feature_idx = current_corr.second; 
                if (image_corrs[current_corr_image_id].insert(current_corr_feature_idx).second) {
                    correspondences.emplace_back(current_corr_image_id, current_corr_feature_idx); 
                }
            }
        }

        last_level_begin = last_level_end; 
        last_level_end = correspondences.size(); 

        if (last_level_begin == last_level_end) {
            break;
        }
    }

    if (correspondences.size() > 1) {
        correspondences.front() = correspondences.back(); 
    }
    correspondences.pop_back(); 

    return correspondences; 
}

void Scene::removeIsolatedNodes() {
    std::unordered_map<size_t, Node*>::iterator it = nodes_.begin();
    while (it != nodes_.end()) {
        if (it->second->connected_edges.size() == 0) {
            delete it->second;
            it = nodes_.erase(it); 
        } else {
            it++;
        }
    }
}

bool Scene::hasTrack(const size_t &node_id, 
                     const size_t &feature_idx, 
                     size_t *track_id) const {
    if (!existsNode(node_id)) {
        std::cout << "[WARNING] " << __PRETTY_FUNCTION__ 
                  << "Node " << node_id << " does not exist. " << std::endl;
        return false; 
    }

    bool has_track = nodes_.at(node_id)->hasTrack(feature_idx, track_id);

    return has_track; 
}

bool Scene::hasNodeBogusCameraParameters(const size_t &node_id, 
                                         const double &min_focal_length_ratio, 
                                         const double &max_focal_length_ratio) const {
    if (!existsNode(node_id)) {
        return true; 
    }

    const base::Scene::Node* const node = nodes_.at(node_id); 

    return node->hasBogusCameraParams(min_focal_length_ratio, max_focal_length_ratio); 
}

double Scene::computeReprojectionError(const size_t &node_id, 
                                       const size_t &feature_idx, 
                                       const Eigen::Vector3d &point_3D) const {
    if (!existsNode(node_id)) {
        return -1.0; 
    }

    return nodes_.at(node_id)->computeReprojectionError(point_3D, feature_idx); 
}

bool Scene::isTwoViewObservation(const size_t &node_id, 
                                 const size_t &point_2D_idx) const {
    assert(existsNode(node_id));  
    const Node *node = nodes_.at(node_id); 
    assert(point_2D_idx < node->points_2D.size()); 
    
    const auto corrs1 = 
        findDirectCorrespondences(node_id, point_2D_idx); 
    if (corrs1.size() != 1) {
        return false; 
    } 

    const auto corrs2 = 
        findDirectCorrespondences(corrs1[0].first, corrs1[0].second); 

    return corrs2.size() == 1;
}

}
}
