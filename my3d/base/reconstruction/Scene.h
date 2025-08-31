/*
 * filename: Scene.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MY3D_BASE_RECONSTRUCTION_SCENE_H_
#define _MY3D_BASE_RECONSTRUCTION_SCENE_H_

#include <algorithm>
#include <array>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/camera/RadialPinHoleCamera.h"
#include "base/camera/projection.h"
#include "base/pose.h"
#include "base/reconstruction/Track.h"
#include "utils/eigen_types.h"

namespace my3d {
namespace base {

struct Point2D {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector2d position;
  Eigen::Matrix<uint8_t, 3, 1> color;

  Point2D() {}
  Point2D(const Eigen::Vector2d &_position,
          const Eigen::Matrix<uint8_t, 3, 1> &_color)
      : position{_position}, color{_color} {}
};

/**
 * @brief
 */
class Scene {
public:
  struct Node;
  struct Edge;

  /**
   * @brief Information of a camera including pose and intrinsics.
   */
  struct CameraInfo {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CameraInfo();
    Eigen::Quaterniond rotation;
    Eigen::Vector3d translation;
    double fx{500.0}, fy{500.0};
    double cx{0.0}, cy{0.0};
    double alpha{0.0};
    double k1{0.0}, k2{0.0};
    bool is_valid{true};
  };

  /**
   * @brief Two-view geometry information.
   */
  struct TwoViewGeometryInfo {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    TwoViewGeometryInfo();
    Eigen::Matrix3d fundamental_matrix;
    Eigen::Matrix3d homography;
    Eigen::Matrix3d essential_matrix;
    EigenVec<Eigen::Vector3d> triangulated_points;
    std::vector<double> triangulation_angles;
    double median_triangulation_angle{0.0};
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
  };

  /* A node is corresponded to a specific image. */
  struct Node {

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Node(size_t _id, int _width, int _height,
         const EigenVec<Point2D> &_points_2D,
         const std::unordered_set<size_t> &_connected_node_ids);

    inline void setPose(const Eigen::Matrix3x4d &pose) {
      camera_info.rotation = Eigen::Quaterniond(pose.leftCols<3>());
      camera_info.translation = pose.rightCols<1>();
    }

    inline void setPose(const Eigen::Quaterniond &rotation,
                        const Eigen::Vector3d &translation) {
      camera_info.rotation = rotation;
      camera_info.translation = translation;
    }

    inline void setIntrinsics(double fx, double fy, double cx, double cy,
                              double alpha, double k1, double k2) {
      camera_info.fx = fx;
      camera_info.fy = fy;
      camera_info.cx = cx;
      camera_info.cy = cy;
      camera_info.alpha = alpha;
      camera_info.k1 = k1;
      camera_info.k2 = k2;
    }

    inline Eigen::Matrix3x4d getPose() const {
      return base::composePose(
          base::quaternion2RotationMatrix(camera_info.rotation),
          camera_info.translation);
    }

    inline Eigen::Quaterniond getRotation() const {
      return camera_info.rotation;
    }

    inline Eigen::Vector3d getTranslation() const {
      return camera_info.translation;
    }

    inline Eigen::Matrix3d getCameraMatrix() const {
      Eigen::Matrix3d K;
      K << camera_info.fx, camera_info.alpha, camera_info.cx, 0, camera_info.fy,
          camera_info.cy, 0, 0, 1;

      return K;
    }

    Eigen::Matrix3x4d getProjectionMatrix() const;

    inline bool hasTrack(const size_t &feature_idx, size_t *track_id) const {
      for (const auto &obs : observations) {
        if (obs.second == feature_idx) {
          if (track_id != nullptr) {
            *track_id = obs.first;
          }
          return true;
        }
      }
      return false;
    }

    double computeReprojectionError(const Eigen::Vector3d &point_3D,
                                    const size_t &point_2D_idx) const;

    inline Point2D getPoint2D(const size_t &idx) const {
      assert(idx < points_2D.size());
      return points_2D[idx];
    }

    inline Eigen::Vector2d getPoint2DPosition(const size_t &idx) const {
      assert(idx < points_2D.size());
      return points_2D[idx].position;
    }

    inline Eigen::Matrix<uint8_t, 3, 1>
    getPoint2DColor(const size_t &idx) const {
      assert(idx < points_2D.size());
      return points_2D[idx].color;
    }

    inline Eigen::Vector2d pix2Norm(const Eigen::Vector2d &pix) const {
      const base::RadialPinHoleCamera camera(getCameraMatrix());
      return camera.pix2Norm(pix);
    }

    inline double thresholdPix2Norm(const double &threshold) const {
      const base::RadialPinHoleCamera camera(getCameraMatrix());
      return camera.thresholdPix2Norm(threshold);
    }

    bool hasBogusCameraParams(const double &min_focal_length_ratio,
                              const double &max_focal_length_ratio) const;
    // std::vector<size_t> findObservationsOfFeature(size_t feature_idx) {

    // }

    /* The ID of a node in a scene should be unique. */
    size_t id{0};
    /* The width of the corresponded image */
    int width{-1};
    /* The width of the corresponded image */
    int height{-1};
    /* List of 2D pixel coordinates (normalized by max(width, height)) of
     * feature points observed in the image, */
    EigenVec<Point2D> points_2D;
    /* List of the ID of nodes that have connection with this. */
    std::unordered_set<size_t> connected_node_ids;
    /* List of connected edges. [(connected_node_id, edge)] */
    std::unordered_map<size_t, Edge *> connected_edges;
    // /* The IDs of tracks containing this node. */
    // std::unordered_set<size_t> track_ids;
    /* The observations of 3D points observed by this node. (track_id,
     * feature_idx) */
    std::unordered_map<size_t, size_t> observations;
    /* Information of the corresponding camera */
    CameraInfo camera_info;
    /* Name of the node. Usually image path */
    std::string name{""};
  };

  /* Binary edge connecting 2 different nodes. */
  struct Edge {
    enum SceneEdgeModelType {
      INVALID = -1,
      UNDETERMINED = 0,
      GENERAL_AND_CALIBRATED = 1,
      GENERAL_AND_UNCALIBRATED = 2,
      PANORAMIC = 3,
      PLANNAR = 4,
      PANORAMIC_OR_PLANNAR = 5
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Edge(size_t _id, const std::array<size_t, 2> &_node_ids,
         const std::vector<std::pair<size_t, size_t>> &_correspondences,
         SceneEdgeModelType _model_type = UNDETERMINED)
        : id{_id}, node_ids{_node_ids}, correspondences{_correspondences},
          model_type{_model_type} {}

    /* The ID of an edge in a scene should be unique. */
    size_t id;
    /* The nodes connected by the edge */
    std::array<size_t, 2> node_ids;
    /* The matching correspondences between 2D points of two nodes.
     * [(point_idx_1, point_idx_2)] */
    std::vector<std::pair<size_t, size_t>> correspondences;
    // /* The IDs of tracks containing this edge. */
    // std::vector<size_t> track_ids;
    /* Model type */
    SceneEdgeModelType model_type{SceneEdgeModelType::UNDETERMINED};
    /* Two-view geometry information */
    TwoViewGeometryInfo two_view_geom_info;
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Scene();
  ~Scene();

  void clear();

  /**
   * @brief Add an node into the scene. The function will do nothing and return
   *  false if there already exists an edge with the same ID
   */
  bool addNode(const size_t &id, const size_t &width, const size_t &height,
               const EigenVec<Point2D> &points_2D, const std::string &name = "",
               const CameraInfo &camera_info = CameraInfo());

  /**
   * @brief Add an edge into the scene. The function will do nothing and return
   * false if:
   *   (i) there already exists an edge with the same ID
   *   (ii) one of the node id in node_ids does not exist
   */
  bool addEdge(
      const size_t &id, const std::array<size_t, 2> &node_ids,
      const std::vector<std::pair<size_t, size_t>> &correspondences,
      const Edge::SceneEdgeModelType &model_type,
      const TwoViewGeometryInfo &two_view_geom_info = TwoViewGeometryInfo());

  /**
   * @brief Remove the node with ID id. This function will not remove the
   * corresponded edges. If the node with the given ID does not exist, this
   * function will do nothing.
   */
  void removeNode(const size_t &id);

  /**
   * @brief Remove the node(s) with no edge connected to it/them.
   */
  void removeIsolatedNodes();

  /**
   * @brief Remove the node with ID id and the corresponding edges. If the node
   * with the given ID does not exist, this function will do nothing.
   */
  void removeNodeWithEdges(const size_t &id);

  /**
   * @brief Remove the edge with ID id. If the edge with the given ID does not
   * exist, this function will do nothing. Also, this function will never remove
   * any nodes.
   */
  void removeEdge(const size_t &id);

  /**
   * @brief Add an observation to the scene. Note that if there already exists
   * an observation of track_id in node_id, this function will override the
   * original observation
   *
   */
  void addObservation(const size_t &track_id, const size_t &node_id,
                      const size_t &feature_idx);

  /**
   * @brief Remove an observation from node_id. Note that this function will do
   * nothing if the observation of track_id in node_id does not exist
   */
  void removeObservation(const size_t &track_id, const size_t &node_id);

  /**
   * @brief Add a track to the scene. Note that if there already exists an
   * observation of track_id in node_id, this function will override the
   * original observation
   */
  void addTrack(const base::Track &track);

  /**
   * @brief Remove a track to the scene
   */
  void removeTrack(const base::Track &track);

  /**
   * @brief Check whether the given track is visible in the given node
   * (transitivity = 1)
   *
   * @param track             the track to be checked
   * @param node_id           the id of node to be checked
   * @param point_2D_idx[out]  if the track is visible in node[node_id], it will
   * be the index of the track's observation in node[node_id]
   */
  bool checkTrackDirectVisibilityInNode(const base::Track &track,
                                        const size_t &node_id,
                                        size_t &point_2D_idx) const;

  Eigen::Matrix<uint8_t, 3, 1>
  computeTrackColor(const base::Track &track) const {
    int color[3] = {0, 0, 0};
    int num_elems = 0;
    for (const auto &elem : track.getElements()) {
      if (!existsNode(elem.image_id)) {
        continue;
      }
      if (elem.feature_idx >= getNode(elem.image_id)->points_2D.size()) {
        continue;
      }
      const Eigen::Matrix<uint8_t, 3, 1> elem_color =
          getNodePoint2DColor(elem.image_id, elem.feature_idx);
      for (Eigen::Index i = 0; i < 3; ++i) {
        color[i] += static_cast<int>(elem_color(i));
      }
      ++num_elems;
    }

    if (num_elems == 0) {
      return Eigen::Matrix<uint8_t, 3, 1>::Zero();
    }

    return Eigen::Matrix<uint8_t, 3, 1>{
        static_cast<uint8_t>(util::clamp<int>(color[0] / num_elems, 0, 255)),
        static_cast<uint8_t>(util::clamp<int>(color[1] / num_elems, 0, 255)),
        static_cast<uint8_t>(util::clamp<int>(color[2] / num_elems, 0, 255))};
  }

  /**
   * @brief Find direct correspondences (i.e., transitivity = 1) of the given
   * feature point
   *
   * @retval the correspondences represented by a list of (node_id, feature_idx)
   * pairs
   */
  std::vector<std::pair<size_t, size_t>>
  findDirectCorrespondences(const size_t &node_id,
                            const size_t &feature_idx) const;

  /**
   * @brief Find transitive correspondences of the given feature point. Note
   * that this function does not check ambiguity of correspondences, i.e., two
   * correspondences with the same image id but different feature indices may
   * exist in the result
   *
   * @retval the correspondences represented by a list of (node_id, feature_idx)
   * pairs
   */
  std::vector<std::pair<size_t, size_t>>
  findTransitiveCorrespondences(const size_t &node_id,
                                const size_t &feature_idx,
                                const size_t &max_transitivity) const;

  /**
   * @brief Check if the feature point in a node belongs to a track
   *
   */
  bool hasTrack(const size_t &node_id, const size_t &feature_idx,
                size_t *track_id) const;

  /**
   * @brief Check if a node has bogus camera parameters
   *
   * @retval true if the node has bogus camera params or node_id does not exist,
   * false otherwise
   */
  bool hasNodeBogusCameraParameters(const size_t &node_id,
                                    const double &min_focal_length_ratio,
                                    const double &max_focal_length_ratio) const;

  /**
   * @brief Compute reprojection error of a 3D point in a node. This funciton
   * will return -1 if the given node or feature point does not exist
   */
  double computeReprojectionError(const size_t &node_id,
                                  const size_t &feature_idx,
                                  const Eigen::Vector3d &point_3D) const;

  inline std::unordered_map<size_t, Node *> getNodes() const { return nodes_; }

  inline std::unordered_map<size_t, Edge *> getEdges() const { return edges_; }

  inline Node *getNode(const size_t &id) {
    if (!existsNode(id)) {
      return nullptr;
    }
    return nodes_.at(id);
  }

  inline const Node *getNode(const size_t &id) const {
    if (!existsNode(id)) {
      return nullptr;
    }
    return nodes_.at(id);
  }

  inline Eigen::Vector3d getNodeTranslation(const size_t &id) const {
    if (!existsNode(id)) {
      return Eigen::Vector3d::Zero();
    }

    return nodes_.at(id)->getTranslation();
  }

  inline Eigen::Quaterniond getNodeRotation(const size_t &id) const {
    if (!existsNode(id)) {
      return Eigen::Quaterniond::Identity();
    }

    return nodes_.at(id)->getRotation();
  }

  inline Eigen::Matrix3x4d getNodePose(const size_t &id) const {
    if (!existsNode(id)) {
      return Eigen::Matrix3x4d::Identity();
    }

    return nodes_.at(id)->getPose();
  }

  inline Eigen::Matrix3d getNodeCameraMatrix(const size_t &id) const {
    if (!existsNode(id)) {
      return Eigen::Matrix3d::Identity();
    }

    return nodes_.at(id)->getCameraMatrix();
  }

  inline Eigen::Matrix3x4d getNodeProjectionMatrix(const size_t &id) const {
    if (!existsNode(id)) {
      return Eigen::Matrix3x4d::Identity();
    }

    return nodes_.at(id)->getProjectionMatrix();
  }

  inline bool existsNode(const size_t &id) const {
    return nodes_.find(id) != nodes_.end();
  }

  // const Node* getNode(size_t id) const;

  inline Edge *getEdge(const size_t &id) {
    if (!existsEdge(id)) {
      return nullptr;
    }
    return edges_.at(id);
  }

  inline const Edge *getEdge(const size_t &id) const {
    if (!existsEdge(id)) {
      return nullptr;
    }
    return edges_.at(id);
  }

  inline bool existsEdge(const size_t &id) const {
    return edges_.find(id) != edges_.end();
  }

  // const Edge* getEdge(size_t id) const;

  inline size_t getNumNodes() const { return nodes_.size(); }

  inline size_t getNumEdges() const { return edges_.size(); }

  /**
   * @brief Get the 2D point in a node.
   *
   * @retval the found 2D point. (zero if node_id does not exist)
   */
  inline Point2D getNodePoint2D(const size_t &node_id,
                                const size_t &point_2D_idx) const {
    assert(existsNode(node_id));
    return nodes_.at(node_id)->getPoint2D(point_2D_idx);
  }

  inline Eigen::Vector2d
  getNodePoint2DPosition(const size_t &node_id,
                         const size_t &point_2D_idx) const {
    assert(existsNode(node_id));
    return nodes_.at(node_id)->getPoint2DPosition(point_2D_idx);
  }

  inline Eigen::Matrix<uint8_t, 3, 1>
  getNodePoint2DColor(const size_t &node_id, const size_t &point_2D_idx) const {
    assert(existsNode(node_id));
    return nodes_.at(node_id)->getPoint2DColor(point_2D_idx);
  }

  bool isTwoViewObservation(const size_t &node_id,
                            const size_t &point_2D_idx) const;

private:
  /* [(Node/Image_ID, Node)] */
  std::unordered_map<size_t, Node *> nodes_;
  /* [(Edge_ID, Node)] */
  std::unordered_map<size_t, Edge *> edges_;
};

} // namespace base
} // namespace my3d

#endif // _MY3D_BASE_RECONSTRUCTION_SCENE_H_
