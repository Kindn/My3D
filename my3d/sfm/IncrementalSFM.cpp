/*
 * filename: IncrementalSFM.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "sfm/IncrementalSFM.h"

namespace my3d {
namespace sfm {

IncrementalSFM::IncrementalSFM(const Config &config) : config_{config} {}

IncrementalSFM::~IncrementalSFM() { reset(); }

double
IncrementalSFM::computeInitialPairScore(const EdgeInfo &edge_info) const {
  const base::Scene *const scene = reconstruction_->getScene();
  const double score =
      0.3 * static_cast<double>(edge_info.num_connected_nodes_1) /
          (static_cast<double>(scene->getNumNodes()) - 1) +
      0.3 * static_cast<double>(edge_info.num_connected_nodes_2) /
          (static_cast<double>(scene->getNumNodes()) - 1) +
      0.4 * edge_info.median_triangulated_angle / M_PI;

  return score;
}

void IncrementalSFM::config(const Config &config) { config_ = config; }

void IncrementalSFM::beginReconstruction(
    const std::shared_ptr<base::Reconstruction> reconstruction) {
  assert(reconstruction->getScene() != nullptr);

  // reset();
  reconstruction_ = reconstruction;
  // scene_ = scene;
}

void IncrementalSFM::endReconstruction(Report &report) {
  report.camera_infos.clear();
  report.tracks.clear();

  for (const auto &image_id : reconstruction_->getRegisteredImageIDs()) {
    report.camera_infos.insert(std::make_pair(
        image_id, reconstruction_->getScene()->getNode(image_id)->camera_info));
  }

  for (const auto &track : reconstruction_->getTracks()) {
    if (track.second->isTriangulated()) {
      report.tracks.insert(std::make_pair(track.first, *(track.second)));
    }
  }
}

void IncrementalSFM::computeInitialTracks() {}

void IncrementalSFM::reset() { reconstruction_->reset(); }

IncrementalSFM::Report IncrementalSFM::reconstruct(
    const std::shared_ptr<base::Reconstruction> reconstruction) {
  std::cout << "----------------------------Reconstruction "
               "Started----------------------------"
            << std::endl;
  beginReconstruction(reconstruction);

  Report report;
  report.success = true;
  std::vector<std::pair<size_t, size_t>> initial_image_pairs;
  std::vector<size_t> initial_edge_ids;
  std::pair<size_t, size_t> initial_image_pair;
  size_t initial_edge_id;
  if (!findInitialImagePairCandidates(initial_image_pairs, initial_edge_ids)) {
    std::cout
        << "[ERROR] Failed to find initial image pair. Reconstruction aborted. "
        << std::endl;
    report.success = false;
    return report;
  } else {
    std::cout << "[INFO] Found " << initial_image_pairs.size()
              << " initial image pair candidates. "
              << "Trying to register initial image pair... " << std::endl;
    bool success_registration = false;
    for (size_t idx = 0; idx < initial_image_pairs.size(); ++idx) {
      initial_image_pair = initial_image_pairs[idx];
      initial_edge_id = initial_edge_ids[idx];
      // const base::Scene::Edge *init_edge = scene_->getEdge(initial_edge_id_);

      // std::cout << "Trying to register edge " << initial_edge_id << " as
      // initial pair. " << std::endl;
      reset();
      if (registerInitialImagePair(initial_image_pair, initial_edge_id)) {
        const auto initial_image_pair = reconstruction_->getInitialImagePair();
        const size_t initial_edge_id = reconstruction_->getInitialEdgeID();
        const base::Scene *const scene = reconstruction_->getScene();
        std::cout << "[INFO] Initial image pair registered. "
                  << "initial_pair: (" << initial_image_pair.first << ", "
                  << initial_image_pair.second
                  << "), initial edge id: " << initial_edge_id << std::endl;
        std::cout << "[INFO] Model type: "
                  << scene->getEdge(initial_edge_id)->model_type << std::endl;
        std::cout << "[INFO] Median triangulation angle: "
                  << util::rad2Deg(
                         scene->getEdge(initial_edge_id)
                             ->two_view_geom_info.median_triangulation_angle)
                  << " deg(s) " << std::endl;
        std::cout
            << "[INFO] Initial image 0 pose: ["
            << scene->getNodeTranslation(initial_image_pair.first).transpose()
            << "]"
               " "
            << scene->getNodeRotation(initial_image_pair.first)
                   .coeffs()
                   .transpose()
            << "] " << std::endl;
        std::cout
            << "[INFO] Initial image 1 pose: ["
            << scene->getNodeTranslation(initial_image_pair.second).transpose()
            << " "
            << scene->getNodeRotation(initial_image_pair.second)
                   .coeffs()
                   .transpose()
            << "] " << std::endl;

        BundleAdjustmentReport gba_report = solveGlobalBundleAdjustment(false);
        // gba_report.print();
        if (gba_report.success) {
          report.reprojection_rmse =
              gba_report.final_squared_reprojection_error;
          std::cout << "[INFO] Initial Global BA succeeded. " << std::endl;
          const size_t num_filtered_observations = filterAllTracks(true);
          std::cout << "[INFO] Filtered " << num_filtered_observations
                    << " observations " << std::endl;
          // const size_t num_filtered_images = filterAllImages();
          // std::cout << "[INFO] Filtered " << num_filtered_images << " images
          // " << std::endl;
          if (reconstruction_->getRegisteredImageIDs().size() == 0 ||
              reconstruction_->getTracks().size() < config_.min_num_points) {
            std::cout << "[ERROR] Initial image registration failed. Trying "
                         "anothor image pair. "
                      << std::endl;
            success_registration = false;
          } else {
            success_registration = true;
          }
        } else {
          std::cout << "[ERROR] Initial Global BA failed. " << std::endl;
          success_registration = false;
        }
      } else {
        success_registration = false;
      }

      if (success_registration) {
        break;
      } else {
        std::cout << "[WARNING] Failed to resgister inital pair ("
                  << initial_image_pair.first << ", "
                  << initial_image_pair.second << "), "
                  << "Trying another pair... " << std::endl;
        reconstruction_->deregisterImage(initial_image_pair.first);
        reconstruction_->deregisterImage(initial_image_pair.second);
      }
    }

    if (!success_registration) {
      std::cout << "[ERROR] Failed to register initial image pair. "
                   "Reconstruction aborted. "
                << std::endl;
      endReconstruction(report);
      report.success = false;

      return report;
    }
  }

  std::vector<size_t> next_image_ids;
  std::vector<std::unordered_map<size_t, size_t>> corrs_3D2Ds;
  const size_t kGlobalBAInterval = config_.global_ba_interval;
  size_t step = 0;
  while (reconstruction_->getNumRegisteredImages() +
             reconstruction_->getNumFilteredImages() <
         reconstruction_->getScene()->getNumNodes()) {
    std::cout << "================================" << std::endl;
    if (!findNextImageCandidates(next_image_ids, corrs_3D2Ds)) {
      std::cout << "[WARNING] Failed to find next image "
                << reconstruction_->getRegisteredImageIDs().size()
                << ". Stopped adding image to reconstruction. " << std::endl;
      report.success = false;
      break;
    } else {
      bool success = false;
      for (size_t i = 0; i < next_image_ids.size(); ++i) {
        const size_t next_image_id = next_image_ids[i];
        const std::unordered_map<size_t, size_t> correspondences_3D2D =
            corrs_3D2Ds[i];
        std::cout << "================================" << std::endl;
        std::cout << "[INFO] Trying to register image " << next_image_id << "("
                  << reconstruction->getScene()->getNode(next_image_id)->name
                  << "); Image sees " << correspondences_3D2D.size() << "/"
                  << reconstruction_->getTracks().size() << " 3D points. "
                  << std::endl;
        reconstruction_->clearModifiedTrackIDs();
        reconstruction_->clearNewTrackIDs();
        reconstruction_->clearNewImageIDs();
        if (!registerNextImage(next_image_id, correspondences_3D2D)) {
          std::cout << "[WARNING] Failed to register next image "
                    << next_image_id << std::endl;
          reconstruction_->deregisterImage(next_image_id);
          continue;
        } else {
          std::cout << "[INFO] Registered next image (id = " << next_image_id
                    << ", "
                    << reconstruction->getScene()->getNode(next_image_id)->name
                    << std::endl;

          BundleAdjustmentReport ba_report;
          // gba_report = solveGlobalBundleAdjustment(true);
          // if (gba_report.success) {
          //     report.reprojection_rmse =
          //     gba_report.final_squared_reprojection_error; std::cout <<
          //     "[INFO] Pre Global BA succeeded. " << std::endl;
          // } else {
          //     std::cout << "[INFO] Pre Global BA failed. Reconstruction
          //     aborted " << std::endl;
          // }

          size_t num_triangulated = 0;
          num_triangulated += triangulateImage(next_image_id);

          size_t num_filtered;
          num_filtered = filterObservationsWithNegativeDepth();
          std::cout << "[INFO] Filtered " << num_filtered
                    << " observations with negative depths before BA. "
                    << std::endl;

          const auto registered_image_ids =
              reconstruction_->getRegisteredImageIDs();
          const auto filtered_image_ids =
              reconstruction_->getFilteredImageIDs();
          const auto tracks = reconstruction_->getTracks();

          ba_report = solveLocalBundleAdjustment(next_image_id, false);
          if (ba_report.success) {
            report.reprojection_rmse =
                ba_report.final_squared_reprojection_error;
            std::cout << "[INFO] Local BA succeeded. " << std::endl;
            size_t num_completed = completeTracks(ba_report.track_ids);
            size_t num_merged = mergeTracks(ba_report.track_ids);
            num_completed += completeImage(next_image_id);
            num_filtered = filterTracks(ba_report.track_ids);
            // const size_t num_triangulated = triangulateImage(next_image_id);
            std::cout << "[INFO] Filtered " << num_filtered << " observations "
                      << std::endl;
            std::cout << "[INFO] Completed " << num_completed
                      << " observations " << std::endl;
            std::cout << "[INFO] Merged " << num_merged << " observations "
                      << std::endl;
            // size_t num_triangulated = 0;
            // num_triangulated += triangulateImage(next_image_id);
            std::cout << "[INFO] Triangulated " << num_triangulated
                      << " observations " << std::endl;
            std::cout << "\n\tRegistration for image {" << next_image_id
                      << "} done. " << std::endl;
            std::cout << "\tnum_points_3D: " << tracks.size() << std::endl;
            std::cout << "\tnum_registered_images: "
                      << registered_image_ids.size() << std::endl;
            std::cout << "\tnum_filtered_images: " << filtered_image_ids.size()
                      << std::endl;
            std::cout << "\tnum_remaining_images: "
                      << reconstruction_->getScene()->getNumNodes() -
                             registered_image_ids.size() -
                             filtered_image_ids.size()
                      << std::endl;
            std::cout << "================================\n" << std::endl;
            success = true;
            // const size_t num_filtered_images = filterAllImages();
            // std::cout << "[INFO] Filtered " << num_filtered_images << "
            // images " << std::endl;
          } else {
            success = false;
            std::cout << "[INFO] Local BA failed when register image "
                      << next_image_id << std::endl;
            reconstruction_->deregisterImage(next_image_id);
            // break;
          }

          if (step % kGlobalBAInterval == 0) {
            ba_report = solveGlobalBundleAdjustment(false);
            if (ba_report.success) {
              report.reprojection_rmse =
                  ba_report.final_squared_reprojection_error;
              std::cout << "[INFO] Global BA succeeded. " << std::endl;
              const size_t num_completed = completeImage(next_image_id);
              num_filtered = filterAllTracks(false);
              // const size_t num_triangulated =
              // triangulateImage(next_image_id);
              std::cout << "[INFO] Filtered " << num_filtered
                        << " observations " << std::endl;
              std::cout << "[INFO] Completed " << num_completed
                        << " observations " << std::endl;
              // size_t num_triangulated = 0;
              // num_triangulated += triangulateImage(next_image_id);
              std::cout << "[INFO] Triangulated " << num_triangulated
                        << " observations " << std::endl;
              std::cout << "\n\tRegistration for image {" << next_image_id
                        << "} done. " << std::endl;
              std::cout << "\tnum_points_3D: " << tracks.size() << std::endl;
              std::cout << "\tnum_registered_images: "
                        << registered_image_ids.size() << std::endl;
              std::cout << "\tnum_filtered_images: "
                        << reconstruction_->getFilteredImageIDs().size()
                        << std::endl;
              std::cout << "\tnum_remaining_images: "
                        << reconstruction_->getScene()->getNumNodes() -
                               registered_image_ids.size() -
                               filtered_image_ids.size()
                        << std::endl;
              std::cout << "================================\n" << std::endl;
              success = true;
              // const size_t num_filtered_images = filterAllImages();
              // std::cout << "[INFO] Filtered " << num_filtered_images << "
              // images " << std::endl;
            } else {
              success = false;
              std::cout << "[INFO] Global BA failed when register image "
                        << next_image_id << std::endl;
              reconstruction_->deregisterImage(next_image_id);
              // break;
            }
          } else {
          }
          // gba_report.print();
        }
        if (reconstruction_->getTracks().size() < config_.min_num_points) {
          success = false;
        }

        if (success) {
          ++step;
          break;
        } else {
          std::cout << "[WARNING] Failed to register next image "
                    << next_image_id << std::endl;
          reconstruction_->deregisterImage(next_image_id);
          continue;
        }
      }
      if (!success) {
        report.success = false;
        std::cerr << "[ERROR] Failed to register next image. " << std::endl;
        break;
      }
    }
  }

  if (true) {
    std::cout << "[INFO] Doing final global bundle adjustment ... "
              << std::endl;
    reconstruction_->clearModifiedTrackIDs();
    reconstruction_->clearNewTrackIDs();
    reconstruction_->clearNewImageIDs();
    BundleAdjustmentReport gba_report = solveGlobalBundleAdjustment(false);
    const size_t num_filtered = filterAllTracks(false);
    std::cout << "[INFO] Filtered " << num_filtered << " observations "
              << std::endl;
    // gba_report.print();
    if (gba_report.success) {
      report.reprojection_rmse = gba_report.final_squared_reprojection_error;
      std::cout << "[INFO] Final global BA succeeded. " << std::endl;
    } else {
      std::cout << "[INFO] Final global BA failed. Reconstruction aborted. "
                << std::endl;
      report.success = false;
    }
  }

  endReconstruction(report);

  return report;
}

bool IncrementalSFM::findInitialImagePair(
    std::pair<size_t, size_t> &initial_pair, size_t &initial_edge_id) {
  std::vector<std::pair<size_t, size_t>> initial_pairs;
  std::vector<size_t> initial_edge_ids;
  bool found = findInitialImagePairCandidates(initial_pairs, initial_edge_ids);
  if (!found) {
    return false;
  }

  initial_pair = initial_pairs.front();
  initial_edge_id = initial_edge_ids.front();

  return true;
}

bool IncrementalSFM::findInitialImagePairCandidates(
    std::vector<std::pair<size_t, size_t>> &initial_pairs,
    std::vector<size_t> &initial_edge_ids) {
  const base::Scene *const scene = reconstruction_->getScene();
  const std::unordered_map<size_t, base::Scene::Edge *> edges =
      scene->getEdges();
  const std::unordered_map<size_t, base::Scene::Node *> nodes =
      scene->getNodes();
  EigenVec<EdgeInfo> candidate_infos;
  candidate_infos.reserve(edges.size());
  /* Only determined edges are candidates */
  for (const auto &id_edge : edges) {
    if (id_edge.second->model_type !=
            base::Scene::Edge::SceneEdgeModelType::GENERAL_AND_CALIBRATED &&
        id_edge.second->model_type !=
            base::Scene::Edge::SceneEdgeModelType::GENERAL_AND_UNCALIBRATED &&
        id_edge.second->model_type !=
            base::Scene::Edge::SceneEdgeModelType::PANORAMIC &&
        id_edge.second->model_type !=
            base::Scene::Edge::SceneEdgeModelType::PLANNAR &&
        id_edge.second->model_type !=
            base::Scene::Edge::SceneEdgeModelType::PANORAMIC_OR_PLANNAR) {
      continue;
    }

    if (static_cast<int>(id_edge.second->correspondences.size()) <
        config_.min_num_init_corrs) {
      continue;
    }

    // if
    // (std::abs(id_edge.second->two_view_geom_info.median_triangulation_angle)
    // <
    //     util::deg2Rad(std::abs(config_.min_tri_angle_deg_init))) {
    //     continue;
    // }

    if (static_cast<double>(
            id_edge.second->two_view_geom_info.triangulated_points.size()) /
            static_cast<double>(id_edge.second->correspondences.size()) <
        config_.min_init_two_view_geo_tri_rate) {
      continue;
    }

    size_t num_correspondences = id_edge.second->correspondences.size();
    for (const size_t &node_id :
         {id_edge.second->node_ids[0], id_edge.second->node_ids[1]}) {
      const base::Scene::Node *node = scene->getNode(node_id);
      for (size_t point_2D_idx = 0; point_2D_idx < node->points_2D.size();
           ++point_2D_idx) {
        num_correspondences +=
            scene->findDirectCorrespondences(node_id, point_2D_idx).size();
      }
    }

    EdgeInfo edge_info;
    edge_info.edge_id = id_edge.first;
    edge_info.num_triangulated_points =
        id_edge.second->two_view_geom_info.triangulated_points.size();
    edge_info.median_triangulated_angle =
        id_edge.second->two_view_geom_info.median_triangulation_angle;
    edge_info.num_connected_nodes_1 =
        scene->getNode(id_edge.second->node_ids[0])->connected_node_ids.size();
    edge_info.num_connected_nodes_2 =
        scene->getNode(id_edge.second->node_ids[1])->connected_node_ids.size();
    edge_info.num_correspondences = num_correspondences;
    edge_info.model_type = id_edge.second->model_type;
    // std::cout << "\t" << edge_info.edge_id << " " << id_edge.first <<
    // std::endl;

    candidate_infos.push_back(edge_info);
    // std::cout <<
    // id_edge.second->two_view_geom_info.median_triangulation_angle <<
    // std::endl;
  }
  if (candidate_infos.empty()) {
    std::cout << __PRETTY_FUNCTION__ << ": Found no candidate. " << std::endl;
    return false;
  }

  // // std::cout << "ogay2 " << candidate_infos.size() << std::endl;
  /* Sort the candidates such that edges with more triangulated points, more
     correspondences, and larger baselines are preferred */
  std::sort(
      candidate_infos.begin(), candidate_infos.end(),
      [this](const EdgeInfo &edge_info1, const EdgeInfo &edge_info2) {
        if (edge_info1.model_type ==
                base::Scene::Edge::SceneEdgeModelType::GENERAL_AND_CALIBRATED &&
            edge_info2.model_type !=
                base::Scene::Edge::SceneEdgeModelType::GENERAL_AND_CALIBRATED) {
          return true;
        } else if (
            edge_info1.model_type !=
                base::Scene::Edge::SceneEdgeModelType::GENERAL_AND_CALIBRATED &&
            edge_info2.model_type ==
                base::Scene::Edge::SceneEdgeModelType::GENERAL_AND_CALIBRATED) {
          return false;
        } else if (edge_info1.model_type ==
                       base::Scene::Edge::SceneEdgeModelType::PLANNAR &&
                   edge_info2.model_type ==
                       base::Scene::Edge::SceneEdgeModelType::PANORAMIC) {
          return true;
        } else if (edge_info1.model_type ==
                       base::Scene::Edge::SceneEdgeModelType::PANORAMIC &&
                   edge_info2.model_type ==
                       base::Scene::Edge::SceneEdgeModelType::PLANNAR) {
          return false;
        } else if (edge_info1.model_type ==
                       base::Scene::Edge::SceneEdgeModelType::
                           GENERAL_AND_UNCALIBRATED &&
                   edge_info2.model_type !=
                       base::Scene::Edge::SceneEdgeModelType::
                           GENERAL_AND_UNCALIBRATED) {
          return true;
        } else if (edge_info1.model_type !=
                       base::Scene::Edge::SceneEdgeModelType::
                           GENERAL_AND_UNCALIBRATED &&
                   edge_info2.model_type ==
                       base::Scene::Edge::SceneEdgeModelType::
                           GENERAL_AND_UNCALIBRATED) {
          return false;
        }

        return edge_info1.num_correspondences > edge_info2.num_correspondences;

        // const double score1 = computeInitialPairScore(edge_info1);
        // const double score2 = computeInitialPairScore(edge_info2);

        // return score1 > score2;
      });

  initial_pairs.clear();
  initial_edge_ids.clear();
  // std::cout << "--" << std::endl;
  for (size_t i = 0; i < candidate_infos.size(); ++i) {
    const EdgeInfo &edge_info = candidate_infos[i];
    const size_t &edge_id = edge_info.edge_id;
    const size_t &image_id1 = edges.at(edge_id)->node_ids[0];
    const size_t &image_id2 = edges.at(edge_id)->node_ids[1];
    initial_pairs.push_back(std::make_pair(image_id1, image_id2));
    initial_edge_ids.push_back(edge_id);
    // std::cout << edge_info.median_triangulated_angle << " " <<
    // scene_->getEdge(edge_id)->two_view_geom_info.median_triangulation_angle
    // << std::endl;
  }

  return true;
}

bool IncrementalSFM::registerInitialImagePair(
    const std::pair<size_t, size_t> &initial_pair, size_t initial_edge_id) {
  std::cout << "--------------Trying to register initial pair ("
            << initial_pair.first << ", " << initial_pair.second
            << ")--------------" << std::endl;

  base::Scene *scene = reconstruction_->getScene();
  base::Scene::Edge *const initial_edge = scene->getEdge(initial_edge_id);
  assert((initial_edge->node_ids[0] == initial_pair.first &&
          initial_edge->node_ids[1] == initial_pair.second));

  /* Mark the nodes as registered */
  if (!reconstruction_->registerInitialImagePair(initial_pair,
                                                 initial_edge_id)) {
    return false;
  }
  reconstruction_->clearNewImageIDs();
  reconstruction_->addNewImageID(initial_pair.first);
  reconstruction_->addNewImageID(initial_pair.second);

  /* Recover pose from two-view geometry information. Since the initial pair is
     always general and calibrated, the relative pose is already available in
     the two-view geometry information */
  base::Scene::Node *const node1 = scene->getNode(initial_edge->node_ids[0]);
  base::Scene::Node *const node2 = scene->getNode(initial_edge->node_ids[1]);
  assert(node1 != nullptr);
  assert(node2 != nullptr);
  reconstruction_->setRefImageID0(node1->id);
  reconstruction_->setRefImageID1(node2->id);
  node1->setPose(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  node2->setPose(
      Eigen::Quaterniond(initial_edge->two_view_geom_info.R.inverse()),
      -initial_edge->two_view_geom_info.R.inverse() *
          initial_edge->two_view_geom_info.t);

  /* Compute initial tracks and triangulate them */
  const std::vector<std::pair<size_t, size_t>> &correspondences =
      initial_edge->correspondences;
  const Eigen::Matrix3x4d proj1 = Eigen::Matrix3x4d::Identity();
  const Eigen::Matrix3x4d proj2 = base::composePose(
      initial_edge->two_view_geom_info.R, initial_edge->two_view_geom_info.t);
  size_t num_initial_tracks = 0;
  reconstruction_->clearNewTrackIDs();
  for (auto &corr : correspondences) {
    const Eigen::Vector2d point_2D1 = node1->points_2D[corr.first].position;
    const Eigen::Vector2d point_2D2 = node2->points_2D[corr.second].position;
    const base::RadialPinHoleCamera camera1(node1->getCameraMatrix());
    const base::RadialPinHoleCamera camera2(node2->getCameraMatrix());
    const Eigen::Vector2d point_2D1N = camera1.pix2Norm(point_2D1);
    const Eigen::Vector2d point_2D2N = camera2.pix2Norm(point_2D2);

    const Eigen::Vector3d point_3D =
        base::triangulatePoint(proj1, proj2, point_2D1N, point_2D2N);
    // const double reproj_error1 = base::computeReprojectionError(point_3D,
    // point_2D1, proj1); const double reproj_error2 =
    // base::computeReprojectionError(point_3D, point_2D2, proj2);
    const double tri_angle_rad = base::computeTriangulationAngle(
        node1->getTranslation(), node2->getTranslation(), point_3D);
    if (base::hasPointPositiveDepth(proj1, point_3D) &&
        base::hasPointPositiveDepth(proj2, point_3D) &&
        !std::isnan(tri_angle_rad) &&
        tri_angle_rad >= util::deg2Rad(config_.min_tri_angle_deg_init)) {
      const size_t track_id = reconstruction_->addTrack(
          {base::TrackElement(initial_pair.first, corr.first),
           base::TrackElement(initial_pair.second, corr.second)},
          point_3D, true);
      reconstruction_->addNewTrackID(track_id);
      ++num_initial_tracks;
      // if (reproj_error1 <= config_.max_track_reproj_error && reproj_error2 <=
      // config_.max_track_reproj_error) {
      //     addTrack({base::TrackElement(initial_pair.first, corr.first),
      //               base::TrackElement(initial_pair.second, corr.second)},
      //               point_3D, true);
      // } else {
      //     // std::cout << reproj_error1 << ", " << reproj_error2 <<
      //     std::endl;
      // }
    } else {
      // std::cout << "[INFO] Add initial track failed. ";
      // std::cout << "point3D: [" << point_3D.transpose() << "] ["
      //           << (initial_edge->two_view_geom_info.R * point_3D +
      //           initial_edge->two_view_geom_info.t).transpose() << "] t: ["
      //           << node2->getTranslation().transpose()
      //           << "], tri_angle: " << util::rad2Deg(tri_angle_rad) << "
      //           deg(s)" << std::endl;
    }
  }

  if (num_initial_tracks == 0) {
    std::cout << "[ERROR] No initial tracks! " << std::endl;
    return false;
  }

  // std::cout << proj1 << std::endl;
  // std::cout << proj2 << std::endl;
  std::cout << "median triangulation angle from initial two-view geometry: "
            << util::rad2Deg(
                   initial_edge->two_view_geom_info.median_triangulation_angle)
            << " deg(s)" << std::endl;
  std::cout << "initial_pose1: \n"
            << node1->getPose() << std::endl
            << "initial_pose2: \n"
            << node2->getPose() << std::endl;
  std::cout << "Created " << num_initial_tracks << " initial tracks. "
            << std::endl;

  reconstruction_->setInitialImagePair(initial_pair);
  reconstruction_->setInitialEdgeID(initial_edge_id);

  return true;
}

bool IncrementalSFM::findNextImage(
    size_t &next_image_id,
    std::unordered_map<size_t, size_t> &correspondences_3D2D) {
  bool success = false;
  std::vector<size_t> next_image_ids;
  std::vector<std::unordered_map<size_t, size_t>> corrs_3D2Ds;
  switch (config_.nbv_method) {
  case NextBestViewSelectionMethod::MAX_VISIBLE_TRACK_NUM:
    success = findNextImagesWithMaxVisibleTracks(next_image_ids, corrs_3D2Ds);
    next_image_id = next_image_ids.front();
    correspondences_3D2D = corrs_3D2Ds.front();
    break;

  default:
    success = findNextImagesWithMaxVisibleTracks(next_image_ids, corrs_3D2Ds);
    next_image_id = next_image_ids.front();
    correspondences_3D2D = corrs_3D2Ds.front();
  }

  return success;
}

bool IncrementalSFM::findNextImagesWithMaxVisibleTracks(
    std::vector<size_t> &next_image_ids,
    std::vector<std::unordered_map<size_t, size_t>> &correspondences_3D2Ds) {
  const base::Scene *const scene = reconstruction_->getScene();
  std::unordered_map<size_t, base::Scene::Node *> nodes = scene->getNodes();
  bool found = false;
  std::vector<std::shared_ptr<ImageInfo>> image_infos;
  for (auto &node : nodes) {
    const size_t &image_id = node.first;
    /* Only consider the unregistered images */
    if (isImageRegistered(image_id) || isImageFiltered(image_id)) {
      continue;
    }
    /* Count the number of visible tracks */
    // size_t num_visible_tracks = 0;
    // EigenVec<Eigen::Vector3d> temp_points_3D;
    // EigenVec<Eigen::Vector2d> temp_points_2D;
    std::unordered_set<size_t> added_track_ids;
    std::vector<size_t> temp_track_ids;
    std::vector<size_t> temp_feature_idxs;
    std::unordered_set<size_t> curr_vis_track_ids;
    for (size_t point_2D_idx = 0; point_2D_idx < node.second->points_2D.size();
         ++point_2D_idx) {
      const auto &corrs_2D2D =
          // scene->findTransitiveCorrespondences(image_id, point_2D_idx,
          // config_.max_transitivity);
          scene->findDirectCorrespondences(image_id, point_2D_idx);
      // std::cout << corrs_2D2D.size() << std::endl;
      curr_vis_track_ids.clear();
      for (const auto &corr_2D2D : corrs_2D2D) {
        const size_t &corr_image_id = corr_2D2D.first;
        const size_t &corr_feature_idx = corr_2D2D.second;
        if (!isImageRegistered(corr_image_id) ||
            isImageFiltered(corr_image_id)) {
          continue;
        }
        if (scene->hasNodeBogusCameraParameters(
                corr_image_id, config_.min_focal_length_ratio,
                config_.max_focal_length_ratio)) {
          continue;
        }
        if (corr_image_id == image_id) {
          continue;
        }
        size_t vis_track_id;
        if (scene->hasTrack(corr_image_id, corr_feature_idx, &vis_track_id)) {
          if (!existsTrack(vis_track_id)) {
            continue;
          }
          if (curr_vis_track_ids.count(vis_track_id) > 0) {
            continue;
          }
          temp_track_ids.push_back(vis_track_id);
          temp_feature_idxs.push_back(point_2D_idx);
          curr_vis_track_ids.insert(vis_track_id);
          added_track_ids.insert(vis_track_id);
        }
      }
    }

    const std::shared_ptr<ImageInfo> image_info(new ImageInfo);
    image_info->image_id = node.first;
    image_info->num_visible_tracks = added_track_ids.size();
    image_info->corrs_3D2D.clear();
    for (size_t i = 0; i < temp_track_ids.size(); ++i) {
      image_info->corrs_3D2D.insert(
          std::make_pair(temp_track_ids[i], temp_feature_idxs[i]));
    }
    image_infos.push_back(image_info);
    found = true;
  }

  std::sort(image_infos.begin(), image_infos.end(),
            [](const std::shared_ptr<ImageInfo> &info1,
               const std::shared_ptr<ImageInfo> &info2) {
              if (info1->num_visible_tracks > info2->num_visible_tracks) {
                return true;
              } else {
                return false;
              }
            });

  next_image_ids.resize(image_infos.size());
  correspondences_3D2Ds.resize(image_infos.size());
  for (size_t i = 0; i < image_infos.size(); ++i) {
    next_image_ids[i] = image_infos[i]->image_id;
    correspondences_3D2Ds[i] = image_infos[i]->corrs_3D2D;
  }

  return found;
}

bool IncrementalSFM::findNextImageCandidates(
    std::vector<size_t> &next_image_ids,
    std::vector<std::unordered_map<size_t, size_t>> &correspondences_3D2Ds) {
  bool success = false;

  switch (config_.nbv_method) {
  case NextBestViewSelectionMethod::MAX_VISIBLE_TRACK_NUM:
    success = findNextImagesWithMaxVisibleTracks(next_image_ids,
                                                 correspondences_3D2Ds);
    break;

  default:
    success = findNextImagesWithMaxVisibleTracks(next_image_ids,
                                                 correspondences_3D2Ds);
  }

  return success;
}

bool IncrementalSFM::registerNextImage(
    const size_t &image_id,
    const std::unordered_map<size_t, size_t> &correspondences_3D2D) {
  base::Scene *const scene = reconstruction_->getScene();

  if (!existsImage(image_id)) {
    return false;
  }

  if (isImageRegistered(image_id) || isImageFiltered(image_id)) {
    return false;
  }
  base::Scene::Node *node = scene->getNode(image_id);
  if (node == nullptr) {
    return false;
  }

  /* Find the absolute pose of the next image */
  // Search for 3D-2D correspondences
  std::vector<size_t> track_ids;
  std::vector<size_t> observation_idxs;
  EigenVec<Eigen::Vector3d> points_3D;
  EigenVec<Eigen::Vector2d> points_2D;
  if (correspondences_3D2D.size() <
      estimator::AP3PPoseEstimator::kMinNumSamples) {
    return false;
  }
  for (const auto &corr : correspondences_3D2D) {
    if (!existsTrack(corr.first)) {
      std::cout << "[WARNING] track " << corr.first << " does not exits. "
                << std::endl;
      continue;
    }

    if (corr.second >= node->points_2D.size()) {
      std::cout << "[WARNING] obervation index " << corr.second
                << " exceeds the size limit " << node->points_2D.size()
                << std::endl;
      continue;
    }

    const size_t track_id = corr.first;
    const size_t observation_idx = corr.second;
    track_ids.push_back(track_id);
    observation_idxs.push_back(observation_idx);
    points_3D.push_back(reconstruction_->getTrack(track_id)->get3DPosition());
    points_2D.push_back(
        node->pix2Norm(node->points_2D[observation_idx].position));
  }

  // Solve for the absolute pose using RANSAC P3P
  estimator::RANSACConfig estimator_config;
  estimator_config.max_inlier_error =
      node->thresholdPix2Norm(config_.max_abs_pose_reproj_error);
  estimator_config.min_iter_num = 100;
  estimator_config.max_iter_num = 10000;
  estimator_config.min_inlier_ratio = config_.min_abs_pose_inlier_ratio;
  estimator_config.confidence = 0.99999;
  AbsolutePoseEstimator estimator(estimator_config);
  if (points_3D.size() < estimator.getEstimator().kMinNumSamples ||
      points_3D.size() < estimator.getLocalEstimator().kMinNumSamples) {
    std::cout << "[ERROR] Insufficient points for robust absolute pose "
                 "estimation. Registration for image "
              << image_id << " failed. " << std::endl;
    return false;
  }
  estimator.getEstimator().setEpsilon(config_.abs_pose_epsilon);
  estimator.getEstimator().setK(Eigen::Matrix3d::Identity());
  estimator.getLocalEstimator().setK(Eigen::Matrix3d::Identity());
  AbsolutePoseEstimator::Report report =
      estimator.estimate(points_3D, points_2D);
  if (!report.success) {
    std::cout << "[ERROR] Robust absolute pose estimation failed. Registration "
                 "for image "
              << image_id << " failed. " << std::endl;
    return false;
  }

  const double inlier_ratio = static_cast<double>(report.num_inliers) /
                              static_cast<double>(points_3D.size());
  if (report.num_inliers < config_.min_abs_pose_inlier_num &&
      inlier_ratio < config_.min_abs_pose_inlier_ratio) {
    std::cout << "[ERROR] Insufficient inlier number of robust absolute pose "
                 "estimation "
              << "(inlier number: " << report.num_inliers << "). "
              << "Registration for image " << image_id << " failed. "
              << std::endl;
    return false;
  }

  // const double inlier_ratio = static_cast<double>(report.num_inliers) /
  // static_cast<double>(report.inlier_mask.size()); if (inlier_ratio <
  // config_.min_abs_pose_inlier_ratio) {
  //     std::cout << "[ERROR] Insufficient inlier ratio of robust absolute pose
  //     estimation "
  //               << "(inlier ratio: " << inlier_ratio << "). "
  //               << "Registration for image " << image_id << " failed. " <<
  //               std::endl;
  //     return false;
  // }

  // Update the result pose to the scene node
  Eigen::Matrix3x4d abs_pose = report.model;
  std::cout << "abs_pose\n" << abs_pose << std::endl;
  node->setPose(abs_pose);

  /* Mark the image as registered */
  if (!reconstruction_->registerImage(image_id)) {
    return false;
  }
  reconstruction_->clearNewImageIDs();
  reconstruction_->addNewImageID(image_id);

  /* Update the existing tracks */
  std::unordered_set<size_t> added_track_ids;
  reconstruction_->clearModifiedTrackIDs();
  for (size_t i = 0; i < report.inlier_mask.size(); ++i) {
    if (report.inlier_mask[i] && added_track_ids.insert(track_ids[i]).second &&
        !reconstruction_->hasTrack(image_id, observation_idxs[i], nullptr)) {
      reconstruction_->addObservation(track_ids[i], image_id,
                                      observation_idxs[i]);
      reconstruction_->addModifiedTrackID(track_ids[i]);
    }
  }

  /* Refine absolute pose */
  const auto refinement_report = refineAbsolutePose(image_id);
  if (refinement_report.success) {
    std::cout << "[INFO] Pose refinement done. " << std::endl;
    std::cout << "[INFO] Refined absolute pose: \n " << node->getPose()
              << std::endl;
    return true;
  } else {
    std::cout << "[WARNING] Pose refinement failed. " << std::endl;
    return false;
  }

  return true;
}

IncrementalSFM::BundleAdjustmentReport
IncrementalSFM::refineAbsolutePose(const size_t &image_id) {
  BundleAdjustmentReport report;
  if (!isImageRegistered(image_id) || isImageFiltered(image_id)) {
    report.success = false;
    return report;
  }

  base::Scene *const scene = reconstruction_->getScene();
  base::Scene::Node *const node = scene->getNode(image_id);
  base::BundleAdjustment::Config ba_config;
  ba_config.cam_opt_mode =
      base::BundleAdjustment::CameraOptimizationMode::ONLY_POSE;
  ba_config.optimization_algorithm =
      base::BundleAdjustment::ALGORITHM_LEVENBERG_MARQUART_SPARSE_SHUR;
  ba_config.max_num_iters = 10;
  base::BundleAdjustment ba_solver(ba_config);
  // Add the image
  EigenVec<Eigen::Vector2d> points_2D(node->points_2D.size());
  for (size_t i = 0; i < points_2D.size(); ++i) {
    points_2D[i] = node->points_2D[i].position;
  }
  ba_solver.addCamera(image_id, node->width, node->height, false,
                      node->getRotation(), node->getTranslation(), points_2D,
                      base::RadialPinHoleCamera(node->getCameraMatrix(),
                                                node->camera_info.k1,
                                                node->camera_info.k2));
  // Collect all 3D-2D
  for (const auto &obs : node->observations) {
    const size_t &track_id = obs.first;
    const size_t &point_2D_idx = obs.second;
    if (!existsTrack(track_id)) {
      continue;
    }
    const base::Track *const track = reconstruction_->getTracks().at(track_id);
    if (!(track->isTriangulated())) {
      continue;
    }
    std::unordered_map<size_t, size_t> observations;
    observations.insert(std::make_pair(image_id, point_2D_idx));
    ba_solver.addPoint3D(track_id, track->get3DPosition(), observations, true);
  }

  std::cout << "[INFO] Refining absolute pose for image " << image_id
            << std::endl;
  const auto ba_report = ba_solver.optimize();
  report.num_iter = ba_report.num_iter;
  report.final_cost = ba_report.final_cost;
  report.final_squared_reprojection_error =
      ba_report.final_squared_reprojection_error;
  report.success = ba_report.success;
  report.ret_val = ba_report.ret_val;
  report.error_info = ba_report.error_info;
  report.num_cameras = ba_report.num_cameras;
  report.num_points_3D = ba_report.num_points_3D;
  report.num_observations = ba_report.num_observations;

  if (report.success) {
    const auto ba_camera = ba_solver.getCameras().at(image_id);
    node->setPose(ba_camera->rotation, ba_camera->translation);
    node->setIntrinsics(ba_camera->camera.fx_, ba_camera->camera.fy_,
                        ba_camera->camera.cx_, ba_camera->camera.cy_,
                        ba_camera->camera.alpha_, ba_camera->camera.k1_,
                        ba_camera->camera.k2_);
    // updateStructureAndMotionFromBA(ba_solver);
  }

  return report;
}

IncrementalSFM::BundleAdjustmentReport
IncrementalSFM::solveGlobalBundleAdjustment(const bool &should_update) {
  BundleAdjustmentReport gba_report;

  const auto tracks = reconstruction_->getTracks();
  const auto registered_image_ids = reconstruction_->getRegisteredImageIDs();
  const base::Scene *const scene = reconstruction_->getScene();
  if (tracks.empty()) {
    std::cout
        << __PRETTY_FUNCTION__
        << " : There is no available track. Global bundle adjustment aborted. "
        << std::endl;
    return gba_report;
  }
  if (registered_image_ids.empty()) {
    std::cout
        << __PRETTY_FUNCTION__
        << " : There is no available image. Global bundle adjustment aborted. "
        << std::endl;
    return gba_report;
  }

  auto all_nodes = scene->getNodes();

  /* Bundle adjustment */
  base::BundleAdjustment ba_solver(config_.gba_config.ba_config);
  // Add registered cameras
  for (auto &image_id : registered_image_ids) {
    if (!existsImage(image_id)) {
      continue;
    }

    if (isImageFiltered(image_id)) {
      continue;
    }

    const base::Scene::Node *const node = scene->getNode(image_id);
    EigenVec<Eigen::Vector2d> points_2D;
    for (size_t point_2D_idx = 0; point_2D_idx < node->points_2D.size();
         ++point_2D_idx) {
      const Eigen::Vector2d &point_2D = node->points_2D[point_2D_idx].position;
      points_2D.push_back(point_2D);
    }
    bool fix_in{false}, fix_t{false}, fix_r{false};
    if (config_.gba_config.ba_config.cam_opt_mode ==
            base::BundleAdjustment::CameraOptimizationMode::FULL ||
        config_.gba_config.ba_config.cam_opt_mode ==
            base::BundleAdjustment::CameraOptimizationMode::
                FULL_AND_SEPARATED) {
      if (image_id == reconstruction_->getRefImageID0()) {
        fix_t = true;
        fix_r = true;
      } else if (image_id == reconstruction_->getRefImageID1()) {
        fix_t = true;
      }
    } else {
      if (image_id == reconstruction_->getRefImageID0()) {
        fix_in = true;
        fix_t = true;
        fix_r = true;
      }
    }
    if (!ba_solver.addCamera(image_id, node->width, node->height, fix_in, fix_t,
                             fix_r, node->getRotation(), node->getTranslation(),
                             points_2D,
                             base::RadialPinHoleCamera(node->getCameraMatrix(),
                                                       node->camera_info.k1,
                                                       node->camera_info.k2))) {
      std::cout << __PRETTY_FUNCTION__ << " : Adding camera " << image_id
                << " to BA problem failed. Global bundle adjustment aborted. "
                << std::endl;
      return gba_report;
    }
  }
  // Add 3D points with their observations
  std::vector<size_t> track_ids;
  for (auto &track : tracks) {
    if (!(track.second->isTriangulated())) {
      continue;
    }
    // Collect observations
    const base::TrackElementList &elements = track.second->getElements();
    std::unordered_map<size_t, size_t> observations;
    for (auto &element : elements) {
      if (isImageRegistered(element.image_id) &&
          !isImageFiltered(element.image_id)) {
        observations.insert(
            std::make_pair(element.image_id, element.feature_idx));
      }
    }
    const size_t kMinFixedTrackLength = 15;
    const bool fixed =
        (observations.size() < kMinFixedTrackLength) ? false : true;
    if (!ba_solver.addPoint3D(track.first, track.second->get3DPosition(),
                              observations, fixed)) {
      std::cout << __PRETTY_FUNCTION__ << " : Adding track " << track.first
                << " to BA problem failed. Global bundle adjustment aborted. "
                << std::endl;
      return gba_report;
    }
    track_ids.push_back(track.first);
  }
  // Solve BA
  auto ba_report = ba_solver.optimize();
  gba_report.num_iter = ba_report.num_iter;
  gba_report.final_cost = ba_report.final_cost;
  gba_report.final_squared_reprojection_error =
      ba_report.final_squared_reprojection_error;
  gba_report.success = ba_report.success;
  gba_report.ret_val = ba_report.ret_val;
  gba_report.error_info = ba_report.error_info;
  gba_report.num_cameras = ba_report.num_cameras;
  gba_report.num_points_3D = ba_report.num_points_3D;
  gba_report.num_observations = ba_report.num_observations;

  if (!ba_report.success) {
    std::cout << __PRETTY_FUNCTION__
              << ": Optimization failed. Global bundle adjustment aborted. "
              << std::endl;
  }
  // Update parameters of cameras and coordinates of tracks according to the
  // result of BA Filter images with bogus camera parameters
  if (ba_report.success) {
    updateStructureAndMotionFromBA(ba_solver);
  }

  gba_report.num_merged_observations = 0;
  gba_report.num_completed_observations = 0;
  if (should_update) {
    /* Merge tracks */
    gba_report.num_merged_observations += mergeTracks(track_ids);

    /* Complete tracks */
    gba_report.num_completed_observations += completeTracks(track_ids);
  }

  if (config_.gba_config.print_report) {
    gba_report.print();
  }

  reconstruction_->normalize();

  return gba_report;
}

IncrementalSFM::BundleAdjustmentReport
IncrementalSFM::solveLocalBundleAdjustment(const size_t &image_id,
                                           const bool &should_update) {
  BundleAdjustmentReport lba_report;

  const auto tracks = reconstruction_->getTracks();
  const auto registered_image_ids = reconstruction_->getRegisteredImageIDs();
  const base::Scene *const scene = reconstruction_->getScene();
  if (tracks.empty()) {
    std::cout
        << __PRETTY_FUNCTION__
        << " : There is no available track. Local bundle adjustment aborted. "
        << std::endl;
    return lba_report;
  }
  if (registered_image_ids.empty()) {
    std::cout
        << __PRETTY_FUNCTION__
        << " : There is no available image. Local bundle adjustment aborted. "
        << std::endl;
    return lba_report;
  }
  if (!existsImage(image_id)) {
    std::cout << __PRETTY_FUNCTION__ << " : Image " << image_id
              << " does not exist. Local bundle adjustment aborted. "
              << std::endl;
    return lba_report;
  }

  /* Bundle adjustment */
  base::BundleAdjustment::Config ba_config = config_.gba_config.ba_config;
  ba_config.max_num_iters = 10;
  base::BundleAdjustment ba_solver(ba_config);

  //* Collect data
  const base::Scene::Node *node = scene->getNode(image_id);
  std::unordered_map<size_t, size_t> nums_covisible;
  for (const auto &obs : node->observations) {
    const size_t &track_id = obs.first;

    if (!existsTrack(track_id)) {
      continue;
    }
    const base::Track *track = tracks.at(track_id);
    if (!track->isTriangulated()) {
      continue;
    }

    for (const auto &elem : track->getElements()) {
      if (elem.image_id == image_id) {
        continue;
      }
      if (!existsImage(elem.image_id) || isImageFiltered(elem.image_id)) {
        continue;
      }
      nums_covisible[elem.image_id] += 1;
    }
  }

  const size_t kMaxNumRelatedImages = 5;
  std::vector<std::pair<size_t, size_t>> nums_covisible_vec;
  for (const auto &img : nums_covisible) {
    nums_covisible_vec.emplace_back(img.first, img.second);
  }
  const size_t num_related_images =
      std::min(kMaxNumRelatedImages, nums_covisible_vec.size());
  if (num_related_images < 1) {
    lba_report.success = false;
    return lba_report;
  }
  std::partial_sort(nums_covisible_vec.begin(),
                    nums_covisible_vec.begin() + num_related_images,
                    nums_covisible_vec.end(),
                    [](const std::pair<size_t, size_t> &img1,
                       const std::pair<size_t, size_t> &img2) {
                      return img1.second > img2.second;
                    });

  const size_t local_ref_image_id_0 = image_id;
  size_t local_ref_image_id_1;
  std::unordered_set<size_t> camera_ids;
  camera_ids.insert(image_id);
  for (size_t i = 0; i < num_related_images; ++i) {
    if (i == 0) {
      local_ref_image_id_1 = nums_covisible_vec[i].first;
    }
    camera_ids.insert(nums_covisible_vec[i].first);
  }
  lba_report.camera_ids.clear();
  lba_report.track_ids.clear();
  std::unordered_set<size_t> track_ids;
  for (const size_t &camera_id : camera_ids) {
    const base::Scene::Node *n = scene->getNode(camera_id);
    EigenVec<Eigen::Vector2d> points_2D;
    for (size_t point_2D_idx = 0; point_2D_idx < n->points_2D.size();
         ++point_2D_idx) {
      points_2D.push_back(n->points_2D[point_2D_idx].position);
    }
    bool fix_in{false}, fix_t{false}, fix_r{false};
    if (config_.gba_config.ba_config.cam_opt_mode ==
            base::BundleAdjustment::CameraOptimizationMode::FULL ||
        config_.gba_config.ba_config.cam_opt_mode ==
            base::BundleAdjustment::CameraOptimizationMode::
                FULL_AND_SEPARATED) {
      if (camera_id == reconstruction_->getRefImageID0() ||
          camera_id == local_ref_image_id_0) {
        fix_t = true;
        fix_r = true;
      } else if (image_id == reconstruction_->getRefImageID1() ||
                 local_ref_image_id_1) {
        fix_t = true;
      }
    } else {
      if (camera_id == reconstruction_->getRefImageID0() ||
          local_ref_image_id_0) {
        fix_in = true;
        fix_t = true;
        fix_r = true;
      }
    }
    if (!ba_solver.addCamera(
            camera_id, n->width, n->height, fix_in, fix_t, fix_r,
            n->getRotation(), n->getTranslation(), points_2D,
            base::RadialPinHoleCamera(n->getCameraMatrix(), n->camera_info.k1,
                                      n->camera_info.k2))) {
      std::cout << __PRETTY_FUNCTION__ << " : Adding camera " << image_id
                << " to BA problem failed. Global bundle adjustment aborted. "
                << std::endl;
      return lba_report;
    }
    lba_report.camera_ids.push_back(camera_id);

    for (const auto &obs : n->observations) {
      const size_t &track_id = obs.first;
      if (!existsTrack(track_id)) {
        continue;
      }
      const base::Track *track = tracks.at(track_id);
      if (!track->isTriangulated()) {
        continue;
      }
      if (!(reconstruction_->isTrackNew(track_id) ||
            reconstruction_->isTrackModified(track_id))) {
        continue;
      }
      track_ids.insert(track_id);
    }
  }

  for (const size_t &track_id : track_ids) {
    const base::Track *track = tracks.at(track_id);
    std::unordered_map<size_t, size_t> observations;
    for (const auto &elem : track->getElements()) {
      if (!existsImage(elem.image_id) || isImageFiltered(elem.image_id)) {
        continue;
      }
      if (camera_ids.count(elem.image_id) == 0) {
        continue;
      }
      observations.insert(std::make_pair(elem.image_id, elem.feature_idx));
    }
    if (!ba_solver.addPoint3D(track_id, track->get3DPosition(), observations)) {
      std::cout << __PRETTY_FUNCTION__ << " : Adding track " << track_id
                << " to BA problem failed. Global bundle adjustment aborted. "
                << std::endl;
      return lba_report;
    }
    lba_report.track_ids.push_back(track_id);
  }

  // Solve BA
  auto ba_report = ba_solver.optimize();
  lba_report.num_iter = ba_report.num_iter;
  lba_report.final_cost = ba_report.final_cost;
  lba_report.final_squared_reprojection_error =
      ba_report.final_squared_reprojection_error;
  lba_report.success = ba_report.success;
  lba_report.ret_val = ba_report.ret_val;
  lba_report.error_info = ba_report.error_info;
  lba_report.num_cameras = ba_report.num_cameras;
  lba_report.num_points_3D = ba_report.num_points_3D;
  lba_report.num_observations = ba_report.num_observations;

  if (!ba_report.success) {
    std::cout << __PRETTY_FUNCTION__
              << ": Optimization failed. Global bundle adjustment aborted. "
              << std::endl;
  }
  // Update parameters of cameras and coordinates of tracks according to the
  // result of BA Filter images with bogus camera parameters
  if (ba_report.success) {
    updateStructureAndMotionFromBA(ba_solver);
  }

  lba_report.num_merged_observations = 0;
  lba_report.num_completed_observations = 0;
  if (should_update) {
    /* Merge tracks */
    lba_report.num_merged_observations += mergeTracks(lba_report.track_ids);

    /* Complete tracks */
    lba_report.num_completed_observations +=
        completeTracks(lba_report.track_ids);
  }

  if (config_.gba_config.print_report) {
    lba_report.print();
  }

  return lba_report;
}

void IncrementalSFM::updateStructureAndMotionFromBA(
    const base::BundleAdjustment &ba_solver) {
  const auto &ba_cameras = ba_solver.getCameras();
  const auto &ba_points_3D = ba_solver.getPoints3D();

  for (auto &ba_camera : ba_cameras) {
    const size_t camera_id = ba_camera.first;
    if (!existsImage(camera_id)) {
      continue;
    }

    if (!isImageRegistered(camera_id) || isImageFiltered(camera_id)) {
      continue;
    }

    if (ba_camera.second->hasBogusParams(config_.min_focal_length_ratio,
                                         config_.max_focal_length_ratio)) {
      reconstruction_->removeImage(camera_id);
      std::cout << "[INFO] Removed camera " << camera_id
                << " due to bogus parameters. " << std::endl;
      continue;
    }

    base::Scene::Node *const node =
        reconstruction_->getScene()->getNode(camera_id);
    if (node != nullptr) {
      node->setPose(ba_camera.second->rotation, ba_camera.second->translation);
      node->setIntrinsics(
          ba_camera.second->camera.fx_, ba_camera.second->camera.fy_,
          ba_camera.second->camera.cx_, ba_camera.second->camera.cy_,
          ba_camera.second->camera.alpha_, ba_camera.second->camera.k1_,
          ba_camera.second->camera.k2_);
    }
  }
  for (auto &ba_point_3D : ba_points_3D) {
    const size_t track_id = ba_point_3D.first;
    if (!existsTrack(track_id)) {
      continue;
    }
    base::Track *const track = reconstruction_->getTrack(track_id);
    track->set3DPosition(ba_point_3D.second->position);
    track->setTriangulated(true);
  }
}

size_t IncrementalSFM::mergeTrack(const size_t &track_id) {
  if (!existsTrack(track_id)) {
    return 0;
  }

  base::Track *const track = reconstruction_->getTrack(track_id);
  if (!(track->isTriangulated())) {
    return 0;
  }

  for (auto &element : track->getElements()) {
    if (!existsImage(element.image_id)) {
      reconstruction_->removeObservation(track_id, element.image_id,
                                         element.feature_idx);
      continue;
    }

    const size_t image_id = element.image_id;
    const size_t point_2D_idx = element.feature_idx;
    std::vector<std::pair<size_t, size_t>> correspondences =
        // scene_->findDirectCorrespondences(image_id, point_2D_idx);
        reconstruction_->getScene()->findTransitiveCorrespondences(
            image_id, point_2D_idx, config_.max_transitivity);
    for (auto &corr : correspondences) {
      size_t corr_track_id;
      if (!(reconstruction_->getScene()->hasTrack(corr.first, corr.second,
                                                  &corr_track_id))) {
        continue;
      }

      if (corr_track_id == track_id) {
        continue;
      }

      if (!existsTrack(corr_track_id)) {
        continue;
      }

      base::Track *const corr_track = reconstruction_->getTrack(corr_track_id);
      if (!(corr_track->isTriangulated())) {
        continue;
      }
      const size_t len_track = track->length();
      const size_t len_corr_track = corr_track->length();
      const Eigen::Vector3d merge_pos =
          (len_track * track->get3DPosition() +
           len_corr_track * corr_track->get3DPosition()) /
          (len_track + len_corr_track);

      bool success = true;
      base::TrackElementList merge_elements;
      std::unordered_set<size_t> added_image_ids;
      for (auto &elems : {track->getElements(), corr_track->getElements()}) {
        for (auto &elem : elems) {
          if (reconstruction_->getScene()->computeReprojectionError(
                  elem.image_id, elem.feature_idx, merge_pos) >
              config_.max_merge_reproj_error) {
            success = false;
            break;
          }
          if (added_image_ids.insert(elem.image_id).second) {
            merge_elements.push_back(elem);
          }
        }
        if (!success) {
          break;
        }
      }

      if (success) {
        // Remove original tracks
        reconstruction_->removeTrack(track_id);
        reconstruction_->removeTrack(corr_track_id);
        // Add new track
        const size_t merge_track_id =
            reconstruction_->addTrack(merge_elements, merge_pos, true);
        // Merge the new track recursively
        const size_t num_merged_recursive = mergeTrack(merge_track_id);
        if (num_merged_recursive > 0) {
          return num_merged_recursive;
        } else {
          return merge_elements.size();
        }
      }
    }
  }

  return 0;
}

size_t IncrementalSFM::mergeTracks(const std::vector<size_t> &track_ids) {
  size_t num_merged = 0;
  for (size_t i = 0; i < track_ids.size(); ++i) {
    num_merged += mergeTrack(track_ids[i]);
    std::cout << "\r[INFO] Merging tracks... " << i + 1 << "/"
              << track_ids.size();
  }
  std::cout << std::endl;

  return num_merged;
}

size_t IncrementalSFM::triangulateImage(const size_t &image_id) {
  size_t num_tris = 0;

  if (!existsImage(image_id)) {
    return num_tris;
  }

  if (!isImageRegistered(image_id) || isImageFiltered(image_id)) {
    return num_tris;
  }

  base::Scene::Node *const node =
      reconstruction_->getScene()->getNode(image_id);
  if (node->hasBogusCameraParams(config_.min_focal_length_ratio,
                                 config_.max_focal_length_ratio)) {
    return num_tris;
  }
  /* Try to create new track from each feature point */
  reconstruction_->clearNewTrackIDs();
  const base::Scene *const scene = reconstruction_->getScene();
  for (size_t point_2D_idx = 0; point_2D_idx < node->points_2D.size();
       ++point_2D_idx) {
    // Skip features that is already triangulated
    size_t existing_track_id;
    if (node->hasTrack(point_2D_idx, &existing_track_id)) {
      if (existsTrack(existing_track_id)) {
        base::Track *existing_track =
            reconstruction_->getTrack(existing_track_id);
        // If a track already exists but has not been triangulated, triangulate
        // it
        if (!existing_track->isTriangulated()) {
          num_tris += triangulateTrack(existing_track_id);
        }
      }

      continue;
    }

    if (scene->isTwoViewObservation(image_id, point_2D_idx)) {
      continue;
    }

    // Try to create new tracks from observations that have no tracks
    const std::vector<std::pair<size_t, size_t>> correspondences =
        // scene_->findDirectCorrespondences(image_id, point_2D_idx);
        scene->findTransitiveCorrespondences(image_id, point_2D_idx,
                                             config_.max_transitivity);
    // std::cout << "[INFO] Found " << correspondences.size() << "
    // correspondences for feature (index = "
    //           << point_2D_idx << ") in image (id = " << image_id << ")" <<
    //           std::endl;
    base::TrackElementList track_elems;
    std::unordered_set<size_t> added_image_ids;
    track_elems.emplace_back(image_id, point_2D_idx);
    added_image_ids.insert(image_id);
    for (const auto &corr : correspondences) {
      // Skip unregistered or filtered images
      if (!isImageRegistered(corr.first)) {
        continue;
      }
      if (isImageFiltered(corr.first)) {
        continue;
      }
      // Skip observations that already in tracks
      if (scene->hasTrack(corr.first, corr.second, nullptr)) {
        continue;
      }
      // Skip myself
      if (image_id == corr.first) {
        continue;
      }

      // if (added_image_ids.insert(corr.first).second) {
      //     track_elems.emplace_back(corr.first, corr.second);
      // }
      track_elems.emplace_back(corr.first, corr.second);
    }

    if (track_elems.size() < std::max(std::max(config_.min_tri_num_obs, 2UL),
                                      config_.min_track_length)) {
      continue;
    } else {
      const size_t track_id = reconstruction_->addTrack(
          track_elems, Eigen::Vector3d::Zero(), false);
      // std::cout << "[INFO] Trying to triangulate track " << track_id <<
      // std::endl;
      num_tris += triangulateTrack(track_id);
    }
  }

  return num_tris;
}

size_t IncrementalSFM::triangulateTrack(const size_t &track_id) {
  size_t num_tris = 0;

  if (!existsTrack(track_id)) {
    return num_tris;
  }

  base::Track *const track = reconstruction_->getTrack(track_id);
  base::TrackElementList elements = track->getElements();
  base::TrackElementList inliers, outliers;
  /* Initial triangulation */
  estimator::RANSACConfig tri_config;
  tri_config.max_inlier_error =
      util::deg2Rad(config_.max_tri_inlier_angular_error_deg);
  const size_t kMaxNumExhaustion = 15;
  if (elements.size() <= kMaxNumExhaustion) {
    tri_config.min_iter_num =
        util::computeCombinationNumber(elements.size(), 2);
    tri_config.max_iter_num = 2 * tri_config.min_iter_num;
  } else {
    tri_config.min_iter_num = 10;
    tri_config.max_iter_num = 10000;
  }
  tri_config.confidence = 0.9999;
  tri_config.min_inlier_ratio = 0.02;

  tri_config.verbose = false;
  estimator::RobustTriangulationEstimator::Report report =
      triangulateObservations(tri_config, elements);

  if (!report.success) {
    track->setTriangulated(false);
    reconstruction_->removeTrack(track_id);
    // std::cout << "[ERROR] Robust triangulation failed for track " << track_id
    // << std::endl;
    return num_tris;
  }
  // Extract inliers and outliers
  std::unordered_set<size_t> added_image_ids;
  bool exists_duplicated_images = false;
  for (size_t i = 0; i < report.inlier_mask.size(); ++i) {
    if (report.inlier_mask[i]) {
      inliers.push_back(elements[i]);
      if (!added_image_ids.insert(elements[i].image_id).second) {
        exists_duplicated_images = true;
      }
    } else {
      outliers.push_back(elements[i]);
    }
  }

  // If number of inliers is less than 3, remove the track. Otherwise, set
  // outliers as new elements
  double inlier_ratio = static_cast<double>(inliers.size()) /
                        static_cast<double>(elements.size());
  if (exists_duplicated_images || inlier_ratio < config_.min_tri_inlier_ratio ||
      inliers.size() < config_.min_track_length) {
    reconstruction_->removeTrack(track_id);
    // std::cout << "[INFO] track " << track_id << " deleted due to too few
    // inliers. " << std::endl;

    return num_tris;
  } else {
    // const size_t prev_len = track->length();
    track->setElements(inliers);
    track->setTriangulated(true);
    track->set3DPosition(report.model);
    num_tris += inliers.size();
    // std::cout << "[INFO] Filtered " << prev_len - inliers.size() << "
    // outliers in track " << track_id << std::endl;
  }
  reconstruction_->addNewTrackID(track_id);

  if (config_.use_recursive_triangulation) {
    /* Recursively triangulate observations and create new tracks in outliers */
    elements = outliers;
    while (elements.size() >= 3UL) {
      report = triangulateObservations(tri_config, elements);
      if (!report.success) {
        break;
      } else {
        // Extract inliers and outliers
        inliers.clear();
        outliers.clear();
        added_image_ids.clear();
        exists_duplicated_images = false;
        for (size_t i = 0; i < report.inlier_mask.size(); ++i) {
          if (report.inlier_mask[i]) {
            inliers.push_back(elements[i]);
            if (!added_image_ids.insert(elements[i].image_id).second) {
              exists_duplicated_images = true;
            }
          } else {
            outliers.push_back(elements[i]);
          }
        }

        inlier_ratio = static_cast<double>(inliers.size()) /
                       static_cast<double>(elements.size());
        // Create new track from inliers
        if (!exists_duplicated_images &&
            inlier_ratio >= config_.min_tri_inlier_ratio &&
            inliers.size() >= config_.min_track_length) {
          const size_t added_id =
              reconstruction_->addTrack(inliers, report.model, true);
          reconstruction_->addNewTrackID(added_id);
          // std::cout << "[INFO] Added a new track (id = " << added_id << ")"
          //           << " from " << elements.size() << " outliers of track "
          //           << track_id << std::endl;
          num_tris += inliers.size();
        } else {
          break;
        }

        // Use outliers as data of next triangulation
        elements = outliers;
      }
    }
  }

  return num_tris;
}

estimator::RobustTriangulationEstimator::Report
IncrementalSFM::triangulateObservations(
    const typename estimator::RANSACConfig &tri_config,
    const base::TrackElementList &elements) {
  // Prepare data for triangulation
  base::Scene *const scene = reconstruction_->getScene();
  typename estimator::TriangulationEstimator::VecXData point_data;
  typename estimator::TriangulationEstimator::VecYData pose_data;
  std::vector<const base::CameraBase *> cameras;
  for (size_t elem_idx = 0; elem_idx < elements.size(); ++elem_idx) {
    const auto &element = elements[elem_idx];
    if (!existsImage(element.image_id)) {
      std::cout << "[WARNING] [Observation Triangulation] image "
                << element.image_id << " does not exist " << std::endl;
      continue;
    }
    if (!isImageRegistered(element.image_id)) {
      std::cout << "[WARNING] [Observation Triangulation] image "
                << element.image_id << " has not been registered " << std::endl;
      continue;
    }
    if (isImageFiltered(element.image_id)) {
      continue;
    }
    const Eigen::Vector2d point_2D =
        scene->getNodePoint2DPosition(element.image_id, element.feature_idx);
    const Eigen::Matrix3d rotation = base::quaternion2RotationMatrix(
        scene->getNodeRotation(element.image_id));
    const Eigen::Vector3d translation =
        scene->getNodeTranslation(element.image_id);
    const Eigen::Matrix3x4d proj = base::composePose(
        rotation.inverse(), -rotation.inverse() * translation);
    const base::CameraBase *camera = new base::RadialPinHoleCamera(
        scene->getNodeCameraMatrix(element.image_id));
    point_data.emplace_back(point_2D, camera->pix2Norm(point_2D));
    pose_data.emplace_back(proj, translation, camera);
    cameras.push_back(camera);
  }

  estimator::RobustTriangulationEstimator::Report report;
  if (point_data.size() < 2) {
    std::cout << "[ERROR] [Observation Triangulation] View number is less than "
                 "2. Cannot triangulate. "
              << std::endl;
    report.success = false;
    return report;
  }

  estimator::RobustTriangulationEstimator estimator(tri_config);
  estimator.getEstimator().setMinTriAngleDeg(config_.min_tri_angle_deg);
  estimator.getEstimator().setEstimationErrorType(
      estimator::TriangulationEstimator::EstimationErrorType::ANGULAR_ERROR);
  estimator.getLocalEstimator().setMinTriAngleDeg(config_.min_tri_angle_deg);
  estimator.getLocalEstimator().setEstimationErrorType(
      estimator::TriangulationEstimator::EstimationErrorType::ANGULAR_ERROR);
  report = estimator.estimate(point_data, pose_data);

  for (size_t i = 0; i < cameras.size(); ++i) {
    delete cameras[i];
    cameras[i] = nullptr;
  }

  return report;
}

size_t IncrementalSFM::completeTrack(const size_t &track_id) {
  size_t num_completed = 0;
  if (!existsTrack(track_id)) {
    return num_completed;
  }

  if (!(reconstruction_->getTrack(track_id)->isTriangulated())) {
    return num_completed;
  }

  base::Scene *const scene = reconstruction_->getScene();
  base::TrackElementList queue =
      reconstruction_->getTrack(track_id)->getElements();
  std::unordered_set<size_t> added_image_ids;
  for (const auto &element : queue) {
    if (!added_image_ids.insert(element.image_id).second) {
      // This indicates that the track has conflict elements
      reconstruction_->removeTrack(track_id);
      return 0;
    }
  }

  for (size_t transitivity = 0; transitivity < config_.max_transitivity;
       ++transitivity) {
    if (queue.empty()) {
      break;
    }

    const base::TrackElementList prev_queue = queue;
    queue.clear();
    for (const auto &element : prev_queue) {
      if (!existsImage(element.image_id)) {
        reconstruction_->removeObservation(track_id, element.image_id,
                                           element.feature_idx);
        continue;
      }

      const std::vector<std::pair<size_t, size_t>> corrs =
          scene->findDirectCorrespondences(element.image_id,
                                           element.feature_idx);
      for (const auto &corr : corrs) {
        const size_t corr_image_id = corr.first;
        const size_t corr_feature_idx = corr.second;
        // Avoid conflict observations
        if (reconstruction_->getTrack(track_id)->hasImage(corr_image_id)) {
          continue;
        }
        // Skip nonexistent images
        if (!existsImage(corr_image_id)) {
          continue;
        }
        // Skip unregisterd images
        if (!isImageRegistered(corr_image_id)) {
          continue;
        }
        // Skip filterd images
        if (isImageFiltered(corr_image_id)) {
          continue;
        }
        // Skip images that have bogus camera parameters
        if (scene->hasNodeBogusCameraParameters(
                corr_image_id, config_.min_focal_length_ratio,
                config_.max_focal_length_ratio)) {
          continue;
        }
        // Skip features that already in tracks
        if (scene->hasTrack(corr_image_id, corr_feature_idx, nullptr)) {
          continue;
        }

        const base::Scene::Node *const corr_node =
            scene->getNode(corr_image_id);
        if (corr_node == nullptr) {
          continue;
        }
        // Check reprojection error
        const double reproj_error = corr_node->computeReprojectionError(
            reconstruction_->getTrack(track_id)->get3DPosition(),
            corr_feature_idx);
        if (reproj_error <= config_.max_track_reproj_error) {
          if (added_image_ids.insert(corr_image_id).second &&
              !reconstruction_->hasTrack(corr_image_id, corr_feature_idx,
                                         nullptr)) {
            reconstruction_->addObservation(track_id, corr_image_id,
                                            corr_feature_idx);
            if (transitivity < config_.max_transitivity - 1) {
              queue.emplace_back(corr_image_id, corr_feature_idx);
            }
            num_completed++;
          }
        }
      }
    }
  }

  return num_completed;
}

size_t IncrementalSFM::completeTracks(const std::vector<size_t> &track_ids) {
  size_t num_completed = 0;
  for (size_t i = 0; i < track_ids.size(); ++i) {
    num_completed += completeTrack(track_ids[i]);
    std::cout << "\r[INFO] Completing tracks... " << i + 1 << "/"
              << track_ids.size();
  }
  std::cout << std::endl;

  return num_completed;
}

size_t IncrementalSFM::completeImage(const size_t &image_id) {
  size_t num_completed = 0;

  if (!existsImage(image_id)) {
    return num_completed;
  }

  if (!isImageRegistered(image_id) || isImageFiltered(image_id)) {
    return num_completed;
  }

  base::Scene *const scene = reconstruction_->getScene();
  base::Scene::Node *const node = scene->getNode(image_id);
  if (node->hasBogusCameraParams(config_.min_focal_length_ratio,
                                 config_.max_focal_length_ratio)) {
    return num_completed;
  }
  /* Try to create a new track from each feature point */
  for (size_t point_2D_idx = 0; point_2D_idx < node->points_2D.size();
       ++point_2D_idx) {
    // Skip features that is already triangulated
    size_t existing_track_id;
    if (node->hasTrack(point_2D_idx, &existing_track_id)) {
      num_completed += completeTrack(existing_track_id);
      continue;
    }

    if (scene->isTwoViewObservation(image_id, point_2D_idx)) {
      continue;
    }

    // Try to create new tracks from observations that have no tracks
    const std::vector<std::pair<size_t, size_t>> correspondences =
        // scene_->findDirectCorrespondences(image_id, point_2D_idx);
        scene->findTransitiveCorrespondences(image_id, point_2D_idx,
                                             config_.max_transitivity);
    // std::cout << "[INFO] Found " << correspondences.size() << "
    // correspondences for feature (index = "
    //           << point_2D_idx << ") in image (id = " << image_id << ")" <<
    //           std::endl;
    base::TrackElementList track_elems;
    std::unordered_set<size_t> added_image_ids;
    track_elems.emplace_back(image_id, point_2D_idx);
    added_image_ids.insert(image_id);
    for (const auto &corr : correspondences) {
      // Skip unregistered or filtered images
      if (!isImageRegistered(corr.first)) {
        continue;
      }
      if (isImageFiltered(corr.first)) {
        continue;
      }
      // Skip observations that already in tracks
      if (scene->hasTrack(corr.first, corr.second, nullptr)) {
        continue;
      }
      // Skip myself
      if (image_id == corr.first) {
        continue;
      }

      // if (added_image_ids.insert(corr.first).second) {
      //     track_elems.emplace_back(corr.first, corr.second);
      // }
      track_elems.emplace_back(corr.first, corr.second);
    }

    if (track_elems.size() < std::max(std::max(config_.min_tri_num_obs, 2UL),
                                      config_.min_track_length)) {
      continue;
    } else {
      estimator::RANSACConfig tri_config;
      tri_config.max_inlier_error =
          util::deg2Rad(config_.max_tri_inlier_angular_error_deg);
      const size_t kMaxNumExhaustion = 15;
      if (track_elems.size() <= kMaxNumExhaustion) {
        tri_config.min_iter_num =
            util::computeCombinationNumber(track_elems.size(), 2);
        tri_config.max_iter_num = 2 * tri_config.min_iter_num;
      } else {
        tri_config.min_iter_num = 10;
        tri_config.max_iter_num = 10000;
      }
      tri_config.confidence = 0.99;
      tri_config.verbose = false;
      const auto report = triangulateObservations(tri_config, track_elems);
      if (!report.success) {
        continue;
      }
      base::TrackElementList inliers;
      std::unordered_set<size_t> added_elem_image_ids;
      bool exists_duplicated_images = false;
      for (size_t i = 0; i < report.inlier_mask.size(); ++i) {
        if (report.inlier_mask[i]) {
          inliers.push_back(track_elems[i]);
          if (!added_elem_image_ids.insert(track_elems[i].image_id).second) {
            exists_duplicated_images = true;
          }
        }
      }
      if (exists_duplicated_images ||
          inliers.size() < config_.min_track_length) {
        continue;
      }

      reconstruction_->addTrack(inliers, report.model, true);
      num_completed += inliers.size();
    }
  }

  return num_completed;
}

size_t IncrementalSFM::filterTracks(const std::vector<size_t> &track_ids,
                                    const bool &init) {
  size_t num_filtered = 0;
  num_filtered += filterTracksWithLargeReprojectionError(
      config_.max_track_reproj_error, track_ids, init);
  std::cout << "[INFO] Filtered " << num_filtered
            << " observations with large reprojection error. " << std::endl;
  size_t num_filtered_small_tri_ang = filterTracksWithSmallTriangulationAngle(
      util::deg2Rad(config_.min_tri_angle_deg), track_ids, init);
  std::cout << "[INFO] Filtered " << num_filtered_small_tri_ang
            << " observations with small triangulation angles. " << std::endl;
  num_filtered += num_filtered_small_tri_ang;
  return num_filtered;
}

size_t IncrementalSFM::filterTracksWithLargeReprojectionError(
    const double &max_reproj_error, const std::vector<size_t> &track_ids,
    const bool &init) {
  size_t num_filtered = 0;
  // size_t num_checked_tracks = 0;
  base::Scene *const scene = reconstruction_->getScene();
  for (const size_t &track_id : track_ids) {
    // std::cout << "\r[INFO] Filtering tracks... " << num_checked_tracks++ <<
    // "/" << track_ids.size();
    if (!existsTrack(track_id)) {
      continue;
    }
    base::Track *const track = reconstruction_->getTrack(track_id);
    if (track == nullptr) {
      continue;
    }

    if (!(track->isTriangulated())) {
      continue;
    }

    if (!init && track->length() < config_.min_track_length) {
      // std::cout << "[WARNING] Removed track " << track_id << " due to
      // insufficient length. " << std::endl;
      num_filtered += track->length();
      reconstruction_->removeTrack(track_id);
      continue;
    }

    const Eigen::Vector3d &position = track->get3DPosition();
    // Filter tracks with too large reprojection error
    double rmse = 0.0;
    for (const auto &element : track->getElements()) {
      // Filtered observations in filtered images
      if (isImageFiltered(element.image_id) ||
          !isImageRegistered(element.image_id)) {
        reconstruction_->removeObservation(track_id, element.image_id,
                                           element.feature_idx);
        num_filtered++;
        continue;
      }
      if (!scene->hasTrack(element.image_id, element.feature_idx, nullptr)) {
        continue;
      }
      const double reproj_error = scene->computeReprojectionError(
          element.image_id, element.feature_idx, position);
      if (reproj_error < 0 || reproj_error > max_reproj_error) {
        // std::cout << "(" << reproj_error << ", " <<
        //              scene->getNode(element.image_id)->camera_info.alpha <<
        //              ", " << scene->getNode(element.image_id)->camera_info.k1
        //              << ", " <<
        //              scene->getNode(element.image_id)->camera_info.k2 << ")
        //              ";
        reconstruction_->removeObservation(track_id, element.image_id,
                                           element.feature_idx);
        rmse += reproj_error * reproj_error;
        num_filtered++;
      }
    }
    if (!init && track->length() < config_.min_track_length) {
      // std::cout << "[WARNING] Removed track " << track_id << " due to large
      // reprojection errors. " << std::endl;
      num_filtered += track->length();
      reconstruction_->removeTrack(track_id);
    } else if (init && track->length() < 2) {
      // std::cout << "[WARNING] Removed track " << track_id << " due to large
      // reprojection errors. " << std::endl;
      num_filtered += track->length();
      reconstruction_->removeTrack(track_id);
    }
  }
  std::cout << std::endl;

  return num_filtered;
}

size_t IncrementalSFM::filterTracksWithSmallTriangulationAngle(
    const double &min_tri_angle_rad, const std::vector<size_t> &track_ids,
    const bool &init) {
  size_t num_filtered = 0;
  // size_t num_checked_tracks = 0;
  base::Scene *const scene = reconstruction_->getScene();
  for (const size_t &track_id : track_ids) {
    // std::cout << "\r[INFO] Filtering tracks... " << num_checked_tracks++ <<
    // "/" << track_ids.size();
    if (!existsTrack(track_id)) {
      continue;
    }
    base::Track *const track = reconstruction_->getTrack(track_id);
    if (!(track->isTriangulated())) {
      continue;
    }

    const Eigen::Vector3d &position = track->get3DPosition();
    // Filter tracks with no pair of observations that has sufficient
    // triangulation angle
    bool should_keep = false;
    const base::TrackElementList &elements = track->getElements();
    // Indices of elements with nonexistent or filtered image_id
    std::unordered_set<size_t> nonex_elem_idxs;
    for (int64_t elem_idx1 = 0;
         elem_idx1 < static_cast<int64_t>(elements.size()) - 1; ++elem_idx1) {
      const base::TrackElement &elem1 = elements[elem_idx1];
      if (!existsImage(elem1.image_id) || isImageFiltered(elem1.image_id)) {
        nonex_elem_idxs.insert(elem_idx1);
        continue;
      }
      const Eigen::Vector3d t1 = scene->getNodeTranslation(elem1.image_id);
      for (int64_t elem_idx2 = elem_idx1 + 1;
           elem_idx2 < static_cast<int64_t>(elements.size()); ++elem_idx2) {
        const base::TrackElement &elem2 = elements[elem_idx2];
        if (!existsImage(elem2.image_id) || isImageFiltered(elem2.image_id)) {
          nonex_elem_idxs.insert(elem_idx2);
          continue;
        }
        const Eigen::Vector3d t2 = scene->getNodeTranslation(elem2.image_id);

        const double tri_angle_rad =
            base::computeTriangulationAngle(t1, t2, position);
        if (std::abs(tri_angle_rad) >= std::abs(min_tri_angle_rad)) {
          should_keep = true;
          break;
        }
      }
      if (should_keep) {
        break;
      }
    }
    if (!init &&
        track->length() < nonex_elem_idxs.size() + config_.min_track_length) {
      should_keep = false;
    }

    if (!should_keep) {
      // std::cout << "[WARNING] Removed track " << track_id << " due to small
      // triangulation angles. " << std::endl;
      num_filtered += track->length();
      reconstruction_->removeTrack(track_id);
    } else {
      // Remove invalid elements
      for (const size_t &nonex_elem_idx : nonex_elem_idxs) {
        const base::TrackElement nonex_elem = elements[nonex_elem_idx];
        reconstruction_->removeObservation(track_id, nonex_elem.image_id,
                                           nonex_elem.feature_idx);
        num_filtered++;
      }
    }
  }
  std::cout << std::endl;

  return num_filtered;
}

size_t IncrementalSFM::filterObservationsWithNegativeDepth(const bool &init) {
  size_t num_filtered = 0;
  base::Scene *const scene = reconstruction_->getScene();
  for (const auto &id_track : reconstruction_->getTracks()) {
    const size_t track_id = id_track.first;
    base::Track *track = id_track.second;
    if (track == nullptr) {
      continue;
    }
    if (!(track->isTriangulated())) {
      continue;
    }
    const base::TrackElementList &elements = track->getElements();
    const Eigen::Vector3d point_3D = track->get3DPosition();
    for (const auto &element : elements) {
      const size_t &image_id = element.image_id;
      const size_t &point_2D_idx = element.feature_idx;
      if (!existsImage(image_id)) {
        continue;
      }
      if (!isImageRegistered(image_id) || isImageFiltered(image_id)) {
        continue;
      }
      const base::Scene::Node *const node = scene->getNode(image_id);
      if (point_2D_idx >= node->points_2D.size()) {
        continue;
      }
      const Eigen::Matrix3x4d proj = node->getProjectionMatrix();
      if (!base::hasPointPositiveDepth(proj, point_3D)) {
        reconstruction_->removeObservation(track_id, image_id, point_2D_idx);
        ++num_filtered;
      }
    }

    const size_t length = track->length();
    if (!init && length < config_.min_track_length) {
      num_filtered += length;
      reconstruction_->removeTrack(track_id);
    }
  }

  return num_filtered;
}

size_t IncrementalSFM::filterAllTracks(const bool &init) {
  std::vector<size_t> track_ids;
  for (const auto &track : reconstruction_->getTracks()) {
    track_ids.push_back(track.first);
  }

  return filterTracks(track_ids, init);
}

} // namespace sfm
} // namespace my3d
