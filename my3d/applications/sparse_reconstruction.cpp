/*
 * filename: sparse_reconstruction.cpp
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

bool savePoint3D(const std::string &directory, 
                 const std::vector<base::Image> &images, 
                 const base::Scene &scene, 
                 const EigenUMap<size_t, base::Track> &tracks) {
    boost::filesystem::path path(directory); 
    path.append("MY3D_Points_3D.txt"); 
    std::ofstream ofs(path.string()); 
    std::cout << path.string() << std::endl;
    if (!ofs.is_open()) {
        return false; 
    } 
    for (const auto &track : tracks) {
        const Eigen::Vector3d pos = track.second.get3DPosition(); 
        Eigen::Vector3i rgb(0, 0, 0); 
        int num_elems = 0; 
        for (const auto &elem : track.second.getElements()) {
            ++num_elems; 
            // const base::Image &image = images[elem.image_id]; 
            // const Eigen::Vector2d point_2D = scene.getNodePoint2DPosition(elem.image_id, elem.feature_idx); 
            const Eigen::Matrix<uint8_t, 3, 1> color = scene.getNodePoint2DColor(elem.image_id, elem.feature_idx); 
            
            rgb.x() += static_cast<int>(color.x()); 
            rgb.y() += static_cast<int>(color.y()); 
            rgb.z() += static_cast<int>(color.z()); 
        }
        rgb.x() = util::clamp<int>(rgb.x() / num_elems, 0, 255); 
        rgb.y() = util::clamp<int>(rgb.y() / num_elems, 0, 255); 
        rgb.z() = util::clamp<int>(rgb.z() / num_elems, 0, 255); 
        ofs << pos.x() << " " << pos.y() << " " << pos.z() << " " 
            << rgb.x() << " " << rgb.y() << " " << rgb.z() << std::endl;
    }
    ofs.close();

    return true;
}

bool saveCameras(const std::string &directory, 
                      const EigenUMap<size_t, base::Scene::CameraInfo> &camera_infos) {
    std::ofstream ofs; 
    // Save camera poses
    const boost::filesystem::path path_poses = 
        boost::filesystem::path(directory) / "MY3D_Camera_Poses.txt";
    ofs.open(path_poses.string());
    if (!ofs.is_open()) {
        return false; 
    }
    for (const auto &camera_info : camera_infos) {
        const Eigen::Vector3d pos = camera_info.second.translation; 
        const Eigen::Quaterniond rot = camera_info.second.rotation; 
        ofs << camera_info.first << " " 
            << pos.x() << " " << pos.y() << " " << pos.z() << " " 
            << rot.w() << " " << rot.x() << " " << rot.y() << " " << rot.z()
            << std::endl;
    }
    ofs.close(); 
    // Save camera intrinsics 
    const boost::filesystem::path path_intrinsics = 
        boost::filesystem::path(directory) / "MY3D_Camera_Intrinsics.txt"; 
    ofs.open(path_intrinsics.string()); 
    if (!ofs.is_open()) {
        return false; 
    }
    for (const auto &camera_info : camera_infos) {
        const auto &camera = camera_info.second; 
        ofs << camera_info.first << " " 
            << camera.fx << " " << camera.fy << " " << camera.cx << " " << camera.cy << " " 
            << camera.alpha << " " << camera.k1 << " " << camera.k2 
            << std::endl; 
            
    }
    ofs.close();

    return true; 
} 

bool saveFeaturesAsJSON(const std::string &directory, 
                        const std::vector<base::Image> &images, 
                        const feature::FeatureImageList<feature::SIFTKeyPointList, feature::SIFTDescriptorList> &features) {
    boost::filesystem::path path(directory); 
    path.append("MY3D.feat"); 

    const size_t num_imgs = images.size(); 
    if (num_imgs != features.size()) { 
        std::cout << "[ERROR] Image number and feature image number does not match. " 
                  << "Failed to save features. " << std::endl; 
        return false; 
    }

    sjson::Json file; 
    sjson::JsonNode &root = file.getRoot(); 
    try {
        if (root["feature_images"].set_array(num_imgs)) {
            for (size_t i = 0; i < num_imgs; ++i) {
                root["feature_images"].array_get(i)["image_path"] = images[i].getName(); 
                const auto camera = std::dynamic_pointer_cast<base::RadialPinHoleCamera>(images[i].getCameraModel()); 
                root["feature_images"].array_get(i)["focal_length_x"] = camera->fx_; 
                root["feature_images"].array_get(i)["focal_length_y"] = camera->fy_; 
                root["feature_images"].array_get(i)["center_x"] = camera->cx_; 
                root["feature_images"].array_get(i)["center_y"] = camera->cy_; 
                root["feature_images"].array_get(i)["alpha"] = camera->alpha_; 
                root["feature_images"].array_get(i)["k1"] = camera->k1_; 
                root["feature_images"].array_get(i)["k2"] = camera->k2_; 
                const size_t num_feats = features[i].key_points.size(); 
                root["feature_images"].array_get(i)["features"].set_array(num_feats); 
                for (size_t j = 0; j < num_feats; ++j) {
                    auto &feats_json = root["feature_images"].array_get(i)["features"][j]; 
                    feats_json.set_array(133); 
                    feats_json.array_get(0) = features[i].points_2D[j].x(); 
                    feats_json.array_get(1) = features[i].points_2D[j].y(); 
                    feats_json.array_get(2) = features[i].colors[j].x(); 
                    feats_json.array_get(3) = features[i].colors[j].y(); 
                    feats_json.array_get(4) = features[i].colors[j].z(); 
                    for (size_t col = 0; col < 16; ++col) {
                        for (size_t row = 0; row < 8; ++row) {
                            const size_t idx = col * 8 + row; 
                            feats_json.array_get(5 + idx) = features[i].descriptors[j].histograms(row, col); 
                        }
                    }
                }
            }

            if (!file.save(path.string(), false)) {
                return false; 
            }
        } else {
            return false; 
        }

    } catch (const std::exception& e) {
        std::cout << e.what() << '\n';
        return false;
    }

    return true; 
} 


bool loadFeaturesFromJSON(const std::string &directory, 
                          const std::vector<base::Image> &images, 
                          feature::FeatureImageList<feature::SIFTKeyPointList, feature::SIFTDescriptorList> &features) {
    boost::filesystem::path path(directory); 
    path.append("MY3D.feat"); 
    if (!boost::filesystem::exists(path)) {
        std::cout << "[WARNING] Feature file " << path.string() << " does not exists. " << std::endl;
        return false; 
    }
    
    std::ifstream ifs(path.string()); 
    sjson::Json file(ifs); 
    ifs.close(); 
    if (file.fail()) {
        std::cout << "[ERROR] Failed to parse file " << path.string() << ". " << std::endl;
        return false; 
    }

    const size_t num_imgs = images.size(); 
    const sjson::JsonNode &root = file.getRoot(); 
    try
    {
        if (root.obj_has_item("feature_images")) {
            const auto &feats_json = root["feature_images"]; 
            const auto feats_json_vec = feats_json.as_vector(); 
            if (num_imgs != feats_json_vec.size()) {
                std::cout << "[ERROR] Numbers of images does not match. " << std::endl; 
                return false; 
            }
            features.clear(); 
            features.resize(num_imgs);
            std::unordered_set<size_t> processed_image_indices; 
            for (size_t i = 0; i < num_imgs; ++i) {
                const auto &feat_json = feats_json_vec[i]; 
                if (!feat_json->obj_has_item("image_path") || 
                    !feat_json->obj_has_item("features") || 
                    !feat_json->obj_has_item("focal_length_x") || 
                    !feat_json->obj_has_item("focal_length_x") || 
                    !feat_json->obj_has_item("center_x") || 
                    !feat_json->obj_has_item("center_y") || 
                    !feat_json->obj_has_item("alpha") || 
                    !feat_json->obj_has_item("k1") || 
                    !feat_json->obj_has_item("k2")) {
                    std::cout << "[ERROR] Invalid feature file. " << std::endl; 
                    return false; 
                }

                const std::string img_path = feat_json->obj_get_item("image_path").as_string(); 
                bool found = false; 
                size_t img_idx = 0; 
                for (; img_idx < num_imgs; ++img_idx) {
                    if (img_path == images[img_idx].getName()) {
                        found = true; 
                        break;
                    }
                }
                if (!found) {
                    std::cout << "[ERROR] Image information mismatches. " << std::endl; 
                    return false; 
                }
                if (!processed_image_indices.insert(img_idx).second) {
                    std::cout << "[ERROR] Image information conflicts. " << std::endl; 
                    return false; 
                }

                const base::Image &src = images[img_idx]; 
                auto camera = std::dynamic_pointer_cast<base::RadialPinHoleCamera>(src.getCameraModel()); 
                camera->fx_ = feat_json->obj_get_item("focal_length_x").as_double(); 
                camera->fy_ = feat_json->obj_get_item("focal_length_y").as_double(); 
                camera->cx_ = feat_json->obj_get_item("center_x").as_double(); 
                camera->cy_ = feat_json->obj_get_item("center_y").as_double(); 
                camera->alpha_ = feat_json->obj_get_item("alpha").as_double(); 
                camera->k1_ = feat_json->obj_get_item("k1").as_double(); 
                camera->k2_ = feat_json->obj_get_item("k2").as_double(); 

                feature::FeatureImage<feature::SIFTKeyPointList, feature::SIFTDescriptorList> feat_img; 
                const auto &kpts_json_vec = feat_json->obj_get_item("features").as_vector(); 
                size_t kpt_idx = 0; 
                for (const auto kpt_json : kpts_json_vec) {
                    const auto &kpt_json_vec = kpt_json->as_vector(); 
                    feat_img.image_idx = img_idx; 
                    feat_img.points_2D.emplace_back(kpt_json_vec[0]->as_double(), 
                                                    kpt_json_vec[1]->as_double()); 
                    feat_img.colors.emplace_back(static_cast<uint8_t>(kpt_json_vec[2]->as_int()), 
                                                static_cast<uint8_t>(kpt_json_vec[3]->as_int()), 
                                                static_cast<uint8_t>(kpt_json_vec[4]->as_int())); 
                    Eigen::Matrix<double, 8, 16> hist; 
                    for (size_t col = 0; col < 16; ++col) {
                        for (size_t row = 0; row < 8; ++row) {
                            const size_t idx = col * 8 + row; 
                            hist(row, col) = kpt_json_vec[5 + idx]->as_double(); 
                        }
                    }
                    feat_img.descriptors.emplace_back(kpt_idx, feat_img.points_2D.back(), 
                                                    -1, -1, hist); 
                    feat_img.key_points.emplace_back(feat_img.points_2D.back(), 
                                                    -1, -1, 0, 0, -1); 

                    ++kpt_idx; 
                }
                features[img_idx] = feat_img; 
            }

        } else {
            std::cout << "[ERROR] Invalid feature file. " << std::endl; 
            return false; 
        }
    }
    catch(const std::exception& e)
    {
        std::cout << e.what() << '\n';
        return false; 
    }

    return true; 
}

bool saveFeaturesAsBinary(const std::string &directory, 
                          const std::vector<base::Image> &images, 
                          const feature::FeatureImageList<feature::SIFTKeyPointList, feature::SIFTDescriptorList> &features) {
    boost::filesystem::path path(directory); 
    path.append("MY3D_Features.bin"); 

    const size_t num_imgs = images.size(); 
    if (num_imgs != features.size()) { 
        std::cout << "[ERROR] Image number and feature image number does not match. " 
                  << "Failed to save features. " << std::endl; 
        return false; 
    }

    std::ofstream ofs(path.string(), 
                      std::ios::binary | std::ios::out); 
    if (!ofs.is_open()) {
        return false; 
    }

    ofs.write((const char *)(&num_imgs), sizeof(size_t)); 
    for (size_t img_idx = 0; img_idx < num_imgs; ++img_idx) {
        const base::Image &image = images[img_idx]; 
        const std::string image_name = image.getName(); 
        const size_t name_size = image_name.size(); 
        const auto camera = 
            std::dynamic_pointer_cast<base::RadialPinHoleCamera>(images[img_idx].getCameraModel()); 
        const size_t num_kpts = features[img_idx].key_points.size(); 
        const auto &feat_img = features[img_idx]; 
        ofs.write((const char *)(&name_size), sizeof(size_t)); 
        ofs.write((const char *)(image_name.data()), name_size + 1);  
        // ofs.write((const char *)(&camera->fx_), sizeof(double)); 
        // ofs.write((const char *)(&camera->fy_), sizeof(double)); 
        // ofs.write((const char *)(&camera->cx_), sizeof(double)); 
        // ofs.write((const char *)(&camera->cy_), sizeof(double));
        // ofs.write((const char *)(&camera->alpha_), sizeof(double));  
        // ofs.write((const char *)(&camera->k1_), sizeof(double)); 
        // ofs.write((const char *)(&camera->k2_), sizeof(double)); 
        ofs.write((const char *)(&num_kpts), sizeof(size_t)); 
        for (size_t kpt_idx = 0; kpt_idx < num_kpts; ++kpt_idx) {
            ofs.write((const char *)(feat_img.points_2D[kpt_idx].data()), 2 * sizeof(double)); 
            ofs.write((const char *)(feat_img.colors[kpt_idx].data()), 3); 
            ofs.write((const char *)(feat_img.descriptors[kpt_idx].histograms.data()), 128 * sizeof(double)); 
        }
    }

    ofs.close(); 

    return true;  
} 

bool loadFeaturesFromBinary(const std::string &directory, 
                            const std::vector<base::Image> &images, 
                            feature::FeatureImageList<feature::SIFTKeyPointList, feature::SIFTDescriptorList> &features) {
    boost::filesystem::path path(directory); 
    path.append("MY3D_Features.bin"); 
    if (!boost::filesystem::exists(path)) {
        std::cout << "[WARNING] Feature file " << path.string() << " does not exists. " << std::endl;
        return false; 
    }
    
    std::ifstream ifs(path.string(), 
                      std::ios::binary | std::ios::in); 
    if (!ifs.is_open()) {
        return false; 
    }

    features.clear(); 
    size_t num_imgs; 
    ifs.read((char *)(&num_imgs), sizeof(size_t)); 
    if (!ifs.good()) {
        ifs.close(); 
        return false; 
    }

    if (num_imgs != images.size()) {
        std::cout << "[ERROR] Numbers of images does not match. " << std::endl; 
        ifs.close(); 
        return false; 
    }

    std::unordered_set<size_t> processed_image_indices; 
    features.resize(num_imgs); 
    for (size_t img_idx = 0; img_idx < num_imgs; ++img_idx) {
        size_t name_size; 
        ifs.read((char *)(&name_size), sizeof(size_t)); 
        if (!ifs.good()) {
            features.clear(); 
            ifs.close();
            return false; 
        }

        // std::cout << name_size << std::endl;
        char *name_c_str = new char[name_size + 1]; 
        ifs.read(name_c_str, name_size + 1); 
        if (!ifs.good()) {
            features.clear(); 
            ifs.close();
            return false; 
        }

        std::string image_name(name_c_str); 
        delete [] name_c_str; 
        name_c_str = nullptr; 
        bool found = false; 
        size_t i = 0;
        for (; i < images.size(); ++i) {
            if (images[i].getName() == image_name) {
                found = true; 
                break; 
            }
        }

        if (!found) {
            std::cout << "[ERROR] Image information mismatches. " << std::endl; 
            features.clear(); 
            ifs.close();
            return false; 
        }
        if (!processed_image_indices.insert(i).second) {
            std::cout << "[ERROR] Image information conflicts. " << std::endl; 
            features.clear(); 
            ifs.close();
            return false; 
        }

        // const base::Image &src = images[i]; 
        auto &feat_img = features[img_idx]; 
        feat_img.image_idx = i; 
        feat_img.name = image_name; 
        // auto camera = 
        //     std::dynamic_pointer_cast<base::RadialPinHoleCamera>(src.getCameraModel()); 

        // ifs.read((char *)(&camera->fx_), sizeof(double)); 
        // ifs.read((char *)(&camera->fy_), sizeof(double)); 
        // ifs.read((char *)(&camera->cx_), sizeof(double)); 
        // ifs.read((char *)(&camera->cy_), sizeof(double)); 
        // ifs.read((char *)(&camera->alpha_), sizeof(double)); 
        // ifs.read((char *)(&camera->k1_), sizeof(double)); 
        // ifs.read((char *)(&camera->k2_), sizeof(double)); 
        // if (!ifs.good()) {
        //     features.clear(); 
        //     ifs.close();
        //     return false; 
        // }

        size_t num_kpts; 
        ifs.read((char *)(&num_kpts), sizeof(size_t)); 
        if (!ifs.good()) {
            features.clear(); 
            ifs.close();
            return false; 
        }

        feat_img.key_points.resize(num_kpts);
        feat_img.descriptors.resize(num_kpts); 
        feat_img.points_2D.resize(num_kpts); 
        feat_img.colors.resize(num_kpts); 
        feat_img.num_pnts = num_kpts; 
        for (size_t kpt_idx = 0; kpt_idx < num_kpts; ++kpt_idx) {
            ifs.read((char *)(feat_img.points_2D[kpt_idx].data()), 2 * sizeof(double)); 
            ifs.read((char *)(feat_img.colors[kpt_idx].data()), 3); 
            ifs.read((char *)(feat_img.descriptors[kpt_idx].histograms.data()), 128 * sizeof(double)); 
            feat_img.key_points[kpt_idx].point = feat_img.points_2D[kpt_idx]; 
            feat_img.descriptors[kpt_idx].point = feat_img.points_2D[kpt_idx]; 
            feat_img.descriptors[kpt_idx].key_point_idx = kpt_idx; 
        }
        if (!ifs.good()) {
            features.clear(); 
            ifs.close();
            return false; 
        }

        std::cout << "===================================" << std::endl; 
        std::cout << "Loaded features in image " << image_name << std::endl; 
        // std::cout << "Camera model: " << std::endl; 
        // std::cout << "\tfx: " << camera->fx_ << std::endl; 
        // std::cout << "\tfy: " << camera->fy_ << std::endl; 
        // std::cout << "\tcx: " << camera->cx_ << std::endl; 
        // std::cout << "\tcy: " << camera->cy_ << std::endl; 
        // std::cout << "\talpha: " << camera->alpha_ << std::endl; 
        // std::cout << "\tk1: " << camera->k1_ << std::endl; 
        // std::cout << "\tk2: " << camera->k2_ << std::endl; 
        std::cout << "Feature number: " << num_kpts << std::endl; 
        std::cout << "===================================" << std::endl; 
    }

    ifs.close(); 

    return true; 
}

bool saveMatchesAsBinary(const std::string &directory, 
                         const std::vector<feature::PairWiseMatchingInfo> &pair_wise_matching_infos) {
    boost::filesystem::path path(directory); 
    path.append("MY3D_Pair_Wise_Matching_Infos.bin"); 
    
    std::ofstream ofs(path.string(), 
                      std::ios::binary | std::ios::out); 
    if (!ofs.is_open()) {
        return false; 
    }

    const size_t num_pwms = pair_wise_matching_infos.size(); 
    ofs.write((const char *)(&num_pwms), sizeof(size_t)); 
    for (size_t i = 0; i < num_pwms; ++i) {
        if (!pair_wise_matching_infos[i].serialize(ofs)) {
            ofs.close(); 
            return false; 
        }
    }

    ofs.close(); 

    return ofs.good(); 
}

bool loadMatchesFromBinary(const std::string &directory, 
                           const std::vector<base::Image> &images, 
                           std::vector<feature::PairWiseMatchingInfo> &pair_wise_matching_infos) {
    boost::filesystem::path path(directory); 
    path.append("MY3D_Pair_Wise_Matching_Infos.bin"); 
    
    std::ifstream ifs(path.string(), 
                      std::ios::binary | std::ios::in); 
    if (!ifs.is_open()) {
        return false; 
    } 

    size_t num_pwms; 
    ifs.read((char *)(&num_pwms), sizeof(size_t)); 
    pair_wise_matching_infos.resize(num_pwms); 
    const size_t num_imgs = images.size(); 
    for (size_t i = 0; i < num_pwms; ++i) {
        if (!pair_wise_matching_infos[i].deserialize(ifs)) {
            ifs.close(); 
            return false; 
        } 
        if (pair_wise_matching_infos[i].query_image_idx >= num_imgs || 
            pair_wise_matching_infos[i].train_image_idx >= num_imgs) {
            std::cout << "[ERROR] Image indices out of range. " << std::endl; 
            ifs.close(); 
            return false; 
        } 
        if (images[pair_wise_matching_infos[i].query_image_idx].getName() != 
                pair_wise_matching_infos[i].query_image_name || 
            images[pair_wise_matching_infos[i].train_image_idx].getName() != 
                pair_wise_matching_infos[i].train_image_name) {
            std::cout << "[ERROR] Image information mismatches. " << std::endl; 
            std::cout << "\tquery image name from idx: " << images[pair_wise_matching_infos[i].query_image_idx].getName() << std::endl; 
            std::cout << "\tquery image name from binary: " << pair_wise_matching_infos[i].query_image_name << std::endl; 
            std::cout << "\ttrain image name from idx: " << images[pair_wise_matching_infos[i].train_image_idx].getName() << std::endl; 
            std::cout << "\ttrain image name from binary: " << pair_wise_matching_infos[i].train_image_name << std::endl; 
            ifs.close(); 
            return false; 
        } 

        std::cout << "===================================" << std::endl; 
        std::cout << "Loaded pair-wise matching information. " << std::endl; 
        std::cout << "query image: " << pair_wise_matching_infos[i].query_image_name 
                  << ", idx: " << pair_wise_matching_infos[i].query_image_idx 
                  << ", id: " << images[pair_wise_matching_infos[i].query_image_idx].getId() << std::endl; 
        std::cout << "train image: " << pair_wise_matching_infos[i].train_image_name 
                  << ", idx: " << pair_wise_matching_infos[i].train_image_idx 
                  << ", id: " << images[pair_wise_matching_infos[i].train_image_idx].getId() << std::endl; 
        std::cout << "match number: " << pair_wise_matching_infos[i].matches.size() << std::endl; 
        std::cout << "===================================" << std::endl; 
    }

    return ifs.good(); 
}

void threadReconstruction(base::Scene *scene) {
    
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cerr << "Error! Invalid argument. Usage: \n" << 
                     "\tsparse_reconstruction <image_directory>" << std::endl; 
        return -1;
    }
    const std::string workspace_directory = argv[1]; 

    /* Read images */
    std::cout << "\n-----------------------------Reading Images-----------------------------" << std::endl;
    std::vector<base::Image> images; 
    const boost::filesystem::path images_directory = 
        boost::filesystem::path(workspace_directory) / "images"; 
    const double fx_prior = -1; 
    const double fy_prior = -1; 
    // const double fx_prior = 3205.9; 
    // const double fy_prior = 3185.7; 
    const size_t num_read = 
        base::readImages<base::OpenCVImageReader>(images_directory.string(), 
                                                  fx_prior, 
                                                  fy_prior, 
                                                  images, 
                                                  base::SUPPORTED_IMAGE_EXTENSIONS, 
                                                  base::ImageReaderBase::IMREAD_COLOR); 
    std::cout << "\nRead " << num_read << " images. " << std::endl; 

    /* Extract features */
    std::cout << "\n-----------------------------Extracting Features-----------------------------" << std::endl;
    // feature::FeatureImageList<std::vector<cv::KeyPoint>, cv::Mat> features; 
    // feature::OpenCVORBExtractor::Config extractor_config; 
    // extractor_config.max_feature_num = 10000;
    // extractor_config.fast_threshold = 10;
    // feature::FeatureExtractorBase<std::vector<cv::KeyPoint>, cv::Mat> *extractor = 
    //     new feature::OpenCVORBExtractor(extractor_config); 
    // feature::OpenCVBatchFeatureExtractor batch_extractor(extractor); 
    // batch_extractor.extract(images, features); 

    feature::FeatureImageList<feature::SIFTKeyPointList, feature::SIFTDescriptorList> features;
    if (!loadFeaturesFromBinary(workspace_directory, images, features)) {
        std::cout << "Failed to load features from file. Redoing feature extraction. " << std::endl; 
        feature::SIFTExtractor::Config extractor_config; 
        extractor_config.num_octaves = 4; 
        extractor_config.octave_idx_base_image = -1; 
        extractor_config.min_abs_resp = 0.02 / 3; 
        extractor_config.max_width = 5000; 
        extractor_config.max_height = 5000; 
        feature::FeatureExtractorBase<feature::SIFTKeyPointList, feature::SIFTDescriptorList> *extractor = 
            new feature::SIFTExtractor(extractor_config); 
        // feature::BatchFeatureExtractorBase<feature::SIFTKeyPointList, feature::SIFTDescriptorList> batch_extractor(extractor); 
        const size_t max_num_threads = 3; 
        const size_t min_num_images_per_thread = std::ceil(static_cast<double>(images.size()) / max_num_threads); 
        feature::BatchFeatureExtractorMultiThreadBase<feature::SIFTKeyPointList, feature::SIFTDescriptorList> batch_extractor(extractor, min_num_images_per_thread); 
        batch_extractor.extract(images, features); 
        if (!saveFeaturesAsBinary(workspace_directory, images, features)) {
            std::cout << "[ERROR] Failed to save features. " << std::endl; 
        }
        std::cout << "Feature extraction done. " << std::endl; 
    }

    /* Match features */
    std::cout << "\n-----------------------------Matching Features-----------------------------" << std::endl;
    // std::vector<feature::PairWiseMatchingInfo> pair_wise_matcher_infos; 
    // feature::OpenCVMatcher::Config matcher_config; 
    // matcher_config.ratio_thresh = 0.6;
    // matcher_config.matcher_type = cv::DescriptorMatcher::MatcherType::BRUTEFORCE_HAMMING; 
    // feature::FeatureMatcherBase<cv::Mat> *matcher = 
    //     new feature::OpenCVMatcher(matcher_config);
    // feature::OpenCVBatchFeatureMatcher batch_matcher(matcher); 
    // batch_matcher.match(features, pair_wise_matcher_infos); 
    std::vector<feature::PairWiseMatchingInfo> pair_wise_matcher_infos;
    if (!loadMatchesFromBinary(workspace_directory, images, pair_wise_matcher_infos)) {
        std::cout << "Failed to load matches from file. Redoing feature matching. " << std::endl; 
        typedef feature::BrutalForceMatcher<feature::SIFTDescriptor, feature::SIFTDescriptorList> Matcher; 
        typedef feature::BatchFeatureMatcherBase<feature::SIFTKeyPointList, feature::SIFTDescriptorList> BatchMatcher;
        Matcher::Config matcher_config; 
        matcher_config.ratio_thresh = 0.8; 
        matcher_config.max_inlier_dist = 0.7;
        matcher_config.dist_func = feature::SIFTDescriptor::computeDistance; 
        matcher_config.match_type = Matcher::MatchType::TWO_WAY; 
        Matcher *matcher = new Matcher(matcher_config); 
        BatchMatcher batch_matcher(matcher); 
        batch_matcher.match(features, pair_wise_matcher_infos); 
        if (!saveMatchesAsBinary(workspace_directory, pair_wise_matcher_infos)) {
            std::cout << "[ERROR] Failed to save features. " << std::endl; 
        } 
        std::cout << "Feature matching done. " << std::endl; 
    }

    std::cout << "\n-----------------------------Building Scene-----------------------------" << std::endl;
    EigenVec<EigenVec<Eigen::Vector2d>> points_2D_list(features.size()); 
    for (size_t i = 0; i < features.size(); ++i) {
        EigenVec<Eigen::Vector2d> points_2D(features[i].key_points.size()); 
        for (size_t j = 0; j < features[i].key_points.size(); ++j) {
            points_2D[j] = Eigen::Vector2d(features[i].key_points[j].point.x(), 
                                          features[i].key_points[j].point.y()); 
            // std::cout << points_2D.back().transpose() << std::endl;
        }
        points_2D_list[features[i].image_idx] = points_2D; 
    }
    base::Scene scene; 
    sfm::SceneBuilder::Config scene_builder_config; 
    scene_builder_config.min_inlier_ratio = 0.65; 
    scene_builder_config.min_num_inliers = 100;
    scene_builder_config.min_inlier_ratio = 0.35; 
    scene_builder_config.min_epsilon_EF = 0.95;
    scene_builder_config.ransac_config.max_inlier_error = 4.0; 
    scene_builder_config.ransac_config.min_iter_num = 30; 
    scene_builder_config.ransac_config.min_inlier_ratio = 0.25; 
    scene_builder_config.ransac_config.confidence = 0.999; 
    // scene_builder_config.ransac_config.max_iter_num = 100000; 
    scene_builder_config.ransac_config.verbose = true; 
    sfm::SceneBuilder scene_builder(scene_builder_config); 
    if (scene_builder.build(images, points_2D_list, pair_wise_matcher_infos, scene)) {
        std::cout <<"\nScene building done. " << std::endl; 
    } else {
        std::cout << "\nScene building failed. " << std::endl; 
        return -1;
    }

    /* Incremental Structure-from-Motion */
    std::cout << "\n-----------------------------Reconstructing-----------------------------" << std::endl;
    sfm::IncrementalSFM::Config mapper_config; 
    mapper_config.gba_config.ba_config.max_num_iters = 10; 
    mapper_config.gba_config.ba_config.cam_opt_mode = base::BundleAdjustment::CameraOptimizationMode::FULL; 
    mapper_config.max_track_reproj_error = 3.0;
    mapper_config.max_merge_reproj_error = 3.0; 
    mapper_config.min_tri_angle_deg = 2.0; 
    mapper_config.min_tri_angle_deg_init = 13.0; 
    mapper_config.min_abs_pose_inlier_ratio = 0.25; 
    mapper_config.min_abs_pose_inlier_num = 30; 
    mapper_config.abs_pose_epsilon = 1e-5; 
    mapper_config.min_tri_num_obs = 2; 
    mapper_config.max_tri_inlier_angular_error_deg = 1.0; 
    mapper_config.use_recursive_triangulation = false; 
    mapper_config.min_track_length = 3; 
    mapper_config.max_abs_pose_reproj_error = 12.0; 
    mapper_config.global_ba_interval = 5;  
    sfm::IncrementalSFM mapper(mapper_config);
    std::shared_ptr<base::Reconstruction> reconstruction = 
        std::make_shared<base::Reconstruction>(&scene); 
    std::shared_ptr<base::Visualizer> visualizer = 
        std::make_shared<base::Visualizer>(reconstruction, "My3D Sparse Reconstruction"); 
    visualizer->run(); 
    sfm::IncrementalSFM::Report mapper_report = 
        mapper.reconstruct(reconstruction);
    if (mapper_report.success) {
        std::cout << "Reconstruction succeeded. " << std::endl; 
    } else {
        std::cout << "Reconstruction failed. " << std::endl; 
    }
    if (savePoint3D(workspace_directory, images, scene, mapper_report.tracks)) {
        std::cout << "Saved 3D points. " << std::endl;
    } else {
        std::cout << "Failed to save 3D points. " << std::endl; 
    }

    if (saveCameras(workspace_directory, mapper_report.camera_infos)) {
        std::cout << "Saved Cameras. " << std::endl;
    } else {
        std::cout << "Failed to save cameras. " << std::endl; 
    }
    
    visualizer->join();

    return 0;
}


