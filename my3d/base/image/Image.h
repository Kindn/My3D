/*
 * filename: Image.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _MY3D_BASE_IMAGE_IMAGE_H_
#define _MY3D_BASE_IMAGE_IMAGE_H_

#include <memory>

#include "utils/eigen_types.h"
#include "base/camera/CameraBase.h"

#include "TinyEXIF/TinyEXIF.h"

namespace my3d {
namespace base {

/**
 * @brief Class of byte image.
*/
class Image {
public:
    Image(); 
    Image(const size_t &rows, const size_t &cols, const size_t &channels = 1); 
    Image(const size_t &rows, const size_t &cols, const size_t &channels, uint8_t *data, 
          const size_t &id = 0, const std::string &name = "", 
          CameraBase *camera = nullptr);
    ~Image();

    /**
     * @brief Construct a deep copy of other
    */
    Image(const Image &other);

    /**
     * @brief Return a deep copy of other
    */
    Image &operator = (const Image &other);

    void set(const size_t &rows, const size_t &cols, const size_t &channels, uint8_t *data, 
             const size_t &idx = 0, const std::string &name = "", 
             CameraBase *camera = nullptr);
    
    void setId(const size_t &id) { id_ = id; }

    size_t &getId() { return id_; }

    const size_t &getId() const { return id_; }

    void setName(const std::string &name) { name_ = name; }

    std::string &getName() { return name_; }

    const std::string &getName() const { return name_; }

    void setCameraModel(CameraBase *camera) { camera_.reset(camera); }

    std::shared_ptr<CameraBase> getCameraModel() const { return camera_; }

    size_t rows() const { return rows_; }

    size_t cols() const { return cols_; }

    size_t channels() const { return channels_; }

    size_t size() const { return rows_ * cols_; }

    const uint8_t& at(const size_t &row, const size_t &col, const size_t &channel) const;
    
    uint8_t& at(const size_t &row, const size_t &col, const size_t &channel);

    uint8_t *ptr(const size_t &row);

    const uint8_t *ptr(const size_t &row) const;

    uint8_t *ptr(const size_t &row, const size_t &col);

    const uint8_t *ptr(const size_t &row, const size_t &col) const;

    uint8_t *ptr(const size_t &row, const size_t &col, const size_t &channel);

    const uint8_t *ptr(const size_t &row, const size_t &col, const size_t &channel) const;

    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> getChannel(const size_t &row, const size_t &col) const; 

    Eigen::Vector2d getGrandient(const size_t &x, const size_t &y, const size_t &channel = 0) const; 

    bool isEmpty() const;

    uint8_t *data() { return data_; }

    const uint8_t *data() const { return data_; }

    TinyEXIF::EXIFInfo &getEXIFInfo() { return exif_info_; }

    const TinyEXIF::EXIFInfo &getEXIFInfo() const { return exif_info_; }

    void clear(); 

    template <typename DType>
    EigenVec<Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic>> 
    toEigenMatrices() const {
        EigenVec<Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic>> 
            out(channels_, Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows_, cols_)); 
        for (size_t ch = 0; ch < channels_; ++ch) {
            for (size_t r = 0; r < rows_; ++r) {
                for (size_t c = 0; c < cols_; ++c) {
                    out[ch].operator()(r, c) = static_cast<DType>(at(r, c, ch)); 
                }
            }
        }
        return out; 
    }

private: 
    uint8_t *data_{nullptr};
    size_t step0_{0};
    size_t step1_{0};

    size_t rows_{0};
    size_t cols_{0};
    size_t channels_{0};

    size_t id_{0};
    std::string name_{""};

    TinyEXIF::EXIFInfo exif_info_;

    std::shared_ptr<CameraBase> camera_;
};

}
}

#endif // _MY3D_BASE_IMAGE_IMAGE_H_
