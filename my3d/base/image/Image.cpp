/*
 * filename: Image.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "base/image/Image.h"

namespace my3d {
namespace base {

Image::Image()
    : data_{nullptr}, step0_{0}, step1_{0}, rows_{0}, cols_{0}, id_{0},
      name_{""} {}

Image::Image(const size_t &rows, const size_t &cols, const size_t &channels)
    : step0_{cols * channels}, step1_{channels}, rows_{rows}, cols_{cols},
      channels_(channels), id_{0}, name_{""}, camera_{nullptr} {
  if (rows == 0 || cols == 0 || channels == 0) {
    data_ = nullptr;
  }

  size_t size = rows_ * cols_ * channels_;
  data_ = new uint8_t[size]{0};
}

Image::Image(const size_t &rows, const size_t &cols, const size_t &channels,
             uint8_t *data, const size_t &id, const std::string &name,
             CameraBase *camera)
    : step0_{cols * channels}, step1_{channels}, rows_{rows}, cols_{cols},
      channels_(channels), id_{id}, name_{name}, camera_{camera} {
  size_t size = rows_ * cols_ * channels_;
  if (size > 0) {
    data_ = new uint8_t[size];
    std::copy(data, data + size, data_);
  } else {
    data_ = nullptr;
  }
}

Image::Image(const Image &other) {
  step0_ = other.step0_;
  step1_ = other.step1_;
  rows_ = other.rows_;
  cols_ = other.cols_;
  channels_ = other.channels_;
  id_ = other.id_;
  name_ = other.name_;
  camera_ = other.camera_;
  if (other.data_ != nullptr) {
    if (data_ != nullptr) {
      delete[] data_;
    }
    data_ = new uint8_t[rows_ * cols_ * channels_];
    std::copy(other.data_, other.data_ + other.size() * other.channels_, data_);
  } else {
    if (data_ != nullptr) {
      delete[] data_;
    }
    data_ = nullptr;
  }
}

Image::~Image() {
  if (data_ != nullptr) {
    delete[] data_;
    data_ = nullptr;
  }
}

Image &Image::operator=(const Image &other) {
  step0_ = other.step0_;
  step1_ = other.step1_;
  rows_ = other.rows_;
  cols_ = other.cols_;
  channels_ = other.channels_;
  id_ = other.id_;
  name_ = other.name_;
  camera_ = other.camera_;
  if (other.data_ != nullptr) {
    if (data_ != nullptr) {
      delete[] data_;
    }
    data_ = new uint8_t[rows_ * cols_ * channels_];
    std::copy(other.data_, other.data_ + other.size() * other.channels_, data_);
  } else {
    if (data_ != nullptr) {
      delete[] data_;
    }
    data_ = nullptr;
  }

  return *this;
}

void Image::set(const size_t &rows, const size_t &cols, const size_t &channels,
                uint8_t *data, const size_t &idx, const std::string &name,
                CameraBase *camera) {
  if (rows == 0 || cols == 0 || channels == 0) {
    if (data_ != nullptr) {
      delete[] data_;
    }
    data_ = nullptr;
    rows_ = 0;
    cols_ = 0;
    channels_ = 0;
    step0_ = 0;
    step1_ = 0;
  } else {
    assert(data != nullptr);
    if (data_ != nullptr) {
      delete[] data_;
    }
    const size_t size = rows * cols * channels;
    data_ = new uint8_t[size];
    std::copy(data, data + size, data_);
    rows_ = rows;
    cols_ = cols;
    channels_ = channels;
    step0_ = cols * channels;
    step1_ = channels;
  }
  name_ = name;
  camera_.reset(camera);
}

const uint8_t &Image::at(const size_t &i0, const size_t &i1,
                         const size_t &i2) const {
  assert(i0 < rows_);
  assert(i1 < cols_);
  assert(i2 < channels_);
  size_t addr = step0_ * i0 + step1_ * i1 + i2;
  return data_[addr];
}

uint8_t &Image::at(const size_t &i0, const size_t &i1, const size_t &i2) {
  assert(i0 < rows_);
  assert(i1 < cols_);
  assert(i2 < channels_);
  size_t addr = step0_ * i0 + step1_ * i1 + i2;
  return data_[addr];
}

uint8_t *Image::ptr(const size_t &i0) {
  assert(i0 < rows_);
  return data_ + step0_ * i0;
}

const uint8_t *Image::ptr(const size_t &i0) const {
  assert(i0 < rows_);
  return data_ + step0_ * i0;
}

uint8_t *Image::ptr(const size_t &i0, const size_t &i1) {
  assert(i0 < rows_);
  assert(i1 < cols_);
  return data_ + step0_ * i0 + step1_ * i1;
}

const uint8_t *Image::ptr(const size_t &i0, const size_t &i1) const {
  assert(i0 < rows_);
  assert(i1 < cols_);
  return data_ + step0_ * i0 + step1_ * i1;
}

uint8_t *Image::ptr(const size_t &i0, const size_t &i1, const size_t &i2) {
  assert(i0 < rows_);
  assert(i1 < cols_);
  assert(i2 < channels_);
  return data_ + step0_ * i0 + step1_ * i1 + i2;
}

const uint8_t *Image::ptr(const size_t &i0, const size_t &i1,
                          const size_t &i2) const {
  assert(i0 < rows_);
  assert(i1 < cols_);
  assert(i2 < channels_);
  return data_ + step0_ * i0 + step1_ * i1 + i2;
}

Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>
Image::getChannel(const size_t &row, const size_t &col) const {
  assert(row <= rows_);
  assert(col <= cols_);

  Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> channel(channels_);
  uint8_t *ptr = data_ + step0_ * row + step1_ * col;
  for (size_t i = 0; i < channels_; ++i) {
    channel(i) = ptr[i];
  }

  return channel;
}

Eigen::Vector2d Image::getGrandient(const size_t &x, const size_t &y,
                                    const size_t &channel) const {
  assert(!isEmpty());
  assert(x >= 0 && x < cols_);
  assert(y >= 0 && y < rows_);
  assert(channel >= 0 && channel < channels_);

  double dfx, dfy;
  if (x == 0) {
    dfx = static_cast<double>(at(y, x + 1UL, channel)) -
          static_cast<double>(at(y, x, channel));
  } else if (x == cols_ - 1UL) {
    dfx = static_cast<double>(at(y, x, channel)) -
          static_cast<double>(at(y, x - 1UL, channel));
  } else {
    dfx = 0.5 * (static_cast<double>(at(y, x + 1, channel)) -
                 static_cast<double>(at(y, x - 1UL, channel)));
  }

  if (y == 0) {
    dfy = static_cast<double>(at(y + 1UL, x, channel)) -
          static_cast<double>(at(y, x, channel));
  } else if (y == rows_ - 1UL) {
    dfy = static_cast<double>(at(y, x, channel)) -
          static_cast<double>(at(y - 1UL, x, channel));
  } else {
    dfy = 0.5 * (static_cast<double>(at(y + 1UL, x, channel)) -
                 static_cast<double>(at(y - 1UL, x, channel)));
  }

  return Eigen::Vector2d(dfx, dfy);
}

bool Image::isEmpty() const { return data_ == nullptr; }

void Image::clear() {
  if (data_ != nullptr) {
    delete[] data_;
    data_ = nullptr;
  }
  rows_ = 0;
  cols_ = 0;
  channels_ = 0;
  step0_ = 0;
  step1_ = 0;
  id_ = 0;
  name_ = "";
  exif_info_.clear();
  camera_.reset();
}

} // namespace base
} // namespace my3d
