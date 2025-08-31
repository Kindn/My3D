/*
 * filename: Track.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "base/reconstruction/Track.h"

namespace my3d {
namespace base {

void Track::addElement(const TrackElement &element) {
  elements_.push_back(element);
}

void Track::addElement(const size_t &image_id, const size_t &feature_idx) {
  elements_.emplace_back(image_id, feature_idx);
}

void Track::addElements(const TrackElementList &elements) {
  for (auto &element : elements) {
    addElement(element);
  }
}

void Track::removeElement(const size_t &image_id, const size_t &feature_idx) {
  for (auto it = elements_.begin(); it != elements_.end(); ++it) {
    if (it->image_id == image_id && it->feature_idx == feature_idx) {
      elements_.erase(it);
      break;
    }
  }
}

bool Track::hasImage(const size_t &image_id) const {
  for (const auto &element : elements_) {
    if (element.image_id == image_id) {
      return true;
    }
  }

  return false;
}

} // namespace base
} // namespace my3d
