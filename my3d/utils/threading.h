/*
 * filename: threading.h
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MY3D_UTIL_THREADING_H_
#define _MY3D_UTIL_THREADING_H_

#include <iostream>
#include <mutex>
#include <string>
#include <thread>

namespace my3d {
namespace util {

class ThreadGuard {
private:
  std::thread &t;

public:
  explicit ThreadGuard(std::thread &t_) : t(t_) {}

  ~ThreadGuard() {
    if (t.joinable()) {
      t.join();
    }
  }
};

} // namespace util
} // namespace my3d

#endif // _MY3D_UTIL_THREADING_H_
