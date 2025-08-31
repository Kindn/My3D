/*
 * filename: io.h
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Some useful file operations
 */

#ifndef _MY3D_UTIL_IO_H_
#define _MY3D_UTIL_IO_H_

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <boost/filesystem.hpp>

namespace my3d {
namespace util {

std::vector<std::string> collectFiles(const std::string &directory,
                                      const std::string &extension = "");

std::vector<std::string>
collectFiles(const std::string &directory,
             const std::unordered_set<std::string> &extensions = {});

bool readStringFromFStream(std::string &str, std::ifstream &ifs);

} // namespace util
} // namespace my3d

#endif // _MY3D_UTIL_IO_H_
