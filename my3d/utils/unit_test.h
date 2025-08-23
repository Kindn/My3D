/*
 * filename: unit_test.h
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Class of random number generator.
 */

#ifndef _MY3D_UTIL_UNIT_TEST_H_
#define _MY3D_UTIL_UNIT_TEST_H_

#include <iostream>

namespace my3d {
namespace util {
}
}

#define BOOST_TEST_MAIN

#ifndef TEST_NAME
#error "TEST_NAME not defined"
#endif

#define BOOST_TEST_MODULE TEST_NAME

#include <boost/test/unit_test.hpp>

#endif // _MY3D_UTILS_UNIT_TEST_H_
