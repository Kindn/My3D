/*
 * filename: RandomNumberGenerator.h 
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Class of random number generator.
 */

#ifndef _MY3D_UTIL_RANDOM_NUMBER_GENERATOR_H_
#define _MY3D_UTIL_RANDOM_NUMBER_GENERATOR_H_

#include <iostream>
#include <memory>
#include <assert.h>
#include <ctime>
#include <random>
#include <cmath>

namespace my3d {
namespace util {
class RandomNumberGenerator {
public: 
    RandomNumberGenerator(): 
    seed_(time(NULL)), engine_(seed_)
    {}

    RandomNumberGenerator(std::uint_fast32_t seed): 
    seed_(seed), engine_(seed_)
    {}

    ~RandomNumberGenerator() 
    {}

public: 
    void setSeed(std::uint_fast32_t seed) {
        seed_ = seed;
        engine_.seed(seed_);
    }

    double uniform01() {
        return urd01_(engine_);
    }

    double uniformReal(double lower, double upper) {
        assert(lower <= upper);
        return lower + (upper - lower) * urd01_(engine_);
    }

    int uniformInteger(int lower, int upper) {
        assert(lower <= upper);
        std::uniform_int_distribution<int> uid(lower, upper);
        return uid(engine_);
    }

    size_t uniformIndex(size_t lower, size_t upper) {
        assert(lower <= upper);
        std::uniform_int_distribution<size_t> uid(lower, upper);
        return uid(engine_);
    }

    // From: "Uniform Random Rotations", Ken Shoemake, Graphics Gems III,
    //       pg. 124-132
    void uniformQuaternion(double &x, double &y, double &z, double &w) {
        double t = urd01_(engine_);
        double r1 = sqrt(1.0 - t), r2 = sqrt(t);
        double th1 = 2.0 * M_PI * urd01_(engine_);
        double th2 = 2.0 * M_PI * urd01_(engine_);
        double c1 = cos(th1), s1 = sin(th1);
        double c2 = cos(th2), s2 = sin(th2);
        x = s1 * r1;
        y = c1 * r1;
        z = s2 * r2;
        w = c2 * r2;
    }

private:
    std::uint_fast32_t seed_;
    std::mt19937 engine_;
    std::uniform_real_distribution<double> urd01_{0.0, 1.0};
};

}
} // namespace base

#endif // _MY3D_UTILS_RANDOM_NUMBER_GENERATOR_H_
