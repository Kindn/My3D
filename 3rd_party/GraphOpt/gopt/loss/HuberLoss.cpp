/*
 * filename: HuberLoss.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "loss/HuberLoss.h"

namespace gopt {

double HuberLoss::operator () (double error2, double *grad, double *grad2) {
    double const error{std::sqrt(std::abs(error2))};
    double output = std::abs(error2) <= delta2_ ? 
                        error2 : 
                        2.0 * delta_ * error - delta2_;
    double constexpr kEps{1.0e-6};
    if (grad) {
        *grad = std::abs(error2) <= delta2_ ? 
               1.0 : 
               delta_ / (error + kEps);
    }
    
    return output;
}

}