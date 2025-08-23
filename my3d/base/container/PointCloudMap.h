/*
 * filename: PointCloudMap.hpp
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _MY3D_BASE_CONTAINER_POINT_CLOUD_MAP_H_
#define _MY3D_BASE_CONTAINER_POINT_CLOUD_MAP_H_

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <string>
#include <functional>
#include <fstream>

#include <eigen3/Eigen/Eigen>

#include "base/container/GNATree.h"

namespace my3d {
namespace base
{
class PointCloudMap
{
public: 
    PointCloudMap(): 
    map_bound_l_(-25.0, -25.0, 0.0), 
    map_bound_u_(25.0, 25.0, 6.0)
    {
        setUpCloud();
    }

    PointCloudMap(const Eigen::Vector3d &lb, 
                  const Eigen::Vector3d &ub, 
                  double resolution = 0.1):
    map_bound_l_(lb), 
    map_bound_u_(ub), 
    resolution_(resolution)
    {
        setUpCloud();
    }

    ~PointCloudMap() {}

public: 
    
    double getResolution() const
    {
        return resolution_;
    }

    void setResolution(double res)
    {
        assert(res > std::numeric_limits<double>::epsilon());
        resolution_ = res;
    }

    void clear()
    {
        cloud_->clear();
    }

    void reset(double res)
    {
        clear();
        assert(res > std::numeric_limits<double>::epsilon());
        resolution_ = res;
    }

    void add(const Eigen::Vector3d &point)
    {
        cloud_->add(point);
    }

    void add(const std::vector<Eigen::Vector3d> &points)
    {
        cloud_->add(points);
    }

    void toVector(std::vector<Eigen::Vector3d> &vec) const
    {
        cloud_->toVector(vec);
    }

    void nearest(const Eigen::Vector3d &p, Eigen::Vector3d &out)
    {
        cloud_->nearest(p, out);
    }

    void nearestR(const Eigen::Vector3d &p, double r, std::vector<Eigen::Vector3d> &out)
    {
        cloud_->nearestR(p, r, out);
    }

    void addVerticleCylinder(double bottom_center_x, 
                             double bottom_center_y, 
                             double bottom_center_z, 
                             double radius, 
                             double height)
    {
        const double cx = bottom_center_x, cy = bottom_center_y, cz = bottom_center_z;
        
        if (cx - radius < map_bound_l_(0) || cx + radius > map_bound_u_(0) || 
            cy - radius < map_bound_l_(0) || cy + radius > map_bound_u_(0) || 
            cz < map_bound_l_(2) || cz + height > map_bound_u_(2))
        {
            std::cout << "Cylinder out of bound.Failed to add cylinder." << std::endl;
            return;
        }

        double x, y;
        for (double theta = 0.0; theta < 2 * M_PI; theta += resolution_ / radius)
        {
            for (double r = 0.0; r <= radius; r += resolution_)
            {
                for (double z = cz; z <= cz + height; z += resolution_)
                {
                    x = cx + r * cos(theta);
                    y = cy + r * sin(theta);
                    cloud_->add(Eigen::Vector3d(x, y, z));
                }
            }
        }
    }

    void addVerticleCube(double bottom_center_x, 
                         double bottom_center_y, 
                         double bottom_center_z, 
                         double lx, 
                         double ly, 
                         double height, 
                         double yaw, 
                         bool only_surface = true)
    {
        const double cx = bottom_center_x, cy = bottom_center_y, cz = bottom_center_z;
        
        if (cx < map_bound_l_(0) || cx > map_bound_u_(0) || 
            cy < map_bound_l_(1) || cy > map_bound_u_(1) || 
            cz < map_bound_l_(2) || cz > map_bound_u_(2) || 
            cz + height > map_bound_u_(2))
        {
            std::cout << "Cube out of bound.Failed to add cube." << std::endl;
            return;
        }

        Eigen::Vector3d bv1(-lx / 2.0, -ly / 2.0, cz), 
                        bv2(-lx / 2.0, ly / 2.0, cz), 
                        bv3(lx / 2.0, -ly / 2.0, cz), 
                        bv4(lx / 2.0, ly / 2.0, cz), 
                        bc(cx, cy, cz);
        Eigen::Quaterniond q(cos(yaw / 2.0), 0, 0, sin(yaw / 2.0));

        bv1 = bc + q * bv1; 
        bv2 = bc + q * bv2;
        bv3 = bc + q * bv3;
        bv4 = bc + q * bv4;

        if (bv1(0) < map_bound_l_(0) || bv1(0) > map_bound_u_(0) || 
            bv2(0) < map_bound_l_(0) || bv2(0) > map_bound_u_(0) || 
            bv3(0) < map_bound_l_(0) || bv3(0) > map_bound_u_(0) || 
            bv4(0) < map_bound_l_(0) || bv4(0) > map_bound_u_(0) || 
            bv1(1) < map_bound_l_(1) || bv1(1) > map_bound_u_(1) || 
            bv2(1) < map_bound_l_(1) || bv2(1) > map_bound_u_(1) || 
            bv3(1) < map_bound_l_(1) || bv3(1) > map_bound_u_(1) || 
            bv4(1) < map_bound_l_(1) || bv4(1) > map_bound_u_(1) || 
            bv1(2) < map_bound_l_(2) || bv1(2) > map_bound_u_(2) || 
            bv2(2) < map_bound_l_(2) || bv2(2) > map_bound_u_(2) || 
            bv3(2) < map_bound_l_(2) || bv3(2) > map_bound_u_(2) || 
            bv4(2) < map_bound_l_(2) || bv4(2) > map_bound_u_(2))
        {
            std::cout << "[" << bv1.transpose() << "]" << std::endl;
            std::cout << "[" << bv2.transpose() << "]" << std::endl;
            std::cout << "[" << bv3.transpose() << "]" << std::endl;
            std::cout << "[" << bv4.transpose() << "]" << std::endl;
            std::cout << "Cube out of bound.Failed to add cylinder." << std::endl;
            return;
        }

        Eigen::Vector3d p;
        if (only_surface)
        {
            double a, b, c;
            a = -lx / 2.0;
            for (b = -ly / 2.0; b <= ly / 2.0; b += resolution_)
            {
                for (c = 0.0; c <= height; c += resolution_)
                {
                    p(0) = cx + a * cos(yaw) - b * sin(yaw);
                    p(1) = cy + a * sin(yaw) + b * cos(yaw);
                    p(2) = cz + c;
                    cloud_->add(p);
                }
            }
            a = lx / 2.0;
            for (b = -ly / 2.0; b <= ly / 2.0; b += resolution_)
            {
                for (c = 0.0; c <= height; c += resolution_)
                {
                    p(0) = cx + a * cos(yaw) - b * sin(yaw);
                    p(1) = cy + a * sin(yaw) + b * cos(yaw);
                    p(2) = cz + c;
                    cloud_->add(p);
                }
            }

            b = -ly / 2.0;
            for (a = -lx / 2.0; a <= lx / 2.0; a += resolution_)
            {
                for (c = 0.0; c <= height; c += resolution_)
                {
                    p(0) = cx + a * cos(yaw) - b * sin(yaw);
                    p(1) = cy + a * sin(yaw) + b * cos(yaw);
                    p(2) = cz + c;
                    cloud_->add(p);
                }
            }
            b = ly / 2.0;
            for (a = -lx / 2.0; a <= lx / 2.0; a += resolution_)
            {
                for (c = 0.0; c <= height; c += resolution_)
                {
                    p(0) = cx + a * cos(yaw) - b * sin(yaw);
                    p(1) = cy + a * sin(yaw) + b * cos(yaw);
                    p(2) = cz + c;
                    cloud_->add(p);
                }
            }

            c = 0.0;
            for (a = -lx / 2.0; a <= lx / 2.0; a += resolution_)
            {
                for (b = -ly / 2.0; b <= ly / 2.0; b += resolution_)
                {
                    p(0) = cx + a * cos(yaw) - b * sin(yaw);
                    p(1) = cy + a * sin(yaw) + b * cos(yaw);
                    p(2) = cz + c;
                    cloud_->add(p);
                }
            }
            c = height;
            for (a = -lx / 2.0; a <= lx / 2.0; a += resolution_)
            {
                for (b = -ly / 2.0; b <= ly / 2.0; b += resolution_)
                {
                    p(0) = cx + a * cos(yaw) - b * sin(yaw);
                    p(1) = cy + a * sin(yaw) + b * cos(yaw);
                    p(2) = cz + c;
                    cloud_->add(p);
                }
            }
        }
        else
        {
            for (double a = -lx / 2.0; a <= lx / 2.0; a += resolution_)
            {
                for (double b = -ly / 2.0; b <= ly / 2.0; b += resolution_)
                {
                    for (double c = 0.0; c <= height; c += resolution_)
                    {
                        p(0) = cx + a * cos(yaw) - b * sin(yaw);
                        p(1) = cy + a * sin(yaw) + b * cos(yaw);
                        p(2) = cz + c;
                        cloud_->add(p);
                    }
                }
            }
        }
        
    }

protected: 
    void setUpCloud()
    {
        cloud_ = std::make_shared<GNATree<Eigen::Vector3d>>();
        auto dist_func = [](const base::Point3D &p1, const base::Point3D &p2)
        {
            return (p1 - p2).norm();
        };
        cloud_->setDistanceFunction(dist_func);
    }

protected:
    GNATree<Eigen::Vector3d>::Ptr cloud_;
    double resolution_{0.1};

public: 
    Eigen::Vector3d map_bound_l_, map_bound_u_;
};

} // namespace base
}

#endif // _MY3D_BASE_CONTAINER_POINT_CLOUD_MAP_H_
