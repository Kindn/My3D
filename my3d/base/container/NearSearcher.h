/*
 * filename: NearSearcher.hpp 
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Abstract class of near neighbor searcher.
 */

#ifndef _MY3D_BASE_CONTAINER_NEAR_SEARCHER_H_
#define _MY3D_BASE_CONTAINER_NEAR_SEARCHER_H_

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <cassert>

namespace my3d {
namespace base
{
    
/* Abstract class of near searcher */
template <typename _T>
class NearSearcher : std::enable_shared_from_this<NearSearcher<_T>>
{
public: 
    typedef std::shared_ptr<NearSearcher<_T>> Ptr;
    using DistanceFunction = std::function<double(const _T &, const _T &)>;

    NearSearcher() {}
    virtual ~NearSearcher() {}

    /* 
     * Set user-defined distance function.It is necessary to 
     * call this function before calling methods of the searcher.
     */
    virtual void setDistanceFunction(const DistanceFunction &func)
    {
        assert(func);
        dist_func_ = func;
    }

    /* 
     * Check if is empty.
     */
    bool empty() const
    {
        return data_size_ == 0;
    }

    /* 
     * Get number of elements.
     */
    size_t size() const
    {
        return data_size_;
    }

    /* 
     * Add one element to the searcher.
     */
    virtual void add(const _T &element) = 0;

    /* 
     * Add a list of elements to the searcher.
     */
    virtual void add(const std::vector<_T> &data)
    {
        for (auto &element : data)
            add(element);
    }

    //virtual void add(const std::vector<_T> &) = 0;

    //virtual void remove(const _T &) = 0;

    /* 
     * List the data in the searcher.
     */
    virtual void toVector(std::vector<_T> &vec) = 0;

    /* 
     * Clear the searcher.Data in it will be lost.
     */
    virtual void clear() = 0;
    
    /* 
     * Find the nearest element in the searcher w.r.t the given element
     * according to the user-defined distance function.
     */
    virtual void nearest(const _T &x, _T &result) = 0;

    /* 
     * Find the nearest element in the searcher w.r.t the given element
     * according to the user-defined distance function.In addition, get 
     * the distance value.
     */
    virtual void nearest(const _T &x, _T &result, double &val) = 0;

    /* 
     * Find elements in the searcher within distance radius in ascending order.
     */
    virtual void nearestR(const _T &x, double radius, std::vector<_T> &out) = 0;

protected: 
    /* User-defined distance function. */
    DistanceFunction dist_func_;

    /* Total number of elements stored in the tree. */
    size_t data_size_{0};

};

} // namespace base
}

#endif // _MY3D_BASE_CONTAINER_NEAR_SEARCHER_H_
