/*
 * filename: GNATree.hpp 
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Class of Geometric Nearest-Access Tree(GNAT).
 */

#ifndef _MY3D_BASE_CONTAINER_GNA_TREE_H_
#define _MY3D_BASE_CONTAINER_GNA_TREE_H_

#include "base/container/NearSearcher.h"
#include "utils/RandomNumberGenerator.h"

namespace my3d {
namespace base{
/* Reference: 
    Paper: 
            Brin S . Near Neighbor Search in Large Metric Spaces.  1995.
    The structure of the code is partly borrowed from OMPL.

   Author: Peiyan Liu, nROS-LAB, HITSZ
*/

/* 
 * Split point selector using greedy algorithm optimized by Dynamic Programming.
 */
template <typename _T>
class CenterSelector
{
public: 
    CenterSelector(){}
    ~CenterSelector() {}

    void setDistanceFunction(const std::function<double(const _T &, const _T &)> &dist_func)
    {
        assert(dist_func);
        dist_func_ = dist_func;
    }

    /* Select center_num split points (centers) from candidates using greedy method. */
    void select(const std::vector<_T> &candidates, 
                size_t center_num, 
                std::vector<size_t> &center_indices)
    {
        assert(dist_func_);
        center_indices.clear();
        if (center_num >= candidates.size())
        {
            /* If candidates' size is less than required center_num, 
                then all the candidates are selected. */
            center_indices.reserve(candidates.size());
            for (size_t i = 0; i < candidates.size(); i++)
                center_indices.push_back(i);

            return;
        }
        else
        {
            center_indices.reserve(center_num);

            size_t cdd_sz = candidates.size(); // candidates' size
            std::vector<double> min_dists(cdd_sz, std::numeric_limits<double>::infinity());
            /* randomly choose one center from candidates */
            size_t ind = static_cast<size_t>((cdd_sz - 1) * rng_.uniform01());
            center_indices.push_back(ind);
            for (size_t i = 1; i < center_num; i++)
            {
                double max_dist = -std::numeric_limits<double>::infinity();
                const _T &last_center = candidates[center_indices[i - 1]];
                for (size_t j = 0; j < cdd_sz; j++)
                {
                    double dist = dist_func_(candidates[j], last_center);
                    if (dist < min_dists[j])
                        min_dists[j] = dist;

                    if (min_dists[j] > max_dist)
                    {
                        max_dist = min_dists[j];
                        ind = j;
                    }
                }
                // if no more element could be chosen as split point
                if (max_dist < std::numeric_limits<double>::epsilon())  
                    break;

                center_indices.push_back(ind);
            }
        }
    }

protected: 
    util::RandomNumberGenerator rng_;
    std::function<double(const _T &, const _T &)> dist_func_;
};

/* 
 * The searcher based on Geometric Near-Access Tree(GNAT).
 */
template <typename _T>
class GNATree : public NearSearcher<_T>
{
public: 
    typedef std::shared_ptr<GNATree<_T>> Ptr;

    /* constructor */
    GNATree(uint32_t degree = 8, 
            uint32_t min_degree = 4, uint32_t max_degree = 12, 
            size_t max_size_per_leaf = 50, 
            bool need_rebalancing = false): 
    NearSearcher<_T>(), 
    degree_(degree), 
    min_degree_(min_degree), max_degree_(max_degree), 
    max_size_per_leaf_(max_size_per_leaf), 
    rebuild_thresh_(need_rebalancing ? max_size_per_leaf * degree : std::numeric_limits<size_t>::max()), 
    root_(nullptr)
    {
    }

    ~GNATree()
    {
        delete root_;
    }

    /* 
     * Set user-defined distance function.It is necessary to 
     * call this function before calling methods of the searcher.
     */
    virtual void setDistanceFunction(const typename NearSearcher<_T>::DistanceFunction &func) 
    {
        // need ‘typename’ before ‘base::NearSearcher<_T>::DistanceFunction’ because ‘base::NearSearcher<_T>’ is a dependent scope
        NearSearcher<_T>::setDistanceFunction(func);
        center_selector_.setDistanceFunction(func);
        if (root_)
            rebuild();
    }

    /* 
     * Add one element to the searcher.
     */
    void add(const _T &element) override
    {
        this->data_size_++;
        if (root_ == nullptr)
        {
            root_ = new Node(degree_, max_size_per_leaf_, element);
        }
        else
        {
            root_->add(*this, element);
            if (this->data_size_ > rebuild_thresh_)
            {
                rebuild();
                rebuild_thresh_ *= 2;
            }
        }
    }

    /* 
     * Add a list of elements to the searcher.
     */
    virtual void add(const std::vector<_T> &elements)
    {
        if (root_ == nullptr)
        {
            root_ = new Node(degree_, max_size_per_leaf_, elements[0]);
            root_->elements_.insert(root_->elements_.end(), elements.begin() + 1, elements.end());
            this->data_size_ += elements.size();
            
            if (root_->needToSplit(*this))
                root_->split(*this);
        }
        else if (!elements.empty())
        {
            NearSearcher<_T>::add(elements);
        }
    }

    /* 
     * Clear the searcher.Data in it will be lost.
     */
    void clear() override
    {
        if (root_)
            delete root_;
        root_ = nullptr;
        this->data_size_ = 0;
        if (rebuild_thresh_ != std::numeric_limits<size_t>::max())
            rebuild_thresh_ = degree_ * max_size_per_leaf_;
    }

    /* 
     * List the data in the searcher.
     */
    void toVector(std::vector<_T> &vec) override
    {
        vec.clear();
        if (this->data_size_ > 0 && root_ != nullptr)
        {
            vec.reserve(this->data_size_);
            root_->list(vec);
        }
    }

    /* Rebuild the data structure */
    void rebuild()
    {
        std::vector<_T> data;
        toVector(data);
        clear();
        add(data);
    }

    /* 
     * Find the nearest element in the searcher w.r.t the given element
     * according to the user-defined distance function.
     */
    void nearest(const _T &x, _T &result) override
    {
        if (root_ == nullptr || this->data_size_ <= 0)
            return;
        
        Node *leaf = root_->findLeaf(*this, x);
        if (leaf->elements_.empty())
        {
            result = leaf->center_;
            return;
        }
        else
        {
            double min_dist = this->dist_func_(x, leaf->center_);
            result = leaf->center_;
            for (auto &element : leaf->elements_)
            {
                double dist = this->dist_func_(x, element);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    result = element;
                }
            }

            std::vector<_T> nb;
            nearestR(x, min_dist, nb);
            for (auto &p : nb)
            {
                double dist = this->dist_func_(x, p);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    result = p;
                }
            }
        }
    }

    /* 
     * Find the nearest element in the searcher w.r.t the given element
     * according to the user-defined distance function. In addition, get 
     * the distance value.
     */
    void nearest(const _T &x, _T &result, double &val) override
    {
        if (root_ == nullptr || this->data_size_ <= 0)
        {
            val = std::numeric_limits<double>::infinity();
            return;
        }
        
        Node *leaf = root_->findLeaf(*this, x);
        
        double min_dist = this->dist_func_(x, leaf->center_);
        result = leaf->center_;
        for (auto &element : leaf->elements_)
        {
            double dist = this->dist_func_(x, element);
            if (dist < min_dist)
            {
                min_dist = dist;
                result = element;
            }
        }

        std::vector<_T> nb;
        nearestR(x, min_dist, nb);
        for (auto &p : nb)
        {
            double dist = this->dist_func_(x, p);
            if (dist < min_dist)
            {
                min_dist = dist;
                result = p;
            }
        }
            
        val = min_dist;
        
    }

    /* 
     * Find elements in the searcher within distance radius in ascending order.
     */
    void nearestR(const _T &x, double radius, std::vector<_T> &out) override
    {
        out.clear();
        if (!root_)
            return;

        std::multimap<double, _T> result;
        root_->nearestR(*this, x, radius, result);
        if (!result.empty())
        {
            while (!result.empty())
            {
                out.push_back(result.begin()->second);
                result.erase(result.begin());
            }
        }
    }

    /* 
     * Reset ignorance of all nodes of GNAT.
     */
    void resetIgnorance()
    {
        if (root_)
            root_->resetIgnorance();

        ignored_cache_.clear();
    }

protected: 
    /* the datastucture 
     * A node indicates a Dirichlet domain w.r.t its split(center) point.
     */
    class Node
    {
    public: 
        Node(uint32_t degree, 
             size_t capacity, 
             _T center): 
        degree_(degree), 
        center_(std::move(center)),  
        min_range_(degree, std::numeric_limits<double>::infinity()), 
        max_range_(degree, -std::numeric_limits<double>::infinity())
        {
            elements_.reserve(capacity + 1);
            //min_range_[0] = min_range_[1] = std::numeric_limits<double>::max();
        }

        ~Node() 
        {
            for (auto &child : children_)
                delete child;
        }

    public: 
        void add(GNATree &gnat, const _T &element)
        {
            if (children_.empty())
            {
                elements_.push_back(element);
                if (needToSplit(gnat))
                    split(gnat);
            }
            else
            {
                double min_dist = gnat.dist_func_(element, children_[0]->center_);
                children_[0]->dist_to_center_ = min_dist;
                double ind_min = 0;

                for (unsigned i = 1; i < children_.size(); i++)
                {
                    children_[i]->dist_to_center_ = gnat.dist_func_(element, children_[i]->center_);
                    if (children_[i]->dist_to_center_ < min_dist)
                    {
                        min_dist = children_[i]->dist_to_center_;
                        ind_min = i;
                    }
                }
                for (unsigned i = 0; i < children_.size(); i++)
                    children_[i]->updateRange(ind_min, children_[i]->dist_to_center_);
                children_[ind_min]->add(gnat, element);
            }
        }

        void split(GNATree &gnat)
        {
            if (!children_.empty() || elements_.size() <= 0)
                return;

            children_.reserve(degree_);
            /* Select centers */
            std::vector<size_t> centers_indices;
            gnat.center_selector_.select(elements_, degree_, centers_indices);
            
            for (auto i : centers_indices)
                children_.push_back(new Node(degree_, gnat.max_size_per_leaf_, elements_[i]));
            degree_ = centers_indices.size();

            /* Assign each element to corresponding child */
            for (size_t i = 0; i < elements_.size(); i++)
            {
                _T &element = elements_[i];
                double min_dist = gnat.dist_func_(element, children_[0]->center_);
                children_[0]->dist_to_center_ = min_dist;
                double child_ind = 0;

                for (unsigned i = 1; i < children_.size(); i++)
                {
                    children_[i]->dist_to_center_ = gnat.dist_func_(element, children_[i]->center_);
                    if (children_[i]->dist_to_center_ < min_dist)
                    {
                        min_dist = children_[i]->dist_to_center_;
                        child_ind = i;
                    }
                }

                if (i != centers_indices[child_ind])
                    children_[child_ind]->elements_.push_back(element);
                for (size_t i = 0; i < degree_; i++)
                    children_[i]->updateRange(child_ind, children_[i]->dist_to_center_);
            }

            /* Reset children's degree. */
            for (auto &child : children_)
            {
                size_t degree = degree_ * child->elements_.size() / elements_.size();
                if (degree < gnat.min_degree_)
                    degree = gnat.min_degree_;
                else if (degree > gnat.max_degree_)
                    degree = gnat.max_degree_;
                
                child->degree_ = degree;
            }

            /* Free memory of elements_ */
            std::vector<_T> tmp;
            elements_.swap(tmp);

            /* Check if any children need to split. */
            for (auto &child : children_)
                if (child->needToSplit(gnat))
                    child->split(gnat);
        }

        bool needToSplit(GNATree &gnat)
        {
            size_t size = elements_.size();

            return size > gnat.max_size_per_leaf_ && size > degree_;
        }

        void updateRange(unsigned ind, double dist)
        {
            if (dist < min_range_[ind])
                min_range_[ind] = dist;
            if (dist > max_range_[ind])
                max_range_[ind] = dist;
        }

        Node *findLeaf(GNATree &gnat, const _T &x)
        {
            if (isLeaf())
                return this;
            
            double min_dist = gnat.dist_func_(x, children_[0]->center_);
            double child_ind = 0;
            for (size_t i = 1; i < children_.size(); i++)
            {
                double dist = gnat.dist_func_(x, children_[i]->center_);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    child_ind = i;
                }
            }

            return children_[child_ind]->findLeaf(gnat, x);
        }

        bool isLeaf() const
        {
            return children_.empty();
        }

        void nearestR(GNATree &gnat, const _T &x, double radius, std::multimap<double, _T> &out)
        {
            if (this->isLeaf())
            {
                double dist = 0.0;
                /* We don't check center point here because parent node 
                   has already done this. */
                for (auto &element : elements_)
                {
                    if ((dist = gnat.dist_func_(x, element)) <= radius)
                       out.insert(std::make_pair(dist, element)); 
                }
            }
            else
            {
                /* Prune children according to the ranges. */
                std::vector<bool> remaining(degree_, true); // marked remaining
                for (size_t i = 0; i < degree_; i++)
                {
                    if (!remaining[i])
                    {
                        continue;
                    }
                    
                    double dist = gnat.dist_func_(x, children_[i]->center_);
                    if (dist <= radius)
                        out.insert(std::make_pair(dist, children_[i]->center_));
                    double lower = dist - radius, upper = dist + radius;
                    for (size_t j = 0; j < degree_; j++)
                    {
                        if (!remaining[j])
                            continue;
                        
                        if (lower > children_[i]->max_range_[j] || 
                            upper < children_[i]->min_range_[j])
                            remaining[j] = false;
                    }
                }

                /* Recursively search all remaining children. */
                for (size_t i = 0; i < degree_; i++)
                    if (remaining[i])
                        children_[i]->nearestR(gnat, x, radius, out);
            }
        }

        void list(std::vector<_T> &vec)
        {
            vec.push_back(center_);
            for (auto &element : elements_)
                vec.push_back(element);

            for (auto &child : children_)
                child->list(vec);
        }

        void resetIgnorance()
        {
            ignored_ = false;
            for (auto &child : children_)
                child->resetIgnorance();
        }

        /* Degree of the node */
        uint32_t degree_;

        /* Children of the node */
        std::vector<Node *> children_;

        /* center of the node. */
        _T center_;

        /* The elements in addition to the center that are stored in the node. 
           It is not empty only if the node is a leaf. */
        std::vector<_T> elements_;

        /* Ranges of distance values from split points to the data points 
           associated with other split points. */
        std::vector<double> min_range_, max_range_;

        /* Cache the distance from a point to the center. */
        double dist_to_center_;

        /* Used for conditional search. If set true, that means the node (together with
           its children) will be ignored when doing conditional search. */
        bool ignored_{false};
    };

protected: 
    /* Desired average degree of each node. */
    uint32_t degree_;

    /* Minimum and maximum degrees of each node. */
    uint32_t min_degree_, max_degree_;

    /* Root node of GNAT. */
    Node *root_;

    /* Maximum number of elements allowed to be stored in a node before it is split. */
    size_t max_size_per_leaf_; 

    /* When size_ exceeds rebuild_thresh_, the tree will be rebuild and rebuild_thresh_ 
       will be doubled */
    size_t rebuild_thresh_;

    /* split point selector in GNAT using greedy method */
    CenterSelector<_T> center_selector_;

    /* Cache the elements that should be ignored in leaves that are not marked ignored. */
    std::unordered_set<_T *> ignored_cache_;
};

} // namespace base
}


#endif // _MY3D_BASE_CONTAINER_GNA_TREE_H_