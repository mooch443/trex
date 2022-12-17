#include "PairingGraph.h"
#include "Tracker.h"
#include <tracking/Individual.h>
#include <file/CSVExport.h>
#include <file/Path.h>
#include <misc/Timer.h>
#include "Hungarian.h"

/**
 Essentially, the algorithm may be thought of as competing paths through a weighted directed acyclic graph. The paths are all possible permutations of object-identity combinations, taking into account that not all combinations are possible / have edges connecting them. A paths' performance is measured by its accumulative probability score, adding up edge-weights along the path. Paths, which are "lagging behind" in performance, are discarded as soon as the best accumulative probability score they can achieve becomes smaller than the best currently known alternative.
 
 All individuals and their edges are pre-sorted previous to traversal, optimized to perform as few iterations as possible. This means that there is a dynamic computational complexity range, having quite different best- and worst-case times.
*/

namespace track {
namespace Match {
    static GenericThreadPool *pool = NULL;
    
    //IMPLEMENT(PairingGraph::unused);
    //IMPLEMENT(PairingGraph::unused_mutex);
   
PairedProbabilities::row_t::value_type PairedProbabilities::row(fish_index_t rdx) const {
    assert(rdx < n_rows());
    return _rows[(index_t)rdx];
}

PairedProbabilities::col_t::value_type PairedProbabilities::col(blob_index_t cdx) const {
    assert(cdx < n_cols());
    return _cols[(index_t)cdx];
}

std::string PairedProbabilities::toStr() const {
    std::stringstream ss;
    for(auto &row : _rows) {
        ss << row->identity().name() << ":\n";
        for(auto idx : edges_for_row(index(row))) {
            ss << "\t" << (col(idx.cdx).valid() ? col(idx.cdx).toStr() : "null") << ":" << idx.p << "\n";
        }
    }
    return ss.str();
}

size_t PairedProbabilities::degree(fish_index_t rdx) const {
    assert((Match::index_t)rdx < _degree.size());
    return _degree[(index_t)rdx];
}

blob_index_t PairedProbabilities::index(col_t::value_type col) const {
    if(!_col_index.count(col))
        throw U_EXCEPTION("Cannot find blob ",col," in map.");
    return _col_index.at(col);
}

fish_index_t PairedProbabilities::index(row_t::value_type row) const {
    if(!_row_index.count(row))
        throw U_EXCEPTION("Cannot find individual ",row->identity().ID()," in map.");
    return _row_index.at(row);
}

bool PairedProbabilities::has(row_t::value_type row) const {
    if(!_row_index.count(row))
        return false;
    return true;
}
bool PairedProbabilities::has(col_t::value_type col) const {
    if(!_col_index.count(col))
        return false;
    return true;
}

const decltype(PairedProbabilities::_col_edges)::mapped_type& PairedProbabilities::edges_for_col(blob_index_t cdx) const {
    return _col_edges.at(col(cdx));
}

const decltype(PairedProbabilities::_col_edges)& PairedProbabilities::col_edges() const {
    return _col_edges;
}

std::span<const PairedProbabilities::Edge> PairedProbabilities::edges_for_row(fish_index_t rdx) const {
    size_t current = _offsets[(index_t)rdx];
    size_t next = (index_t)rdx + 1 < (index_t)_offsets.size()
        ? _offsets[(index_t)rdx+1]
        : _probabilities.size();
    return std::span<const Edge>(_probabilities.data() + _offsets[(index_t)rdx], next - current);
    //return std::vector<Edge>(_probabilities.begin() + _offsets[(index_t)rdx], _probabilities.begin() + next);
}

bool PairedProbabilities::empty() const {
    return (index_t)_num_rows == 0 && (index_t)_num_cols == 0;
}

prob_t PairedProbabilities::max_prob(fish_index_t rdx) const {
    return _row_max_probs.at((index_t)rdx);
}

void PairedProbabilities::erase(col_t::value_type col) {
    throw U_EXCEPTION("erase(col) not implemented");
    if(!_col_index.count(col))
        return; //! not found
    
    size_t offset_offset = 0;
    auto oit = _offsets.begin();
    size_t j=0;
        
    auto index = _col_index.at(col);
    auto it = std::find(_cols.begin(), _cols.end(), col);
    
    size_t ridx = 0;
    size_t next = oit < _offsets.end() - 1 ? *(oit+1) : _probabilities.size();
    
    if(it != _cols.end()) {
        for(auto pit = _probabilities.begin(); pit != _probabilities.end(); ++j) {
            if(next == j) {
                print("Increasing offset since we reached ", next," (for row ", std::distance(_offsets.begin(), oit),")");
                ++oit;
                ++ridx;
                
                if(oit != _offsets.end())
                    next = oit < _offsets.end() - 1 ? *(oit+1) : _probabilities.size();
                else {
                    print("Reached end of offsets.");
                }
                
                if(oit != _offsets.end() && offset_offset > 0) {
                    print("\tSubtracting ", offset_offset," from ", std::distance(_offsets.begin(), oit)," (", *oit,")");
                    *oit -= offset_offset;
                }
            }
            
            if(pit->cdx == index) {
                auto cidx = std::distance(_offsets.begin(), oit);
                print("Row ", cidx,"/", ridx," removing 1 degree (previously ",_degree.at(ridx),")");
                assert(_degree.at(ridx) > 0);
                --_degree.at(ridx);
                
                pit = _probabilities.erase(pit);
                ++offset_offset;
            } else {
                ++pit;
            }
        }
        
        _col_edges.erase(col);
        _cols.erase(it);
        _col_index.erase(col);
        
        for (auto i=index; (index_t)i < _cols.size(); ++i)
            _col_index.at(_cols.at((index_t)i)) = i;
        
        _num_cols = blob_index_t(_cols.size());
    }
    
#ifndef NDEBUG
    for(auto o : _offsets) {
        assert(o <= _probabilities.size());
    }
#endif
    
    for(size_t i=0; i<_rows.size();) {
        if(_degree.at(i) == 0) {
            erase(_rows.at(i));
        } else
            ++i;
    }
}

void PairedProbabilities::erase(row_t::value_type row) {
    if(!_row_index.count(row))
        return; //! not found
    
    auto rdx = _row_index[row];
    auto offset = _offsets[(index_t)rdx];
    auto next_offset = (index_t)rdx + 1 < _offsets.size() ? _offsets[(index_t)rdx+1] : _probabilities.size();
    _probabilities.erase(_probabilities.begin() + offset, _probabilities.begin() + next_offset);
    
    for(auto it = _offsets.begin() + (index_t)rdx + 1; it != _offsets.end(); ++it)
        *it = *it - (next_offset - offset);
    
    for(auto & [r, idx] : _row_index) {
        if(idx > rdx)
            --idx;
    }
    
    for(auto & [col, edges] : _col_edges) {
        for(auto it = edges.begin(); it != edges.end(); ) {
            if(*it == rdx) {
                it = edges.erase(it);
            } else if(*it > rdx) {
                --(*it);
                ++it;
            } else
                ++it;
        }
    }
    
    _degree.erase(_degree.begin() + (index_t)rdx);
    _row_max_probs.erase(_row_max_probs.begin() + (index_t)rdx);
    _rows.erase(_rows.begin() + (index_t)rdx);
    _offsets.erase(_offsets.begin() + (index_t)rdx);
    _row_index.erase(row);
    
    _num_rows = fish_index_t(_rows.size());
}

prob_t PairedProbabilities::probability(row_t::value_type row, col_t::value_type col) const {
    return probability(index(row), index(col));
}

prob_t PairedProbabilities::probability(fish_index_t rdx, blob_index_t cdx) const {
    size_t next = (index_t)rdx + 1 < (index_t)_num_rows ? _offsets[(index_t)rdx+1] : _probabilities.size();
    auto it = std::find(_probabilities.begin() + _offsets[(index_t)rdx], _probabilities.begin() + next, cdx);
    if(it != _probabilities.end()) {
        return it->p;
    }
    
    return 0;
}

fish_index_t PairedProbabilities::add(
      row_t::value_type row,
      const pairing_map_t<col_t::value_type, prob_t>& edges)
{
    fish_index_t rdx;
    size_t offset;
    if(_row_index.count(row))
        throw U_EXCEPTION("Already added individual ",row->identity().ID()," to map.");
    
    rdx = fish_index_t(_rows.size());
    _row_index[row] = rdx;
    _rows.push_back(row);
    _num_rows = fish_index_t(_rows.size());
    
    offset = _probabilities.size();
    _offsets.push_back(offset);
    
    _probabilities.resize(_probabilities.size() + edges.size());
    
    //! now add the edges / cols needed
    prob_t maximum = 0;
    size_t degree = 0;
    for(auto && [col, p] : edges) {
        auto cdx = add(col);
        _probabilities[offset++] = Edge(cdx, p);
        _col_edges[col].push_back(rdx);
        if(p > maximum)
            maximum = p;
        if(p > FAST_SETTING(matching_probability_threshold))
            ++degree;
    }
    _row_max_probs.push_back(maximum);
    _degree.push_back(degree);
    
    return rdx;
}

blob_index_t PairedProbabilities::add(col_t::value_type col) {
    auto it = _col_index.find(col);
    if(it != _col_index.end())//if(contains(_cols, col))
        return it->second; // already added this row
    
    auto index = blob_index_t(_cols.size());
    _col_index[col] = index;
    _cols.push_back(col);
    _num_cols = blob_index_t(_cols.size());
    return index;
}

void PairedProbabilities::init() {
}

bool PairedProbabilities::Edge::operator<(const Edge &j) const {
    return p > j.p || (p == j.p && (!j.cdx.valid() || (cdx.valid() && cdx > j.cdx)));
}
    
PairingGraph::Stack::Stack(fish_index_t number_individuals, blob_index_t number_blobs, const multiset::const_iterator& node, pset_ptr_t set) : Stack()
{
    construct(number_individuals, number_blobs, node, set);
}

PairingGraph::Stack::Stack(fish_index_t number_individuals, const Stack& parent, const multiset::const_iterator& next, pset::const_iterator combination, prob_t p, pset_ptr_t set)
{
    blobs.reserve((index_t)number_individuals);
    //blobs_present.resize(number_blobs);
    clear(parent, next, combination, p, set);
}

void PairingGraph::Stack::construct(fish_index_t number_individuals, blob_index_t number_blobs, const multiset::const_iterator& node, pset_ptr_t set)
{
    _probs = set;
    fish_it = node;
    prob_it = set->begin();
    acc_p = 0;
    
    blobs.clear();
    //probs.clear();
    blobs_present.resize((index_t)number_blobs);
    blobs.reserve((index_t)number_individuals);
    //per_individual.resize(number_individuals);
    
    for(size_t i=0; i<blobs_present.size(); ++i)
        blobs_present[i] = false;
}

void PairingGraph::Stack::clear(const Stack& parent, const multiset::const_iterator& next, pset::const_iterator combination, prob_t p, pset_ptr_t set)
{
    blobs.clear();
    blobs.insert(blobs.end(), parent.blobs.begin(), parent.blobs.end());
    
    blobs_present = parent.blobs_present;
    //per_individual.resize(number_individuals);
    
    clear_reuse(parent, next, combination, p, set);
}

void PairingGraph::Stack::clear_reuse(const Stack&, const multiset::const_iterator& next, pset::const_iterator , prob_t new_p, pset_ptr_t set)
{
    //blobs.push_back(p->blob_index);
    //if(p->blob_index != -1) {
        //assign_blob(this, p->blob_index);
        //blobs_present[p->blob_index] = true;
    //}
    
    fish_it = next;
    
    _probs = set;
    prob_it = set->begin();
    
    acc_p = new_p;
    
    //probs.clear();
}

long_t PairingGraph::Stack::idx(const multiset& fish_set, const multiset::const_iterator& fish_pos)
{
    return (long_t)std::distance(fish_set.begin(), fish_pos);
}

    void PairingGraph::prepare_shutdown() {
        if(pool)
            delete pool;
        pool = NULL;
    }

    bool PairingGraph::Node::operator<(const Node& other) const {
        //return max_prob * degree > other.max_prob * other.degree || (max_prob * degree == other.max_prob * other.degree && fish->identity().ID() < other.fish->identity().ID());
        //return degree > other.degree || (degree == other.degree && fish->identity().ID() < other.fish->identity().ID());
        return degree > other.degree || (degree == other.degree && (max_prob > other.max_prob || (max_prob == other.max_prob && fish->identity().ID() < other.fish->identity().ID())));
    }
    
    /*void PairingGraph::work(gq_t_& global_stack, std::deque<Stack*>& local_stack, Stack &current) {
        auto& node = current.fish_it;
        //auto &path = current.path;
        auto &blobs = current.blobs;
        
        // the next node
        auto next = std::next(current.fish_it);
        
        const float hierarchy_best_p = _ordered_max_probs.at(blobs.size());
        auto insert_globally = [hierarchy_best_p, this](gq_t_& queue, Stack& stack, const multiset::const_iterator& next, pset::const_iterator p)
        {
            
            std::lock_guard<std::mutex> lock(unused_mutex);
            if(stack.acc_p + p->second + hierarchy_best_p > _optimal_pairing->p) {
                long_t ID = Stack::idx(_optimal_pairing->set, next);
                if(unused.empty())
                    queue.push(new Stack(ID, stack, next, p, stack.acc_p + p->second));
                else {
                    auto obj = unused.front();
                    unused.pop();
                    
                    obj->clear(ID, stack, next, p, stack.acc_p + p->second);
                    queue.push(obj);
                }
                
                return true;
            }
            
            return false;
        };
        
        auto insert_locally = [&queue = local_stack, hierarchy_best_p, this](Stack& stack, const multiset::const_iterator& next, pset::const_iterator p)
        {
            // if its the last arm, insert first item into local queue
            if(stack.acc_p + p->second + hierarchy_best_p > _optimal_pairing->p) {
                long_t ID = Stack::idx(_optimal_pairing->set, next);
                stack.clear_reuse(ID, stack, next, p, stack.acc_p + p->second);
                queue.push_front(&stack);
                return true;
            }
            
            std::lock_guard<std::mutex> lock(unused_mutex);
            unused.push(&stack);
            return false;
        };
        
        // initialize stack element
        if(!current.init) {
            current.init = true;
            //current.it++; // increase current.it to next layer
            
            if(current.acc_p + 0 + hierarchy_best_p > _optimal_pairing->p) {
                current.probs.insert({Edge(NULL), 0});
                //null_values++;
            }
            
            auto it = _edges.find(node->fish);
            
            // if this object has edges, insert them
            if(it != _edges.end()) {
                for(auto &e : it->second) {
                    if(contains(blobs, e.blob))
                        continue;
                    //if(blobs.find(e.blob) != blobs.end())
                    //    continue;
                    
                    float p = prob(node->fish, e.blob);
                    
                    // is this worth traversing, or is the maxium that can be reached in terms
                    // of probabilities already smaller than the current best value
                    if(current.acc_p + p + hierarchy_best_p < _optimal_pairing->p)
                        continue;
                    
                    current.probs.insert({e, p});
                }
            }
            
            if(current.probs.empty()) {
                std::lock_guard<std::mutex> lock(unused_mutex);
                unused.push(&current);
                return;
                
            } else if(current.probs.size() == 1 && next != _optimal_pairing->set.end()) {
                insert_locally(current, next, current.probs.begin());
                return;
            }
            
            current.prob_it = std::next(current.probs.begin());
        }
        
        assert(next == _optimal_pairing->set.end() || current.prob_it != current.probs.end());
        
        if(next != _optimal_pairing->set.end()) {
            if(current.acc_p + 0 + hierarchy_best_p < _optimal_pairing->p) {
                std::lock_guard<std::mutex> lock(unused_mutex);
                unused.push(&current);
                return;
            }
            
            // look at next combination of fish/blob and
            // evaluate branch weight
            auto p = current.prob_it++;
            
            if(current.prob_it != current.probs.end())
                local_stack.push_front(&current);
            
            // put new element (if found) into global queue
            if(current.ID > 1) {
                std::lock_guard<std::mutex> lock(unused_mutex);
                if(current.acc_p + p->second + hierarchy_best_p > _optimal_pairing->p) {
                    long_t ID = current.ID+1;//Stack::idx(_optimal_pairing->set, next);
                    if(unused.empty())
                        local_stack.push_front(new Stack(ID, current, next, p, current.acc_p + p->second));
                    else {
                        auto obj = unused.front();
                        unused.pop();
                        
                        obj->clear(ID, current, next, p, current.acc_p + p->second);
                        local_stack.push_front(obj);
                    }
                }
            } else
            //if(std::distance(_optimal_pairing->set.begin(), current.fish_it) <= _optimal_pairing->set.size()*0.5) {
                insert_globally(global_stack, current, next, p);
            //} else {
            //    insert_globally(local_stack, current, next, p);
            //}
            
            // if we reached the end of the probs list, insert first element
            // back into local queue
            if(current.prob_it == current.probs.end()) {
                //auto p = current.probs.begin();
                
                insert_locally(current, next, current.probs.begin());
            }
            
        } else {
            blobs.resize(blobs.size() + 1);
            
            for(auto &p : current.probs) {
                if(current.acc_p + p.second > _optimal_pairing->p) {
                    //path.back().fish = node->fish;
                    blobs.back() = p.first.blob;
                    
                    std::vector<Combination> comb_path;
                    comb_path.resize(blobs.size());
                    
                    auto it = _optimal_pairing->set.begin();
                    for(size_t i=0; i<blobs.size(); i++, ++it) {
                        comb_path[i].blob = blobs[i];
                        comb_path[i].fish = it->fish;
                    }
                    
                    std::lock_guard<std::mutex> lock(unused_mutex);
                    if(current.acc_p + p.second > _optimal_pairing->p) {
                        _optimal_pairing->p = current.acc_p + p.second;
                        _optimal_pairing->path = comb_path;
                    }
                }
            }
            
            std::lock_guard<std::mutex> lock(unused_mutex);
            unused.push(&current);
        }
    }*/

template <typename Queue>
static auto private_top(Queue& queue) -> decltype(queue.top()) { return queue.top();}
template <typename Queue>
static auto private_top(Queue& queue) -> decltype(queue.front()) { return queue.front();}

template <typename Queue>
static auto private_pop(Queue& queue) -> decltype(queue.pop()) { return queue.pop();}
template <typename Queue>
static auto private_pop(Queue& queue) -> decltype(queue.pop_front()) { return queue.pop_front();}

template <typename Queue, typename Item>
static
auto private_push(Queue& queue, Item item) -> decltype(queue.push(item)) {
    //if(!queue.empty() && queue.top()->acc_p > item->acc_p)
    //    FormatWarning(queue.top()->acc_p," > ",item->acc_p);
    return queue.push(item);
}
template <typename Queue, typename Item>
static
auto private_push(Queue& queue, Item item) -> decltype(queue.push_front(item)) {
    //if(!queue.empty() && queue.front()->acc_p > item->acc_p)
    //    FormatWarning(queue.front()->acc_p," > ",item->acc_p);
    return queue.push_front(item);
}

void PairingGraph::assign_blob(Stack *stack, blob_index_t blob_index) {
    stack->blobs.push_back(blob_index);
    
    if(blob_index.valid()) {
        stack->blobs_present[(index_t)blob_index] = true;
        
        /*for(auto &i : _blob_edges_idx[blob_index]) {
            if(i > stack->blobs.size()) {
                auto && [start, end] = stack->per_individual[i];
                if(start != end && start->blob_index == blob_index) {
                    do ++start;
                    while(start != end && start->blob_index != -1 && stack->blobs_present[start->blob_index]);
                }
            }
        }*/
    }
    
    
    /*for(size_t i=stack->blobs.size()+1; i<stack->per_individual.size(); ++i) {
        auto && [start, end] = stack->per_individual[i];
    //for(auto && [start, end] : stack->per_individual) {
        if(start != end && start->blob_index == blob_index) {
            do ++start;
            while(start != end && start->blob_index != -1 && stack->blobs_present[start->blob_index]);
        }
    //}
    }*/
}

void PairingGraph::initialize_stack(Stack* ) {
    /*if(per_individual.empty()) {
        per_individual.resize(_optimal_pairing->set.size());
        
        auto it = per_individual.begin();
        auto kit = _optimal_pairing->psets.begin();
        for(; it != per_individual.end(); ++it, ++kit) {
            *it = {kit->begin(), kit->end()};
        }
    }
    
    stack->per_individual = per_individual;*/
}

PairingGraph::Stack* PairingGraph::work_single(queue_t& stack, Stack &current, const bool debug)
{
    UNUSED(debug);
    
    const prob_t hierarchy_best_p = _ordered_max_probs.at(current.blobs.size());
    
    /*if(current.blobs.empty()) {
        hierarchy_best_p = _ordered_max_probs.at(current.blobs.size());
        
    } else {
        prob_t p = 0;
        auto i = current.blobs.size()+1;//std::distance(_optimal_pairing->set.begin(), fit);
        for(; i < current.per_individual.size(); ++i) {
            auto &tup = current.per_individual.at(i);
            auto kit = std::get<0>(tup);
            if(kit != std::get<1>(tup))
                p += kit->p;
        }
        
        hierarchy_best_p = p;
    }*/
    
    //const prob_t hierarchy_best_p = _ordered_max_probs.at(current.blobs.size());
    /**
     
     */
        
#ifndef NDEBUG
    auto local_best_p = current.prob_it != current._probs->end() && (!current.prob_it->cdx.valid() || !current.blobs_present[(Match::index_t)current.prob_it->cdx]) ? current.prob_it->p : 0;
    
    auto print_stack = [&]() {
        auto next = std::next(current.fish_it);
        
        if(current.fish_it != _optimal_pairing->set.end()) {
            size_t degree = current.fish_it->degree;
            for(auto &e : *current._probs) {
                if(e.cdx.valid() && current.blobs_present[(Match::index_t)e.cdx]) {
                    --degree;
                }
            }
            cmn::print("\tidentity:", current.fish_it->fish->identity().ID(),", blob:",current.prob_it != current._probs->end() ? (Match::index_t)current.prob_it->cdx : -2);
            cmn::print("\tdegree:", degree,"/",current.fish_it->degree);
            cmn::print("\tacc_p:",current.acc_p + ((next != _optimal_pairing->set.end()) ? hierarchy_best_p : 0) + local_best_p,"/",_optimal_pairing->p.load()," (",current.acc_p,", ",hierarchy_best_p,", ",local_best_p,")");
            if(_optimal_pairing->p.load() > 0 && current.acc_p + ((next != _optimal_pairing->set.end()) ? hierarchy_best_p : 0) + local_best_p > _optimal_pairing->p.load()) {
                FormatWarning("This is weird.");
            }
        } else
            cmn::print("\tinvalid node");
    };
#endif
    
    //! Select next edge by looking whether the probability that we found is already worse than the current optimum (given the maximum probability that is known to be possible from here on out)
    /// adding at most O(C_u) for the current u\in U
    /// the probability that an improvement can be made is dependent on the ratio between probabilities on the edges and the best case on each layer
    while(current.prob_it != current._probs->end() &&
          (current.acc_p + current.prob_it->p + hierarchy_best_p <= _optimal_pairing->p
           || (current.prob_it->cdx.valid() && current.blobs_present[(index_t)current.prob_it->cdx])))
    {
        ++current.prob_it;
    }
    
    auto next = std::next(current.fish_it);
    
    if(current.prob_it == current._probs->end()) {
        //! the current edge of our individual is not valid anymore, which means that we are finished with our edges and can assume that no viable path has been found (meaning we probably ended our search at the current depth by going through the above "while" loop multiple times)
        /// would be adding O(1)
        
        unused.push(&current);
        
    } else if(next != _optimal_pairing->set.end()) {
        //! we are not out of edges yet, and we are not at leaf depth yet, so we have to continue iterating both
        const auto edge = current.prob_it;
        
        /// adding O(1), per element, with maximum C_u elements (number of edges of current individual), if we have to go through all elements
        /// this is the breadth-second part of the search, extending on the x-axis along the edges
        ++current.prob_it;
        if(current.prob_it != current._probs->end()) {
            private_push(stack, &current);
        }
        else {
            //! we can directly reuse this object, as its not needed anymore
            
            //if(edge->blob_index != -1) {
                //current.blobs_present[edge->blob_index] = true;
                assign_blob(&current, edge->cdx);
            //} else
            //    current.blobs.push_back(edge)
            
            current.acc_p = current.acc_p + edge->p;
            current.fish_it = next;
            ++current._probs;
            current.prob_it = current._probs->begin();
            
            //current.blobs.push_back(edge->blob_index);
            
            return &current;
        }
        
        /// the depth-first part of the algorithm, implemented here by reusing objects from a queue with O(1). so overall the maximum length of one path is O(N).
        /// this goes one level deeper and inserts it as the first element to be evaluated
        Stack *ptr = nullptr;
        
        if(unused.empty()) {
            //! we have to create a new object
            ptr = new Stack(_paired.n_rows(), current, next, edge, current.acc_p + edge->p, std::next(current._probs));
        }
        else {
            ptr = unused.front();
            unused.pop();
            ptr->clear(current, next, edge, current.acc_p + edge->p, std::next(current._probs));
        }
        initialize_stack(ptr);
        assign_blob(ptr, edge->cdx);
        return ptr;
        
    } else {
        //! here we have apparently reached a leaf of our tree, meaning that theres a likelihood that our best probability will be accepted (otherwise we would have reached a bound before)
        blob_index_t blob{ -2 };
        prob_t max_p = _optimal_pairing->p;
        
        /// same as going through edges in a queue, so O(C_u)
        for(auto &edge : *current._probs) {
            if((!edge.cdx.valid() || !current.blobs_present[(index_t)edge.cdx]) 
                && current.acc_p + edge.p > max_p)
            {
                blob = edge.cdx;
                max_p = current.acc_p + edge.p;
#ifndef NDEBUG
                local_best_p = edge.p;
#endif
            }
        }
        
        ++_optimal_pairing->leafs_visited;
        
        //! check if we have found something better, the next part will be executed with moderate probability (only if the probability we found is better than our previous optimum)
        if((index_t)blob != -2) {
            current.blobs.push_back(blob);
            if(blob.valid())
                current.blobs_present[(index_t)blob] = true;
            
            _optimal_pairing->path.clear();
            _optimal_pairing->path.resize(current.blobs.size());
            
            //! generate & save the pairings
            auto it = _optimal_pairing->set.begin();
            for(size_t i=0; i<current.blobs.size(); i++, ++it) {
                if(current.blobs[i].valid())
                    _optimal_pairing->path[i].blob = _paired.col(current.blobs[i]);
                else
                    _optimal_pairing->path[i].blob = pv::bid::invalid;
                _optimal_pairing->path[i].fish = it->fish;
            }
            
            _optimal_pairing->p = max_p;
            ++_optimal_pairing->improvements_made;
        }
                
#ifndef NDEBUG
        if(debug) {
            cmn::print("[Graph::work] Stopping path of length ", current.blobs.size());
            print_stack();
        }
#endif
        
        unused.push(&current);
    }
    
    return nullptr;
}
    
    size_t PairingGraph::traverse(const bool debug)
    {
        //! If there are no individuals, we can escape already:
        if(_optimal_pairing->set.empty())
            return 0;
        
#ifndef NDEBUG
        Timer timer; // only for debug mode
#endif
        
        /**
         Here, we generate a starting point for the traversal. The algorithm works using Stack frames, which are analogous to recursive functions, but with a queue instead.

         A stack frame contains
            - i the current individual, iterating through the individuals array (y-layer)
            - e the current edge, iterating through the individuals adjacent edge array (x-layer)
            - p_acc a probability variable that stores the current accumulated probability values along the path it has taken

         They help performing a depth-first search through the graphs' edges.
         */
        
        //queue_t stack([this](Stack* A, Stack* B) -> bool {
        //    return A->acc_p < B->acc_p;
        //});
        queue_t stack;
        
        // part of the object caching, which saves stack frames and allows us to reuse them
        Stack *ptr;
        if(unused.empty()) {
            ptr = new Stack(_paired.n_rows(), _paired.n_cols(), _optimal_pairing->set.begin(), _optimal_pairing->psets.begin());
        } else {
            ptr = unused.front();
            unused.pop();
            
            ptr->construct(_paired.n_rows(), _paired.n_cols(), _optimal_pairing->set.begin(), _optimal_pairing->psets.begin());
        }
        
        private_push(stack, ptr);
        initialize_stack(ptr);
        
        //! Instead of using a recursive algorithm, we are using a while loop that only stops if no further stack frames are left in a queue. This queue is filled by the work_single method with new branches as soon as they're discovered.
        _optimal_pairing->objects_looked_at = 0;
        //size_t maximal_stack_size = 0;
        Stack *current;
        
        while (!stack.empty()) {
            // retrieve and remove the first element
            current = private_top(stack);
            private_pop(stack);
            
            //maximal_stack_size = max(maximal_stack_size, stack.size());
            
            // walk through stack frames, potentially skipping pushing to the queue if we already know, which frame is to be looked at next
            do ++_optimal_pairing->objects_looked_at;
            while((current = work_single(stack, *current, debug)) != nullptr);
            
            if(_optimal_pairing->objects_looked_at >= 20000000) {
#ifndef NDEBUG
              /*  if(debug) {
                    //! ignore this condition if in debug mode anyway
                    continue;
                }
                
                print("Too many combinations...");
                FILE *f = fopen("failures.txt", "a+b");
                if(f) {
                    size_t counter = 0;
                    for(auto && [blob, edges] : _edges) {
                        counter += edges.size();
                    }
                    
                    std::string str = "video "+SETTING(filename).value<file::Path>().str()+" "+std::to_string(frame())
                    +" "+Meta::toStr(_individuals.size())+"fish "+Meta::toStr(_blobs.size())+"blobs  "+Meta::toStr(counter)+"edges"
                    +"\n";
                    fwrite(str.data(), 1, str.length(), f);
                    fclose(f);
                    
                    auto props = Tracker::properties(frame());
                    auto prev_props = Tracker::properties(frame()-1);
                    if(props && prev_props) {
                        auto tdelta = DurationUS{uint64_t(1000 * 1000 * (props->time - prev_props->time))};
                        auto tdelta_soll = DurationUS{uint64_t(1000 * 1000 * 1.f / FAST_SETTING(frame_rate))};
                        auto str = "Too many combinations in frame %d (tdelta "+tdelta.to_string()+" and should be "+tdelta_soll.to_string()+").";
                        Warning(str.c_str(), frame());
                    } else {
                        print("Too many combinations in frame ",frame(),".");
                    }
                }*/
#endif
                
                throw U_EXCEPTION("Too many combinations (",frame(),", ",_optimal_pairing->objects_looked_at,").");
                break;
            }
        }
        
        
#ifndef NDEBUG
        if(FAST_SETTING(debug)) {
            auto elapsed = timer.elapsed();
            
            std::ofstream file;
            file.open("single.txt", std::ofstream::out | std::ofstream::app);
            file << frame().toStr() << ":\t[" << _optimal_pairing->objects_looked_at <<",\t"<<elapsed*1000<<",\t"<<elapsed*1000/ _optimal_pairing->objects_looked_at <<"" << ",[";
            for(auto &a: _optimal_pairing->path)
            {
                file << "(" << a.fish->identity().ID() << "," << a.blob.toStr() << "," << (a.blob.valid() ? prob(a.fish, a.blob) : 0) << "), ";
            }
            file << "],"<< _optimal_pairing->p <<"],\n";
            file.close();
            cmn::print("(single) ",frame(),": ",_optimal_pairing->objects_looked_at," steps in ",elapsed*1000,"ms => ",elapsed*1000/ _optimal_pairing->objects_looked_at,"ms/step");
        } //else
#endif
        
        return _optimal_pairing->objects_looked_at;
    }
    
    /*void PairingGraph::traverse_threads(size_t threads)
    {
        Timer timer;
        
        if(_optimal_pairing->set.empty())
            return;
        
        gq_t_ stack(Stack::compare);
        
        if(unused.empty()) {
            stack.push(new Stack(_blobs.size(), 0, _optimal_pairing->set.begin()));
        } else {
            auto obj = unused.front();
            unused.pop();
            
            obj->construct(_blobs.size(), 0, _optimal_pairing->set.begin());
            stack.push(obj);//push_front(obj);
        }
        
        if(threads > 1 && !pool)
            pool = new GenericThreadPool(threads, "traverse_threads");
        
        std::atomic<size_t> currently = 0;
        size_t objects = 0;
        
        auto fn_thread = [this, &stack, &currently, &objects]() {
            std::deque<Stack*> local_queue;
            Stack *current;
            
            {
                std::lock_guard<std::mutex> guard(unused_mutex);
                if(!stack.empty()) {
                    local_queue.push_front(stack.top());
                    stack.pop();//pop_front();
                }
            }
            
            while(!local_queue.empty()) {
                current = local_queue.front();
                local_queue.pop_front();
                
                objects++;
                
                assert(current->blobs.size() < _ordered_max_probs.size());
                this->work(stack, local_queue, *current);
            }
            
            currently--;
        };
        
        while (true) {
            {
                std::lock_guard<std::mutex> guard(unused_mutex);
                if(stack.empty()) {
                    if(pool) {
                        while(currently || pool->working()) {
                            unused_mutex.unlock();
                            pool->wait_one();
                            unused_mutex.lock();
                            
                            if(!stack.empty())
                                break;
                        }
                    }
                    
                    if(stack.empty())
                        break;
                }
            }
            
            if(pool) {
                while(currently >= threads)
                    pool->wait_one();
                
                currently++;
                pool->enqueue(fn_thread);
                
            } else
                fn_thread();
        }
        
        if(pool)
            pool->wait();
        
        if(SETTING(debug)) {
            auto elapsed = timer.elapsed();
            
            std::ofstream file;
            file.open("multi.txt", std::ofstream::out | std::ofstream::app);
            file << frame() << ":\t[" << objects<<",\t"<<elapsed*1000<<",\t"<<elapsed*1000/objects<<"" << ",[";
            for(auto &a: _optimal_pairing->path)
            {
                file << "(" << a.fish->identity().ID() << "," << (a.blob ? long_t(a.blob->bounds().pos().x) : LONG_MIN) << "," << (a.blob ? long_t(a.blob->bounds().pos().y) : LONG_MIN) <<"," << (a.blob ? prob(a.fish, a.blob) : 0) << "), ";
            }
            file << "],"<< _optimal_pairing->p <<"],\n";
            file.close();
            print("(multi) ",frame(),": ",objects," steps in ",elapsed*1000,"ms => ",elapsed*1000/objects,"ms/step");
        } //else
    }*/
    
    const PairingGraph::Result& PairingGraph::get_optimal_pairing(bool debug, default_config::matching_mode_t::Class match_mode) {
        static std::mutex _mutex;
        std::lock_guard<std::mutex> guard(_mutex);
        
        if(_optimal_pairing)
            delete _optimal_pairing;
        //if (!_optimal_pairing) {
            _optimal_pairing = new Result;
            
            using namespace default_config;
            struct Benchmark_t {
                double time_acc;
                long_t samples;
                Result *ptr;
                
                bool operator!= (const Benchmark_t& other) const {
                    return time_acc != other.time_acc || samples != other.samples;
                }
                bool operator== (const Benchmark_t& other) const {
                    return time_acc == other.time_acc && samples == other.samples;
                }
            };
            
            static std::mutex mutex;
            static ska::bytell_hash_map<matching_mode_t::Class, Benchmark_t> benchmarks;
            
            if(is_in(match_mode, matching_mode_t::hungarian, matching_mode_t::benchmark))
            {
                Timer timer;
                //HungarianAlgorithm alg;
                std::unordered_map<const Individual*, fish_index_t> individual_index;
                size_t num = 0;
                for (fish_index_t i{ 0 }; i < _paired.n_rows(); ++i) {
                    individual_index[_paired.row(i)] = i;
                    //if(_paired.degree(i) > 0)
                    {
                        ++num;
                    }
                }
                
                //! the number of individuals with edges
                size_t n = num;
                
                //! the number of columns in the distance matrix.
                /// additional n columns representing NULL values
                size_t m = n + (index_t)_paired.n_cols();
                
#ifndef NDEBUG
                if(debug) {
                    cmn::print("frame ",frame()," -- individuals: ",num," blobs: ",_paired.n_cols()," resulting in ",n,"x",m);
                }
#endif
                
                static constexpr prob_t scaling = 10000000.0;
                static_assert(((size_t(scaling) * 2) << 6L) < size_t(std::numeric_limits<Hungarian_t>::max()), "Scaling must be within range.");
                Hungarian_t** dist_matrix = (Hungarian_t**)malloc(n * sizeof(Hungarian_t*));
                
                for(size_t i=0; i<n; ++i) {
                    dist_matrix[i] = (Hungarian_t*)malloc(m * sizeof(Hungarian_t));
                    
                    //! achieving a symmetric matrix by adding additional nodes / edges
                    /// so that the assignment problem is balanced
                    std::fill(dist_matrix[i], dist_matrix[i] + m, Hungarian_t(1));
                    //! each individual gets another object (null) that it can be assigned to, so a perfect matching is guaranteed
                    dist_matrix[i][i + (index_t)_paired.n_cols()] = Hungarian_t(0);
                }
                
                for(auto &&[row, i] : individual_index) {
                    for (auto &e : _paired.edges_for_row(i)) {
                        assert(e.cdx.valid());
                        //auto blob_edges = _blob_edges.at(_blobs[b.blob_index]);
                        if(e.p >= FAST_SETTING(matching_probability_threshold)) {
                            assert((Match::index_t)i < n);
                            assert(e.cdx < _paired.n_cols());
                            dist_matrix[(index_t)i][(index_t)e.cdx] = Hungarian_t(-(scaling * e.p + 0.5)); //! + 0.5 to ensure proper rounding
                        }
                    }
                }
                
                if(!_paired.empty()) {
                    //std::vector<int> assignment;
                    ssize_t** assignment = kuhn_match(dist_matrix, n, m);
                    //alg.Solve(dist_matrix, assignment);
                    //auto str = Meta::toStr(assignment);
                    
                    _optimal_pairing->path.clear();
                    //for(size_t i=0; i<assignment.size(); ++i) {
                    for(size_t i=0; i<n; ++i) {
                        auto j = assignment[i][0];
                        auto idx = assignment[i][1];
                        assert(j < (index_t)_paired.n_rows());
                        if(idx != -1 && idx < (index_t)_paired.n_cols())
                            _optimal_pairing->pairings.push_back({_paired.row(fish_index_t(j)), _paired.col(blob_index_t(idx))});
                        free(assignment[i]);
                    }
                    free(assignment);
                    
                    if(match_mode == matching_mode_t::benchmark) {
                        auto hs = timer.elapsed();
                        std::lock_guard<std::mutex> guard(mutex);
                        auto &val = benchmarks[matching_mode_t::hungarian];
                        val.ptr = _optimal_pairing;
                        _optimal_pairing = new Result;
                        ++val.samples;
                        val.time_acc += hs;
                    }
                }
                
                for(size_t i=0; i<n; ++i) {
                    free(dist_matrix[i]);
                }
                free(dist_matrix);
            }
            
            if(is_in(match_mode, matching_mode_t::approximate, matching_mode_t::benchmark))
            {
                //! approximate matching
                using namespace Match;
                Result *ptr;
                
                if(match_mode == matching_mode_t::approximate)
                    ptr = _optimal_pairing;
                else {
                    std::lock_guard<std::mutex> guard(mutex);
                    auto &val = benchmarks[matching_mode_t::approximate];
                    val.ptr = new Result;
                    ptr = val.ptr;
                }
                
                Timer timer;
                set_of_individuals_t used_blobs;
                
                for(auto && [blob, edges] : _paired.col_edges()) {
                    prob_t max_p = 0;
                    Individual* max_fish = nullptr;
                    
                    for(auto & fish : edges) {
                        if(used_blobs.find(_paired.row(fish)) == used_blobs.end()) {
                            auto p = _paired.probability(fish, _paired.index(blob));
                            if(p > max_p) {
                                max_p = p;
                                max_fish = _paired.row(fish);
                            }
                        }
                    }
                    
                    if(max_fish) {
                        used_blobs.insert(max_fish);
                        ptr->pairings.push_back({max_fish, blob});
                    }
                }
                
                if(match_mode == matching_mode_t::benchmark) {
                    std::lock_guard<std::mutex> guard(mutex);
                    auto &val = benchmarks[matching_mode_t::approximate];
                    auto s = timer.elapsed();
                    val.time_acc += s;
                    ++val.samples;
                }
            }
            
            //! Compares the degree of two nodes
            if(is_in(match_mode, matching_mode_t::tree, matching_mode_t::benchmark))
            {
                Timer timer;
                _optimal_pairing->calculations = 0;
                
                //! save all node indices to a set ordered by the nodes degree
                //! and then by the nodes maximum probability (decreasing), to reduce branching
                /*for (auto &e : _paired.edges()) {
                    size_t degree = 0;
                    for(auto &b : e.second) {
                        assert(b.blob_index != -1);
                        degree += _paired.edges_for_col(b.blob_index).size();
                    }
                    if(degree > 0)
                        _optimal_pairing->set.insert(Node{e.first, _paired.degree(_paired.index(e.first)), _paired.max_prob(e.first)});
                    else
                        print("Individual ",e.first->identity().ID()," is empty");
                }*/
                for(auto row : _paired.rows()) {
                    auto rdx = _paired.index(row);
                    auto degree = _paired.degree(rdx);
                    
                    if(degree > 0)
                        _optimal_pairing->set.insert(Node{row, degree, _paired.max_prob(rdx)});
                    else
                        cmn::print("Individual ",row->identity().ID()," is empty");
                }
                
               /* {
                    size_t i = 0;
                    for(auto &fish : _optimal_pairing->set) {
                        _fish_2_idx[fish.fish] = i++;
                    }
                    
                    for(size_t i=0; i<_blobs.size(); ++i) {
                    
                        auto & edges = _blob_edges.at(_blobs[i]);
                    //for(auto && [blob, edges] : _blob_edges) {
                        std::vector<size_t> indexes;
                        for(auto &fish : edges) {
                            indexes.push_back(_fish_2_idx[fish]);
                        }
                        _blob_edges_idx.push_back(indexes);
                    //}
                    }
                }*/
                
#ifndef NDEBUG
                if(debug)
                    cmn::print("[Graph] Generating edge / probability set for ", _optimal_pairing->set.size()," nodes...");
#endif
                
                //! Ordered max probs is an array containing the maximum achievable probability after a given step (index). In this case, we can generate this array by accumulating the maximum probability for each individual (step by step) as an accum-sum. We start from the back (where the value added is 0) and walk towards the front.
                _ordered_max_probs.clear();
                prob_t p = 0;
                //_ordered_max_probs = {0};
                
                //! psets contains a vector of multisets per individual (index), each multiset is a sorted list of all edges that need to be explored in the given order, including one for the NULL case
                size_t current_set = _optimal_pairing->set.size() - 1;
                _optimal_pairing->psets.resize(_optimal_pairing->set.size());
                
                for(auto it = _optimal_pairing->set.rbegin(); it != _optimal_pairing->set.rend(); ++it, --current_set)
                {
                    _ordered_max_probs.push_back(p);
                    
                    //! Find maximum probability for all edges of the current individual
                    prob_t max_prob = 0;
                    for(auto &k : _paired.edges_for_row(_paired.index(it->fish))) {
                        max_prob = max(max_prob, k.p);
                        
                        //! also collect all edges
                        _optimal_pairing->psets[current_set].insert(k);
                    }
                    
                    //! add the NULL case
                    _optimal_pairing->psets[current_set].insert(PairedProbabilities::Edge(blob_index_t::invalid(), 0));
                    
                    //! save maximum probability also in a map for easy access
                    //_max_probs[it->fish] = max_prob;
                    
                    //! accumulate for next _ordered_max_probs
                    p += max_prob;
                }
                
                //! Reverse order because we walked backwards
                std::reverse(_ordered_max_probs.begin(), _ordered_max_probs.end());
                
                /* ======================================== */
                /** Run the algorithm / traverse the graph  */
                /* ======================================== */
                _optimal_pairing->objects_looked_at = traverse(debug);
                
                //! Collect assignments and save them as pairings:
                for (auto &node : _optimal_pairing->path) {
                    if (node.blob != NULL) {
                        _optimal_pairing->pairings.push_back({node.fish, node.blob});
                    }
                }
                
                if(match_mode == matching_mode_t::benchmark) {
                    auto s = timer.elapsed();
                    std::lock_guard<std::mutex> guard(mutex);
                    auto &val = benchmarks[matching_mode_t::tree];
                    ++val.samples;
                    val.time_acc += s;
                    val.ptr = _optimal_pairing;
                }
            }
            
            if(match_mode == matching_mode_t::benchmark) {
                std::lock_guard<std::mutex> guard(mutex);
                static size_t print_counter = 0;
                decltype(benchmarks) previous_benchmarks;
                
                if((++print_counter) % 100 == 0 && benchmarks != previous_benchmarks) {
                    previous_benchmarks = benchmarks;
                    
                    for(auto && [key, values] : benchmarks) {
                        cmn::print(key.name(),": ",values.time_acc / double(values.samples) * 1000,"ms (",values.samples," samples)");
                    }
                }
                
                ska::bytell_hash_map<matching_mode_t::Class, ska::bytell_hash_map<const Individual*, Blob_t>> assignments;
                for(auto &fish : _paired.rows()) {
                    for(auto && [key, values] : benchmarks)
                        assignments[key][fish] = pv::bid::invalid;
                }
                
                for(auto && [key, values] : benchmarks) {
                    if (values.ptr) {
                        for (auto&& [fish, blob] : values.ptr->pairings)
                            assignments[key][fish] = blob;
                    }
                }
                
                std::set<matching_mode_t::Class> different, agree;
                for(auto && [key, assignment] : assignments) {
                    for(auto && [other, o_assignment] : assignments) {
                        if(key == other)
                            continue;
                        if(assignment != o_assignment)
                            different.insert(other);
                        else
                            agree.insert(other);
                    }
                }
                
                if(!different.empty() && ((agree.find(matching_mode_t::hungarian) != agree.end()) ^ (agree.find(matching_mode_t::tree) != agree.end()))) {
                    FormatWarning("Assignments in frame ", frame()," are not identical (", different,") these agree (",agree,"):");
                    for(auto fish : _paired.rows()) {
                        std::set<matching_mode_t::Class> printed;
                        if(!agree.empty()) {
                            assert(agree.size() > 1);
                            printed.insert(*agree.begin());
                        }
                        
                        for(auto &key_0 : different) {
                            printed.insert(key_0);
                            
                            for(auto &key_1 : different) {
                                if(printed.find(key_1) != printed.end())
                                    continue;
                                
                                if(assignments[key_0][fish] != assignments[key_1][fish])
                                {
                                    auto p0 = _paired.probability(fish, assignments[key_0][fish]);
                                    auto p1 = _paired.probability(fish, assignments[key_1][fish]);
                                    FormatWarning("\tindividual ",fish->identity(),":",
                                        "(", key_0.name(), ")",
                                        assignments[key_0][fish].valid() ?
                                            (assignments[key_0][fish]) : pv::bid::invalid,
                                        " (", p0, ") != "
                                        "(", key_1.name(), ")",
                                        assignments[key_1][fish].valid() ?
                                            (assignments[key_1][fish]) : pv::bid::invalid,
                                        " (",p1,")");
                                }
                            }
                        }
                    }
                }
                
                for(auto && [key, values] : benchmarks) {
                    if(values.ptr) {
                        if(values.ptr != _optimal_pairing) {
                            delete values.ptr;
                            values.ptr = nullptr;
                        }
                    }
                }
            }
        //}
        
        return *_optimal_pairing;
    }
    
    void PairingGraph::print() {
        if(_optimal_pairing)
            delete _optimal_pairing;
        _optimal_pairing = NULL;
        
        get_optimal_pairing(true);
        
        {
            /*std::stringstream ss;
            auto it = _optimal_pairing->set.begin();
            for (auto s : _optimal_pairing->path) {
                //ss << char(char(*it)+'A') << ":";
				ss << std::hex << *it << ":";
                if(s != -1)
                    ss << s;
                else
                    ss << "NULL";
                ss << " ";
                ++it;
            }
            auto s = ss.str();
            print("Best path is: ",s.c_str()," (",_optimal_pairing->p,") (calculations: ",_optimal_pairing->calculations,", permutation elements: ",_optimal_pairing->permutation_elements,")");*/
        }
    }
    
    void PairingGraph::print_summary() {
        if(_optimal_pairing)
            delete _optimal_pairing;
        _optimal_pairing = NULL;
        
#ifndef NDEBUG
        try {
            get_optimal_pairing(true);
        } catch(...) {
            
        }
#endif
    }
    
PairingGraph::PairingGraph(const FrameProperties& props, Frame_t frame, const decltype(_paired)& paired)
    : _frame(frame), _time(props.time), _paired(paired), _optimal_pairing(NULL)
{
}

PairingGraph::~PairingGraph() {
    if(_optimal_pairing)
        delete _optimal_pairing;
    
    while(!unused.empty()) {
        delete unused.front();
        unused.pop();
    }
}
    
    /*cv::Point2f PairingGraph::pos(const Individual* o) const {
     const auto centroid = o->centroid(_frame);
     if(centroid) {
     return centroid->pos(track::PX_AND_SECONDS);
     
     } else {
     return o->estimate_position_at(_frame, _time);
     }
     }
     
     cv::Point2f PairingGraph::pos(const Blob_t b) const {
     return b->pos() + b->center();
     }*/
    
    Match::prob_t PairingGraph::prob(Individual* o, Blob_t b) const {
        /*auto it = _paired.find(o);
        if(it != _paired.end()) {
            auto jt = it->second.find(b);
            if (jt != it->second.end()) {
                return jt->second;
            }
        }
        
        return Match::PairingGraph::prob_t(0);*/
        return _paired.probability(o, b);
    }
    
    bool PairingGraph::connected_to(Individual *o, Blob_t b) const {
        return prob(o, b) > FAST_SETTING(matching_probability_threshold);
    }
    
    /*void PairingGraph::add(Individual *o) {
        if(!_blobs.empty())
            throw U_EXCEPTION("Blobs must be added after individuals.");
        _individuals.push_back(o);
    }
    
    void PairingGraph::add(Blob_t o) {
        std::vector<Individual*> edges;
        
        for (auto && [fish, map] : _paired) {
            auto jt = map.find(o);
            if (jt != map.end()) {
                _edges[fish].push_back(Edge(_blobs.size(), jt->second));
                edges.push_back(fish);
            }
        }
        
        _blob_edges[o] = edges;
        _blobs.push_back(o);
    }*/
}
}
