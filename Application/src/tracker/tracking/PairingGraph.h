#ifndef _PAIRING_GRAPH_H
#define _PAIRING_GRAPH_H

#include <misc/defines.h>
#include <misc/PVBlob.h>
#include <tracker/misc/default_config.h>
#include <misc/ranges.h>

//! Can transport Individual/Blob
namespace track {
class Individual;

namespace Match
{
    using prob_t = double;
    using Blob_t = const pv::BlobPtr*;
    template<typename K, typename V>
    using pairing_map_t = robin_hood::unordered_map<K, V>;

    class PairedProbabilities {
    public:
        using row_t = std::vector<Individual*>;
        using col_t = std::vector<Blob_t>;
        
        struct Edge {
            long_t cdx;
            prob_t p;
            
            explicit Edge(long_t cdx = -1, prob_t p = -1)
                : cdx(cdx), p(p)
            {}
            operator std::string() const {
                return cdx >= 0
                    ? std::to_string(cdx)+"["+std::to_string(p)+"]"
                    : "null";
            }
            bool operator==(const Edge& other) const {
                return other.cdx == cdx;
            }
            
            bool operator==(long_t cdx) const {
                return this->cdx == cdx;
            }
            bool operator<(const Edge& other) const;
        };
        
    protected:
        GETTER(row_t, rows)
        GETTER(col_t, cols)
        
        size_t _num_rows;
        size_t _num_cols;
        
        std::vector<size_t> _offsets;
        std::vector<size_t> _degree;
        std::vector<prob_t> _row_max_probs;
        GETTER(std::vector<Edge>, probabilities) //! size is individuals + edges per individual
        
        pairing_map_t<row_t::value_type, size_t> _row_index;
        pairing_map_t<col_t::value_type, size_t> _col_index;
        
        pairing_map_t<col_t::value_type, std::vector<size_t>> _col_edges;
        //std::vector<std::vector<size_t>> _blob_edges_idx;
        //std::unordered_map<row_t::value_type, size_t> _fish_2_idx;
        
    public:
        PairedProbabilities();
        
        size_t add(row_t::value_type, const pairing_map_t<col_t::value_type, prob_t>&);
        void erase(row_t::value_type);
        void erase(col_t::value_type);
        
        size_t n_rows() const { return _num_rows; }
        size_t n_cols() const { return _num_cols; }
        
        void init();
        
        row_t::value_type row(size_t rdx) const;
        col_t::value_type col(size_t cdx) const;
        
        size_t index(col_t::value_type) const;
        size_t index(row_t::value_type) const;
        
        bool has(row_t::value_type) const;
        bool has(col_t::value_type) const;
        
        //! return -1 if invalid assignment
        prob_t probability(row_t::value_type, col_t::value_type) const;
        prob_t probability(size_t row, size_t col) const;
        
        prob_t max_prob(size_t) const;
        
        //const decltype(_col_edges)::mapped_type& edges_for_col(col_t::value_type) const;
        const decltype(_col_edges)::mapped_type& edges_for_col(size_t) const;
        const decltype(_col_edges)& col_edges() const;
        
        std::vector<Edge> edges_for_row(size_t) const;
        
        size_t degree(size_t) const;
        
        bool empty() const; // no elements in the graph
    private:
        size_t add(col_t::value_type);
    };

    class PairingGraph {
    public:
        //! this is the "queue" for permutations from this node on
        typedef PairedProbabilities::Edge _value_t;
        //typedef std::multiset<_value_t, std::function<bool(const _value_t&, const _value_t&)>> pset;
        typedef std::multiset<_value_t> pset;
        
        typedef std::vector<pset> psets_t;
        typedef psets_t::const_iterator pset_ptr_t;
        
        struct Node {
            Individual* fish;
            size_t degree;
            prob_t max_prob;
            
            bool operator==(const Node& other) const {
                return other.fish == fish;
            }
            bool operator<(const Node& other) const;
        };
        
        struct Combination {
            Individual* fish;
            Blob_t blob;
        };
        
        //typedef std::unordered_map<Individual*, std::vector<_value_t>> EdgeMap;
        typedef std::multiset<Node> multiset;
        
        struct Result {
            //! multiset with node indices ordered by degree
            multiset set;
            
            psets_t psets;
            
            //! Individuals and Blobs paired in the optimal way.
            //  Does not necessarily contain all Individuals/Blobs
            //  (as some might have improved the result by not being used)
            std::vector<std::pair<Individual*, Blob_t>> pairings;
            
            //! Optimal path down the tree (indicies of nodes)
            std::vector<Combination> path;
            
            //! Overall probability that this is the right choice.
            std::atomic<prob_t> p;
            
            //! Overall comparisons performed.
            size_t calculations;
            
            //! Permutation elements calculated.
            size_t permutation_elements;
            
            size_t objects_looked_at;
            size_t improvements_made, leafs_visited;
            
            Result()//const std::function<bool(const Node&, const Node&)>& c)
                : p(0.0), calculations(0), permutation_elements(0), objects_looked_at(0), improvements_made(0), leafs_visited(0)
            {}
        };
        
    protected:
        GETTER(Frame_t, frame)
        GETTER(float, time)
        PairedProbabilities _paired;
        
        std::vector<prob_t> _ordered_max_probs;
        GETTER_PTR(Result*, optimal_pairing)
        
        //GETTER(EdgeMap, edges)
        
    public:
        PairingGraph(Frame_t frame, const decltype(_paired)& paired);
        ~PairingGraph();
        
        static void prepare_shutdown();
        
       // void add(Individual* o);
        //void add(pv::Blob* o);
        void print();
        void print_summary();
        
        //cv::Point2f pos(const Individual*) const;
        //cv::Point2f pos(const Blob*) const;
        
        const Result& get_optimal_pairing(bool print = false, default_config::matching_mode_t::Class mode = default_config::matching_mode_t::automatic);
        
    public:
        //psets_t _psets;
        
        struct Stack {
            multiset::const_iterator fish_it;
            prob_t acc_p;
            //pset probs;
            pset_ptr_t _probs;
            //bool init;
            pset::const_iterator prob_it;
            
            //std::vector<std::tuple<pset::const_iterator, pset::const_iterator>> per_individual;
            std::vector<long_t> blobs;
            std::vector<bool> blobs_present;
            
            bool operator==(const Stack& other) const {
                return other.acc_p == acc_p /*&& other.probs == probs*/ && other.fish_it == fish_it && other._probs == _probs;
            }
            
        private:
            Stack()
            {
                
            }
            
        public:
            Stack(size_t number_individuals, size_t number_blobs, long_t ID, const multiset::const_iterator& node, pset_ptr_t);
            Stack(size_t number_individuals, size_t number_blobs, long_t ID, const Stack& parent, const multiset::const_iterator& next, pset::const_iterator combination, prob_t p, pset_ptr_t);
            
            void construct(size_t number_individuals, size_t number_blobs, long_t ID, const multiset::const_iterator& node, pset_ptr_t);
            
            void clear(size_t number_individuals, size_t number_blobs, long_t ID, const Stack& parent, const multiset::const_iterator& next, pset::const_iterator combination, prob_t p, pset_ptr_t);
            
            void clear_reuse(long_t ID, const Stack&, const multiset::const_iterator& next, pset::const_iterator p, prob_t new_p, pset_ptr_t);
            static long_t idx(const multiset& fish_set, const multiset::const_iterator& fish_pos);
        };
        
        typedef std::priority_queue<Stack*, std::vector<Stack*>, std::function<bool(const Stack*, const Stack*)>> gq_t_;
        
        class Cache {
        public:
            std::queue<Stack*> unused;
            std::mutex mutex;
            
            Stack* get() {
                std::lock_guard<std::mutex> lock(mutex);
                if(unused.empty()) {
                    
                }
                return NULL;
            }
        };
        
        prob_t prob(Individual*, Blob_t) const;
        bool connected_to(Individual *o, Blob_t b) const;
        
    public:
        std::queue<Stack*> unused;
        
    private:
        //void work(gq_t_&, std::deque<Stack*>&, Stack&);
        //std::vector<std::tuple<pset::const_iterator, pset::const_iterator>> per_individual;
        
        void assign_blob(Stack*, long_t blob_index);
        void initialize_stack(Stack*);
        //typedef std::priority_queue<Stack*, std::vector<Stack*>, std::function<bool(Stack*,Stack*)>> queue_t;
        typedef std::deque<Stack*> queue_t;
        Stack* work_single(queue_t&, Stack&, const bool debug);
        //void traverse_threads(size_t threads);
        size_t traverse(const bool debug);
    };
}
}

#endif
