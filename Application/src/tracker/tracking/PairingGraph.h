#ifndef _PAIRING_GRAPH_H
#define _PAIRING_GRAPH_H

#include <commons.pc.h>
#include <misc/PVBlob.h>
#include <tracker/misc/default_config.h>
#include <misc/ranges.h>
#include <tracking/MotionRecord.h>
#include <misc/idx_t.h>
#include <misc/TrackingSettings.h>

//! Can transport Individual/Blob
namespace track {
class Individual;

namespace Match
{
    //using blob_index_t = long_t;
    //using fish_index_t = long_t;
    using index_t = int32_t;

    struct fish_index_t {
        static constexpr index_t _invalid = -1;
        static constexpr fish_index_t invalid() {
            return fish_index_t(_invalid);
        }
        index_t index;
        constexpr explicit operator index_t() const {
            return index;
        }

        constexpr fish_index_t() = default;
        explicit constexpr fish_index_t(index_t index) : index(index) {}
        constexpr bool operator<(fish_index_t other) const { return index < other.index; }
        constexpr bool operator>(fish_index_t other) const { return index > other.index; }
        constexpr bool operator<=(fish_index_t other) const { return index <= other.index; }
        constexpr bool operator>=(fish_index_t other) const { return index >= other.index; }
        constexpr bool operator==(fish_index_t other) const { return index == other.index; }
        constexpr fish_index_t& operator++() {
            ++index;
            return *this;
        }
        constexpr fish_index_t& operator--() {
            --index;
            return *this;
        }
        constexpr bool valid() const { return index >= 0; }
        std::string toStr() const { return std::to_string(index); }
    };

    struct blob_index_t {
        static constexpr index_t _invalid = -1;
        static constexpr blob_index_t invalid() {
            return blob_index_t(_invalid);
        }
        index_t index;
        constexpr explicit operator index_t() const {
            return index;
        }

        constexpr blob_index_t() = default;
        explicit constexpr blob_index_t(index_t index) : index(index) {}
        constexpr bool operator<(blob_index_t other) const { return index < other.index; }
        constexpr bool operator>(blob_index_t other) const { return index > other.index; }
        constexpr bool operator<=(blob_index_t other) const { return index <= other.index; }
        constexpr bool operator>=(blob_index_t other) const { return index >= other.index; }
        constexpr bool operator==(blob_index_t other) const { return index == other.index; }
        constexpr blob_index_t& operator++() {
            ++index;
            return *this;
        }
        constexpr bool valid() const { return index >= 0; }
        std::string toStr() const { return std::to_string(index); }
    };
}
}

namespace std
{
template <>
struct hash<track::Match::blob_index_t>
{
    size_t operator()(const track::Match::blob_index_t& k) const noexcept
    {
        //return robin_hood::hash<track::Match::index_t>{}((track::Match::index_t)k);
        return std::hash<track::Match::index_t>{}((track::Match::index_t)k);
    }
};

template <>
struct hash<track::Match::fish_index_t>
{
    size_t operator()(const track::Match::fish_index_t& k) const noexcept
    {
        //return robin_hood::hash<track::Match::index_t>{}((track::Match::index_t)k);
        return std::hash<track::Match::index_t>{}((track::Match::index_t)k);
    }
};
}

namespace track {
namespace Match {
    template<typename K, typename V>
    using pairing_map_t = robin_hood::unordered_flat_map<K, V>;

    class PairedProbabilities {
    public:
        using row_t = std::vector<Fish_t>;
        using col_t = std::vector<Blob_t>;
        
        struct Edge {
            blob_index_t cdx;
            prob_t p;
            
            explicit Edge(blob_index_t cdx = blob_index_t::invalid(), prob_t p = -1)
                : cdx(cdx), p(p)
            {}
            operator std::string() const {
                return cdx.valid()
                    ? std::to_string((Match::index_t)cdx)+"["+std::to_string(p)+"]"
                    : "null";
            }
            bool operator==(const Edge& other) const {
                return other.cdx == cdx;
            }
            
            bool operator==(blob_index_t cdx) const {
                return this->cdx == cdx;
            }
            bool operator<(const Edge& other) const;
        };
        
    protected:
        GETTER(row_t, rows);
        GETTER(col_t, cols);
        
        fish_index_t _num_rows{0};
        blob_index_t _num_cols{0};
        
        std::vector<size_t> _offsets;
        std::vector<size_t> _degree;
        std::vector<prob_t> _row_max_probs;
        GETTER(std::vector<Edge>, probabilities); //! size is individuals + edges per individual
        
        pairing_map_t<row_t::value_type, fish_index_t> _row_index;
        pairing_map_t<col_t::value_type, blob_index_t> _col_index;
        
        pairing_map_t<col_t::value_type, std::vector<fish_index_t>> _col_edges;
        //std::vector<std::vector<size_t>> _blob_edges_idx;
        //std::unordered_map<row_t::value_type, size_t> _fish_2_idx;
        
    public:
        const decltype(_row_index)& row_indexes() const { return _row_index;  }
        void clear() {
            _row_index.clear();
            _col_index.clear();
            _col_edges.clear();
            _offsets.clear();
            _degree.clear();
            _row_max_probs.clear();
            _probabilities.clear();
            _num_rows = fish_index_t(0);
            _num_cols = blob_index_t(0);
            _rows.clear();
            _cols.clear();
        }
        void reserve(size_t N) {
            if(_probabilities.capacity() < N)
                _probabilities.reserve(N);
        }
        
        using ordered_assign_map_t = robin_hood::unordered_node_map<col_t::value_type, prob_t>;
        fish_index_t add(row_t::value_type, const ordered_assign_map_t&);
        void erase(row_t::value_type);
        void erase(col_t::value_type);
        
        auto n_rows() const { return _num_rows; }
        auto n_cols() const { return _num_cols; }
        
        void init();
        
        row_t::value_type row(fish_index_t rdx) const;
        col_t::value_type col(blob_index_t cdx) const;
        
        blob_index_t index(col_t::value_type) const;
        fish_index_t index(row_t::value_type) const;
        
        bool has(row_t::value_type) const;
        bool has(col_t::value_type) const;
        
        //! return -1 if invalid assignment
        prob_t probability(row_t::value_type, col_t::value_type) const;
        prob_t probability(fish_index_t row, blob_index_t col) const;
        
        prob_t max_prob(fish_index_t) const;
        
        //const decltype(_col_edges)::mapped_type& edges_for_col(col_t::value_type) const;
        const decltype(_col_edges)::mapped_type& edges_for_col(blob_index_t) const;
        const decltype(_col_edges)& col_edges() const;
        
        std::span<const Edge> edges_for_row(fish_index_t) const;
        
        size_t degree(fish_index_t) const;
        
        bool empty() const; // no elements in the graph
        std::string toStr() const;
        static std::string class_name() { return "PairedProbabilities"; }
        
        PairedProbabilities() noexcept = default;
        PairedProbabilities(const PairedProbabilities&) = delete;
        PairedProbabilities(PairedProbabilities&&) noexcept = default;
        PairedProbabilities& operator=(const PairedProbabilities&) noexcept = delete;
        PairedProbabilities& operator=(PairedProbabilities&&) noexcept = default;
        
    private:
        blob_index_t add(col_t::value_type);
    };

    class PairingGraph {
        std::mutex _mutex;
        
    public:
        using ordered_map_t = robin_hood::unordered_node_map<Blob_t, Fish_t>;
        //! this is the "queue" for permutations from this node on
        typedef PairedProbabilities::Edge _value_t;
        //typedef std::multiset<_value_t, std::function<bool(const _value_t&, const _value_t&)>> pset;
        typedef std::multiset<_value_t> pset;
        
        typedef std::vector<pset> psets_t;
        typedef psets_t::const_iterator pset_ptr_t;
        
        struct Node {
            Fish_t fish;
            size_t degree;
            prob_t max_prob;
            
            bool operator==(const Node& other) const {
                return other.fish == fish;
            }
            bool operator<(const Node& other) const;
        };
        
        struct Combination {
            Fish_t fish;
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
            ordered_map_t pairings;
            
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
        GETTER(Frame_t, frame);
        GETTER(float, time);
        GETTER_NCONST(PairedProbabilities, paired);
        
        std::vector<prob_t> _ordered_max_probs;
        GETTER_PTR(Result*, optimal_pairing);
        
        //GETTER(EdgeMap, edges);
        
    public:
        PairingGraph(const FrameProperties& props, Frame_t frame, PairedProbabilities&& paired);
        ~PairingGraph();
        
        static void prepare_shutdown();
        
       // void add(Individual* o);
        //void add(pv::Blob* o);
        void Print();
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
            std::vector<blob_index_t> blobs;
            std::vector<bool> blobs_present;
            
            bool operator==(const Stack& other) const {
                return other.acc_p == acc_p /*&& other.probs == probs*/ && other.fish_it == fish_it && other._probs == _probs;
            }
            
        private:
            Stack() = default;
            
        public:
            Stack(fish_index_t number_individuals, blob_index_t number_blobs, const multiset::const_iterator& node, pset_ptr_t);
            Stack(fish_index_t number_individuals, const Stack& parent, const multiset::const_iterator& next, pset::const_iterator combination, prob_t p, pset_ptr_t);
            
            void construct(fish_index_t number_individuals, blob_index_t number_blobs, const multiset::const_iterator& node, pset_ptr_t);
            
            void clear(const Stack& parent, const multiset::const_iterator& next, pset::const_iterator combination, prob_t p, pset_ptr_t);
            
            void clear_reuse(const Stack&, const multiset::const_iterator& next, pset::const_iterator p, prob_t new_p, pset_ptr_t);
            static long_t idx(const multiset& fish_set, const multiset::const_iterator& fish_pos);
        };
        
        typedef std::priority_queue<Stack*, std::vector<Stack*>, std::function<bool(const Stack*, const Stack*)>> gq_t_;
        
        prob_t prob(Fish_t, Blob_t) const;
        bool connected_to(Fish_t, Blob_t b) const;
        
    public:
        std::queue<Stack*> unused;
        
    private:
        //void work(gq_t_&, std::deque<Stack*>&, Stack&);
        //std::vector<std::tuple<pset::const_iterator, pset::const_iterator>> per_individual;
        
        void assign_blob(Stack*, blob_index_t blob_index);
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
