#include "CPULabeling.h"
#include <misc/GlobalSettings.h>
#include <misc/Timer.h>
#include <misc/checked_casts.h>
#include <misc/pretty.h>
#include <misc/ranges.h>

namespace cmn {
namespace CPULabeling {

//#define DEBUG_MEM

class Brototype;
class DLList;

class Node {
public:
    DLList *parent = nullptr;
    
private:
    size_t _retains = 0;
    
public:
    void retain() {
        ++_retains;
    }
    
    constexpr bool unique() const {
        return _retains == 0;
    }
    
    void release() {
        assert(_retains > 0);
        --_retains;
    }
    
    struct Ref {
        Node* obj = nullptr;
        
        Ref(const Ref& other) : obj(other.obj) {
            if(obj)
                obj->retain();
        }
        
        Ref(Node* obj = nullptr) : obj(obj) {
            if(obj)
                obj->retain();
        }
        
        constexpr Ref(Ref&& other) noexcept {
            obj = other.obj;
            other.obj = nullptr;
        }
        
        ~Ref() {
            release_check();
        }
        
        Ref& operator=(const Ref& other) {
            if(obj != other.obj) {
                release_check();
                obj = other.obj;
                if(obj)
                    obj->retain();
            }
            return *this;
        }
        
        Ref& operator=(Ref&& other) noexcept {
            if(obj != other.obj) {
                release_check();
                obj = other.obj;
                
            } else if(obj) {
                obj->release();
            }
            
            other.obj = nullptr;
            return *this;
        }
        
        constexpr bool operator==(Node* other) const { return other == obj; }
        constexpr bool operator==(const Ref& other) const { return other.obj == obj; }
        constexpr bool operator!=(Node* other) const { return other != obj; }
        constexpr bool operator!=(const Ref& other) const { return other.obj != obj; }
        
        constexpr Node& operator*() const { assert(obj != nullptr); return *obj; }
        constexpr Node* operator->() const { assert(obj != nullptr); return obj; }
        constexpr Node* get() const { return obj; }
        constexpr operator bool() const { return obj != nullptr; }
        
        void release_check();
    };
    
    Node::Ref prev = nullptr;
    Node::Ref next = nullptr;
    std::unique_ptr<Brototype> obj;
    
    static auto& mutex() {
        static std::mutex _mutex;
        return _mutex;
    }
    
#if defined(DEBUG_MEM)
    static auto& created_nodes() {
        static std::unordered_set<void*> _created_nodes;
        return _created_nodes;
    }
#endif
    static std::vector<Node::Ref>& pool() {
        static std::vector<Node::Ref> p;
        return p;
    }
    
    static void move_to_cache(Node::Ref& node);
    static void move_to_cache(Node* node);
    
    Node(std::unique_ptr<Brototype>&& obj, DLList* parent)
        : parent(parent), obj(std::move(obj))
    {
#if defined(DEBUG_MEM)
        std::lock_guard guard(mutex());
        created_nodes().insert(this);
#endif
    }
    
    void init(DLList* parent) {
        this->parent = parent;
    }
    
    void invalidate();
    
    ~Node() {
        invalidate();
#if defined(DEBUG_MEM)
        std::lock_guard guard(mutex());
        created_nodes().erase(this);
#endif
    }
};

struct Source {
    using Pixel = const uchar*;
    using Line = HorizontalLine;
    
    std::vector<Pixel> _pixels;
    std::vector<const Line*> _lines;
    std::vector<Line> _full_lines;
    std::vector<typename Node::Ref> _nodes;
    
    std::vector<coord_t> _row_y;
    std::vector<size_t> _row_offsets;
    
    coord_t lw = 0, lh = 0;
    
    size_t num_rows() const {
        return _row_offsets.size();
    }
    
    bool empty() const {
        return _row_offsets.empty();
    }
    
    void clear() {
        _pixels.clear();
        _lines.clear();
        _full_lines.clear();
        _nodes.clear();
        _row_y.clear();
        _row_offsets.clear();
        lw = lh = 0;
    }
    
    void push_back(const Line& line, const Pixel& px) {
        if(_row_offsets.empty() || line.y > _row_y.back()) {
            _row_y.push_back(line.y);
            _row_offsets.push_back(_lines.size());
        }
        
        _pixels.emplace_back(px);
        _lines.emplace_back((const Line*)_full_lines.size());
        _full_lines.emplace_back(line);
        _nodes.emplace_back(typename Node::Ref());
    }
    
    void push_back(const Line& line) {
        if(_row_offsets.empty() || line.y > _row_y.back()) {
            _row_y.push_back(line.y);
            _row_offsets.push_back(_lines.size());
        }
        
        assert(_pixels.empty()); // needs to be consistent! dont add pixels "sometimes"
        _lines.emplace_back((const Line*)_full_lines.size());
        _full_lines.emplace_back(line);
        _nodes.emplace_back(typename Node::Ref());
    }
    
    //! assumes external ownership of Line ptr -- needs to stay alive during the process
    void push_back(const Line* line, const Pixel& px) {
        if(_row_offsets.empty() || line->y > _row_y.back()) {
            _row_y.push_back(line->y);
            _row_offsets.push_back(_lines.size());
        }
        
        _pixels.emplace_back(px);
        _lines.emplace_back(line);
        _nodes.emplace_back(typename Node::Ref());
    }
    
    void finalize() {
        if(!_full_lines.empty()) {
            const Line* start = _full_lines.data();
            auto it = _lines.data();
            for(; it != _lines.data() + _lines.size(); ++it)
                *it = &_full_lines[(size_t)*it];
        }
    }
    
    class RowRef {
    protected:
        template<typename ValueType>
        class _iterator {
        public:
            typedef _iterator self_type;
            typedef ValueType value_type;
            typedef ValueType& reference;
            typedef ValueType* pointer;
            typedef std::forward_iterator_tag iterator_category;
            typedef int difference_type;
            constexpr _iterator(const value_type& ptr) : ptr_(ptr) { }
            _iterator& operator=(const _iterator& it) = default;
            
            constexpr self_type operator++() {
                assert(ptr_.Lit+1 <= ptr_.obj->line_end);
                ++ptr_.Lit;
                ++ptr_.Nit;
                ++ptr_.Pit;
                return ptr_;
            }
            
            constexpr reference operator*() { assert(ptr_.Lit < ptr_.obj->line_end); return ptr_; }
            constexpr pointer operator->() { assert(ptr_.Lit < ptr_.obj->line_end); return &ptr_; }
            constexpr bool operator==(const self_type& rhs) const { return ptr_ == rhs.ptr_; }
            constexpr bool operator!=(const self_type& rhs) const { return ptr_ != rhs.ptr_; }
        public:
            value_type ptr_;
        };
        
    public:
        struct end_tag {};
        
        Source* _source = nullptr;
        size_t idx;
        coord_t y;
        
        const Pixel* pixel_start = nullptr;
        const Pixel* pixel_end = nullptr;
        Node::Ref* node_start = nullptr;
        Node::Ref* node_end = nullptr;
        const Line**  line_start = nullptr;
        const Line**  line_end = nullptr;
        
    public:
        struct Combined {
            const RowRef* obj;
            decltype(RowRef::line_start) Lit;
            decltype(RowRef::node_start) Nit;
            decltype(RowRef::pixel_start) Pit;
            
            Combined(const RowRef& obj)
                : obj(&obj), Lit(obj.line_start),
                  Nit(obj.node_start),
                  Pit(obj.pixel_start)
            {}
            
            Combined(const RowRef& obj, end_tag)
                : obj(&obj), Lit(obj.line_end),
                  Nit(obj.node_end),
                  Pit(obj.pixel_end)
            {}
            
            bool operator!=(const Combined& other) const {
                return Lit != other.Lit; // assume all are the same
            }
            bool operator==(const Combined& other) const {
                return Lit == other.Lit; // assume all are the same
            }
        };
        
        bool valid() const {
            return _source != nullptr && line_start != line_end;
        }
        
        typedef _iterator<Combined> iterator;
        typedef _iterator<const Combined> const_iterator;
        
        iterator begin() { return _iterator<Combined>(Combined(*this)); }
        iterator end() { return _iterator<Combined>(Combined(*this, end_tag{} )); }
        
        void inc_row() {
            if (!valid())
                return;

            pixel_start = pixel_end;
            node_start = node_end;
            line_start = line_end;
            
            ++idx;
            
            if(idx+1 < _source->_row_y.size()) {
                size_t idx1 = idx+1 >= _source->_row_offsets.size() ? _source->_lines.size() : _source->_row_offsets.at(idx+1);
                y = _source->_row_y[idx];
                
                line_end  = _source->_lines.data()  + idx1;
                node_end  = _source->_nodes.data()  + idx1;
                pixel_end = _source->_pixels.data() + idx1;
                
            } else {
                y = std::numeric_limits<coord_t>::max();
            }
        }
        
        /**
         * Constructs a RowRef with a given y-coordinate (or bigger).
         * @param source Source object that contains the desired row
         * @param y y-coordinate of the desired row
         */
        static RowRef from_index(Source* source, coord_t y) {
            auto it = std::upper_bound(source->_row_y.begin(), source->_row_y.end(), y);
            
            // beyond the value ranges
            if(it == source->_row_y.end()) {
                return RowRef(); // return nullptr
            }
            
            // if the found row is bigger than the desired y, then it will either be because the y we sought is the previous element, or because it does not exist.
            if(it > source->_row_y.begin() && *(it - 1)  == y) {
                --it;
            }
            
            y = *it;
            
            size_t idx = std::distance(source->_row_y.begin(), it);
            size_t idx0 = source->_row_offsets.at(idx);
            size_t idx1 = idx0+1 == source->_row_offsets.size() ? source->_lines.size() : source->_row_offsets.at(idx+1);
            
            return RowRef{
                source,
                
                idx,
                y,
                
                source->_pixels.data() + idx0,
                source->_pixels.data() + idx1,
                source->_nodes.data() + idx0,
                source->_nodes.data() + idx1,
                source->_lines.data() + idx0,
                source->_lines.data() + idx1
            };
        }
    };
    
    /**
     * Initialize source entity based on an OpenCV image. All 0 pixels are interpreted as background. This function extracts all horizontal lines from an image and saves them inside, along with information about where which y-coordinate is located.
     */
    void init(const cv::Mat& image, bool enable_threads) {
        // assuming the number of threads allowed is < 255
        static const uchar sys_max_threads = max(1u, cmn::hardware_concurrency());
        const uchar max_threads = enable_threads && image.cols*image.rows > 100*100 ? sys_max_threads : 1;
        
        assert(image.cols < USHRT_MAX && image.rows < USHRT_MAX);
        assert(image.type() == CV_8UC1);
        assert(image.isContinuous());
        
        clear();
        
        //! local width, height
        lw = image.cols;
        lh = image.rows;
        
        if(max_threads > 1) {
            /**
             * FIND HORIZONTAL LINES IN ORIGINAL IMAGE
             */
            std::vector<Range<int32_t>> thread_ranges;
            thread_ranges.resize(max_threads);
            int32_t step = max(1, ceil(lh / float(max_threads)));
            int32_t end = 0;
            
            for(uchar i=0; i<max_threads; i++) {
                thread_ranges[i].start = end;
                
                end = min(int32_t(lh), end + step);
                thread_ranges[i].end = end;
            }
            
            std::vector<std::thread*> threads;
            std::vector<Source> thread_sources;
            thread_sources.resize(max_threads);
            
            // start threads:
            // TODO: maybe could use a thread pool here) to extract lines in parallel)
            for(uchar i=0; i<max_threads; i++) {
                thread_sources[i].lw = lw;
                thread_sources[i].lh = lh;
                
                threads.push_back(new std::thread(Source::extract_lines, image, &thread_sources[i], thread_ranges[i]));
            }
            
            // now merge lines (partly) in parallel:
            for(uchar i=0; i<max_threads; ++i) {
                threads[i]->join();
                delete threads[i];
                
                // merge arrays:
                // TODO: can reuse vectors, no need to allocate over an over again...
                size_t S = _lines.size();
                _pixels.insert(_pixels.end(), thread_sources[i]._pixels.begin(), thread_sources[i]._pixels.end());
                _full_lines.insert(_full_lines.end(), thread_sources[i]._full_lines.begin(), thread_sources[i]._full_lines.end());
                for(auto &o : thread_sources[i]._lines) {
                    o = (const HorizontalLine*)((size_t)o + S);
                }
                _lines.insert(_lines.end(), thread_sources[i]._lines.begin(), thread_sources[i]._lines.end());
                _nodes.insert(_nodes.end(), thread_sources[i]._nodes.begin(), thread_sources[i]._nodes.end());
                for(auto &o : thread_sources[i]._row_offsets) {
                    o += S;
                }
                _row_offsets.insert(_row_offsets.end(), thread_sources[i]._row_offsets.begin(), thread_sources[i]._row_offsets.end());
                _row_y.insert(_row_y.end(), thread_sources[i]._row_y.begin(), thread_sources[i]._row_y.end());
            }
            
        } else {
            extract_lines(image, this, Range<int32_t>{0, int32_t(lh)});
        }
        
        finalize();
    }
    
    /**
     * Constructs a RowRef struct for a given y-coordinate (see RowRef::from_index).
     */
    RowRef row(const coord_t y) {
        return RowRef::from_index(this, y);
    }
    
    /**
     * Finds all HorizontalLines of POIs within rows range.start to range.end.
     * @param image the source image
     * @param source this is where the converted input data is written to
     * @param rows the y-range for this call
     */
    static void extract_lines(const cv::Mat& image, Source* source, const Range<int32_t>& rows) {
        const coord_t rstart = rows.start;
        const coord_t rend = rows.end;
        Pixel start, end_ptr, ptr;
        
        bool prev;
        Line current;
        
        for(coord_t i = rstart; i < rend; ++i) {
            start = image.ptr(i);
            end_ptr = start + image.cols;
            ptr = start;
            prev = false;
            
            // walk columns
            for(; ptr != end_ptr; ++ptr) {
                if(prev) {
                    // previous is set, but current is not?
                    // (the last hline just ended)
                    if(!*ptr) {
                        assert(ptr >= start);
                        assert(coord_t(ptr - start) >= 1);
                        current.x1 = coord_t(ptr - start) - 1; // -1 because we went past x1 already
                        source->push_back(current, start + current.x0);
                        
                        prev = false;
                    }
                    
                } else if(*ptr) {// !prev && curr (hline starts)
                    coord_t col = coord_t(ptr - start);
                    current.y = i;
                    current.x0 = col;
                    
                    prev = true;
                }
            }
            
            // if prev is set, the last hline ended when the
            // x-dimension ended, so set x1 accordingly
            if(prev) {
                assert(current.x0 <= source->lw - 1);
                current.x1 = source->lw - 1;
                source->push_back(current, start + current.x0);
            }
        }
    }
};

class DLList {
    template<typename ValueType, typename NodeType>
    class _iterator
    {
    public:
        typedef _iterator self_type;
        typedef ValueType value_type;
        typedef ValueType& reference;
        typedef ValueType* pointer;
        typedef std::forward_iterator_tag iterator_category;
        typedef int difference_type;
        constexpr _iterator(pointer ptr) : ptr_(ptr), next(ptr_ && ptr_->next ? ptr_->next.get() : nullptr) { }
        
        constexpr self_type operator++() {
            ptr_ = next;
            next = ptr_ && ptr_->next ? ptr_->next.get() : nullptr;
            return ptr_;
        }
        
        constexpr pointer& operator*() { return ptr_; }
        constexpr pointer& operator->() { return ptr_; }
        constexpr bool operator==(const self_type& rhs) const { return ptr_ == rhs.ptr_; }
        constexpr bool operator!=(const self_type& rhs) const { return ptr_ != rhs.ptr_; }
        
    public:
        pointer ptr_;
        pointer next;
    };
    
public:
    typedef _iterator<Node, Node*> iterator;
    typedef _iterator<const Node, const Node*> const_iterator;
    
    typename Node::Ref _begin = nullptr;
    typename Node::Ref _end = nullptr;
    
    struct Cache {
        //std::mutex _mutex;
        std::vector<typename Node::Ref> _nodes;
        std::vector<std::unique_ptr<Brototype>> _brotos;
        
        std::unique_ptr<Brototype> broto() {
            //std::lock_guard guard(_mutex);
            if(_brotos.empty())
                return nullptr;
            auto ptr = std::move(_brotos.back());
            _brotos.pop_back();
            return ptr;
        }
        
        void node(typename Node::Ref& ptr) {
            //std::lock_guard guard(_mutex);
            if(_nodes.empty())
                return;
            ptr = std::move(_nodes.back());
            _nodes.pop_back();
        }
        
        void receive(typename Node::Ref&& ref) {
            //std::lock_guard guard(_mutex);
            assert(!ref.obj || ref->next == nullptr);
            assert(!ref.obj || ref->prev == nullptr);
            if(ref.obj) ref->parent = nullptr;
            _nodes.emplace_back(std::move(ref));
        }
        void receive(std::unique_ptr<Brototype>&& ptr) {
            //std::lock_guard guard(_mutex);
            _brotos.emplace_back(std::move(ptr));
        }
    };
    
    GETTER_NCONST(Cache, cache)
    GETTER_NCONST(Source, source)
    
public:
    static auto& cmutex() {
        static std::mutex _mutex;
        return _mutex;
    }
    
    static auto& caches() {
        static std::vector<std::unique_ptr<DLList>> _caches;
        return _caches;
    }
    
    static std::unique_ptr<DLList> from_cache() {
        std::lock_guard g(cmutex());
        if(!caches().empty()) {
            auto ptr = std::move(caches().back());
            caches().pop_back();
            return ptr;
        }
        return std::make_unique<DLList>();
    }
    
    static void to_cache(std::unique_ptr<DLList>&& ptr) {
        ptr->clear();
        
        std::lock_guard g(cmutex());
        caches().emplace_back(std::move(ptr));
    }
    
    const typename Node::Ref& insert(const typename Node::Ref& ptr) {
        assert(!ptr->prev && !ptr->next);
        
        if(_end) {
            assert(!_end->next);
            assert(_end != ptr);
            ptr->prev = _end;
            _end->next = ptr;
            _end = ptr;
            
            assert(!_end->next);
            
        } else {
            _begin = ptr;
            _end = _begin;
            assert(!ptr->next);
            assert(!ptr->prev);
        }
        
        if(!ptr->parent)
            ptr->init(this);
        
        return ptr;
    }
    
    void insert(Node::Ref& ptr, std::unique_ptr<Brototype>&& obj) {
        ptr.release_check();
        cache().node(ptr);
        
        bool created = false;
        if(!ptr) {
            ptr = Node::Ref(new Node(std::move(obj), this));
            created = true;
        } else {
            assert(!ptr->next);
            assert(!ptr->prev);
            
            ptr->init(this);
            ptr->obj = std::move(obj);
        }
        
        insert(ptr);
    }
    
    ~DLList() {
        clear();
    }
    
    void clear() {
        _source.clear();
        
        auto ptr = std::move(_begin);
        while (ptr != nullptr) {
            auto next = ptr->next;
            auto p = ptr;
            Node::move_to_cache(p);
            ptr = std::move(next);
        }
        _end = nullptr;
    }
    
    constexpr iterator begin() { return _iterator<Node, Node*>(_begin.get()); }
    constexpr iterator end() { return _iterator<Node, Node*>(nullptr); }
};

using List_t = DLList;
using Node_t = Node;

//! A pair of a blob and a HorizontalLine
class Brototype {
private:
    GETTER_NCONST(std::vector<const uchar*>, pixel_starts)
    GETTER_NCONST(std::vector<const HorizontalLine*>, lines)
    
public:
    static std::unordered_set<Brototype*> brototypes() {
        static std::unordered_set<Brototype*> _brototypes;
        return _brototypes;
    }
    
    Brototype() {
#if defined(DEBUG_MEM)
        std::lock_guard guard(mutex());
        brototypes().insert(this);
#endif
    }
    
    Brototype(const HorizontalLine* line, const uchar* px)
        : _pixel_starts({px}), _lines({line})
    {
#if defined(DEBUG_MEM)
        std::lock_guard guard(mutex());
        brototypes().insert(this);
#endif
    }
    
    ~Brototype() {
#if defined(DEBUG_MEM)
        std::lock_guard guard(mutex());
        brototypes().erase(this);
#endif
    }
    
    static std::mutex& mutex() { static std::mutex m; return m; }
    static void move_to_cache(List_t *list, typename std::unique_ptr<Brototype>& node);
    
    inline bool empty() const {
        return _lines.empty();
    }
    
    inline size_t size() const {
        return _lines.size();
    }
    
    inline void push_back(const HorizontalLine* line, const uchar* px) {
        _lines.emplace_back(line);
        _pixel_starts.emplace_back(px);
    }
    
    void merge_with(const std::unique_ptr<Brototype>& b) {
        auto&        A = pixel_starts();
        auto&       AL = lines();
        
        const auto&  B = b->pixel_starts();
        const auto& BL = b->lines();
        
        if(A.empty()) {
            A .insert(A .end(), B .begin(), B .end());
            AL.insert(AL.end(), BL.begin(), BL.end());
            return;
        }
        
        A .reserve(A .size()+B .size());
        AL.reserve(AL.size()+BL.size());
        
        // special cases
        if(AL.back() < BL.front()) {
            A .insert(A .end(), B .begin(), B .end());
            AL.insert(AL.end(), BL.begin(), BL.end());
            return;
        }
        
        auto it0=A .begin();
        auto Lt0=AL.begin();
        auto it1=B .begin();
        auto Lt1=BL.begin();
        
        for (; it1!=B.end() && it0!=A.end();) {
            if((*Lt1) < (*Lt0)) {
                const auto start = it1;
                const auto Lstart = Lt1;
                do {
                    ++Lt1;
                    ++it1;
                }
                while (Lt1 != BL.end() && it1 != B.end()
                       && (*Lt1) < (*Lt0));
                it0 = A .insert(it0, start , it1) + (it1 - start);
                Lt0 = AL.insert(Lt0, Lstart, Lt1) + (Lt1 - Lstart);
                
            } else {
                ++it0;
                //++Nt0;
                ++Lt0;
            }
        }
        
        if(it1!=B.end()) {
            A.insert(A.end(), it1, B.end());
            AL.insert(AL.end(), Lt1, BL.end());
        }
    }
    
    struct Combined {
        decltype(Brototype::_lines)::iterator Lit;
        decltype(Brototype::_pixel_starts)::iterator Pit;
        
        Combined(Brototype& obj)
            : Lit(obj.lines().begin()),
              Pit(obj._pixel_starts.begin())
        {}
        Combined(Brototype& obj, size_t index)
            : Lit(obj.lines().end()),
              Pit(obj._pixel_starts.end())
        {
            
        }
        
        bool operator!=(const Combined& other) const {
            return Lit != other.Lit; // assume all are the same
        }
        bool operator==(const Combined& other) const {
            return Lit == other.Lit; // assume all are the same
        }
    };
    
    template<typename ValueType>
    class _iterator {
    public:
        typedef _iterator self_type;
        typedef ValueType value_type;
        typedef ValueType& reference;
        typedef ValueType* pointer;
        typedef std::forward_iterator_tag iterator_category;
        typedef int difference_type;
        constexpr _iterator(const value_type& ptr) : ptr_(ptr) { }
        _iterator& operator=(const _iterator& it) = default;
        
        constexpr self_type operator++() {
            ++ptr_.Lit;
            ++ptr_.Pit;
            return ptr_;
        }
        //constexpr self_type operator++(int) { value ++; return *this; }
        constexpr reference operator*() { return ptr_; }
        constexpr pointer operator->() { return &ptr_; }
        constexpr bool operator==(const self_type& rhs) const { return ptr_ == rhs.ptr_; }
        constexpr bool operator!=(const self_type& rhs) const { return ptr_ != rhs.ptr_; }
    public:
        value_type ptr_;
    };
    
    typedef _iterator<Combined> iterator;
    typedef _iterator<const Combined> const_iterator;
    
    iterator begin() { return _iterator<Combined>(Combined(*this)); }
    iterator end() { return _iterator<Combined>(Combined(*this, size())); }
};

void Node::invalidate() {
    Brototype::move_to_cache(parent, obj);
    
    auto p = std::move(parent);
    auto pre = std::move(prev);
    auto nex = std::move(next);
    
    assert(!next);
    assert(!prev);
    
    if(nex) {
        nex->prev = pre;
    } else if(p && p->_end && p->_end == this)
        p->_end = pre;
    
    if(pre) {
        pre->next = nex;
    } else if(p && p->_begin && p->_begin == this)
        p->_begin = nex;
    
    assert(!next);
    assert(!prev);
    //assert(!parent);
}

void Node::move_to_cache(Node::Ref& node) {
    move_to_cache(node.obj);
    node = nullptr;
}

void Node::move_to_cache(Node* node) {
    if(!node)
        return;
    
    if(node->parent)
        node->invalidate();
}

void Brototype::move_to_cache(List_t* list, typename std::unique_ptr<Brototype>& node) {
    if(!node) {
        return;
    }
    
    node->lines().clear();
    node->pixel_starts().clear();
    
    if(list)
        list->cache().receive(std::move(node));
    else
        Warning("No list");
    node = nullptr;
}

void Node::Ref::release_check() {
    if(!obj)
        return;
    
    obj->release();
    if(obj->unique()) {
        if(obj->parent) {
            obj->parent->cache().receive(Ref(obj));
        } else
            delete obj;
    }
    obj = nullptr;
}

/**
 * Merges arrays of HorizontalLines into Blobs.
 * This function expects to be given a previous line and the focal line in
 * the source image (processed, so that the input are actually arrays of HorizontalLines).
 * It then tries to merge the HorizontalLines within the current line with the HL in the
 * previous line, if they are connected, and inserts them into Blobs.
 * It also merges Blobs if necessary.
 *
 * @param previous_vector contains all HorizontalLines from y-1
 * @param current_vector contains all HorizontalLines from y
 * @param blobs output / input vector for the currently existing Blobs
 */
void merge_lines(Source::RowRef &previous_vector,
                 Source::RowRef &current_vector,
                 List_t &blobs)
{
    // walk along both lines, trying to match pairs
    auto current = current_vector.begin();
    auto previous = previous_vector.begin();
    
    while (current != current_vector.end()) {
        if(previous == previous_vector.end()
           || (*current->Lit)->y > (*previous->Lit)->y + 1
           || (*current->Lit)->x1+1 < (*previous->Lit)->x0)
        {
            // case 0: current line ends before previous line starts
            // case 1: lines are more than 1 apart in y direction,
            // case 2: the previous line has ended before the current line started-1
            //
            // -> create new blobs for all elements
            // add new blob, next element in current line
            assert(previous == previous_vector.end()
                   || (*current->Lit)->y > (*previous->Lit)->y + 1
                   || !(*current->Lit)->overlap_x(**previous->Lit));
            
            if(!*current->Nit
               || !(*current->Nit)->parent)
            {
                auto p = blobs.cache().broto();
                if(p)
                    p->push_back(*current->Lit, *current->Pit);
                else
                    p = std::make_unique<Brototype>(*current->Lit, *current->Pit);
                
                blobs.insert(*current->Nit, std::move(p));
                
            }
            
            ++current;
            
        } else if((*current->Lit)->x0 > (*previous->Lit)->x1+1) {
            // case 3: previous line ends before current
            // next element in previous line
            assert(!(*current->Lit)->overlap_x(**previous->Lit));
            ++previous;
            
        } else {
            // case 4: lines intersect
            // merge elements, next in line that ends first
            assert((*current->Lit)->overlap_x(**previous->Lit));
            assert(*previous->Nit);
            
            auto pblob = (*previous->Nit);
            
            if(!*current->Nit) {
                // current line isnt part of a blob yet
                // nit is null!
                pblob->obj->push_back((*current->Lit), *current->Pit);
                *current->Nit = (*previous->Nit);
                
            } else if(*current->Nit != *previous->Nit) {
                // current line is part of a blob
                // (merge blobs)
                assert(*current->Nit != *previous->Nit);
                
                auto p = previous;
                auto c = current;
                
                // copy all lines from the blob in "current"
                // into the blob in "previous"
                if((*p->Nit)->obj->size() <= (*c->Nit)->obj->size()) {
                    std::swap(p, c);
                }
                
                auto cblob = (*c->Nit);
                auto pblob = (*p->Nit);
                
                assert(cblob != pblob);
                
                if(!cblob->obj->empty())
                    pblob->obj->merge_with(cblob->obj);
                
                // replace blob pointers in current_ and previous_vector
                for(auto cit = current_vector.begin(); cit != current_vector.end(); ++cit) {
                    if((*cit->Nit) == cblob) {
                        *cit->Nit = *p->Nit;
                    }
                }
                
                for(auto cit = previous; cit != previous_vector.end(); ++cit) {
                    if((*cit->Nit) == cblob) {
                        *cit->Nit = *p->Nit;
                    }
                }
                
                if(cblob->obj)
                    Brototype::move_to_cache(&blobs, cblob->obj);
                Node_t::move_to_cache(cblob);
            }
            
            /*
             * increase the line pointer of the line that
             * ends first. given the following situation
             *
             * previous  >|---------|  |------|      |----|
             * current         >|-------|   |-----|
             *
             * previous would be increased and the line
             * following it can be merged with current->first
             * as well. (current line connects n previous lines)
             *
             * in the following steps it would increase current
             * and then terminate.
             */
            if((*current->Lit)->x1 <= (*previous->Lit)->x1)
                ++current;
            else
                ++previous;
        }
    }
}

blobs_t run_fast(List_t* blobs)
{
    blobs_t result;
    auto& source = blobs->source();
    
    if(source.empty())
        return {};
    
    /**
     * SORT HORIZONTAL LINES INTO BLOBS
     * tested cases:
     *      - empty image
     *      - only one line
     *      - only two lines (one HorizontalLine in each of the two y-arrays)
     */
        
    // iterators for current row (y-coordinate in the original image)
    // and for the previous row (the last one above current that contains objects)
    auto current_row = source.row(0);
    auto previous_row = source.row(0);
    
    // create blobs for the current line if its not empty
    if(current_row.valid()) {
        auto start = current_row.begin();
        auto end = current_row.end();
        for(auto it = start; it != end; ++it) {
            auto &[o,l,n,p] = *it;
            auto bob = blobs->cache().broto();
            if(!bob)
                bob = std::make_unique<Brototype>(*l, *p);
            else
                bob->push_back(*l, *p);
            
            blobs->insert(*n, std::move(bob));
        }
    }
    
    // previous_row remains the same, but current_row has to go to the next one (all blobs for the first row have already been created):
    current_row.inc_row();
    
    // loop until the current_row iterator reaches the end of all arrays
    while (previous_row.valid()) {
        merge_lines(previous_row, current_row, *blobs);
        
        previous_row = current_row;
        current_row.inc_row();
    }
    
    /**
     * FILTER BLOBS FOR SIZE, transform them into proper format
     */
    result.reserve(std::distance(blobs->begin(), blobs->end()));

    for(auto it=blobs->begin(); it != blobs->end(); ++it) {
        if(it->obj && !it->obj->empty()) {
            result.emplace_back(std::make_shared<std::vector<HorizontalLine>>(), std::make_shared<std::vector<uchar>>());
            auto &lines = std::get<0>(result.back());
            auto &pixels = std::get<1>(result.back());
            
            ptr_safe_t L = 0;
            for(auto & [l, px] : *it->obj)
                L += ptr_safe_t((*l)->x1) - ptr_safe_t((*l)->x0) + ptr_safe_t(1);
            
            pixels->resize(L);
            lines->resize(it->obj->lines().size());
            
#ifndef NDEBUG
            coord_t y = 0;
#endif
            auto current = lines->data();
            auto pixel = pixels->data();
            for(auto & [l, px] : *it->obj) {
                assert((*l)->y >= y);
                *current++ = **l; // assign **l to *current; inc current
                
                if(pixels) {
                    assert(*px);
                    auto start = *px;
                    auto end = start + (ptr_safe_t((*l)->x1) - ptr_safe_t((*l)->x0) + ptr_safe_t(1));
                    
                    pixel = std::copy(start, end, pixel);
                }
                
#ifndef NDEBUG
                y = (*l)->y;
#endif
            }
            
            Node_t::move_to_cache(*it);
        }
    }
    
#if defined(DEBUG_MEM)
    if(!Brototype::brototypes().empty())
        Warning("Still have %lu brototypes in memory", Brototype::brototypes().size());
    
    source._nodes.clear();
    
    {
        Node_t::pool().clear();
        
        std::lock_guard guard(Node_t::mutex());
        if(!Node_t::created_nodes().empty())
            Warning("Still have %lu nodes in memory", Node_t::created_nodes().size());
    }
    
    Brototype::brototypes().clear();
#endif
    
    return result;
}

// called by user
blobs_t run(const cv::Mat &image, bool enable_threads) {
    auto list = List_t::from_cache();
    list->source().init(image, enable_threads);
    
    blobs_t results = run_fast(list.get());
    List_t::to_cache(std::move(list));
    return results;
}

// called by user
blobs_t run(const std::vector<HorizontalLine>& lines, const std::vector<uchar>& pixels)
{
    if(lines.empty())
        return {};
    
    auto px = pixels.data();
    auto list = List_t::from_cache();
    
    auto start = lines.begin();
    auto end = lines.end();
    
    coord_t y = 0;
    
    for (auto it = start; it != end; ++it) {
        list->source().push_back(&(*it), px);
        
        if(px)
            px += ptr_safe_t(it->x1) - ptr_safe_t(it->x0) + ptr_safe_t(1);
    }
    
    blobs_t results = run_fast(list.get());
    List_t::to_cache(std::move(list));
    return results;
}

}
}
