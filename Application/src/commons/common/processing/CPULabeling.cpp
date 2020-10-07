#include "CPULabeling.h"
#include <misc/GlobalSettings.h>
#include <misc/Timer.h>

namespace cmn {
    namespace CPULabeling {
        class Brototype;
    
    template<typename T>
    class DLList {
    public:
        class Node {
        public:
            DLList *parent;
            typedef std::shared_ptr<Node> Ptr;
            Ptr prev, next;
            T obj;
            
            Node(const T& obj, DLList* parent) : parent(parent), prev(nullptr), next(nullptr), obj(obj) {
                
            }
            
            void init(DLList* parent) {
                this->parent = parent;
            }
            
            void invalidate() {
                if(prev) prev->next = next;
                if(next) next->prev = prev;
                
                if(parent && parent->_end && parent->_end.get() == this)
                    parent->_end = prev;
                if(parent && parent->_begin && parent->_begin.get() == this)
                    parent->_begin = next;
                
                prev = next = nullptr;
                
                parent = nullptr;
            }
            
            ~Node() {
                invalidate();
            }
        };
        
        typename Node::Ptr _begin, _end;
        
        DLList() : _begin(nullptr), _end(nullptr) {}
        void clear() {
            _begin = _end = nullptr;
        }
        
        typename Node::Ptr insert(typename Node::Ptr& ptr) {
            if(_end) {
                assert(!_end->next);
                _end->next = ptr;
                ptr->prev = _end;
                _end = ptr;
            } else {
                _begin = _end = ptr;
            }
            if(!ptr->parent)
                ptr->init(this);
            return ptr;
        }
        typename Node::Ptr insert(T obj) {
            typename Node::Ptr ptr = std::make_shared<Node>(obj, this);
            return insert(ptr);
        }
        
        ~DLList() {}
        
        
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
            //constexpr self_type operator++(int) { value ++; return *this; }
            constexpr reference operator*() { return *ptr_; }
            constexpr pointer operator->() { return ptr_; }
            constexpr bool operator==(const self_type& rhs) const { return ptr_ == rhs.ptr_; }
            constexpr bool operator!=(const self_type& rhs) const { return ptr_ != rhs.ptr_; }
            //constexpr bool operator<(const self_type& rhs) { return ptr_ < rhs.ptr_; }
            //constexpr bool operator>(const self_type& rhs) { return ptr_ > rhs.ptr_; }
        public:
            pointer ptr_;
            pointer next;
        };
        
        typedef _iterator<Node, typename Node::Ptr> iterator;
        typedef _iterator<const Node, const typename Node::Ptr> const_iterator;
        
        constexpr iterator begin() { return _iterator<Node, typename Node::Ptr>(_begin.get()); }
        constexpr iterator end() { return _iterator<Node, typename Node::Ptr>(nullptr); }
    };
        
    struct Pair {
        DLList<std::shared_ptr<Brototype>>::Node::Ptr blob;
        
        //std::vector<uchar> pixels;
        const uchar* pixel_start;
        //uchar *pixels0, *pixels1;
        HorizontalLine line;
        
        Pair(ushort y=0, ushort x0=0, ushort x1=0)
            : pixel_start(nullptr), line(y, x0, x1)
        {}
        Pair(const HorizontalLine& l)
            : pixel_start(nullptr), line(l)
        {}
    };
    
        //! A vector of pairs of Blobs and HorizontalLines is what will be collected
        //  by each thread
        typedef std::vector<Pair> BlobNLine;
        
        //! A pair of a blob and a HorizontalLine
        class Brototype {
        private:
            //GETTER(std::shared_ptr<std::vector<HorizontalLine>>, lines)
            //std::shared_ptr<std::vector<uchar>> _pixels;
            GETTER(std::unique_ptr<std::vector<Pair*>>, lines)
            
        public:
            Brototype()
            {
                
            }
            
            Brototype(Pair* line)
                : _lines(std::make_unique<std::vector<Pair*>>(std::vector<Pair*>{line}))
            {
            }
            
            inline bool empty() const {
                return _lines->empty();
            }
            
            inline size_t size() const {
                return _lines->size();
            }
            
            inline void push_back(Pair* line) {
                
                assert(_lines->empty() || line->line.y >= _lines->back()->line.y);
                _lines->push_back(line);
            }
        };
        
        //template<typename T>
        void merge_sorted(std::shared_ptr<Brototype>& a, const std::shared_ptr<Brototype>& b) {
            assert(a->lines() && b->lines());
            
            auto& A = *a->lines();
            const auto& B = *b->lines();
            
            if(A.empty()) {
                A.insert(A.end(), B.begin(), B.end());
                return;
            }
            
            A.reserve(A.size()+B.size());
            
            // special cases
            if(A.back()->line < B.front()->line) {
                A.insert(A.end(), B.begin(), B.end());
                return;
            }
            
            auto it0=A.begin();
            auto it1=B.begin();
            
            for (; it1!=B.end() && it0!=A.end();) {
                if((*it1)->line < (*it0)->line) {
                    const auto start = it1;
                    while (++it1 != B.end() && (*it1)->line < (*it0)->line);
                    it0 = A.insert(it0, start, it1) + (it1 - start);
                    
                } else
                    ++it0;
            }
            
            if(it1!=B.end())
                A.insert(A.end(), it1, B.end());
        }
        
        /**
         * Finds all HorizontalLines of POIs within rows range.start to range.end.
         * @param lines a pointer to the output array (assumed to be empty)
         * @param range the y-range for this call
         * @param lw the width of a row
         * @param image the source image
         */
        void extract_lines(std::vector<BlobNLine>* lines, const Range<uint32_t>& range, const ushort lw, const cv::Mat& image, Method method)
        {
            BlobNLine current;
            const ushort end = range.end;
            bool prev;
            const uchar *start, *end_ptr, *ptr;
            
            // walk rows
            for(ushort row = range.start; row < end; row++) {
                start = image.ptr(row);
                end_ptr = start + lw;
                ptr = start;
                prev = false;
                
                // walk columns
                for(; ptr != end_ptr; ++ptr) {
                    if(prev) {
                        // previous is set, but current is not?
                        // (the last hline just ended)
                        if(!*ptr) {
                            ushort col = ushort(ptr - start);
                            current.back().line.x1 = col - 1;
                            if(method == WITH_PIXELS)
                                current.back().pixel_start = start + current.back().line.x0;
                            prev = false;
                        }
                        
                    } else if(*ptr) {// !prev && curr (hline starts)
                        ushort col = ushort(ptr - start);
                        current.push_back(Pair(row, col, col));
                        prev = true;
                    }
                }
                
                // we found horizontal lines, so add current to the
                // lines vector
                if(!current.empty()) {
                    // if prev is set, the last hline ended when the
                    // x-dimension ended, so set x1 accordingly
                    if(prev) {
                        current.back().line.x1 = lw-1;
                        if(method == WITH_PIXELS)
                            current.back().pixel_start = start + current.back().line.x0;
                    }
                    
                    lines->push_back(current);
                    current.clear();
                }
            }
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
        void merge_lines(BlobNLine &previous_vector,
                         BlobNLine &current_vector,
                         DLList<std::shared_ptr<Brototype>> &blobs,
                         std::vector<DLList<std::shared_ptr<Brototype>>::Node::Ptr>& reusable)
        {
            // walk along both lines, trying to match pairs
            auto current = current_vector.begin();
            auto previous = previous_vector.begin();
            
            while (current != current_vector.end()) {
                if(previous == previous_vector.end()
                   || current->line.y > previous->line.y + 1
                   || current->line.x1+1 < previous->line.x0)
                {
                    // case 0: current line ends before previous line starts
                    // case 1: lines are more than 1 apart in y direction,
                    // case 2: the previous line has ended before the current line started-1
                    //
                    // -> create new blobs for all elements
                    // add new blob, next element in current line
                    assert(previous == previous_vector.end()
                           || current->line.y > previous->line.y + 1
                           || !current->line.overlap_x(previous->line));
                    
                    if(!current->blob) {
                        if(!reusable.empty()) {
                            current->blob = blobs.insert(reusable.back());
                            reusable.pop_back();
                        } else
                            current->blob = blobs.insert(std::make_shared<Brototype>(&(*current)));
                    } else if(!current->blob->parent) {
                        blobs.insert(current->blob);
                    }
                    
                    ++current;
                    
                } else if(current->line.x0 > previous->line.x1+1) {
                    // case 3: previous line ends before current
                    // next element in previous line
                    assert(!current->line.overlap_x(previous->line));
                    ++previous;
                    
                } else {
                    // case 4: lines intersect
                    // merge elements, next in line that ends first
                    assert(current->line.overlap_x(previous->line));
                    assert(previous->blob);
                    
                    auto pblob = previous->blob;
                    
                    if(!current->blob) {
                        // current line isnt part of a blob yet
                        current->blob = previous->blob;
                        pblob->obj->push_back(&(*current));
                        
                    } else if(current->blob != previous->blob) {
                        // current line is part of a blob
                        // (merge blobs)
                        auto cblob = current->blob;
                        assert(current->blob != previous->blob);
                        
                        // copy all lines from the blob in "current"
                        // into the blob in "previous"
                        if(pblob->obj->size() <= cblob->obj->size()) {
                            std::swap(pblob, cblob);
                        }
                        
                        if(!cblob->obj->empty())
                            merge_sorted(pblob->obj, cblob->obj);

                        // delete the blob "current"
                        //auto it = blobs.find(cblob);
                        //assert(it != blobs.end());
                        //auto it = blobs.begin() + cblob->index;
                        //assert(*it == cblob);
                        //blobs.erase(it);
                        //Debug("%d", std::distance(it, blobs.end()));
                        //for(; it != blobs.end(); ++it)
                        //    --(*it)->index;
                        cblob->invalidate();
                        
                        // replace blob pointers in current_ and previous_vector
                        for(auto cit = current_vector.begin(); cit != current_vector.end(); ++cit) {
                            if(cit->blob == cblob)
                                cit->blob = pblob;
                        }
                        
                        for(auto cit = previous; cit != previous_vector.end(); ++cit) {
                            if(cit->blob == cblob)
                                cit->blob = pblob;
                        }
                        
                        cblob->obj->lines()->clear();
                        reusable.push_back(cblob);
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
                    if(current->line.x1 <= previous->line.x1)
                        ++current;
                    else
                        ++previous;
                }
            }
        }
        
        blobs_t run_fast(std::vector<std::vector<BlobNLine>>& lines_array, Method method)
        {
            static std::mutex cool_mutex;
            static std::vector<DLList<std::shared_ptr<Brototype>>*> unused_lists;
            DLList<std::shared_ptr<Brototype>> *blobs = nullptr;
            
            static std::vector<std::unique_ptr<std::vector<DLList<std::shared_ptr<Brototype>>::Node::Ptr>>> reusable_deques;
            const auto max_threads = lines_array.size();
            decltype(reusable_deques)::value_type reusable;
            
            /**
             * SORT HORIZONTAL LINES INTO BLOBS
             * tested cases:
             *      - empty image
             *      - only one line
             *      - only two lines (one HorizontalLine in each of the two y-arrays)
             */
            
            // thread-index for current row iterator
            uchar ldx1 = 0;
            
            // iterators for current row (y-coordinate in the original image)
            // and for the previous row (the last one above current that contains objects)
            std::remove_reference<decltype(lines_array)>::type::value_type
                *current_row = &lines_array[ldx1],
                *previous_row = current_row;
            
            // try and find the first thread that found data
            while(ldx1 != max_threads && current_row->empty() && ldx1 < max_threads - 1)
                current_row = previous_row = &lines_array[++ldx1];
            
            // no lines can be found in this frame
            // (if so, return empty)
            if(ldx1 == max_threads)
                return {};
            
            //! fetch a blobs store if available
            {
                std::lock_guard<std::mutex> guard(cool_mutex);
                if(!unused_lists.empty()) {
                    blobs = unused_lists.back();
                    unused_lists.pop_back();
                }
                if(!reusable_deques.empty()) {
                    reusable = std::move(reusable_deques.back());
                    reusable_deques.pop_back();
                }
            }
            
            if(!blobs)
                blobs = new std::remove_pointer<decltype(blobs)>::type ();
            
            if(!reusable)
                reusable = std::make_unique<decltype(reusable)::element_type>();
            
            // thread-index for previous row
            uchar ldx0 = ldx1;
            
            // create blobs for the current line if its not empty
            if(!current_row->empty()) {
                for(auto &l : *current_row->begin()) {
                    if(!l.blob)
                        l.blob = blobs->insert(std::make_shared<Brototype>(&l));
                    else
                        blobs->insert(l.blob);
                }
            }
            
            // initialize iterators to BlobNLine vectors
            // it0 is the previous line, it1 is the focal line (y)
            auto it0 = current_row->begin();
            auto it1 = it0;
            if(it1 != current_row->end())
                ++it1;
            
            // loop until the current_row iterator reaches the end of all arrays
            while (current_row != &lines_array.back() || it1 != current_row->end()) {
                // check if previous line is out-of-bounds with the
                // threads array
                if(it0 == previous_row->end()) {
                    if(ldx0+1 >= max_threads)
                        break;
                    
                    // try to find valid thread-index
                    // (next thread that found HorizontalLines)
                    do previous_row = &lines_array[++ldx0];
                    while (previous_row->empty() && ldx0+1 < max_threads);
                    
                    it0 = previous_row->begin();
                    assert(it0 != it1);
                }
                
                // check if current line is out-of-bounds with the
                // threads array
                if(it1 == current_row->end()) {
                    // dont go any further
                    if(ldx1+1 >= max_threads)
                        break;

                    // try to find valid thread-index
                    // (next thread that found HorizontalLines)
                    do current_row = &lines_array[++ldx1];
                    while (current_row->empty() && ldx1+1u < max_threads);
                    
                    it1 = current_row->begin();
                    //assert(&*it0 != &*it1);
                    //assert(it0 != it1);
                    
                    // we're at the end of everything
                    if(current_row == &lines_array.back() && it1 == current_row->end())
                        break;
                }
                
                merge_lines(*it0, *it1, *blobs, *reusable);
                
                it0++;
                it1++;
            }
            
            /**
             * FILTER BLOBS FOR SIZE, transform them into proper format
             */
            blobs_t result;
            for(auto it=blobs->begin(); it != blobs->end(); ++it) {
                if(!it->obj->empty()) {
                    auto lines = std::make_shared<std::vector<HorizontalLine>>();
                    auto pixels = method == WITH_PIXELS ? std::make_shared<std::vector<uchar>>() : nullptr;
                    lines->reserve(it->obj->lines()->size());
                    
                    size_t count = 0;
                    ushort y = 0;
                    for(auto &line : *it->obj->lines()) {
                        assert(line->line.y >= y);
                        y = line->line.y;
                        lines->push_back(line->line);
                        if(pixels) {
                            
                                pixels->insert(pixels->end(), line->pixel_start, line->pixel_start + (line->line.x1 - line->line.x0 + 1));
                            
                        }
                            //pixels->insert(pixels->end(), //line->pixels.begin(), line->pixels.end());
                        count += line->line.x1 - line->line.x0 + 1;
                    }
                    
                    assert(!pixels || count == pixels->size());
                    result.push_back({lines, pixels});
                }
                
                it->invalidate();
                    //result.push_back(std::make_shared<cmn::Blob>(*blob->lines()));
            }
            
            {
                blobs->clear();
                
                std::lock_guard<std::mutex> guard(cool_mutex);
                unused_lists.push_back(blobs);
            }
            
            for(auto &l : lines_array) {
                for(auto &k : l) {
                    for(auto &p : k) {
                        if(p.blob)
                            p.blob->invalidate();
                    }
                }
            }
            
            /*for(long i=blobs.size()-1; i>=0; i--) {
                auto b = blobs[i];
                if(!b->empty()) {
                    //blobs.erase(blobs.begin()+i);
                }
            }*/
            
            // assign correct ids
            //for(size_t i=0; i<blobs.size(); i++)
            //    blobs[i]->blob_id() = (uint32_t)i;
            //reusableclear();
            std::lock_guard<std::mutex> guard(cool_mutex);
            reusable_deques.push_back(std::move(reusable));
            //Debug("Currently %lu queues", reusable_deques.size());
            return result;
        }
        
        /**
         * Given a binary image, this function searches it for POI (pixels != 0),
         * finds all other spatially connected POIs and combines them into Blobs.
         *
         * @param image a binary image in CV_8UC1 format
         * @param min_size minimum number of pixels per Blob,
         *        smaller Blobs will be deleted
         * @return an array of the blobs found in image
         */
        blobs_t run_fast(const cv::Mat &image, bool enable_threads, Method method) {
            // assuming the number of threads allowed is < 255
            static const uchar sys_max_threads = max(1u, cmn::hardware_concurrency());
            const uchar max_threads = enable_threads && image.cols*image.rows > 100*100 ? sys_max_threads : 1;
            
            assert(image.cols < USHRT_MAX && image.rows < USHRT_MAX);
            assert(image.type() == CV_8UC1);
            assert(image.isContinuous());
            
            //! local width, height
            const ushort lw = image.cols, lh = image.rows;
            
            /**
             * FIND HORIZONTAL LINES IN ORIGINAL IMAGE
             */
            std::vector<Range<uint32_t>> thread_ranges;
            thread_ranges.resize(max_threads);
            uint32_t step = max(1, ceil(lh / float(max_threads)));
            uint32_t end = 0;
            
            for(uchar i=0; i<max_threads; i++) {
                thread_ranges[i].start = end;
                
                end = min(lh, end + step);
                thread_ranges[i].end = end;
            }
            
            std::vector<std::thread*> threads;
            std::vector<std::vector<BlobNLine>> lines_array;
            lines_array.resize(max_threads);
            
            if(max_threads > 1) {
                for(uchar i=0; i<max_threads; i++)
                    threads.push_back(new std::thread(extract_lines, &lines_array[i], thread_ranges[i], lw, image, method));
                
                for(auto t : threads) {
                    t->join();
                    delete t;
                }
                
            } else {
                extract_lines(&lines_array[0], thread_ranges[0], lw, image, method);
            }
            
            return run_fast(lines_array, method/*, min_size*/);
        }
        
        blobs_t run_fast(const std::vector<HorizontalLine>& lines, std::shared_ptr<std::vector<uchar>> pixels, Method method/*, size_t min_size*/)
        {
            if(lines.empty())
                return {};
            
            auto px = method == WITH_PIXELS && pixels ? pixels->data() : nullptr;
            
            static std::queue<std::shared_ptr<std::vector<std::vector<BlobNLine>>>> unused;
            static std::mutex mutex;
            
            std::shared_ptr<std::vector<std::vector<BlobNLine>>> lines_array;
            {
                std::lock_guard<std::mutex> guard(mutex);
                if(!unused.empty()) {
                    lines_array = unused.front();
                    unused.pop();
                }
            }
            
            if(!lines_array) {
                lines_array = std::make_shared<decltype(lines_array)::element_type>();
                lines_array->resize(1);
            } else {
                lines_array->resize(1);
                //lines_array->front().clear();
            }
            
            auto &vector = lines_array->front();
            size_t N = vector.size(), i = 0;
            
            auto start = lines.begin();
            std::vector<const uchar*> pixel_starts;
            
            for (auto it = start; it != lines.end(); ++it) {
                if (it->y > start->y) {
                    if(i < N) {
                        vector[i].clear();
                        vector[i].insert(vector[i].end(), start, it);
                    } else
                        vector.emplace_back(start, it);
                    
                    if(method == WITH_PIXELS) {
                        auto kit = pixel_starts.begin();
                        for(auto &e : vector[i]) {
                            e.pixel_start = *kit;
                            ++kit;
                        }
                        pixel_starts.clear();
                    }
                    
                    ++i;
                    start = it;
                }
                
                if(method == WITH_PIXELS)
                    pixel_starts.push_back(px);
                px += it->x1 - it->x0 + 1;
            }
            
            if(i < N)
                vector.resize(i);
            
            assert((!px && !pixels) || (pixels && px == pixels->data() + pixels->size()));
            
            auto result = run_fast(*lines_array, method/*, min_size*/);
            {
                std::lock_guard<std::mutex> guard(mutex);
                unused.push(lines_array);
            }
            return result;
        }
    }
}
