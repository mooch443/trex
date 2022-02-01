
#pragma once

#include <sys/stat.h>
#include <misc/defines.h>

namespace gui {
    class Color;
}

namespace cmn {
    class Image;

    class CrashProgram {
    public:
        static std::thread::id crash_pid, main_pid;
        static bool do_crash;
        static void crash() { char *s = nullptr;
            *s = 0;}
    };
    
    /**
     * ======================
     * COMMON custom types
     * ======================
     */
    
    //! If this is set to true, horizontal lines will be ordered by CPULabeling according
    //  to their x/y coordinates. If not, then dilate for HorizontalLines is disabled, but
    //  CPULabeling runs a bit faster.
#define ORDERED_HORIZONTAL_LINES true
    
    /**
     * A structure that represents a horizontal line on an image.
     */
    using coord_t = uint16_t;
    using ptr_safe_t = uint64_t;
    struct HorizontalLine {
        coord_t y;
        coord_t x0, x1;
        
        HorizontalLine() = default;
        HorizontalLine(coord_t y_, coord_t x0_, coord_t x1_)
        : y(y_), x0(x0_), x1(x1_) {
            //assert(x0 <= x1);
        }
        
        bool inside(coord_t x_, coord_t y_) const {
            return y_ == y && x_ >= x0 && x_ <= x1;
        }
        
        bool overlap_x(const HorizontalLine& other) const {
            return other.x1 >= x0-1 && other.x0 <= x1+1;
        }
        
        bool overlap(const HorizontalLine& other) const {
            return other.y == y && overlap_x(other);
        }
        
        bool operator==(const HorizontalLine& other) const {
            return other.x0 == x0 && other.y == y && other.x1 == x1;
        }
        
        bool operator<(const HorizontalLine& other) const {
            //! Compares two HorizontalLines and sorts them by y and x0 coordinates
            //  (top-bottom, left-right)
            return y < other.y || (y == other.y && x0 < other.x0);
        }
        
        HorizontalLine merge(const HorizontalLine& other) const {
            //if(other.y != y)
            //    U_EXCEPTION("Cannot merge lines from y=%d and %d", y, other.y);
            //assert(overlap(other));
            return HorizontalLine(y, cmn::min(x0, other.x0), cmn::max(x1, other.x1));
        }
        
        std::string str() const {
            std::stringstream ss;
            ss << "HL<" << y << ", " << x0 << " - " << x1 << ">";
            return ss.str();
        }
        
        static void repair_lines_array(std::vector<HorizontalLine>&, std::vector<uchar>&);
        static void repair_lines_array(std::vector<HorizontalLine>&);
        
        std::string toStr() const;
        static std::string class_name() {
            return "HorizontalLine";
        }
    };
    
#if ORDERED_HORIZONTAL_LINES
    //! Returns an offset for y if it was needed to keep the values within 0<value<USHRT_MAX
    void dilate(std::vector<HorizontalLine>& array, int times=1, int max_cols = 0, int max_rows = 0);
#endif
    
    template <typename T>
    class NoInitializeAllocator : public std::allocator< T > {
    public:
        template <typename U>
        struct rebind {
            typedef NoInitializeAllocator<U> other;
        };
        
        //provide the required no-throw constructors / destructors:
        NoInitializeAllocator() throw() : std::allocator<T>() { };
        NoInitializeAllocator(const NoInitializeAllocator<T>& rhs) throw() : std::allocator<T>(rhs) { };
        ~NoInitializeAllocator() throw() { };
        
        //import the required typedefs:
        typedef T& reference;
        typedef const T& const_reference;
        typedef T* pointer;
        typedef const T* const_pointer;
        
        //redefine the construct function (hiding the base-class version):
        /*void construct( pointer p, const_reference cr) {
         Debug("Construct called!");
         //else, do nothing.
         };*/
        
        template <class _Up, class... _Args>
        void
        construct(_Up*, _Args&&... )
        {
            // do nothing!
        }
    };
    template <class T, class U>
    bool operator==(const NoInitializeAllocator<T>&, const NoInitializeAllocator<U>&) { return true; }
    template <class T, class U>
    bool operator!=(const NoInitializeAllocator<T>&, const NoInitializeAllocator<U>&) { return false; }
    
    //! Converts a lines array to a mask or greyscale (or both).
    //  requires pixels to contain actual greyscale values
    std::pair<cv::Rect2i, size_t> imageFromLines(const std::vector<HorizontalLine>& lines,
                                                 cv::Mat* output_mask,
                                                 cv::Mat* output_greyscale = NULL,
                                                 cv::Mat* output_differences = NULL,
                                                 const std::vector<uchar>* pixels = NULL,
                                                 const int threshold = 0,
                                                 const Image* average = NULL,
                                                 int padding = 0);
    
    class LuminanceGrid;
    std::pair<cv::Rect2i, size_t> imageFromLines(const std::vector<HorizontalLine>& lines, cv::Mat* output_mask, cv::Mat* output_greyscale, cv::Mat* output_differences, const std::vector<uchar>& pixels, int base_threshold, const LuminanceGrid& grid, const cv::Mat& average, int padding);
    
    /**
     * Converts a lines array to a mask or greyscale.
     * Expects pixels to contain difference values instead of actual greyscale.
     * @deprecated backwards compatibility
     */
    /*std::pair<cv::Rect2i, size_t> imageFromLines_old(const std::vector<HorizontalLine>& lines,
     cv::Mat* output_mask,
     cv::Mat* output_greyscale = NULL,
     const std::vector<uchar>* pixels = NULL,
     const char threshold = 0,
     const cv::Mat* average = NULL);*/
    
    cv::Rect2i lines_dimensions(const std::vector<HorizontalLine>& lines);
    void lines2mask(const std::vector<HorizontalLine>& lines, cv::Mat& output_mask, const int value = 255);
    
    inline std::pair<cv::Rect2i, size_t> lines2greyscale(const std::vector<HorizontalLine>& lines, cv::Mat& output_greyscale, const std::vector<uchar>* pixels, const int threshold = 0, const Image* average = NULL)
    {
        return imageFromLines(lines, NULL, &output_greyscale, NULL, pixels, threshold, average);
    }
    
    inline std::pair<cv::Rect2i, size_t> lines2mask(const std::vector<HorizontalLine>& lines, cv::Mat& output_mask, const std::vector<uchar>* pixels, const int threshold = 0, const Image* average = NULL)
    {
        return imageFromLines(lines, &output_mask, NULL, NULL, pixels, threshold, average);
    }
    
    template <typename T = double>
    T normalize_angle(T angle) {
        while (angle < T(0.0)) angle += T(M_PI * 2);
        while (angle >= T(M_PI * 2)) angle -= T(M_PI * 2);
        return angle;
    }
    
    /**
     * The difference in angle to get from angle to vangle.
     */
    template <typename T = double, typename K = T>
    T angle_difference(T angle, K vangle) {
        T difference;
        
        if(std::abs(vangle-angle) < std::abs(T(M_PI*2)+angle-vangle) && std::abs(vangle-angle) < std::abs(T(M_PI*2)+vangle - angle))
            difference = vangle-angle;
        else {
            if(angle < vangle)
                difference = (T(M_PI*2)+angle) - vangle;
            else
                difference = (T(M_PI*2)+vangle) - angle;
        }
        
        return difference;
    }
    
    template <typename T = cv::Vec3b>
    T BGR2HSV(const T& rgb) {
        cv::Mat tmp(1, 1, CV_8UC3);
        tmp = rgb;
        cv::cvtColor(tmp, tmp, cv::COLOR_BGR2HSV);
        
        return tmp.at<cv::Vec3b>(0, 0);
    }
    
    template <typename T = cv::Vec3b>
    T HSV2RGB(const T& hsv) {
        cv::Mat tmp(1, 1, CV_8UC3);
        tmp.at<cv::Vec3b>(0, 0) = hsv;
        cv::cvtColor(tmp, tmp, cv::COLOR_HSV2BGR);
        
        return tmp.at<cv::Vec3b>(0, 0);
    }
    
    inline int ggt(int x, int y) { /* gibt ggt(x,y) zurueck, falls x oder y nicht 0 */
        int c;                /* und gibt 0 zurueck fuer x=y=0.   */
        if ( x < 0 ) x = -x;
        if ( y < 0 ) y = -y;
        while ( y != 0 ) {          /* solange y != 0 */
            c = x % y; x = y; y = c;  /* ersetze x durch y und
                                       y durch den Rest von x modulo y */
        }
        return x;
    }
    
    inline int isPowerOfTwo (unsigned int x)
    {
        return x && !(x & (x - 1));
        //return ((x != 0) && !(x & (x - 1)));
    }
    
    inline int gt(int x, int y) {
        for(int i=(x < y ? x : y); i>1; i--)
            if(x%i == 0)
                return i;
        return 1;
    }
    
    template<typename T>
    T lerp(T start, T end, ScalarType percent)
    {
        return (start + (end - start) * percent);
    }
    
    template<typename K, typename T = K>
    constexpr inline T saturate(K val, T min = 0, T max = 255) {
        return std::clamp(T(val), min, max);
    }
    
    /**
     * ======================
     * INITIALIZER methods
     * ======================
     */
    
    template<int m, int n>
    inline cv::Matx<ScalarType, m, n> identity() {
        cv::Matx<ScalarType, m, n> mat;
        cv::setIdentity(mat);
        return mat;
    }
    
    template<int m=3, int n=1>
    inline cv::Matx<ScalarType, m, n> zeros() {
        return cv::Matx<ScalarType, m, n>::zeros();
    }
    
    template<int m=3, int n=1>
    inline cv::Matx<ScalarType, m, n> ones() {
        return cv::Matx<ScalarType, m, n>::ones();
    }
    
    template<int m=3, int n=1>
    inline cv::Matx<ScalarType, m, n> fill(ScalarType number) {
        cv::Matx<ScalarType, m, n> ret;
        for(int i=0; i<m; i++)
            for(int j=0; j<n; j++)
                ret(i, j) = number;
        
        return ret;
    }
    
    /**
     * ======================
     * PRINTING methods
     * ======================
     */
    
    template <typename _Tp = ScalarType, int m, int n>
    void print_mat(const char*name, const cv::Matx<_Tp, m, n>& mat) {
        printf("%s(%dx%d) = \n", name, m, n);
        for(int i=0; i<mat.rows; i++) {
            printf("[");
            for (int j=0; j<mat.cols; j++) {
                printf("\t%.3f", mat(i, j));
            }
            printf("\t]\n");
        }
        printf("\n");
    }
    
    inline std::string getImgType(int imgTypeInt);

    inline void print_mat(const char*name, const cv::Mat& mat) {
        auto type = getImgType(mat.type());
        printf("%s(%dx%d, %s) = \n", name, mat.rows, mat.cols, type.c_str());
        for(int i=0; i<mat.rows; i++) {
            printf("[");
            for (int j=0; j<mat.cols; j++) {
                switch (mat.type()) {
                    case CV_64FC1:
                        printf("\t%.6f", mat.at<double>(i, j));
                        break;
                    case CV_32FC1:
                        printf("\t%.6f", mat.at<float>(i, j));
                        break;
                    case CV_32SC2:
                        printf("\t(%d,%d)", mat.at<cv::Vec2i>(i, j)(0), mat.at<cv::Vec2i>(i, j)(1));
                        break;
                        
                    default:
                        type = getImgType(mat.type());
                        U_EXCEPTION("Unknown data type: %S", &type);
                        break;
                }
            }
            printf("\t]\n");
        }
        printf("\n");
    }
    
    template <typename _Tp = ScalarType, int m, int n>
    void print_mat(const char*name, const std::initializer_list<cv::Matx<_Tp, m, n>>& l, int = m) {
        std::vector<cv::Matx<_Tp, m, n>> mats(l);
        
        printf("%s(%dx%d) = \n", name, m, n);
        for(int i=0; i<m; i++) {
            for(int k=0; k < mats.size(); k++) {
                auto& mat = mats[k];
                
                printf("[");
                for (int j=0; j<n; j++) {
                    printf("\t%.3f", mat(i, j));
                }
                printf("\t]");
            }
            printf("\n");
        }
        printf("\n");
    }
    
    template <typename _Tp = ScalarType, int m, int n>
    void print_mat(const char*name, const std::vector<cv::Matx<_Tp, m, n>>& mats) {
        printf("%s(%dx%d) = \n", name, m, n);
        for(int i=0; i<m; i++) {
            for(long k=0; k < (long)mats.size(); k++) {
                auto& mat = mats[k];
                
                printf("[");
                for (int j=0; j<n; j++) {
                    printf("\t%.3f", mat(i, j));
                }
                printf("\t]");
            }
            printf("\n");
        }
        printf("\n");
    }
    
    /**
     * ======================
     * IS_NAN methods
     * ======================
     */
    
    template <typename _Tp = ScalarType, int m, int n>
    bool is_nan(const cv::Matx<_Tp, m, n>& mat) {
        for(int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                if(std::isnan(mat(i, j))) {
                    return true;
                }
            }
        }
        return false;
    }
    
    /**
     * ======================
     * MISC methods
     * ======================
     */
    
    constexpr inline bool between_equals(float x, float lower, float upper) {
        return x >= lower && x <= upper;
    }
    
    inline int is_big_endian(void)
    {
        union {
            uint32_t i;
            char c[4];
        } bint = {0x01020304};
        
        return bint.c[0] == 1;
    }
    
    inline std::ostream&
    duration_to_string(std::ostream& os, std::chrono::nanoseconds ns)
    {
        using namespace std;
        using namespace std::chrono;
        typedef duration<int, ratio<86400>> days;
        char fill = os.fill();
        os.fill('0');
        auto d = duration_cast<days>(ns);
        ns -= d;
        auto h = duration_cast<hours>(ns);
        ns -= h;
        auto m = duration_cast<minutes>(ns);
        ns -= m;
        auto s = duration_cast<seconds>(ns);
        os << setw(2) << d.count() << "d:"
        << setw(2) << h.count() << "h:"
        << setw(2) << m.count() << "m:"
        << setw(2) << s.count() << 's';
        os.fill(fill);
        return os;
    }
    
    // take number image type number (from cv::Mat.type()), get OpenCV's enum string.
    inline std::string getImgType(int imgTypeInt)
    {
        int numImgTypes = 35; // 7 base types, with five channel options each (none or C1, ..., C4)
        
        static const int enum_ints[] =       {CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
            CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
            CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
            CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
            CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
            CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
            CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4};
        
        static const std::string enum_strings[] = {"CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",
            "CV_8S",  "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",
            "CV_16U", "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
            "CV_16S", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
            "CV_32S", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
            "CV_32F", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
            "CV_64F", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"};
        
        for(int i=0; i<numImgTypes; i++)
        {
            if(imgTypeInt == enum_ints[i]) return enum_strings[i];
        }
        return "unknown image type";
    }
    
    inline bool file_exists(const std::string &path) {
#ifdef WIN32
		DWORD attr = GetFileAttributes(path.c_str());
		if (INVALID_FILE_ATTRIBUTES == attr /*|| (attr & FILE_ATTRIBUTE_DIRECTORY)*/)
		{
			return false;
		}
		return true;
#else
        struct stat buffer;
        return (stat (path.c_str(), &buffer) == 0);
#endif
    }
    
    template<typename T>
    inline void resize_image(T& mat, double factor, int flags = cv::INTER_NEAREST)
    {
        cv::resize(mat, mat, cv::Size(), factor, factor, flags);
    }
    
    template<typename T>
    inline void resize_image(const T& mat, T& output, double factor, int flags = cv::INTER_NEAREST)
    {
        cv::resize(mat, output, cv::Size(), factor, factor, flags);
    }
    
    // set all mat values at given channel to given value
    inline void setAlpha(cv::Mat &mat, unsigned char value, cv::Scalar only_this = cv::Scalar(0, 0, 0, -1))
    {
        // make sure have enough channels
        if (mat.channels() != 4)
            return;
        
        const int cols = mat.cols;
        const int step = mat.channels();
        const int rows = mat.rows;
        for (int y = 0; y < rows; y++) {
            // get pointer to the first byte to be changed in this row
            unsigned char *p_row = mat.ptr(y) + 3;
            unsigned char *row_end = p_row + cols*step;
            
            if(only_this(3) >= 0) {
                for (; p_row != row_end; p_row += step) {
                    if((*(p_row-3) == only_this(2)*255 && *(p_row-2) == only_this(1)*255) && *(p_row-1) == only_this(0)*255) {
                        *p_row = value;
                    }
                }
            } else {
                for (; p_row != row_end; p_row += step)
                    *p_row = value;
            }
        }
    }
    
    template< typename tPair >
    struct first_t {
        typename tPair::first_type operator()( const tPair& p ) const { return p.first; }
    };
    
    template< typename tMap >
    first_t< typename tMap::value_type > first( const tMap& ) { return first_t< typename tMap::value_type >(); }
    
    template<typename Map, typename KeyType = typename Map::key_type>
    inline std::set<KeyType> extract_keys(const Map& m) {
        std::set<KeyType> v;
        std::transform( m.begin(), m.end(), std::inserter( v, v.end() ), first(m) );
        return v;
    }
    
    template< typename tPair >
    struct second_t {
        typename tPair::second_type operator()( const tPair& p ) const { return p.second; }
    };
    
    template< typename tMap >
    second_t< typename tMap::value_type > second( const tMap& ) { return second_t< typename tMap::value_type >(); }
    
    template<typename KeyType, typename ValueType>
    inline void extract_values(const std::map<KeyType, ValueType>& m, std::vector<ValueType>& v) {
        std::transform( m.begin(), m.end(), std::back_inserter( v ), second(m) );
    }
    
    template<typename KeyType, typename ValueType>
    inline std::vector<ValueType> extract_values(const std::map<KeyType, ValueType>& m) {
        std::vector<ValueType> v;
        extract_values(m, v);
        return v;
    }
    
    //! This interface adds a "minimize_memory()" function that should
    //  make the class more compressed and reduce memory usage to a minimum.
    class Minimizable {
    public:
        virtual void minimize_memory() = 0;
        virtual ~Minimizable() {}
    };
    
    //! Escapes html reserved characters in a string
    inline std::string escape_html(const std::string& data) {
        std::string buffer;
        buffer.reserve(size_t(data.size()*1.1f));
        for(size_t pos = 0; pos != data.size(); ++pos) {
            switch(data[pos]) {
                case '&':  buffer.append("&amp;");       break;
                case '\"': buffer.append("&quot;");      break;
                case '\'': buffer.append("&apos;");      break;
                case '<':  buffer.append("&lt;");        break;
                case '>':  buffer.append("&gt;");        break;
                default:   buffer.append(&data[pos], 1); break;
            }
        }
        return buffer;
    }
    
    /**
     * T is the destination type, V is a primitive Type that can be copied easily.
     * V has to have a method convert(T&) for assignment.
     * Objects of type T are constructed from elements of type V in compare.
     * Excess objects within vector are deleted.
     * @requires Source type needs a method convert(Target*),
     *           Target needs empty constructor
     */
    /*template<typename K, typename T = K, typename V>
    void update_vector_elements(std::vector<K*>& vector,
                                const std::vector<V*>& compare,
                                const std::function<void(T*, V*)>& prepare = [](T*,V*){})
    {
        // Delete elements from the end of vector if its too long
        for(size_t i=vector.size()-1; !vector.empty() && i>=compare.size(); i--) {
            delete vector[i];
            vector.erase(vector.begin() + i);
        }
        
        // set elements in the beginning
        for(size_t i=0; i<vector.size(); i++) {
            prepare((T*)vector[i], compare[i]);
            compare[i]->convert((T*)vector[i]);
        }
        
        // add missing elements
        for(size_t i=vector.size(); i<compare.size(); i++) {
            T* obj = new T;
            vector.push_back(obj);
            
            prepare((T*)vector[i], compare[i]);
            compare[i]->convert((T*)obj);
        }
    }*/
    
    template<typename K, typename T = K, typename V>
    inline void update_vector_elements(std::vector<std::shared_ptr<K>>& vector,
                                const std::vector<std::shared_ptr<V>>& compare,
                                const std::function<void(std::shared_ptr<K>, std::shared_ptr<V>)>& prepare = nullptr)
    {
        // Delete elements from the end of vector if its too long
        for(int64_t i=int64_t(vector.size())-1; !vector.empty() && i>=(int64_t)compare.size(); i--) {
            vector.erase(vector.begin() + i);
        }
        
        // set elements in the beginning
        for(size_t i=0; i<vector.size(); i++) {
            if(prepare)
                prepare(vector[i], compare[i]);
            compare[i]->convert(vector[i]);
        }
        
        // add missing elements
        for(size_t i=vector.size(); i<compare.size(); i++) {
            auto obj = std::make_shared<T>();
            vector.push_back(obj);
            
            if(prepare)
                prepare(vector[i], compare[i]);
            compare[i]->convert(obj);
        }
    }
    
    inline uint8_t hardware_concurrency() {
#if TRACKER_GLOBAL_THREADS
        return TRACKER_GLOBAL_THREADS;
#else
        auto c = (uint8_t)saturate(std::thread::hardware_concurrency());
        if(!c)
            return 1;
        return c;
#endif
    }
    
    namespace sprite {
        class Map;
        
        //! Parses a JSON object from string {"key": "value", "key2": 123, "key3": ["test","strings"]}
        Map parse_values(std::string str);
        std::set<std::string> parse_values(Map&, std::string);
    }
    
    void set_thread_name(const std::string& name);
    std::string get_thread_name();
    
    class Viridis {
    public:
        using value_t = std::tuple<double, double, double>;
    private:
        static const std::array<value_t, 256> data_bgr;
    public:
        static gui::Color value(double percent);
    };
}
