#include <misc/defines.h>
#include <misc/vec2.h>
#include <misc/Image.h>

namespace cmn {

ENUM_CLASS(averaging_method_t, mean, mode, max, min)
ENUM_CLASS_HAS_DOCS(averaging_method_t)

class AveragingAccumulator {
public:
    using Mat = cv::Mat;
    
protected:
    GETTER(averaging_method_t::Class, mode)
    Mat _accumulator;
    Mat _float_mat;
    Mat _local;
    double count = 0;
    bool use_mean;
    Size2 _size;
    
    std::mutex _accumulator_mutex;
    std::vector<std::array<uint8_t, 256>> spatial_histogram;
    std::vector<std::unique_ptr<std::mutex>> spatial_mutex;
    
public:
    AveragingAccumulator();
    AveragingAccumulator(averaging_method_t::Class mode);
    
    void add(const Mat &f);
    void add_threaded(const Mat &f);
    
    std::unique_ptr<cmn::Image> finalize();
    
private:
    template<bool threaded> void _add(const Mat& f);
};

}
