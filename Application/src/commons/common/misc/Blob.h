#ifndef _BLOB_H
#define _BLOB_H

#include <misc/defines.h>
#include <misc/vec2.h>
#include <gui/colors.h>

namespace cmn {
    struct Moments {
        float m[3][3];
        float mu[3][3];
        float mu_[3][3];
        
        bool ready;
        
        Moments() {
            ready = false;
            
            for(int i=0; i<3; i++) {
                for(int j=0; j<3; j++) {
                    m[i][j] = mu[i][j] = mu_[i][j] = 0;
                }
            }
        }
    };

    class Blob {
    protected:
        friend class DataFormat;
        friend class DataPackage;
        
        struct {
            float angle;
            Vec2 center;
            uint64_t _num_pixels;
            bool ready;
            
        } _properties;
        
        Bounds _bounds;
        
        std::shared_ptr<std::vector<HorizontalLine>> _hor_lines;
        GETTER(Moments, moments)
        
    public:
        const decltype(_properties)& properties() const { return _properties; }
        static size_t all_blobs();
    /*private:
        Blob(const std::vector<HorizontalLine>& points)
            : _hor_lines(std::make_shared<std::vector<HorizontalLine>>(points))
        { _properties.ready = false; }*/
        
    public:
        Blob();
        Blob(const std::shared_ptr<std::vector<HorizontalLine>>& points);
        virtual ~Blob();
        
        virtual void add_offset(const Vec2& off);
        
        inline std::shared_ptr<std::vector<HorizontalLine>> lines() const {
            return _hor_lines;
        }
        inline const std::vector<HorizontalLine>& hor_lines() const {
            return *_hor_lines;
        }
        
        float orientation() const {
            if(!_moments.ready)
                U_EXCEPTION("Moments aren't ready yet.");
            return _properties.angle;
        }
        const decltype(_properties.center)& center() const {
            if(!_properties.ready)
                U_EXCEPTION("Properties aren't ready yet.");
            return _properties.center;
        }
        const decltype(_bounds)& bounds() const {
            if(!_properties.ready)
                U_EXCEPTION("Properties have not been calculated yet.");
            return _bounds;
        }
        uint64_t num_pixels() const {
            if(!_properties.ready)
                U_EXCEPTION("Properties have not been calculated yet.");
            
            return _properties._num_pixels;
        }
        void sort() {
            std::sort(_hor_lines->begin(), _hor_lines->end(), std::less<HorizontalLine>());
        }
        
        std::shared_ptr<std::vector<uchar>> calculate_pixels(const cv::Mat& background) const {
            auto res = std::make_shared<std::vector<uchar>>();
            res->resize(num_pixels());
            auto ptr = res->data();
            
            for(auto &hl : *_hor_lines) {
                for (ushort x=hl.x0; x<=hl.x1; ++x, ++ptr) {
                    assert(ptr < res->data() + res->size());
                    *ptr = background.at<uchar>(hl.y, x);
                }
            }
            return res;
        }
        
        bool properties_ready() const { return _properties.ready; }
        
        virtual std::string toStr() const;
        static std::string class_name() {
            return "Blob";
        }
        
        //UTILS_TOSTRING("Blob<pos:" << (cv::Point2f)center() << " size:" << (cv::Size)_bounds.size() << ">");
        
    public:
        void calculate_properties();
        void calculate_moments();
};
}

#endif
