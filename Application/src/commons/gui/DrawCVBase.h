#ifndef _DRAW_CV_BASE_H
#define _DRAW_CV_BASE_H

#include "DrawBase.h"
#include "GuiTypes.h"

namespace gui {
    class CVBase : public Base {
        cv::Mat& _window;
        cv::Mat _overlay;
        
        static std::vector<std::pair<Image*, Vec2>> _static_pixels;
        
    public:
        CVBase(cv::Mat& window, const cv::Mat& background = cv::Mat());
        
        virtual void paint(DrawStructure& s) override;
        void display();
        void set_title(std::string) override {}
    private:
        void draw_image(gui::ExternalImage* ptr);
        void draw(DrawStructure &s, Drawable* o);
        
    };
}

#endif
