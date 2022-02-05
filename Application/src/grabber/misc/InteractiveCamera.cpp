#include "InteractiveCamera.h"
#include <misc/GlobalSettings.h>
#include <grabber/gui.h>
#include <misc/GlobalSettings.h>
#include <misc/SpriteMap.h>

namespace fg {
constexpr bool use_dynamic = true;

void InteractiveCamera::Fish::update(float dt, const Vec2& poi, const std::vector<Fish>& individuals) {
    std::vector<std::tuple<Vec2, float, float, float>> forces;
    forces.push_back({poi, 5, euclidean_distance(position, poi)*0.9, 1});
    
    for(auto &other : individuals) {
        if(&other != this) {
            auto d = euclidean_distance(other.position, position);
            forces.push_back({other.position, d > 100 ? 1 : 10, 100, 1});
        }
    }
    
    /*velocity += Vec2(cos(angle), -sin(angle)) * d * dt;*/
    
    //const float spring_stiffness = 200;
    const float spring_damping = 20;
    const float mass = 10;
    
    for(auto && [target, spring_stiffness, spring_L, attraction] : forces) {
        auto distance = (position - target) * attraction;
        auto CL = distance.length();
        if(CL > 0)
            distance /= CL;
        if(CL == 0)
            CL = 0.00001;
        
        Vec2 f = distance;
        f *= - (spring_stiffness * (CL - spring_L)
             + spring_damping * (velocity).dot(distance) / CL);
        _force += f;
    }
    
    velocity += _force / mass * dt;
    
    auto vl = velocity.length();
    static const float max_speed = SETTING(track_max_speed).value<float>() / SETTING(cm_per_pixel).value<float>();
    if(vl >= max_speed) {
        velocity = velocity / vl * max_speed;
    }
    
    position += velocity * dt;
    
    if(position.x > boundary.x) position.x = boundary.x;
    if(position.y > boundary.y) position.y = boundary.y;
    if(position.x < 0) position.x = 0;
    if(position.y < 0) position.y = 0;
    
    const double damping_linear = .5;
    _force = velocity * (-damping_linear);
}

void InteractiveCamera::Fish::draw(cv::Mat& img) {
    //cv::circle(img, position, 5, gui::White, -1);
    //auto a = atan2(velocity);
    /*for(int i=0; i<10; ++i) {
        float percent =  1 - float(i) / float(10);
        cv::ellipse(img, position, cv::Size(25 * percent, width * percent), DEGREE(a), 0, 360, cv::Scalar(200 * (1-percent) + 55), -1);
    }*/
    auto v = velocity.normalize(); //Vec2(cos(a), sin(a));
    auto iterations = max(1, ceil(L * 0.5));
    for(int j=0; j<10; ++j) {
        float clr_percent =  1 - float(j) / float(10);
        for (int i=0; i<iterations; ++i) {
            float percent =  1 - float(i) / float(iterations);
            cv::circle(img, position + v * L * (percent - 0.5), width * clr_percent * SQR(percent) +1, cv::Scalar(200 * (1-SQR(clr_percent))*SQR(percent) + 55), -1);
        }
    }
}

InteractiveCamera::InteractiveCamera() {
    _size = cv::Size(SETTING(cam_resolution).value<cv::Size>().width, SETTING(cam_resolution).value<cv::Size>().height);
    
    if constexpr(use_dynamic) {
        const auto number_individuals = SETTING(track_max_individuals).value<uint32_t>();
        constexpr auto random_number = [](const Rangel& range) {
            return Float2_t(rand()) / Float2_t(RAND_MAX) * range.length() + range.start;
        };
        
        _fishies.resize(number_individuals ? number_individuals : 3u);
        for(auto &fish : _fishies) {
            fish.position = Vec2(random_number(Rangel(0, _size.width)),
                                 random_number(Rangel(0, _size.height)));
            //fish.angle = random_number(Rangel(- M_PI, M_PI));
            fish.boundary = Vec2(_size.width, _size.height);
            fish.width = random_number(Rangel(3, 7));
            fish.L = random_number(Rangel(30,40));
        }
    }
}

bool InteractiveCamera::next(cmn::Image &image) {
    static cv::Mat img = cv::Mat::zeros(_size.height, _size.width, CV_8UC1);
    img = cv::Scalar(0);
    
    static Timer timer;
    if(timer.elapsed() < 1 / float(SETTING(frame_rate).value<int>()))
        return false;
    
    Vec2 target;
    {
        std::lock_guard<std::recursive_mutex> guard(grab::GUI::instance()->gui().lock());
        target = grab::GUI::instance()->gui().mouse_position();
    }
    
    if constexpr(use_dynamic) {
        auto dt = timer.elapsed();
        for(auto &fish : _fishies) {
            fish.update(dt, target, _fishies);
            fish.draw(img);
        }
        
    } else {
        
        static Vec2 pos;
        
        auto tv = (target - pos);
        auto d = tv.length();
        const float max_speed = _size.width * 0.1;
        if(d)
            tv = tv / d;
        if(d > max_speed) d = max_speed;
        
        pos += tv * d * 10 * timer.elapsed();
        
        static float angle = 0;
        angle += 5 * timer.elapsed();
        
        for(auto && [f, a] : std::vector<std::tuple<float, float>>{ {-50, 1}, {50, 1}, {-150, 0.5}, {150, 0.5} }) {
            auto v = Vec2(sin(angle*a), cos(angle*a)) * f;
            
            float scale_len = (1 - SQR(1 - a));
            cv::ellipse(img, pos + v, cv::Size(25 * scale_len, f < 0 ? 3 : 5), - DEGREE(angle*a), 0, 360, gui::White, -1);
            cv::circle(img, pos + v + Vec2(-cos(angle*a), sin(angle*a)) * 25 * scale_len, 10 * scale_len, gui::White, -1);
        }
    }
    
    image.create(img, image.index());
    timer.reset();
    
    return true;
}
}
