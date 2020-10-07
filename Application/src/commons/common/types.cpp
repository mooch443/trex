#include <png.h>
#include <types.h>
#include <misc/Image.h>
#include <misc/GlobalSettings.h>

namespace tf {
    using namespace cmn;
    
    class ThreadSafety {
        static std::mutex _opencv_lock;
        static std::queue<std::pair<std::string, cmn::Image*>> images;
        static std::map<std::string, std::queue<cmn::Image*>> waiting;
        
    public:
        struct LockGuard {
            std::unique_lock<std::mutex> _lock;
            
            LockGuard() : _lock(ThreadSafety::_opencv_lock)
            { }
        };
        
        static void add(std::string name, const cv::Mat& matrix, std::string label);
        static void show();
        static void waitKey(std::string name);
        
        ThreadSafety() {}
        ~ThreadSafety();
    };
    
    decltype(ThreadSafety::waiting) ThreadSafety::waiting;
    std::mutex ThreadSafety::_opencv_lock;
    decltype(ThreadSafety::images) ThreadSafety::images;
    
    void imshow(const std::string& name, const cv::Mat& mat, std::string label) {
        ThreadSafety::add(name, mat, label);
    }
    
    void show() {
        ThreadSafety::show();
    }
    
    void waitKey(std::string name) {
        ThreadSafety::waitKey(name);
    }
    
    ThreadSafety::~ThreadSafety() {
        LockGuard guard;
        while(!images.empty()) {
            auto &img = images.front();
            delete img.second;
            images.pop();
        }
    }
    
    void ThreadSafety::add(std::string name, const cv::Mat& matrix, std::string label) {
		LockGuard guard;

		assert(matrix.isContinuous());
		assert(matrix.type() == CV_8UC(matrix.channels()));
		
		Image *ptr = new Image(matrix);
		images.push({ name , ptr });
        
        if(!label.empty()) {
            cv::Mat mat = ptr->get();
            cv::putText(mat, label, cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(255, 255, 255));
        }
	}

	void ThreadSafety::show() {
		std::unique_lock<std::mutex> lock(_opencv_lock);
        std::map<std::string, Image*> display_map;
        
        static const bool nowindow = GlobalSettings::map().has("nowindow") ? SETTING(nowindow).value<bool>() : false;

		while(!images.empty()) {
			auto p = images.front();
			images.pop();

			std::string name = p.first;
			Image *ptr = p.second;

            if(waiting.count(name)) {
                waiting[name].push(ptr);
                continue;
            }
            
            if(ptr == NULL) {
                waiting[name] = std::queue<Image*>();
                continue;
            }
            
            if(display_map.count(name)) {
                delete display_map.at(name);
            }
            
            display_map[name] = ptr;
		}
        
        for(auto &p: display_map) {
			cv::Mat mat = p.second->get();
#ifdef __linux__
			resize_image(mat, 1.5, cv::INTER_AREA);
#endif
            if(!nowindow)
                cv::imshow(p.first, mat);
            delete p.second;
        }
        
        if(!nowindow && !display_map.empty()) {
            lock.unlock();
//#ifdef __linux__
            cv::waitKey(1);
//#endif
            lock.lock();
        }
        
        for (auto &pair : waiting) {
            auto &queue = pair.second;
            Image* show = NULL;
            
            while(!queue.empty()) {
                auto first = queue.front();
                queue.pop();
                
                if(first) {
                    if(show)
                        delete show;
                    show = first;
                    
                } else {
                    break;
                }
            }
            
            if(show) {
                cv::Mat mat = show->get();
#ifdef __linux__
                resize_image(mat, 1.5, cv::INTER_AREA);
#endif
                if(!nowindow)
                    cv::imshow(pair.first, mat);
                delete show;
                if(!nowindow) {
                    lock.unlock();
                    cv::waitKey();
                    lock.lock();
                }
            }
        }
        
        for (auto &pair : waiting) {
            if(pair.second.empty()) {
                waiting.erase(pair.first);
                break;
            }
        }
	}
    
    void ThreadSafety::waitKey(std::string name) {
        LockGuard guard;
        images.push({ name, NULL });
    }
}
