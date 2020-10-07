#include "ocl.h"
#include <commons/common/commons.pc.h>

//#include <opencv/cv.hpp>
#if CV_MAJOR_VERSION >= 3
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv_modules.hpp>
#endif

#if CV_MAJOR_VERSION >= 3
#include <opencv2/core/cuda.hpp>
#endif

namespace ocl {
    static std::mutex mutex;
#if CV_MAJOR_VERSION >= 3
    static std::map<decltype(std::this_thread::get_id()), cv::ocl::Context> ocl_context;
#endif
    
    void init_ocl() {
#if CV_MAJOR_VERSION >= 3
        auto id = std::this_thread::get_id();
        std::lock_guard<std::mutex> m(mutex);
        
        if(ocl_context.find(id) != ocl_context.end()) {
            return;
        }
        
        auto &context = ocl_context[id];
        
        if(cv::cuda::getCudaEnabledDeviceCount() > 0) {
            static bool printed = false;
            
            if(!printed) {
                Debug("Using CUDA device:");
                cv::cuda::printCudaDeviceInfo(0);
                printed = true;
            }
            cv::cuda::setDevice(0);
        }
        
        if(!cv::ocl::haveOpenCL()) {
            Except("OpenCL cannot be used.");
            return;
        }
        assert(cv::ocl::haveOpenCL());
        cv::ocl::setUseOpenCL(true);
        
        using namespace std;
        
        if (context.create(cv::ocl::Device::TYPE_DGPU))
        {
            //cout << "Failed creating the context..." << endl;
           // return;
            // create a dedicated GPU device
        } else if(!context.create(cv::ocl::Device::TYPE_IGPU)) {
            // integrated didnt work either
            cout << "Failed creating integrated GPU context." << std::endl;
            
            if(!context.create(cv::ocl::Device::TYPE_ALL)) {
                cout << "Failed creating any OCL context." << std::endl;
                return;
            }
        }
       /* cout << context.ndevices() << " GPU devices are detected." << endl;
        for (size_t i = 0; i < context.ndevices(); i++)
        {
            cv::ocl::Device device = context.device(i);
            cout << "name                 : " << device.name() << endl;
            cout << "available            : " << device.available() << endl;
            cout << "imageSupport         : " << device.imageSupport() << endl;
            cout << "OpenCL_C_Version     : " << device.OpenCL_C_Version() << endl;
            cout << endl;
        }
        
        Debug("Choosing device 0.");*/
        cv::ocl::Device(context.device(0));
#endif
    }
}
