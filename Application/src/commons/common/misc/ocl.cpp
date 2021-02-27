#include "ocl.h"
#include <commons/common/commons.pc.h>

//#include <opencv/cv.hpp>
#if CV_MAJOR_VERSION >= 3
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core/utils/logger.hpp>
#endif

#if CV_MAJOR_VERSION >= 3
#include <opencv2/core/cuda.hpp>
#endif

namespace ocl {
    static std::mutex mutex;
#if CV_MAJOR_VERSION >= 3
    static std::map<decltype(std::this_thread::get_id()), cv::ocl::Context> ocl_context;
#endif
    
    bool init_ocl() {
#if CV_MAJOR_VERSION >= 3
        auto id = std::this_thread::get_id();
        std::lock_guard<std::mutex> m(mutex);
        
        if(ocl_context.find(id) != ocl_context.end()) {
            return true;
        }
        
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
			Except("No OCL devices available. Please check your graphics drivers and point the PATH variable to the appropriate folders.");
            return false;
        }
        assert(cv::ocl::haveOpenCL());
        cv::ocl::setUseOpenCL(true);
        
		using namespace std;

#ifndef NDEBUG
		std::vector<cv::ocl::PlatformInfo> platforms;
		cv::ocl::getPlatfomsInfo(platforms);
		for (auto &platform : platforms) {
			for (int i = 0; i < platform.deviceNumber(); ++i) {
				cv::ocl::Device device;
				platform.getDevice(device, i);

				cout << "Device: " << i << endl;
				cout << "Vendor ID: " << device.vendorName() << endl;
				cout << "available: " << device.available() << endl << endl;
			}
		}
#endif
		auto context = cv::ocl::Context::getDefault();

		cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
		if (!context.create(cv::ocl::Device::TYPE_DGPU) && !context.create(cv::ocl::Device::TYPE_GPU)) {
			cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
			cout << "Failed creating a GPU context. DGPU:" << (context.create(cv::ocl::Device::TYPE_DGPU)) << " GPU:" << (context.create(cv::ocl::Device::TYPE_GPU)) << std::endl;

			cout << context.ndevices() << " GPU devices are detected." << endl;
			for (size_t i = 0; i < context.ndevices(); i++)
			{
				cv::ocl::Device device = context.device(i);
				cout << "name                 : " << device.name() << endl;
				cout << "available            : " << device.available() << endl;
				cout << "imageSupport         : " << device.imageSupport() << endl;
				cout << "OpenCL_C_Version     : " << device.OpenCL_C_Version() << endl;
				cout << "OpenCLVersion        : " << device.OpenCLVersion() << endl;
				cout << "NVIDIA               : " << device.isNVidia() << endl;
				cout << "DGPU                 : " << (device.type() == cv::ocl::Device::TYPE_GPU) << endl;
				cout << endl;
			}

			if (!context.create(cv::ocl::Device::TYPE_ALL)) {
				cout << "Failed creating any OCL context." << std::endl;
				return false;
            } else {
                cout << "Created a generic OCL context." << std::endl;
            }
		}
        
		cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
        
        cv::ocl::Device(context.device(0));
        cv::BufferPoolController* c = cv::ocl::getOpenCLAllocator()->getBufferPoolController();
        if (c)
        {
            c->setMaxReservedSize(0);
        }
        
        ocl_context[id] = std::move(context);
        
        return true;
#endif
    }
}
