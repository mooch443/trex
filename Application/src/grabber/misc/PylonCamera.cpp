#if WITH_PYLON

#include "PylonCamera.h"
#include <misc/GlobalSettings.h>

using namespace Pylon;

namespace fg {
    PylonCamera::PylonCamera()
        : _camera(NULL)
    {
        std::string serial_number = GlobalSettings::map().has("cam_serial_number") ? SETTING(cam_serial_number) : std::string("");
        
        DeviceInfoList list;
        CTlFactory::GetInstance().EnumerateDevices(list);
        for(size_t i=0; i<list.size(); i++) {
            auto &device = list[i];
            std::string name(device.GetFriendlyName());
            Print("[", i,"] Camera: ",name, " SN:",std::string(device.GetSerialNumber()));
            
            if(std::string(device.GetSerialNumber()) == serial_number) {
                _camera = new Camera_t(CTlFactory::GetInstance().CreateDevice(device));
            }
        }
        
        if(serial_number.empty())
            _camera = new Camera_t(CTlFactory::GetInstance().CreateFirstDevice());
        
        if(_camera == NULL)
            throw U_EXCEPTION("Cannot find camera with serial number ",serial_number,".");
        
        std::string name(_camera->GetDeviceInfo().GetFriendlyName());
        Print("Using camera ", name,".");
        
        _camera->RegisterConfiguration( new CAcquireContinuousConfiguration, RegistrationMode_ReplaceAll, Cleanup_Delete);
        //_camera->RegisterConfiguration( new CSoftwareTriggerConfiguration, RegistrationMode_ReplaceAll, Cleanup_Delete);
        _camera->GrabCameraEvents = false;
        _camera->Open();
        
        _camera->DeviceLinkSelector.SetValue(0);
        
        if(GenApi::IsWritable(_camera->DeviceLinkThroughputLimitMode)) {
            Print("Disabling USB throughput limit.");
            _camera->DeviceLinkThroughputLimitMode.SetValue(DeviceLinkThroughputLimitMode_Off);
        }
        
        if(GenApi::IsWritable(_camera->OffsetX)
           && GenApi::IsWritable(_camera->OffsetY))
        {
            _camera->OffsetX.SetValue(0);
            _camera->OffsetY.SetValue(0);
        }
        
        Size2 target_res = SETTING(cam_resolution);
        if(target_res.width == -1) {
            target_res = Size2(_camera->WidthMax.GetValue(),
                               _camera->HeightMax.GetValue());
            SETTING(cam_resolution) = target_res;
        }

        const int64_t offx = (_camera->WidthMax.GetValue() - target_res.width) * 0.5,
        offy = (_camera->HeightMax.GetValue() - target_res.height) * 0.5;
        Print("Setting dimensions to ",target_res.width,"x",target_res.height," (offsets ",offx,",",offy,")");
        
        _camera->CenterX.SetValue(true);
        _camera->CenterY.SetValue(true);
        
        _camera->Width.SetValue(target_res.width);
        _camera->Height.SetValue(target_res.height);
        
        _camera->ExposureTime.SetValue(READ_SETTING(cam_limit_exposure, int));
        
        if (READ_SETTING(cam_framerate, int) > 0) {
            _camera->AcquisitionFrameRateEnable.SetValue(true);
            _camera->AcquisitionFrameRate.SetValue(READ_SETTING(cam_framerate, int));
        }
        else {
            _camera->AcquisitionFrameRateEnable.SetValue(false);
            Print("Setting frame_rate from camera = ",_camera->ResultingFrameRate.GetValue());
            SETTING(cam_framerate) = int(_camera->ResultingFrameRate.GetValue());
        }
        
        // determine camera resolution
        while(true) {
            //_camera->StartGrabbing(1);
            _camera->StartGrabbing(1);
            //-cam_framerate 20 -cam_limit_exposure 2000
            if(_camera->GrabCameraEvents.GetValue() == true)
                _camera->GrabCameraEvents.SetValue(false);
            
            CGrabResultPtr ptrGrabResult;
            _camera->RetrieveResult(5000, ptrGrabResult, TimeoutHandling_ThrowException);
            
            if (ptrGrabResult && ptrGrabResult->GrabSucceeded()) {
                _size = cv::Size(ptrGrabResult->GetWidth(), ptrGrabResult->GetHeight());
                break;
                
            } else
                throw U_EXCEPTION("Could not grab frame for determining the resolution.");
        }
        
        if(_camera->IsGrabbing())
            _camera->StopGrabbing();
    }
    
    PylonCamera::~PylonCamera() {
        std::unique_lock<std::recursive_mutex> lock(_mutex);
        
        if(open())
            close();
    }
    
    bool PylonCamera::next(Image &current) {
        std::unique_lock<std::recursive_mutex> lock(_mutex);
        
        try {
            
            if(!_camera->IsGrabbing())
                _camera->StartGrabbing(GrabStrategy_LatestImageOnly);
            //_camera->StartGrabbing(GrabStrategy_OneByOne, GrabLoop_ProvidedByUser);//GrabStrategy_LatestImageOnly);
            //_camera->OutputQueueSize = _analysis->cache() - 1;
            
            if(_camera->GrabCameraEvents.GetValue() == true) {
                if ( _camera->WaitForFrameTriggerReady( 1000, TimeoutHandling_ThrowException))
                {
                    _camera->ExecuteSoftwareTrigger();
                }
            }
            
            CGrabResultPtr ptrGrabResult;
            
            while(!_camera->RetrieveResult( 5000, ptrGrabResult, TimeoutHandling_Return))
            { }
            
            if (ptrGrabResult && ptrGrabResult->GrabSucceeded())
            {
                CImageFormatConverter fc;
                fc.OutputPixelFormat = PixelType_Mono8;
                
                CPylonImage image;
                fc.Convert(image, ptrGrabResult);
                
                //if(rand()/float(RAND_MAX) >= 0.999)
                //    throw GenericException("Test", "test", 1);
                /*if ((_crop.x != 0 || _crop.y != 0) && (_crop.width != 0 || _crop.height != 0))
                {
                    // using crop offsets
                    cv::Mat tmp(image.GetHeight(), image.GetWidth(), CV_8UC1, (uchar*)image.GetBuffer());
                    tmp(_crop).copyTo(current.get());
                    
                } else {*/
                    current.create(image.GetHeight(), image.GetWidth(), 1, (uchar*)image.GetBuffer(), current.index());

                    
                //}
                auto t = ptrGrabResult->GetTimeStamp() / 1000;
                //static uint64_t previous = 0;
                //previous = t;
                
                current.set_timestamp(t);
                //current.set_timestamp(Image::now());
                
                return true;
                
            } else {
                FormatError{ "Grabbing failed with ", ptrGrabResult->GetErrorCode(),": ",ptrGrabResult->GetErrorDescription() };
            }
            
        } catch(const GenericException& g) {
            Print("An exception occurred: ",g.GetDescription());
        }
        
        return false;
    }
}

#endif
