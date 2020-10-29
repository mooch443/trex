#ifndef _VIDEO_H
#define _VIDEO_H

#include "types.h"

/*#if CV_MAJOR_VERSION >= 3
    #include <opencv2/opencv_modules.hpp>
    #if defined(HAVE_OPENCV_CUDACODEC) && defined(GRABBER_USE_CUVID)
        //#define VIDEOS_USE_CUDA
        #include <opencv2/cudacodec.hpp>
    #endif
#endif*/

//#define DEBUG_MEMORY

namespace cv {
    class VideoCapture;
    class Mat;
}

namespace cmn { class Video; }

/**
 * @class Video
 */
class cmn::Video {
public:
    typedef std::function<void(const cv::Mat &, int)> frame_callback;
    
    /**
     * Constructor of @class Video.
     */
    Video();
    
    /**
     * Destructor of @class Video.
     */
    ~Video();
    
    /**
     * Opens the file with the given name.
     * @param filename
     */
    bool open(const std::string& filename);
    
    /**
     * Closes the file if opened.
     */
    void close();
    
    /**
     * Framerate of the current video.
     */
    int framerate() const;
    
    /**
     * Length (in frames) of the current video.
     */
    int64_t length() const;
    
    /**
     * True if a video is loaded.
     */
    bool isOpened() const;
    
    /**
     * Returns the video dimensions.
     */
    const cv::Size& size() const;
    
    /**
     * Returns frame 'index' if a video is loaded and it exists.
     *
     * Result is cached.
     * @param index
     * @param lazy if this is set to true grab() will never be used and
     * whenever the next index is != the requested index, PROP_POS_FRAMES
     * will be set instead
     */
    //void frame(long_t index, cv::Mat& output, bool lazy = false);
    void frame(int64_t index, cv::Mat& output, bool lazy = false);
    
    /**
     * Sets a callback function for video playback. If a new frame is ready, this
     * function will be called and passed a cv::Mat and an integer (frame number).
     * @param f
     */
    void onFrame(const Video::frame_callback& f);
    
    /**
     * Sets the intrinsic parameters (focal length, skew, etc.) for this video. They
     * are saved inside the object and combined in a camera matrix.
     * @param focal The focal length in pixels for x and y
     * @param alpha_c Skew coefficient
     * @param cc Principal point (usually center of the image)
     */
    const Mat33& set_intrinsics(const Mat21& focal, const ScalarType alpha_c, const Mat21& cc);
    
    /**
     * Sets the distortion vector for this camera. Used during undistortion.
     * @param distortion Distortion coefficient vector from camera calibration
     */
    const Mat51& set_distortion(const Mat51& distortion);
    
    /**
     * Returns the undistorted version of the frame at given index.
     * @param index
     */
    const cv::Mat& undistorted_frame(int index);
    
    /**
     * Returns the camera matrix (skew, focal length, etc. parameters combined).
     */
    const Mat33& camera_matrix() const;
    
    /**
     * Returns the distortion vector as given by the camera calibration.
     */
    const Mat51& distortion() const;
    
    /**
     * Takes into account the neighbourhood of the given frame index (_mean_window
     * frames to each side) and calculates an averaged image. Substracted from the
     * current frame, it should give a pretty good approximation of all "moving"
     * parts.
     *
     * Results are cached and in the format CV_32F.
     * @param index Index of the frame on which the mean calculation is centered
     */
    const cv::Mat& calculate_mean(int index);
    
    /**
     * Calculates an OpenGL projection matrix based on all given camera parameters.
     *
     * http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-o
     * encv-intrinsic-matrix/
     */
    Mat44 glMatrix() const;
    
    /**
     * Clears all memory allocated for the cache by this video.
     */
    void clear();
    
private:
    int64_t _last_index;
    
    /**
     * The intrinsic parameters of the camera the video was recorded with.
     */
    Mat33 _camera_matrix;
    
    /**
     * The distortion parameter vector (5x1).
     */
    Mat51 _distortion;
    
    /**
     * Distortion maps calculated by initUndistortRectifyMap.
     */
    std::pair<gpuMat, gpuMat> _undistort_maps;
    
    /**
     * Flag that is true if the _undistort_maps have been calculated.
     */
    bool _maps_calculated;
    
    /**
     * Cached, undistorted versions of frames, retrieved from the video file.
     */
    std::map<size_t, cv::Mat> _undistorted_frames;
    
    /**
     * Cached frames retrieved from the video file.
     */
    std::map<size_t, cv::Mat> _frames;
    
    /**
     * A callback function that is called for every frame during playback.
     */
    frame_callback _frame_callback;
    
    /**
     * If a video is opened, this contains the pointer to its VideoCapture instance of
     * OpenCV. Otherwise it will be set to NULL.
     */
    cv::VideoCapture* _cap;
    
    /**
     * Tells the internal thread to stop.
     */
    bool _please_stop;
    
    /**
     * Thread for asynchronous playback.
     */
    std::thread* _thread;
    
    /**
     * Filename of the Video that is currently opened (if isOpened returns true).
     */
    std::string _filename;
    
    /**
     * Size of the video in pixels.
     */
    cv::Size _size;
    
    //! Temporary read cache
    cv::Mat read;
    
    /**
     * Calculates maps using OpenCVs initUndistortRectifyMap method, which are later
     * used to undistort frames of this video.
     */
    const std::pair<gpuMat, gpuMat>& calculate_undistort_maps();
    
#if defined(VIDEOS_USE_CUDA)
    cv::Ptr<cv::cudacodec::VideoReader> d_reader;
#endif
    
};

#endif
