#pragma once
#include <commons.pc.h>
#include <Scene.h>
#include <gui/types/ScrollableList.h>
#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/DynamicGUI.h>
#include <gui/DrawBase.h>
#include <gui/types/ListItemTypes.h>
#include <gui/DynamicVariable.h>
#include <misc/RecentItems.h>
#include <misc/ThreadPool.h>
#include <misc/ConnectedTasks.h>
#include <tracking/Tracker.h>

namespace gui {

class TrackingScene : public Scene {
    /**
     * @struct Data
     *
     * Represents a container for video analysis data and associated utilities.
     */
    struct Data {
        
        /**
         * @brief Represents the video file being analyzed.
         */
        pv::File video;

        /**
         * @brief Tracker used for tracking objects/entities in the video.
         */
        track::Tracker tracker;

        /**
         * @brief Flag indicating whether to stop the analysis process.
         *
         * Setting this to 'true' can be used to request a halt to ongoing analysis.
         */
        std::atomic<bool> please_stop_analysis{false};

        /**
         * @brief Manages the analysis tasks and their inter-dependencies.
         */
        ConnectedTasks analysis;

        /**
         * @brief Represents the current frame ID being processed.
         */
        std::atomic<Frame_t> currentID;

        /**
         * @brief Pool of threads used for parallel processing of analysis tasks.
         */
        GenericThreadPool pool;

        /**
         * @brief A queue of unused frames. Frames can be reused to avoid frequent allocations.
         */
        std::queue<std::unique_ptr<track::PPFrame>> unused;
        
        /**
         * @brief Constructor for the Data struct.
         *
         * Initializes the Data object with provided average image, video, and analysis functions.
         *
         * @param average Pointer to the average image.
         * @param video The video file to be analyzed.
         * @param functions A list of functions representing the analysis stages.
         */
        Data(Image::Ptr&& average,
             pv::File&& video,
             std::vector<std::function<bool(ConnectedTasks::Type&&, const ConnectedTasks::Stage&)>>&& functions);
    };
    
    std::unique_ptr<Data> _data;
    std::mutex _task_mutex;
    
    std::vector<std::function<bool(ConnectedTasks::Type&&, const ConnectedTasks::Stage&)>> tasks;
    
    // The HorizontalLayout for the two buttons and the image
    dyn::DynamicGUI dynGUI;
    
    Size2 window_size;
    Size2 element_size;
    Vec2 left_center;

    std::vector<sprite::Map> _fish_data;
    std::vector<std::shared_ptr<dyn::VarBase_t>> _individuals;
    
public:
    TrackingScene(Base& window);

    void activate() override;
    void deactivate() override;

    void _draw(DrawStructure& graph);
    
private:
    bool stage_0(ConnectedTasks::Type&&);
    bool stage_1(ConnectedTasks::Type&&);
    
    void init_video();
};
}
