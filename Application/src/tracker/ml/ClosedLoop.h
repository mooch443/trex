#pragma once

#include <commons.pc.h>
#include <file/Path.h>

namespace ml {


struct ClosedLoop {
    std::optional<std::future<void>> _python_future;
    
    cmn::file::Path closed_loop_path;
    
    ClosedLoop();
    
    ClosedLoop(ClosedLoop&&) = default;
    ClosedLoop& operator=(ClosedLoop&&) = default;
    
    ~ClosedLoop();
    
private:
    void retrieve_closed_loop(bool blocking);
    
    template<typename... Args>
    [[nodiscard]] std::future<void> module_proxy(Args&&... args);
    
public:
    void update_loop(std::function<glz::json_t()> frame_info);
};

}
