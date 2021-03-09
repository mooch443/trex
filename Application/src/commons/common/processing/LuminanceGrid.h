#pragma once

#include <types.h>
#include <misc/vec2.h>

namespace cmn {
    class LuminanceGrid {
        struct Cell {
            int col, row;
            Bounds bounds;
            float brightness, relative_brightness;
            float threshold;
            
            Cell() {}
            Cell(int col, int row, const Bounds& bounds)
                : col(col), row(row), bounds(bounds)
            {
                
            }
        };
        
        GETTER(Bounds, bounds)
        
        static constexpr int cells_per_row = 100;
        static constexpr int cell_count = cells_per_row * cells_per_row;
        
        const Vec2 factors;
        std::array<Cell, cell_count> _cells;
        GETTER(std::vector<float>, thresholds)
        
        gpuMat _gpumat;
        gpuMat _corrected_average;
        GETTER(gpuMat, relative_brightness)
        
        std::mutex buffer_mutex;
        gpuMat _buffer;
        
    public:
        LuminanceGrid(const cv::Mat& background);
        float relative_threshold(int x, int y) const;
        const gpuMat& gpumat() const { return _gpumat; }
        const gpuMat& corrected_average() const { return _corrected_average; }
        
        void correct_image(const gpuMat& input, cv::Mat& output);
#ifdef USE_GPU_MAT
        void correct_image(gpuMat& input_output);
#endif
        void correct_image(cv::Mat& input_output);
    };
}
