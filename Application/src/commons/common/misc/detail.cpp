#include <misc/detail.h>
#include <types.h>
#include <processing/LuminanceGrid.h>
#include <misc/Image.h>
#include <gui/colors.h>
#include <misc/metastring.h>

namespace cmn {
    IMPLEMENT(CrashProgram::do_crash) = false;
    IMPLEMENT(CrashProgram::crash_pid);
    IMPLEMENT(CrashProgram::main_pid);
    
    void check_conditions(std::vector<HorizontalLine>& array) {
        if(array.empty())
            return;
        
        auto check_obj = [](HorizontalLine&, bool cond, const std::string& cond_name) {
            if(!cond) {
                Error("The program is not pleased. '%S'", &cond_name);
            }
        };
#define _assert(obj, COND) { check_obj(obj, (COND), #COND); }
        
        std::string wrong;
        
        for (size_t i=0; i<array.size()-1; i++) {
            auto &a = array.at(i);
            auto &b = array.at(i+1);
            
            _assert(a, a.x1 >= a.x0);
            _assert(b, b.x1 >= b.x0);
            
            _assert(a, a.y <= b.y);
            _assert(a, !a.overlap(b));
            _assert(a, a.y != b.y || a.x1+1 < b.x0);
        }
    }
    
    void HorizontalLine::repair_lines_array(std::vector<HorizontalLine> &ls, std::vector<uchar>& pixels)
    {
        std::set<HorizontalLine> lines(ls.begin(), ls.end());
        std::vector<uchar> corrected;
        corrected.reserve(pixels.size());
        if(lines.empty())
            return;
        
        auto prev = lines.begin();
        auto it = prev;
        ++it;
        auto pxptr = pixels.data();
        
        std::vector<HorizontalLine> result{ls.front()};
        corrected.insert(corrected.end(), pixels.begin(), pixels.begin() + ptr_safe_t(ls.front().x1) - ptr_safe_t(ls.front().x0) + 1);
        
        for(; it != lines.end();) {
            if(result.back().y == it->y) {
                if(result.back().x1 >= it->x0) {
                    // they do overlap in x and y
                    //Debug("(%d) Merging %d-%d + %d-%d", it->y, result.back().x0, result.back().x1, it->x0, it->x1);
                    //*prev = prev->merge(*it);
                    
                    if(result.back().x0 > it->x0) {
                        auto offset = ptr_safe_t(result.back().x0) - ptr_safe_t(it->x0);
                        // need to add [offset] pixels to the front
                        corrected.insert(corrected.end() - (ptr_safe_t(result.back().x1) - ptr_safe_t(result.back().x0) + 1), pxptr, pxptr + offset);
                    }
                    
                    if(result.back().x1 < it->x1) {
                        auto offset = ptr_safe_t(it->x1) - ptr_safe_t(result.back().x1);
                        corrected.insert(corrected.end(), pxptr + ptr_safe_t(it->x1) - ptr_safe_t(it->x0) + 1 - offset, pxptr + ptr_safe_t(it->x1) - ptr_safe_t(it->x0) + 1);
                    }
                    
                    result.back() = result.back().merge(*it);
                    it = lines.erase(it);
                    
                    pxptr = pxptr + ptr_safe_t(it->x1) - ptr_safe_t(it->x0) + 1;
                    continue;
                }
                
            } else if(result.back().y > it->y)
                U_EXCEPTION("Cannot repair %d > %d", result.back().y, it->y);
            
            prev = it;
            result.push_back(*it);
            
            auto start = pxptr;
            pxptr = pxptr + ptr_safe_t(it->x1) - ptr_safe_t(it->x0) + 1;
            corrected.insert(corrected.end(), start, pxptr);
            ++it;
        }
        
        ls = result;
        
        ptr_safe_t L = 0;
        for(auto &line : result) {
            L += ptr_safe_t(line.x1) - ptr_safe_t(line.x0) + 1;
        }
        
        assert(L == corrected.size());
    }
    
    void HorizontalLine::repair_lines_array(std::vector<HorizontalLine> &ls)
    {
        std::set<HorizontalLine> lines(ls.begin(), ls.end());
        if(lines.empty())
            return;
        
        auto prev = lines.begin();
        auto it = prev;
        ++it;
        
        std::vector<HorizontalLine> result{ls.front()};
        
        for(; it != lines.end();) {
            if(result.back().y == it->y) {
                if(result.back().x1 >= it->x0) {
                    // they do overlap in x and y
                    result.back() = result.back().merge(*it);
                    it = lines.erase(it);
                    continue;
                }
                
            } else if(result.back().y > it->y)
                U_EXCEPTION("Cannot repair %d > %d", result.back().y, it->y);
            
            prev = it;
            result.push_back(*it);
            
            ++it;
        }
        
        ls = result;
    }
    
    enum InsertStatus {
        NONE = 0,
        ADDED,
        MERGED
    };
    
    InsertStatus insert(std::vector<HorizontalLine>& array, const HorizontalLine& p) {
#if !ORDERED_HORIZONTAL_LINES
        array.push_back(p);
#else
        InsertStatus status = NONE;
        
        // find best insertion spot
        uint64_t index = std::numeric_limits<long_t>::max();
        int k = 0;
        for (long_t k=array.size()-1; k>=0; k--) {
            auto &c = array.at(k);
            
            if(c.y < p.y) {
                index = k + 1;
                array.insert(array.begin() + index, p);
                status = ADDED;
                break;
                
            }
            else if(c.y > p.y)
                continue;
            else if (c.overlap_x(p)) {
                // lines intersect. merge them.
                c = c.merge(p);
                
                status = MERGED;
                index = k;
                break;
                
            } else if(c.x0 <= p.x0) {
                index = k + 1;
                array.insert(array.begin() + index, p);
                
                status = ADDED;
                break;
            }
        }
        
        if (status == NONE && k <= 0) {
            array.insert(array.begin(), p);
            index = 0;
            status = ADDED;
            
        } else if (status == NONE) {
            index = array.size();
            array.push_back(p);
            status = ADDED;
        }
        
        bool change = false;
        do {
            change = false;
            
            for (long_t k = long_t(index) - 1; k<long_t(array.size())-1 && k < long_t(index) + 1; k++) {
                if(k < 0)
                    continue;
                
                auto &obj0 = array.at(k);
                auto &obj1 = array.at(k+1);
                
                if (obj0.overlap(obj1)) {
                    obj0 = obj0.merge(obj1);
                    array.erase(array.begin() + k + 1);
                    index = k;
                    change = true;
                    k--;
                }
            }
            
        } while(change);
        
        //check_conditions(array);
        return status;
        
#endif
    }
    
#if ORDERED_HORIZONTAL_LINES
#if true
    void dilate(std::vector<HorizontalLine>& array, int times, int max_cols, int max_rows) {
        if(array.empty())
            return;
        
        assert(times == 1);
        
        std::vector<HorizontalLine> ret;
        ret.reserve(array.size() * 3);
        
        auto it_current = ret.end(), it_before = ret.end();
        HorizontalLine *ptr_current = array.data(), *ptr_before = NULL;
        
        auto ptr = array.data();
        int current_y = array[0].y;
        bool previous_set = false;
        HorizontalLine previous;
        
        auto expand_row_with_row = [&ret](HorizontalLine *s0, HorizontalLine *e0,
                                          decltype(ret)::iterator s1, decltype(ret)::iterator e1,
                                          ushort insert_y)
        -> decltype(ret)::iterator
        {
            auto it = s1;
            HorizontalLine p;
            
            while(it != e1 && s0 < e0) {
                // x-pand multiple times...
                if(it->x0 > s0->x1+1) {
                    p = *s0;
                    p.y = insert_y;
                    
                    assert(ret.capacity() >= ret.size()+1);
                    it = ret.insert(it, p)+1;
                    e1++;
                    s0++;
                    
                } else if(it->x1 >= s0->x0-1) {
                    // merge...
                    *it = it->merge(*s0);
                    s0++;
                    
                } else {
                    ++it;
                }
            }
            
            while(s0 < e0) {
                p = *s0;
                p.y = insert_y;
                
                assert(ret.capacity() >= ret.size()+1);
                it = ret.insert(it, p)+1;
                e1++;
                s0++;
            }
            
            return e1;
        };
        
        //! this function is called whenever a row is completed
        auto complete_row = [&]() {
            if(previous_set) {
                ret.insert(ret.end(), previous);
                previous_set = false;
            }
            
            if(max_rows > 0 && current_y >= max_rows)
                return;
            
            // expand previous row into current row
            if(ptr_before) {
                if(current_y == ptr_before->y+1)
                    expand_row_with_row(ptr_before, ptr_current, it_current, ret.end(), current_y);
                else {
                    it_before = it_current;
                    it_current = expand_row_with_row(ptr_before, ptr_current, it_current, it_current, ptr_before->y+1);
                }
            }
            
            // expand current row into previous row
            if(current_y-1 >= 0) {
                if(it_before->y != current_y-1)
                    it_before = it_current;
                it_current = expand_row_with_row(ptr_current, ptr, it_before, it_current, current_y-1);
            }
            
            ptr_before = ptr_current;
            ptr_current = ptr;
            it_before = it_current;
            it_current = ret.end(); // will be first of current row
            
            if(ptr != array.data()+array.size())
                current_y = ptr->y;
            else
                current_y++;
        };
        
        // expand in x-direction
        for (size_t i=0; i<array.size(); i++, ptr++) {
            if(ptr->y != current_y) {
                complete_row();
            }
            
            if(previous_set && previous.x1 >= int(ptr->x0)-times) {
                // merge with previous line
                previous.x1 = max_cols > 0 
                    ? min(ptr_safe_t(ptr->x1)+ptr_safe_t(times), ptr_safe_t(max_cols-1)) 
                    : ptr_safe_t(ptr->x1)+times;
                
            } else {
                if(previous_set)
                    ret.insert(ret.end(), previous);
                
                previous = *ptr;
                previous.x0 -= min((ptr_safe_t)previous.x0, (ptr_safe_t)times); // expand left
                
                // expand right
                previous.x1 = max_cols > 0
                    ? min(ptr_safe_t(previous.x1)+times, ptr_safe_t(max_cols-1))
                    : ptr_safe_t(previous.x1)+times;
                
                previous_set = true;
            }
        }
        
        complete_row(); // add current row to previous row and vice-versa
        complete_row(); // copy/paste last row
        
        std::swap(array, ret);
    }
#else
    void dilate(std::vector<HorizontalLine>& array, int times, int max_cols, int max_rows) {
        int current_y = array.empty() ? 0 : array.front().y;
        Timer timer;
        
        std::vector<HorizontalLine> ret;
        ret.reserve(array.size()*1.25);
        
        size_t prev_start = 0;
        size_t prev_end = 0;
        size_t current_start = 0;
        
        //! Offset in case values might be lower than zero
        size_t offset = 0;
        
        auto inc_offset = [&]() {
            for (auto &r : ret) {
                r.y++;
            }
            
            offset++;
        };
        
        for (size_t i=0; i<array.size(); i++) {
            auto p = array[i];
            p.y += offset;
            
            if (p.y != current_y) {
                for (size_t j=prev_start; j<=prev_end; j++) {
                    auto p = array[j];
                    p.y++;
                    
                    insert(ret, p);
                }
                
                prev_start = current_start;
                prev_end = i - 1;
                
                current_start = i;
                current_y = p.y;
            }
            
            if(i == array.size())
                break;
            
            // for the first row
            if(current_start == prev_start) {
                auto c = p;
                if(c.y == 0) {
                    inc_offset();
                    c.y++;
                    p.y++;
                }
                c.y--;
                
                insert(ret, c);
                
            } else {
                // every other row
                auto c = p;
                if(c.y == 0) {
                    inc_offset();
                    c.y++;
                    p.y++;
                }
                c.y--;
                
                insert(ret, c);
            }
            
            // dilate in x-direction
            if (p.x0 > 0)
                p.x0--;
            p.x1++;
            
            insert(ret, p);
        }
        
        auto p = array.back();
        p.y+=offset;
        p.y++;
        insert(ret, p);
        
        array = ret;
        
        assert(USHRT_MAX > offset);
        //Debug("Took %fms", timer.elapsed()*1000);
        //return offset;
    }
#endif
#endif
    
    inline void _lines_initialize_matrix(cv::Mat& mat, int w, int h, int type = CV_8UC1) {
        if (mat.empty() || mat.type() != type || mat.rows != h || mat.cols != w)
            mat = cv::Mat::zeros(h, w, type);
        else
            mat = cv::Scalar(0);
    };
    
    cv::Rect2i lines_dimensions(const std::vector<HorizontalLine>& lines) {
        float mx = FLT_MAX, my = FLT_MAX,
              px = -FLT_MAX, py = -FLT_MAX;
        
        for (auto &l : lines) {
            if(mx > l.x0)
                mx = l.x0;
            if(my > l.y)
                my = l.y;
            
            if(px < l.x1)
                px = l.x1;
            if(py < l.y)
                py = l.y;
        }
        
        float w = px - mx + 1,
        h = py - my + 1;
        
        return cv::Rect2i(mx, my, w, h);
    }
    
    void lines2mask(const std::vector<HorizontalLine>& lines, cv::Mat& output_mask, const int value) {
        auto r = lines_dimensions(lines);
        _lines_initialize_matrix(output_mask, r.width, r.height);
        
        for (auto &l : lines) {
            for (int x=l.x0; x<=l.x1; x++)
                output_mask.at<uchar>(l.y - r.y, x - r.x) = value;
        }
    }
    
    std::pair<cv::Rect2i, size_t> imageFromLines(const std::vector<HorizontalLine>& lines, cv::Mat* output_mask, cv::Mat* output_greyscale, cv::Mat* output_differences, const std::vector<uchar>& pixels, int base_threshold, const LuminanceGrid& grid, const cv::Mat& average, int padding)
    {
        auto r = lines_dimensions(lines);
        r.x -= padding;
        r.y -= padding;
        r.width += padding * 2;
        r.height += padding * 2;
        
        // initialize matrices
        if(output_mask)
            _lines_initialize_matrix(*output_mask, r.width, r.height);
        if(output_greyscale)
            _lines_initialize_matrix(*output_greyscale, r.width, r.height);
        if(output_differences)
            _lines_initialize_matrix(*output_differences, r.width, r.height);
        
        size_t recount = 0;
        int c, diff;
        
        auto pixels_ptr = pixels.data();
        for (auto &l : lines) {
            for (int x=l.x0; x<=l.x1; x++) {
                c = *pixels_ptr++;
                diff = min(UCHAR_MAX, cmn::abs(int(average.at<uchar>(l.y,x)) - c));
                //diff = min(UCHAR_MAX, max(0, int(average->at<uchar>(l.y, x)) - c));
                
                if(!base_threshold || diff >= base_threshold * grid.relative_threshold(x, l.y)) {
                    if(output_mask)
                        output_mask->at<uchar>(l.y - r.y, x - r.x) = 255;
                    if(output_greyscale)
                        output_greyscale->at<uchar>(l.y - r.y, x - r.x) = c;
                    if(output_differences)
                        output_differences->at<uchar>(l.y - r.y, x - r.x) = diff;
                    recount++;
                }
            }
        }
        
        return {r, recount};
    }
    
    std::pair<cv::Rect2i, size_t> imageFromLines(const std::vector<HorizontalLine>& lines, cv::Mat* output_mask, cv::Mat* output_greyscale, cv::Mat* output_differences, const std::vector<uchar>* pixels, const int threshold, const Image* average, int padding)
    {
        auto r = lines_dimensions(lines);
        r.x -= padding;
        r.y -= padding;
        r.width += padding * 2;
        r.height += padding * 2;
        
        // initialize matrices
        if(output_mask)
            _lines_initialize_matrix(*output_mask, r.width, r.height);
        if(output_greyscale)
            _lines_initialize_matrix(*output_greyscale, r.width, r.height);
        if(output_differences)
            _lines_initialize_matrix(*output_differences, r.width, r.height);
        
        uint32_t pos = 0;
        size_t recount = 0;
        
        assert(!output_differences || average);
        
        // use individual ifs for cases to have less
        // jumps in assembler code within the loops
        if(!threshold) {
            int c;
            
            for (auto &l : lines) {
                recount += size_t(l.x1) - size_t(l.x0) + 1;
                
                for (int x=l.x0; x<=l.x1; x++) {
                    if(output_mask)
                        output_mask->at<uchar>(l.y - r.y, x - r.x) = 255;
                    if(output_greyscale || output_differences) {
                        c = pixels->at(pos++);
                        
                        if(output_greyscale)
                            output_greyscale->at<uchar>(l.y - r.y, x - r.x) = c;
                        if(output_differences)
                            output_differences->at<uchar>(l.y - r.y, x - r.x) = min(UCHAR_MAX, cmn::abs(int(average->at(l.y,x)) - c));
                            //min(UCHAR_MAX, max(0, int(average->at<uchar>(l.y, x)) - c));
                    }
                }
            }
            
        } else {
            int c, diff;
            assert(average);
            assert(threshold);
            assert(pixels);
            
            auto pixels_ptr = pixels->data();
            for (auto &l : lines) {
                for (int x=l.x0; x<=l.x1; x++) {
                    c = *pixels_ptr++;
                    diff = min(UCHAR_MAX, cmn::abs(int(average->at(l.y,x)) - c));
                    //diff = min(UCHAR_MAX, max(0, int(average->at<uchar>(l.y, x)) - c));
                    
                    if(diff >= threshold) {
                        if(output_mask)
                            output_mask->at<uchar>(l.y - r.y, x - r.x) = 255;
                        if(output_greyscale)
                            output_greyscale->at<uchar>(l.y - r.y, x - r.x) = c;
                        if(output_differences)
                            output_differences->at<uchar>(l.y - r.y, x - r.x) = diff;
                        recount++;
                    }
                }
            }
            
        }
        
        return {r, recount};
    }
}

#include <misc/SpriteMap.h>
namespace cmn {
    namespace sprite {
        enum SupportedDataTypes {
            INT,
            LONG,
            FLOAT,
            STRING,
            VECTOR,
            BOOL,
            
            INVALID
        };
        
        SupportedDataTypes estimate_datatype(const std::string& value) {
            if(value.empty())
                return INVALID;
            
            if(utils::beginsWith(value, '"') && utils::endsWith(value, '"'))
                return STRING;
            if(utils::beginsWith(value, '\'') && utils::endsWith(value, '\''))
                return STRING;
            
            if(utils::beginsWith(value, '[') && utils::endsWith(value, ']'))
                return VECTOR;
            
            if((value.at(0) >= '0' && value.at(0) <= '9')
               || value.at(0) == '-' || value.at(0) == '+')
            {
                if(utils::contains(value, '.') || utils::endsWith(value, 'f'))
                    return FLOAT;
                if(utils::endsWith(value, 'l'))
                    return LONG;
                
                return INT;
            }
            
            if(value == "true" || value == "false")
                return BOOL;
            
            return INVALID;
        }
        
        std::set<std::string> parse_values(Map& map, std::string str) {
            str = utils::trim(str);
            if(str.empty())
                return {};
            
            std::set<std::string> added;
            
            if(utils::beginsWith(str, '{') && utils::endsWith(str, '}'))
                str = str.substr(1, str.length()-2);
            else
                U_EXCEPTION("Malformed map string '%S'", &str);
            
            auto parts = util::parse_array_parts(str);
            for (auto &p : parts) {
                auto key_value = util::parse_array_parts(p, ':');
                auto &key = key_value[0];
                std::string value;
                if(key_value.size() > 1) value = key_value[1];
                
                if(!key.empty() && (key[0] == '\'' || key[0] == '"') && key[0] == key[key.length()-1])
                    key = key.substr(1, key.length()-2);
                
                if(map.has(key)) {
                    // try to set with existing type
                    map[key].get().set_value_from_string(value);
                    
                } else {
                    // key is not yet present in the map, estimate type
                    auto type = estimate_datatype(value);
                    
                    switch (type) {
                        case STRING: {
                            if(value.length() && (value.at(0) == '"' || value.at(0) == '\''))
                                value = value.substr(1, value.length()-1);
                            if(value.length() && (value.at(value.length()-1) == '"' || value.at(value.length()-1) == '\''))
                                value = value.substr(0, value.length()-1);
                            map[key] = value;
                            
                            break;
                        }
                            
                        case INT:
                            map[key] = std::stoi(value);
                            break;
                        case FLOAT:
                            map[key] = std::stof(value);
                            break;
                        case LONG:
                            map[key] = std::stol(value);
                            break;
                            
                        case VECTOR:
                            if(!map.has("quiet") || !map.get<bool>("quiet"))
                                U_EXCEPTION("(Key '%S') Vector not yet implemented.", &key);
                            break;
                            
                        case INVALID:
                            if(!map.has("quiet") || !map.get<bool>("quiet"))
                                Warning("Data of invalid type '%S' for key '%S'", &value, &key);
                            break;
                            
                        case BOOL:
                            map[key] = value == "true" ? true : false;
                            break;
                            
                        default:
                            break;
                    }
                    
                }
                
                added.insert(key);
            }
            
            return added;
        }
        
        Map parse_values(std::string str) {
            Map map;
            parse_values(map, str);
            return map;
        }
    }
    
    void set_thread_name(const std::string& name) {
#if __APPLE__
        pthread_setname_np(name.c_str());
#elif __linux__
        pthread_setname_np(pthread_self(), name.c_str());
#endif
    }

    std::string get_thread_name() {
#ifndef WIN32
        char buffer[1024];
        pthread_getname_np(pthread_self(), buffer, sizeof(buffer));
        return std::string(buffer);
#else
        return "thread";
#endif
    }
    
    IMPLEMENT(Viridis::data_bgr){{
        Viridis::value_t {0.26700401,  0.00487433,  0.32941519},
        Viridis::value_t {0.26851048,  0.00960483,  0.33542652},
        Viridis::value_t {0.26994384,  0.01462494,  0.34137895},
        Viridis::value_t {0.27130489,  0.01994186,  0.34726862},
        Viridis::value_t {0.27259384,  0.02556309,  0.35309303},
        Viridis::value_t {0.27380934,  0.03149748,  0.35885256},
        Viridis::value_t {0.27495242,  0.03775181,  0.36454323},
        Viridis::value_t {0.27602238,  0.04416723,  0.37016418},
        Viridis::value_t {0.2770184 ,  0.05034437,  0.37571452},
        Viridis::value_t {0.27794143,  0.05632444,  0.38119074},
        Viridis::value_t {0.27879067,  0.06214536,  0.38659204},
        Viridis::value_t {0.2795655 ,  0.06783587,  0.39191723},
        Viridis::value_t {0.28026658,  0.07341724,  0.39716349},
        Viridis::value_t {0.28089358,  0.07890703,  0.40232944},
        Viridis::value_t {0.28144581,  0.0843197 ,  0.40741404},
        Viridis::value_t {0.28192358,  0.08966622,  0.41241521},
        Viridis::value_t {0.28232739,  0.09495545,  0.41733086},
        Viridis::value_t {0.28265633,  0.10019576,  0.42216032},
        Viridis::value_t {0.28291049,  0.10539345,  0.42690202},
        Viridis::value_t {0.28309095,  0.11055307,  0.43155375},
        Viridis::value_t {0.28319704,  0.11567966,  0.43611482},
        Viridis::value_t {0.28322882,  0.12077701,  0.44058404},
        Viridis::value_t {0.28318684,  0.12584799,  0.44496   },
        Viridis::value_t {0.283072  ,  0.13089477,  0.44924127},
        Viridis::value_t {0.28288389,  0.13592005,  0.45342734},
        Viridis::value_t {0.28262297,  0.14092556,  0.45751726},
        Viridis::value_t {0.28229037,  0.14591233,  0.46150995},
        Viridis::value_t {0.28188676,  0.15088147,  0.46540474},
        Viridis::value_t {0.28141228,  0.15583425,  0.46920128},
        Viridis::value_t {0.28086773,  0.16077132,  0.47289909},
        Viridis::value_t {0.28025468,  0.16569272,  0.47649762},
        Viridis::value_t {0.27957399,  0.17059884,  0.47999675},
        Viridis::value_t {0.27882618,  0.1754902 ,  0.48339654},
        Viridis::value_t {0.27801236,  0.18036684,  0.48669702},
        Viridis::value_t {0.27713437,  0.18522836,  0.48989831},
        Viridis::value_t {0.27619376,  0.19007447,  0.49300074},
        Viridis::value_t {0.27519116,  0.1949054 ,  0.49600488},
        Viridis::value_t {0.27412802,  0.19972086,  0.49891131},
        Viridis::value_t {0.27300596,  0.20452049,  0.50172076},
        Viridis::value_t {0.27182812,  0.20930306,  0.50443413},
        Viridis::value_t {0.27059473,  0.21406899,  0.50705243},
        Viridis::value_t {0.26930756,  0.21881782,  0.50957678},
        Viridis::value_t {0.26796846,  0.22354911,  0.5120084 },
        Viridis::value_t {0.26657984,  0.2282621 ,  0.5143487 },
        Viridis::value_t {0.2651445 ,  0.23295593,  0.5165993 },
        Viridis::value_t {0.2636632 ,  0.23763078,  0.51876163},
        Viridis::value_t {0.26213801,  0.24228619,  0.52083736},
        Viridis::value_t {0.26057103,  0.2469217 ,  0.52282822},
        Viridis::value_t {0.25896451,  0.25153685,  0.52473609},
        Viridis::value_t {0.25732244,  0.2561304 ,  0.52656332},
        Viridis::value_t {0.25564519,  0.26070284,  0.52831152},
        Viridis::value_t {0.25393498,  0.26525384,  0.52998273},
        Viridis::value_t {0.25219404,  0.26978306,  0.53157905},
        Viridis::value_t {0.25042462,  0.27429024,  0.53310261},
        Viridis::value_t {0.24862899,  0.27877509,  0.53455561},
        Viridis::value_t {0.2468114 ,  0.28323662,  0.53594093},
        Viridis::value_t {0.24497208,  0.28767547,  0.53726018},
        Viridis::value_t {0.24311324,  0.29209154,  0.53851561},
        Viridis::value_t {0.24123708,  0.29648471,  0.53970946},
        Viridis::value_t {0.23934575,  0.30085494,  0.54084398},
        Viridis::value_t {0.23744138,  0.30520222,  0.5419214 },
        Viridis::value_t {0.23552606,  0.30952657,  0.54294396},
        Viridis::value_t {0.23360277,  0.31382773,  0.54391424},
        Viridis::value_t {0.2316735 ,  0.3181058 ,  0.54483444},
        Viridis::value_t {0.22973926,  0.32236127,  0.54570633},
        Viridis::value_t {0.22780192,  0.32659432,  0.546532  },
        Viridis::value_t {0.2258633 ,  0.33080515,  0.54731353},
        Viridis::value_t {0.22392515,  0.334994  ,  0.54805291},
        Viridis::value_t {0.22198915,  0.33916114,  0.54875211},
        Viridis::value_t {0.22005691,  0.34330688,  0.54941304},
        Viridis::value_t {0.21812995,  0.34743154,  0.55003755},
        Viridis::value_t {0.21620971,  0.35153548,  0.55062743},
        Viridis::value_t {0.21429757,  0.35561907,  0.5511844 },
        Viridis::value_t {0.21239477,  0.35968273,  0.55171011},
        Viridis::value_t {0.2105031 ,  0.36372671,  0.55220646},
        Viridis::value_t {0.20862342,  0.36775151,  0.55267486},
        Viridis::value_t {0.20675628,  0.37175775,  0.55311653},
        Viridis::value_t {0.20490257,  0.37574589,  0.55353282},
        Viridis::value_t {0.20306309,  0.37971644,  0.55392505},
        Viridis::value_t {0.20123854,  0.38366989,  0.55429441},
        Viridis::value_t {0.1994295 ,  0.38760678,  0.55464205},
        Viridis::value_t {0.1976365 ,  0.39152762,  0.55496905},
        Viridis::value_t {0.19585993,  0.39543297,  0.55527637},
        Viridis::value_t {0.19410009,  0.39932336,  0.55556494},
        Viridis::value_t {0.19235719,  0.40319934,  0.55583559},
        Viridis::value_t {0.19063135,  0.40706148,  0.55608907},
        Viridis::value_t {0.18892259,  0.41091033,  0.55632606},
        Viridis::value_t {0.18723083,  0.41474645,  0.55654717},
        Viridis::value_t {0.18555593,  0.4185704 ,  0.55675292},
        Viridis::value_t {0.18389763,  0.42238275,  0.55694377},
        Viridis::value_t {0.18225561,  0.42618405,  0.5571201 },
        Viridis::value_t {0.18062949,  0.42997486,  0.55728221},
        Viridis::value_t {0.17901879,  0.43375572,  0.55743035},
        Viridis::value_t {0.17742298,  0.4375272 ,  0.55756466},
        Viridis::value_t {0.17584148,  0.44128981,  0.55768526},
        Viridis::value_t {0.17427363,  0.4450441 ,  0.55779216},
        Viridis::value_t {0.17271876,  0.4487906 ,  0.55788532},
        Viridis::value_t {0.17117615,  0.4525298 ,  0.55796464},
        Viridis::value_t {0.16964573,  0.45626209,  0.55803034},
        Viridis::value_t {0.16812641,  0.45998802,  0.55808199},
        Viridis::value_t {0.1666171 ,  0.46370813,  0.55811913},
        Viridis::value_t {0.16511703,  0.4674229 ,  0.55814141},
        Viridis::value_t {0.16362543,  0.47113278,  0.55814842},
        Viridis::value_t {0.16214155,  0.47483821,  0.55813967},
        Viridis::value_t {0.16066467,  0.47853961,  0.55811466},
        Viridis::value_t {0.15919413,  0.4822374 ,  0.5580728 },
        Viridis::value_t {0.15772933,  0.48593197,  0.55801347},
        Viridis::value_t {0.15626973,  0.4896237 ,  0.557936  },
        Viridis::value_t {0.15481488,  0.49331293,  0.55783967},
        Viridis::value_t {0.15336445,  0.49700003,  0.55772371},
        Viridis::value_t {0.1519182 ,  0.50068529,  0.55758733},
        Viridis::value_t {0.15047605,  0.50436904,  0.55742968},
        Viridis::value_t {0.14903918,  0.50805136,  0.5572505 },
        Viridis::value_t {0.14760731,  0.51173263,  0.55704861},
        Viridis::value_t {0.14618026,  0.51541316,  0.55682271},
        Viridis::value_t {0.14475863,  0.51909319,  0.55657181},
        Viridis::value_t {0.14334327,  0.52277292,  0.55629491},
        Viridis::value_t {0.14193527,  0.52645254,  0.55599097},
        Viridis::value_t {0.14053599,  0.53013219,  0.55565893},
        Viridis::value_t {0.13914708,  0.53381201,  0.55529773},
        Viridis::value_t {0.13777048,  0.53749213,  0.55490625},
        Viridis::value_t {0.1364085 ,  0.54117264,  0.55448339},
        Viridis::value_t {0.13506561,  0.54485335,  0.55402906},
        Viridis::value_t {0.13374299,  0.54853458,  0.55354108},
        Viridis::value_t {0.13244401,  0.55221637,  0.55301828},
        Viridis::value_t {0.13117249,  0.55589872,  0.55245948},
        Viridis::value_t {0.1299327 ,  0.55958162,  0.55186354},
        Viridis::value_t {0.12872938,  0.56326503,  0.55122927},
        Viridis::value_t {0.12756771,  0.56694891,  0.55055551},
        Viridis::value_t {0.12645338,  0.57063316,  0.5498411 },
        Viridis::value_t {0.12539383,  0.57431754,  0.54908564},
        Viridis::value_t {0.12439474,  0.57800205,  0.5482874 },
        Viridis::value_t {0.12346281,  0.58168661,  0.54744498},
        Viridis::value_t {0.12260562,  0.58537105,  0.54655722},
        Viridis::value_t {0.12183122,  0.58905521,  0.54562298},
        Viridis::value_t {0.12114807,  0.59273889,  0.54464114},
        Viridis::value_t {0.12056501,  0.59642187,  0.54361058},
        Viridis::value_t {0.12009154,  0.60010387,  0.54253043},
        Viridis::value_t {0.11973756,  0.60378459,  0.54139999},
        Viridis::value_t {0.11951163,  0.60746388,  0.54021751},
        Viridis::value_t {0.11942341,  0.61114146,  0.53898192},
        Viridis::value_t {0.11948255,  0.61481702,  0.53769219},
        Viridis::value_t {0.11969858,  0.61849025,  0.53634733},
        Viridis::value_t {0.12008079,  0.62216081,  0.53494633},
        Viridis::value_t {0.12063824,  0.62582833,  0.53348834},
        Viridis::value_t {0.12137972,  0.62949242,  0.53197275},
        Viridis::value_t {0.12231244,  0.63315277,  0.53039808},
        Viridis::value_t {0.12344358,  0.63680899,  0.52876343},
        Viridis::value_t {0.12477953,  0.64046069,  0.52706792},
        Viridis::value_t {0.12632581,  0.64410744,  0.52531069},
        Viridis::value_t {0.12808703,  0.64774881,  0.52349092},
        Viridis::value_t {0.13006688,  0.65138436,  0.52160791},
        Viridis::value_t {0.13226797,  0.65501363,  0.51966086},
        Viridis::value_t {0.13469183,  0.65863619,  0.5176488 },
        Viridis::value_t {0.13733921,  0.66225157,  0.51557101},
        Viridis::value_t {0.14020991,  0.66585927,  0.5134268 },
        Viridis::value_t {0.14330291,  0.66945881,  0.51121549},
        Viridis::value_t {0.1466164 ,  0.67304968,  0.50893644},
        Viridis::value_t {0.15014782,  0.67663139,  0.5065889 },
        Viridis::value_t {0.15389405,  0.68020343,  0.50417217},
        Viridis::value_t {0.15785146,  0.68376525,  0.50168574},
        Viridis::value_t {0.16201598,  0.68731632,  0.49912906},
        Viridis::value_t {0.1663832 ,  0.69085611,  0.49650163},
        Viridis::value_t {0.1709484 ,  0.69438405,  0.49380294},
        Viridis::value_t {0.17570671,  0.6978996 ,  0.49103252},
        Viridis::value_t {0.18065314,  0.70140222,  0.48818938},
        Viridis::value_t {0.18578266,  0.70489133,  0.48527326},
        Viridis::value_t {0.19109018,  0.70836635,  0.48228395},
        Viridis::value_t {0.19657063,  0.71182668,  0.47922108},
        Viridis::value_t {0.20221902,  0.71527175,  0.47608431},
        Viridis::value_t {0.20803045,  0.71870095,  0.4728733 },
        Viridis::value_t {0.21400015,  0.72211371,  0.46958774},
        Viridis::value_t {0.22012381,  0.72550945,  0.46622638},
        Viridis::value_t {0.2263969 ,  0.72888753,  0.46278934},
        Viridis::value_t {0.23281498,  0.73224735,  0.45927675},
        Viridis::value_t {0.2393739 ,  0.73558828,  0.45568838},
        Viridis::value_t {0.24606968,  0.73890972,  0.45202405},
        Viridis::value_t {0.25289851,  0.74221104,  0.44828355},
        Viridis::value_t {0.25985676,  0.74549162,  0.44446673},
        Viridis::value_t {0.26694127,  0.74875084,  0.44057284},
        Viridis::value_t {0.27414922,  0.75198807,  0.4366009 },
        Viridis::value_t {0.28147681,  0.75520266,  0.43255207},
        Viridis::value_t {0.28892102,  0.75839399,  0.42842626},
        Viridis::value_t {0.29647899,  0.76156142,  0.42422341},
        Viridis::value_t {0.30414796,  0.76470433,  0.41994346},
        Viridis::value_t {0.31192534,  0.76782207,  0.41558638},
        Viridis::value_t {0.3198086 ,  0.77091403,  0.41115215},
        Viridis::value_t {0.3277958 ,  0.77397953,  0.40664011},
        Viridis::value_t {0.33588539,  0.7770179 ,  0.40204917},
        Viridis::value_t {0.34407411,  0.78002855,  0.39738103},
        Viridis::value_t {0.35235985,  0.78301086,  0.39263579},
        Viridis::value_t {0.36074053,  0.78596419,  0.38781353},
        Viridis::value_t {0.3692142 ,  0.78888793,  0.38291438},
        Viridis::value_t {0.37777892,  0.79178146,  0.3779385 },
        Viridis::value_t {0.38643282,  0.79464415,  0.37288606},
        Viridis::value_t {0.39517408,  0.79747541,  0.36775726},
        Viridis::value_t {0.40400101,  0.80027461,  0.36255223},
        Viridis::value_t {0.4129135 ,  0.80304099,  0.35726893},
        Viridis::value_t {0.42190813,  0.80577412,  0.35191009},
        Viridis::value_t {0.43098317,  0.80847343,  0.34647607},
        Viridis::value_t {0.44013691,  0.81113836,  0.3409673 },
        Viridis::value_t {0.44936763,  0.81376835,  0.33538426},
        Viridis::value_t {0.45867362,  0.81636288,  0.32972749},
        Viridis::value_t {0.46805314,  0.81892143,  0.32399761},
        Viridis::value_t {0.47750446,  0.82144351,  0.31819529},
        Viridis::value_t {0.4870258 ,  0.82392862,  0.31232133},
        Viridis::value_t {0.49661536,  0.82637633,  0.30637661},
        Viridis::value_t {0.5062713 ,  0.82878621,  0.30036211},
        Viridis::value_t {0.51599182,  0.83115784,  0.29427888},
        Viridis::value_t {0.52577622,  0.83349064,  0.2881265 },
        Viridis::value_t {0.5356211 ,  0.83578452,  0.28190832},
        Viridis::value_t {0.5455244 ,  0.83803918,  0.27562602},
        Viridis::value_t {0.55548397,  0.84025437,  0.26928147},
        Viridis::value_t {0.5654976 ,  0.8424299 ,  0.26287683},
        Viridis::value_t {0.57556297,  0.84456561,  0.25641457},
        Viridis::value_t {0.58567772,  0.84666139,  0.24989748},
        Viridis::value_t {0.59583934,  0.84871722,  0.24332878},
        Viridis::value_t {0.60604528,  0.8507331 ,  0.23671214},
        Viridis::value_t {0.61629283,  0.85270912,  0.23005179},
        Viridis::value_t {0.62657923,  0.85464543,  0.22335258},
        Viridis::value_t {0.63690157,  0.85654226,  0.21662012},
        Viridis::value_t {0.64725685,  0.85839991,  0.20986086},
        Viridis::value_t {0.65764197,  0.86021878,  0.20308229},
        Viridis::value_t {0.66805369,  0.86199932,  0.19629307},
        Viridis::value_t {0.67848868,  0.86374211,  0.18950326},
        Viridis::value_t {0.68894351,  0.86544779,  0.18272455},
        Viridis::value_t {0.69941463,  0.86711711,  0.17597055},
        Viridis::value_t {0.70989842,  0.86875092,  0.16925712},
        Viridis::value_t {0.72039115,  0.87035015,  0.16260273},
        Viridis::value_t {0.73088902,  0.87191584,  0.15602894},
        Viridis::value_t {0.74138803,  0.87344918,  0.14956101},
        Viridis::value_t {0.75188414,  0.87495143,  0.14322828},
        Viridis::value_t {0.76237342,  0.87642392,  0.13706449},
        Viridis::value_t {0.77285183,  0.87786808,  0.13110864},
        Viridis::value_t {0.78331535,  0.87928545,  0.12540538},
        Viridis::value_t {0.79375994,  0.88067763,  0.12000532},
        Viridis::value_t {0.80418159,  0.88204632,  0.11496505},
        Viridis::value_t {0.81457634,  0.88339329,  0.11034678},
        Viridis::value_t {0.82494028,  0.88472036,  0.10621724},
        Viridis::value_t {0.83526959,  0.88602943,  0.1026459 },
        Viridis::value_t {0.84556056,  0.88732243,  0.09970219},
        Viridis::value_t {0.8558096 ,  0.88860134,  0.09745186},
        Viridis::value_t {0.86601325,  0.88986815,  0.09595277},
        Viridis::value_t {0.87616824,  0.89112487,  0.09525046},
        Viridis::value_t {0.88627146,  0.89237353,  0.09537439},
        Viridis::value_t {0.89632002,  0.89361614,  0.09633538},
        Viridis::value_t {0.90631121,  0.89485467,  0.09812496},
        Viridis::value_t {0.91624212,  0.89609127,  0.1007168 },
        Viridis::value_t {0.92610579,  0.89732977,  0.10407067},
        Viridis::value_t {0.93590444,  0.8985704 ,  0.10813094},
        Viridis::value_t {0.94563626,  0.899815  ,  0.11283773},
        Viridis::value_t {0.95529972,  0.90106534,  0.11812832},
        Viridis::value_t {0.96489353,  0.90232311,  0.12394051},
        Viridis::value_t {0.97441665,  0.90358991,  0.13021494},
        Viridis::value_t {0.98386829,  0.90486726,  0.13689671},
        Viridis::value_t {0.99324789,  0.90615657,  0.1439362 }
    }};
    
    gui::Color Viridis::value(double percent) {
        size_t index = size_t(min(1.0, cmn::abs(percent)) * 255);
        auto& [r, g, b] = data_bgr[index];
        
        return gui::Color((uint8_t)saturate(r * 255), (uint8_t)saturate(g * 255), (uint8_t)saturate(b * 255), 255);
    }
    
    std::string HorizontalLine::toStr() const {
        return "HorizontalLine("+std::to_string(y)+","+std::to_string(x0)+","+std::to_string(x1)+")";
    }
}
