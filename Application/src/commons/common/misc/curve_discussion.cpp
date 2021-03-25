#include "curve_discussion.h"
#include <misc/vec2.h>

namespace cmn {
    namespace curves {
        float interpolate(const std::vector<float>& values, float index) {
            long first = index;
            const long N = values.size();
            float p = index - first;
            
            if(first < 0) {
                first = N - (fabs(first) - N * floorf(fabs(first) / float(N)));
            }
            //while(first < 0)
            //    first = first + long(values.size());
            
            long next = first + 1;
            while(next >= N)
                next = next - N;
            
            float v0 = values[first];
            float v1 = values[next];
            
            return v0 * (1 - p) + v1 * p;
        }
        
        std::vector<float> derive(const std::vector<float>& values) {
            std::vector<float> derivative;
            derivative.resize(values.size());
            for(size_t i = 0; i < values.size(); i++)
                derivative[i] = values[i] - values[i ? i-1 : values.size()-1];
            return derivative;
        }
        
        Extrema find_extreme_points(const std::vector<float>& values, std::vector<float>& derivative)
        {
            Extrema ret;
            if(derivative.empty())
                derivative = derive(values);
            
            const long L = values.size();
            float y0 = derivative.back();
            float last_extremum = -1;
            int last_extremum_type = -1;
            
            ret.min = FLT_MAX;
            ret.max = -FLT_MAX;
            ret.mean = 0;
            for(auto c : values) {
                ret.mean += abs(c);
                if(ret.max < c) ret.max = c;
                if(ret.min > c) ret.min = c;
            }
            ret.mean /= values.size();
            const float magic_value = ret.mean * 0.3;
            
            std::vector<std::pair<float, bool>> all;
            
            for (long x1=0; x1<L; x1++) {
                float x0 = x1-1;
                float y1 = derivative[x1];
                
                float p = crosses_zero(y0, y1);
                
                if(p >= 0 && p <= 1) {
                    float ky1 = values[x1];
                    float ky0 = values[x1 == 0 ? (values.size()-1) : (x1-1)];
                    
                    // it crosses zero in this segment
                    Vec2 pt(x0+p, y1+(y0-y1)*p);
                    Vec2 org_pt(x0+p, ky1+(ky0-ky1)*p);
                    
                    if(ky0 < ky1) {
                        if(last_extremum_type == 0 && abs(org_pt.y - last_extremum) > magic_value)
                            last_extremum_type = -1;
                        
                        if(last_extremum_type == 0 && org_pt.y < last_extremum) {
                            all.back() = {pt.x, 0};
                            
                        } else if(last_extremum_type != 0) {
                            last_extremum = org_pt.y;
                            last_extremum_type = 0;
                            
                            all.push_back({pt.x, 0});
                        }
                        
                    } else {
                        if(last_extremum_type == 1 && abs(org_pt.y - last_extremum) > magic_value)
                            last_extremum_type = -1;
                        
                        if(last_extremum_type == 1 && org_pt.y > last_extremum) {
                            all.back() = {pt.x, 1};
                            
                        } else if(last_extremum_type != 1) {
                            last_extremum = org_pt.y;
                            last_extremum_type = 1;
                            
                            all.push_back({pt.x, 1});
                        }
                    }
                }
                
                y0 = y1;
            }
            
            std::pair<long,bool> prev(0, false);
            if(!all.empty())
                prev = all.front();
            if(prev.second) {
                for(auto it = all.rbegin(); it != all.rend(); ++it) {
                    if(!it->second) {
                        prev = *it;
                        break;
                    }
                }
            }
            
            auto copy = all;
            
            for(long i=0; i<long(all.size()); i++) {
                auto &pair = all.at(i);
                auto idx = pair.first;
                auto type = pair.second;
                
                float height = interpolate(values, idx);//values[idx];
                
                if(type) {
                    //assert(prev.first >= 0 && prev.first <= values.size());
                    float diff = abs(height-interpolate(values, prev.first));
                    //Debug("Extremum(%d)@%d = %f -> %f", type, idx, height, diff);
                    
                    if(diff < magic_value) {
                        // erase maximum (or erase previous maximum? hmm)
                        long to_check;
                        long prev_maximum = i >= 2 ? i - 2 : all.size() - i - 1;
                        if(prev_maximum < long(all.size()) && all[prev_maximum].second
                           && interpolate(values, all[prev_maximum].first) < height)
                        {
                            // 2 previous is a maximum. check height
                            // if this one is bigger, we have to replace the previous one
                            all.erase(all.begin() + prev_maximum);
                            if(prev_maximum >= long(all.size()))
                                to_check = 0;
                            else
                                to_check = prev_maximum;
                            
                            if(prev_maximum <= i)
                                i--;
                            
                        } else {
                            all.erase(all.begin() + i);
                            if(i >= long(all.size()))
                                to_check = 0;
                            else
                                to_check = i;
                            
                            i--;
                        }
                        
                        if(to_check < long(all.size())) {
                            // merge two consecutive minima
                            long previous_minimum = to_check ? to_check-1 : all.size()-1;
                            
                            if(!all[to_check].second && !all[previous_minimum].second) {
                                if(interpolate(values, all[to_check].first) < interpolate(values, all[previous_minimum].first)) {
                                    all.erase(all.begin() + previous_minimum);
                                    
                                    if(previous_minimum <= i)
                                        i--;
                                }
                                else {
                                    all.erase(all.begin() + to_check);
                                    
                                    if(to_check <= i)
                                        i--;
                                }
                            }
                        }
                    }
                } else {
                    //Debug("Jumping over minimum %d", idx);
                }
                
                if(i < -1)
                    i = -1;
                
                if(i < 0 || i >= long(all.size())) {
                    for(auto it = all.rbegin(); it != all.rend(); ++it) {
                        if(!it->second) {
                            prev = *it;
                            break;
                        }
                    }
                } else
                    prev = all.at(i);
            }
            
            assert(ret.minima.empty() && ret.maxima.empty());
            
            for(auto &a : all) {
                if(a.second) {
                    ret.maxima.push_back(a.first);
                } else
                    ret.minima.push_back(a.first);
            }
            
            return ret;
        }
        
        std::map<float, float> area_under_extreme(const std::vector<float>& values, bool minima, std::vector<float>& derivative)
        {
            Extrema e = find_extreme_points(values, derivative);
            
            std::map<float, float> area;
            
            if(!e.minima.empty()) {
                auto &search_first = minima ? e.minima : e.maxima;
                auto &search_second = minima ? e.maxima : e.minima;
                
                for (float exact : search_first) {
                    long minimum = exact;
                    long fidx = LONG_MAX, lidx = LONG_MAX;
                    
                    for(long maximum : search_second) {
                        // search for the first value after the minimum
                        if(fidx == LONG_MAX)
                            fidx = maximum < minimum ? maximum : maximum - values.size();
                        else {
                            if((maximum < minimum && maximum > fidx)
                               || (maximum > minimum && minimum+long(values.size()) - maximum < minimum - fidx))
                            {
                                if(maximum < minimum)
                                    fidx = maximum;
                                else
                                    fidx = maximum - values.size();
                            }
                        }
                        
                        // search for the last value before the minimum
                        if(lidx == LONG_MAX)
                            lidx = maximum > minimum ? maximum : (maximum + values.size());
                        else {
                            if((maximum > minimum && maximum < lidx)
                               || (maximum < minimum && maximum - (minimum-long(values.size())) < lidx - minimum))
                            {
                                if(maximum > minimum)
                                    lidx = maximum;
                                else
                                    lidx = maximum + values.size();
                            }
                        }
                    }
                    
                    // add up the area
                    float v = 0;
                    float value_min = FLT_MAX, value_max = -FLT_MAX;
                    
                    for(long i=fidx; i<=lidx; i++) {
                        long x = i;
                        while (x < 0) x += values.size();
                        while (x >= long(values.size())) x -= long(values.size());
                        
                        if(values[x] > value_max)
                            value_max = values[x];
                        if(values[x] < value_min)
                            value_min = values[x];
                        
                        v += (values[x]);
                    }
                    
                    //Debug("Area[%d ..%d] = %d - %d (%f)", minimum, prev_minimum, fidx, lidx, v);
                    
                    float distances = 1;
                    
                    if(search_first.size() > 1) {
                        for(long m : search_first) {
                            if(m != minimum) {
                                distances += min(abs(minimum - m),
                                                 min(abs(minimum + (float)values.size() - m),
                                                     abs(m - minimum + (float)values.size())));
                            }
                        }
                        
                        distances /= float(search_first.size()-1);
                    }
                    
                    area[exact] = distances * abs(value_max - value_min);
                }
            }
            
            return area;
        }
        
        std::map<float, float> area_under_minima(const std::vector<float>& values) {
            std::vector<float> derivative;
            return area_under_extreme(values, true, derivative);
        }
        
        std::map<float, float> area_under_maxima(const std::vector<float>& values) {
            std::vector<float> derivative;
            return area_under_extreme(values, false, derivative);
        }
        
        std::map<float, float> area_under_minima(const std::vector<float>& values, std::vector<float>& derivative) {
            return area_under_extreme(values, true, derivative);
        }
        
        std::map<float, float> area_under_maxima(const std::vector<float>& values, std::vector<float>& derivative) {
            return area_under_extreme(values, false, derivative);
        }
    }
}
