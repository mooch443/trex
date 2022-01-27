#include "Blob.h"
#include <misc/stacktrace.h>
#include <misc/metastring.h>

using namespace cmn;
//#define _DEBUG_MEMORY
#ifdef _DEBUG_MEMORY
std::unordered_map<Blob*, std::tuple<int, std::shared_ptr<void*>>> all_blobs_map;
std::mutex all_mutex_blob;
#endif

Blob::Blob() {
    _properties.ready = false;
#ifdef _DEBUG_MEMORY
    std::lock_guard<std::mutex> guard(all_mutex_blob);
    all_blobs_map[this] = retrieve_stacktrace();
#endif
}

Blob::Blob(const std::shared_ptr<std::vector<HorizontalLine>>& points)
    : _hor_lines(points)
{
    if(!points)
        U_EXCEPTION("Blob initialized with NULL lines array");
    _properties.ready = false;
#ifdef _DEBUG_MEMORY
    std::lock_guard<std::mutex> guard(all_mutex_blob);
    all_blobs_map[this] = retrieve_stacktrace();
#endif
}

Blob::~Blob() {
#ifdef _DEBUG_MEMORY
    std::lock_guard<std::mutex> guard(all_mutex_blob);
    auto it = all_blobs_map.find(this);
    if(it == all_blobs_map.end())
        Error("Double delete?");
    else
        all_blobs_map.erase(it);
#endif
}

size_t Blob::all_blobs() {
#ifdef _DEBUG_MEMORY
    std::lock_guard<std::mutex> guard(all_mutex_blob);
    
    std::set<std::string> resolved, drawbles;
    for(auto && [ptr, tuple] : all_blobs_map) {
        resolved.insert(resolve_stacktrace(tuple));
    }
    auto str = "[Blobs]\n"+Meta::toStr(resolved);
    
    auto f = fopen("blobs.log", "wb");
    if(f) {
        fwrite(str.data(), sizeof(char), str.length(), f);
        fclose(f);
    } else
        Error("Cannot write 'blobs.log'");
    return all_blobs_map.size();
#else
    return 0;
#endif
}

void Blob::calculate_moments() {
    if(_moments.ready)
        return;
    
    assert(_properties.ready);
    for (auto &h: *_hor_lines) {
        const uint my = h.y;
        const uint mysq = my * my;
        
        int mx = h.x0;
        
        for (int x=h.x0; x<=h.x1; x++, mx++) {
            const uint mxsq = mx * mx;
            
            _moments.m[0][0] += 1;
            _moments.m[0][1] += 1 * my;
            _moments.m[1][0] += mx * 1;
            _moments.m[1][1] += mx * my;
            
            _moments.m[2][0] += mxsq;
            _moments.m[0][2] += mysq;
            _moments.m[2][1] += mxsq * my;
            _moments.m[1][2] += mx * mysq;
            _moments.m[2][2] += mxsq * mysq;
        }
    }
    
    _properties.center = Vec2(_moments.m[1][0] / _moments.m[0][0],
                              _moments.m[0][1] / _moments.m[0][0]);
    
    for (auto &h: *_hor_lines) {
        const int vy  = (h.y) - _properties.center.y;
        const int vy2 = vy * vy;
        
        int vx = (h.x0) - _properties.center.x;
        
        for (int x=h.x0; x<=h.x1; x++, vx++) {
            const auto vx2 = vx * vx;

            _moments.mu[0][0] += 1;
            _moments.mu[0][1] += 1 * vy;
            _moments.mu[0][2] += 1 * vy2;

            _moments.mu[1][0] += vx * 1;
            _moments.mu[1][1] += float(vx) * float(vy);
            _moments.mu[1][2] += float(vx) * float(vy2);

            _moments.mu[2][0] += vx2 * 1;
            _moments.mu[2][1] += float(vx2) * float(vy);
            _moments.mu[2][2] += float(vx2) * float(vy2);
        }
    }

    float mu00_inv = 1.0f / float(_moments.mu[0][0]);
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            _moments.mu_[i][j] = _moments.mu[i][j] * mu00_inv;
        }
    }
    
    _properties.angle = 0.5 * cmn::atan2(2 * _moments.mu_[1][1], _moments.mu_[2][0] - _moments.mu_[0][2]);
    _moments.ready = true;
}

void Blob::calculate_properties() {
    if(_properties.ready)
        return;
    
    // find x,y and width,height of blob
    int x = INT_MAX, y = INT_MAX, maxx = 0, maxy = 0;
    int num_pixels = 0;
    
    for (auto &h: *_hor_lines) {
        if (h.x0 < x)
            x = h.x0;
        if (h.x1 > maxx)
            maxx = h.x1;
        
        if (h.y > maxy)
            maxy = h.y;
        if (h.y < y)
            y = h.y;
        
        num_pixels += h.x1-h.x0+1;
    }
    
    _bounds = Bounds(x, y, maxx-x+1, maxy-y+1);
    _properties._num_pixels = num_pixels;
    _properties.center = _bounds.pos() + _bounds.size() * 0.5;
    
    _properties.ready = true;
}

std::string Blob::toStr() const {
    return "Blob<pos:" + Meta::toStr(center()) + " size:" + Meta::toStr(_bounds.size()) + ">";
}

void Blob::add_offset(const Vec2& off) {
    int offy = off.y;
    int offx = off.x;
    if(!_hor_lines->empty() && offy < -float(bounds().y)) {
        offy = -float(bounds().y);
    }
    if(!_hor_lines->empty() && offx < -float(bounds().x)) {
        offx = -float(bounds().x);
    }
    
    for (auto &h : *_hor_lines) {
        h.y += offy;
        h.x0 += offx;
        h.x1 += offx;
    }
    
    if(_properties.ready) {
        _properties.ready = false;
        calculate_properties();
    }
    if(_moments.ready) {
        _moments.ready = false;
        calculate_moments();
    }
}
