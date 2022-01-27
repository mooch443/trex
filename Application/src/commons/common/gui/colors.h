#ifndef _COLORS_H
#define _COLORS_H

#include "types.h"

#if __has_include ( <imgui/imgui.h> )
    #define HAS_IMGUI
    #include <imgui/imgui.h>
#endif

namespace gui {
namespace const_funcs {
    //! computes ⌊val⌋, the largest integer value not greater than val
    //! NOTE: not precise for huge values that don't fit into a 64-bit int!
    template <typename fp_type, typename std::enable_if<std::is_floating_point<fp_type>::value>::type* = nullptr>
    constexpr fp_type floor(fp_type val) {
        // casting to int truncates the value, which is floor(val) for positive values,
        // but we have to substract 1 for negative values (unless val is already floored == recasted int val)
        const auto val_int = (int64_t)val;
        const fp_type fval_int = (fp_type)val_int;
        return (val >= (fp_type)0 ? fval_int : (val == fval_int ? val : fval_int - (fp_type)1));
    }
}

class Color {
public:
    uint8_t r, g, b, a;
    
public:
    constexpr Color() : Color(0, 0, 0, 0) {}
    constexpr Color(const Color& other) : Color(other.r, other.g, other.b, other.a) {}
    explicit constexpr Color(uint8_t gray, uint8_t alpha = 255) : Color(gray, gray, gray, alpha) {}
    constexpr Color(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha = 255)
        : r(red),
          g(green),
          b(blue),
          a(alpha)
    {}
    
    operator cv::Scalar() const {
        return cv::Scalar(r, g, b, a);
    }
    
    operator cv::Vec3b() const {
        return cv::Vec3b(r, g, b);
    }
    
    operator cv::Vec4b() const {
        return cv::Vec4b(r, g, b, a);
    }
    
    operator std::tuple<uint8_t, uint8_t, uint8_t, uint8_t>() const {
        return {r,g,b,a};
    }
    
    Color(const cv::Vec2b& v) : Color(v[0], v[0], v[0], v[1]) {}
    Color(const cv::Vec3b& v) : Color(v[0], v[1], v[2]) {}
    Color(const cv::Vec4b& v) : Color(v[0], v[1], v[2], v[3]) {}
    
    static constexpr Color blend(const Color& A, const Color& B) {
        auto alphabg = A.a / 255.0;
        auto alphafg = B.a / 255.0;
        auto alpha = alphabg + alphafg * ( 1 - alphabg );
        return Color(
            (uint8_t)saturate((A.r * alphabg + B.r * alphafg * ( 1 - alphabg )) / alpha),
            (uint8_t)saturate((A.g * alphabg + B.g * alphafg * ( 1 - alphabg )) / alpha),
            (uint8_t)saturate((A.b * alphabg + B.b * alphafg * ( 1 - alphabg )) / alpha),
            (uint8_t)(alpha * 255.0)
        );
    }
    
#ifdef HAS_IMGUI
    constexpr Color(const ImColor& c) : Color(uint8_t(c.Value.x * 255), uint8_t(c.Value.y * 255), uint8_t(c.Value.z * 255), uint8_t(c.Value.w * 255)) {}
    operator ImColor() const { return ImColor(r, g, b, a); }
#endif
    
    constexpr uint8_t operator[] (size_t index) const {
        assert(index < 4);
        return *((&r) + index);
    }
    
    constexpr gui::Color& operator=(const gui::Color&) = default;
    
    constexpr bool operator==(const Color& other) const {
        return r == other.r && g == other.g && b == other.b && a == other.a;
    }
    
    constexpr bool operator!=(const Color& other) const {
        return r != other.r || g != other.g || b != other.b || a != other.a;
    }
    
    constexpr Color float_multiply(const Color& other) const {
        return Color(uint8_t(r * float(other.r) / 255.f),
                     uint8_t(g * float(other.g) / 255.f),
                     uint8_t(b * float(other.b) / 255.f),
                     uint8_t(a * float(other.a) / 255.f));
    }
    
    constexpr uint32_t to_integer() const {
        return (uint32_t)((r << 24) | (g << 16) | (b << 8) | a);
    }
    
    std::ostream &operator <<(std::ostream &os) const {
        return os << to_integer();
    }
    
    constexpr inline Color subtract(uint8_t R, uint8_t G, uint8_t B, uint8_t A) const {
        return Color(r - R, g - G, b - B, a - A);
    }
    
    constexpr inline Color exposure(float factor) const {
        Color hsv(toHSV());
        Color rgb(Color(hsv.r, hsv.g, (uint8_t)saturate(factor * hsv.b), hsv.a).HSV2RGB());
        return Color(rgb.r, rgb.g, rgb.b, this->a);
    }
    
    constexpr inline Color exposureHSL(float factor) const {
        Color hsl(toHSL());
        Color rgb(hsl.blue((uint8_t)saturate(hsl.b * factor)).HSL2RGB());
        return Color(rgb.r, rgb.g, rgb.b, this->a);
    }
    
    constexpr inline Color red(uint8_t red) const {
        return Color(red, this->g, this->b, this->a);
    }
    constexpr inline Color green(uint8_t green) const {
        return Color(this->r, green, this->b, this->a);
    }
    constexpr inline Color blue(uint8_t blue) const {
        return Color(this->r, this->g, blue, this->a);
    }
    constexpr inline Color alpha(uint8_t alpha) const {
        return Color(this->r, this->g, this->b, alpha);
    }
    
    constexpr inline Color saturation(float factor) const {
        Color hsv(toHSV());
        Color rgb(Color(hsv.r, (uint8_t)saturate(factor * hsv.g), hsv.b, hsv.a).HSV2RGB());
        return Color(rgb.r, rgb.g, rgb.b, this->a);
    }
    
    constexpr Color HSV2RGB() const {
        float h = r / 255.f, s = g / 255.f, v = b / 255.f;
        float R = 0, G = 0, B = 0;
        
        auto i = (int)const_funcs::floor(h * 6);
        auto f = h * 6 - i;
        auto p = v * (1 - s);
        auto q = v * (1 - f * s);
        auto t = v * (1 - (1 - f) * s);
        
        switch(i % 6){
            case 0: R = v; G = t; B = p; break;
            case 1: R = q; G = v; B = p; break;
            case 2: R = p; G = v; B = t; break;
            case 3: R = p; G = q; B = v; break;
            case 4: R = t; G = p; B = v; break;
            case 5: R = v; G = p; B = q; break;
        }
        
        return Color(uint8_t(R * 255), uint8_t(G * 255), uint8_t(B * 255), uint8_t(255));
    }
    
    constexpr Color toHSV() const {
        float R = r / 255.f, G = g / 255.f, B = b / 255.f;
        
        auto Cmax = max(R, max(G, B));
        auto Cmin = min(R, min(G, B));
        auto d = Cmax - Cmin;
        
        auto H = d == 0
        ? 0
        : (Cmax == R
           ? 60 * fmodf((G-B)/d, 6)
           : (Cmax == G
              ? 60 * ((B-R)/d+2)
              : 60 * ((R-G)/d+4)
              )
           );
        auto S = Cmax == 0
        ? 0
        : d / Cmax;
        
        return Color((uint8_t)saturate(H / 360.f * 255), uint8_t(S * 255), uint8_t(Cmax * 255));
    }
    
    constexpr Color toHSL() const {
        //R, G and B input range = 0 ÷ 255
        //H, S and L output range = 0 ÷ 1.0
        float H = 0, S = 0;
        
        auto var_R = ( r / 255.f );
        auto var_G = ( g / 255.f );
        auto var_B = ( b / 255.f );
        
        const auto var_Min = min( var_R, var_G, var_B );    //Min. value of RGB
        const auto var_Max = max( var_R, var_G, var_B );    //Max. value of RGB
        const auto del_Max = var_Max - var_Min;             //Delta RGB value
        
        const auto L = ( var_Max + var_Min )/ 2.f;
        
        if ( del_Max == 0 )                     //This is a gray, no chroma...
        {
            H = 0;
            S = 0;
        }
        else                                    //Chromatic data...
        {
            if ( L < 0.5 )
                S = del_Max / ( var_Max + var_Min );
            else
                S = del_Max / ( 2 - var_Max - var_Min );
            
            auto del_R = ( ( ( var_Max - var_R ) / 6 ) + ( del_Max / 2 ) ) / del_Max;
            auto del_G = ( ( ( var_Max - var_G ) / 6 ) + ( del_Max / 2 ) ) / del_Max;
            auto del_B = ( ( ( var_Max - var_B ) / 6 ) + ( del_Max / 2 ) ) / del_Max;
            
            if      ( var_R == var_Max )
                H = del_B - del_G;
            else if ( var_G == var_Max )
                H = ( 1 / 3.f ) + del_R - del_B;
            else
                H = ( 2 / 3.f ) + del_G - del_R;
            if ( H < 0 )
                H += 1;
            if ( H > 1 )
                H -= 1;
        }
        
        return Color(uint8_t(H * 255.f), uint8_t(S * 255.f), uint8_t(L * 255.f));
    }
    
    constexpr static float Hue_2_RGB( float v1, float v2, float vH )             //Function Hue_2_RGB
    {
        if ( vH < 0 ) vH += 1;
        if( vH > 1 ) vH -= 1;
        if ( ( 6 * vH ) < 1 )
            return ( v1 + ( v2 - v1 ) * 6 * vH );
        if ( ( 2 * vH ) < 1 ) return ( v2 );
        if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2 / 3 ) - vH ) * 6 );
        return ( v1 );
    }
    
    constexpr Color HSL2RGB() const {
        //H, S and L input range = 0 ÷ 1.0
        //R, G and B output range = 0 ÷ 255
        float H = r / 255.f,
        S = g / 255.f,
        L = b / 255.f;
        
        float R = 0, G = 0, B = 0;
        
        if ( S == 0 )
        {
            
            R = L * 255.f;
            G = L * 255.f;
            B = L * 255.f;
        }
        else
        {
            float var_2 = 0;
            
            if ( L < 0.5 ) var_2 = L * ( 1 + S );
            else           var_2 = ( L + S ) - ( S * L );
            
            auto var_1 = 2 * L - var_2;
            
            R = 255.f * Hue_2_RGB( var_1, var_2, H + ( 1 / 3.f ) );
            G = 255.f * Hue_2_RGB( var_1, var_2, H );
            B = 255.f * Hue_2_RGB( var_1, var_2, H - ( 1 / 3.f ) );
        }
        
        return Color(uint8_t(R), (uint8_t)saturate(G), (uint8_t)saturate(B));
    }
    
    constexpr float diff(const Color& other) const {
        return cmn::abs(float(r) - float(other.r)) + cmn::abs(float(g) - float(other.g)) + cmn::abs(float(b) - float(other.b)) + cmn::abs(float(a) - float(other.a));
    }

    /**
     * META 
     */

    std::string toStr() const {
        return "[" + std::to_string(r) + "," + std::to_string(g) + "," + std::to_string(b) + "," + std::to_string(a) + "]";
    }

    static Color fromStr(const std::string& str);
    static std::string class_name() { return "color"; }
};

inline std::ostream &operator<<(std::ostream &os, const Color& obj) {
    return obj.operator<<(os);
}

inline std::ostream &operator<<(std::ostream &os, const Color*obj) {
    return operator<<(os, *obj);
}

constexpr static const Color
           White = Color(255, 255, 255, 255),
           Black = Color(0, 0, 0, 255),
            Gray = Color(135, 135, 135, 255),
        DarkGray = Color(50, 50, 50, 255),
        DarkCyan = Color(0, 125, 250, 255),
            Cyan = Color(0, 255, 255, 255),
          Yellow = Color(255, 255, 0, 255),
             Red = Color(255, 0, 0, 255),
            Blue = Color(0, 0, 255, 255),
           Green = Color(0, 255, 0, 255),
          Purple = Color(200, 0, 255, 255),
     Transparent = Color(0, 0, 0, 0);

constexpr inline Color operator*(const Color& c0, const Color& c1) {
    return Color((uint8_t)saturate((int)c0.r * (int)c1.r),
                 (uint8_t)saturate((int)c0.g * (int)c1.g),
                 (uint8_t)saturate((int)c0.b * (int)c1.b),
                 (uint8_t)saturate((int)c0.a * (int)c1.a));
}

constexpr inline Color operator*(const Color& c0, float scalar) {
    return Color((uint8_t)saturate((float)c0.r * scalar),
                 (uint8_t)saturate((float)c0.g * scalar),
                 (uint8_t)saturate((float)c0.b * scalar),
                 (uint8_t)saturate((float)c0.a * scalar));
}

constexpr inline Color operator-(const Color& c0, const Color& c1) {
    return Color((uint8_t)saturate((int)c0.r - (int)c1.r),
                 (uint8_t)saturate((int)c0.g - (int)c1.g),
                 (uint8_t)saturate((int)c0.b - (int)c1.b),
                 (uint8_t)saturate((int)c0.a - (int)c1.a));
}

constexpr inline Color operator+(const Color& c0, const Color& c1) {
    return Color((uint8_t)saturate((int)c0.r + (int)c1.r),
                 (uint8_t)saturate((int)c0.g + (int)c1.g),
                 (uint8_t)saturate((int)c0.b + (int)c1.b),
                 (uint8_t)saturate((int)c0.a + (int)c1.a));
}
}

class ColorWheel {
    uint32_t _index;
    /*constexpr static gui::Color colors[] = {
        gui::Color(0,0,255),
        gui::Color(80,170,0),
        gui::Color(255,100,0),
        gui::Color(255,0,210),
        gui::Color(0,255,255),
        gui::Color(255,170,220),
        gui::Color(80,170,0),
        gui::Color(255,0,85),
        gui::Color(0,170,255),
        gui::Color(255,255,0),
        gui::Color(170,0,255),
        gui::Color(0,255,85),
        gui::Color(255,0,0),
        gui::Color(0,85,255),
        gui::Color(170,255,0),
        gui::Color(255,0,255),
        gui::Color(0,255,170),
        gui::Color(255,85,0),
        gui::Color(0,0,255),
        gui::Color(85,255,0),
        gui::Color(255,0,170),
        gui::Color(0,255,255),
        gui::Color(255,170,0),
        gui::Color(85,0,255),
        gui::Color(80,170,0),
        gui::Color(255,0,85),
        gui::Color(0,170,255),
        gui::Color(255,255,0),
        gui::Color(170,0,255),
        gui::Color(0,255,85),
        gui::Color(255,0,0),
        gui::Color(0,85,255),
        gui::Color(170,255,0),
        gui::Color(255,0,255),
        gui::Color(0,255,170),
        gui::Color(255,85,0),
        gui::Color(0,0,255),
        gui::Color(85,255,0),
        gui::Color(255,0,170),
        gui::Color(0,255,255),
        gui::Color(255,170,0),
        gui::Color(85,0,255),
        gui::Color(80,170,0),
        gui::Color(255,0,85),
        gui::Color(0,170,255),
        gui::Color(255,255,0),
        gui::Color(170,0,255),
        gui::Color(0,255,85),
        gui::Color(255,0,0),
        gui::Color(0,85,255)
    };*/
    
    static constexpr int step = 100;
    int _hue;
    //int _offset;
    
public:
    constexpr ColorWheel(uint32_t index = 0) : _index(index), _hue(int(255 + index * (index + 1) * 0.5 * step)) {
        
    }
    constexpr gui::Color next() {
        //if (_index >= sizeof(colors) / sizeof(gui::Color)) {
        
        const uint8_t s = _hue % 255;
        //const uint32_t h = s % 100;
        const gui::Color hsv(s, 255, 255);
        //_hue += step;
        /*if (_hue >= 255) {
         _hue = _hue - 255 + _offset;
         
         _offset = _offset * 0.25;
         if (_offset < 80) {
         _offset = 0;
         }
         }*/
        
        _index++;
        _hue += step * _index;
        
        return hsv.HSV2RGB();
        //}
        
        //return colors[_index++];
    }

};

#endif
