#ifndef _EXPORT_H
#define _EXPORT_H

#include <types.h>
#include <file/Path.h>
#include <misc/GlobalSettings.h>
#include <misc/SpriteMap.h>
#include <misc/frame_t.h>

namespace file {

class Table;

class Row {
    std::vector<std::string> _values;
    
public:
    Row() {}
    
    void clear() { _values.clear(); }
    size_t size() const { return _values.size(); }
    auto begin() const -> decltype(_values.begin()) { return _values.begin(); }
    auto end() const -> decltype(_values.end()) { return _values.end(); }
    auto operator[](size_t i) const -> decltype(*_values.begin()) { return _values[i]; }
    
    template<typename T>
    const std::string to_string(const T& data, typename std::enable_if<std::is_floating_point<T>::value, bool>::type* = 0, decltype((operator<<(std::ofstream(), data).bad()))* =0)
    {
        auto output_csv_decimals = SETTING(output_csv_decimals).value<uint8_t>();
        std::stringstream ss;
        ss << std::fixed << std::setprecision(output_csv_decimals);
        ss << data;
        return ss.str();
    }
    
    template<typename T>
    const std::string to_string(const T& data, typename std::enable_if<!std::is_floating_point<T>::value, bool>::type* = 0, decltype((operator<<(std::ofstream(), data).bad()))* =0) {
        std::stringstream ss;
        ss << data;
        return ss.str();
    }
    
    const std::string to_string(Frame_t frame) {
        return frame.toStr();
    }
    
    template<typename T>
    Row& add(T value) {
        _values.push_back(to_string(value));
        return *this;
    }
    
    template<typename T>
    Row& repeat(T value, size_t n) {
        _values.reserve(_values.size() + n);
        for(size_t i=0; i<n; ++i)
            _values.push_back(value);
        return *this;
    }
};

class Table {
private:
    std::vector<std::string> _header;
    std::vector<Row> _rows;
    
public:
    Table(const std::vector<std::string> &header);
    
    void reserve(size_t num) { _rows.reserve(num); }
    void add(const Row& row);
    const std::vector<std::string>& header() const { return _header; }
    const Row& row(size_t i) const { return _rows.at(i); }
    const std::vector<Row>& rows() const { return _rows; }
};

class ExportType {
    Table _table;
    
public:
    ExportType(const Table& table) : _table(table) {}
    const Table& table() const { return _table; }
    
protected:
    virtual ~ExportType() {}
    virtual bool save(const Path& filename) const = 0;
};
}

#endif
