#include "pv.h"
#include <minilzo.h>
#include <sys/stat.h>
#include <misc/GlobalSettings.h>
#include <misc/Timer.h>
#include <misc/PVBlob.h>
#include <misc/checked_casts.h>
#include <misc/ranges.h>
#include <misc/SpriteMap.h>

/**
 * =============================
 * ProcessedVideo IMPLEMENTATION
 * -----------------------------
 **/

using namespace file;

namespace pv {
    // used to register for global settings updates
    static std::mutex settings_mutex;
    static std::atomic_bool use_differences(false), settings_registered(false);
    
    static std::mutex location_mutex;
    static std::map<std::string, std::function<file::Path(file::Path)>> location_funcs;

    /**
     * If there is a task that is async (and can be run read-only for example) and e.g. continously calls "read_frame", then a task sentinel can be registered. This prevents the file from being destroyed until the task is done.
     */
    struct TaskSentinel {
        pv::File *ptr = nullptr;
        bool _please_terminate = false;
        
        bool terminate() const {
            return _please_terminate;
        }
        
        TaskSentinel(pv::File* file) : ptr(file) {
            std::lock_guard guard(file->_task_list_mutex);
            file->_task_list[std::this_thread::get_id()] = this;
        }
        
        ~TaskSentinel() {
            {
                std::lock_guard guard(ptr->_task_list_mutex);
                assert(ptr->_task_list.find(std::this_thread::get_id()) != ptr->_task_list.end());
                ptr->_task_list.erase(std::this_thread::get_id());
            }
            
            ptr->_task_variable.notify_all();
        }
    };

    File::~File() {
        std::unique_lock guard(_task_list_mutex); // try to lock once to sync
        for(auto & [i, ptr] : _task_list)
            ptr->_please_terminate = true;
        
        while(!_task_list.empty())
            _task_variable.wait_for(guard, std::chrono::milliseconds(1));
    }


    //! Initialize copy
    /*Frame::Frame(const Frame& other) {
        operator=(other);
    }*/

    Frame::Frame(Frame&& other) {
        operator=(std::move(other));
    }

    void Frame::operator=(Frame && other) {
        _timestamp = other._timestamp;
        _n = other._n;
        _loading_time = other._loading_time;
        
        std::swap(_mask, other._mask);
        std::swap(_pixels, other._pixels);
        std::swap(_blobs, other._blobs);
    }

    void Frame::operator=(const Frame &other) {
        _timestamp = other._timestamp;
        _n = other._n;
        _loading_time = other._loading_time;
        
        _blobs.clear();
        
        _mask.clear();
        _pixels.clear();
        
        for (size_t i=0; i<other.n(); ++i) {
            _mask.push_back(std::make_unique<blob::line_ptr_t::element_type>(*other._mask[i]));
            _pixels.push_back(std::make_unique<blob::pixel_ptr_t::element_type>(*other._pixels[i]));
        }
    }

    Frame::Frame(const uint64_t& timestamp, decltype(_n) n)
        : _timestamp(timestamp),//std::chrono::duration_cast<std::chrono::microseconds>(timestamp).count()),
          _n(0), _loading_time(0)
    {
        _mask.reserve(n);
        _pixels.reserve(n);
    }
    
    Frame::Frame(File& ref, long_t idx) {
        read_from(ref, idx);
    }
    
    const std::vector<pv::BlobPtr>& Frame::get_blobs() const {
        if(_blobs.size() != n())
            throw U_EXCEPTION("Have to call the non-const variant of this function first at some point (",_blobs.size()," != ",n(),").");
        return _blobs;
    }
    
    std::vector<pv::BlobPtr>& Frame::get_blobs() {
        if(_blobs.empty()) {
            for (uint32_t i=0; i<n(); i++) {
                auto &mask = _mask[i];
                auto &px = _pixels[i];
                
                _blobs.push_back(std::make_shared<pv::Blob>(*mask, *px));
            }
        }
        
        return _blobs;
    }
    
    void Frame::clear() {
        _mask.clear();
        _pixels.clear();
        _n = 0;
        _timestamp = 0;
        _loading_time = 0;
        _blobs.clear();
        
        set_index(-1);
    }
    
    void Frame::read_from(pv::File &ref, long_t idx) {
        //for(auto m: _mask)
        //    delete m;
        //for(auto p: _pixels)
        //   delete p;
        
        clear();
        set_index(idx);
        
        Data* ptr = &ref;
        ReadonlyMemoryWrapper *compressed = NULL;
        
        if(!settings_registered){
            std::lock_guard<std::mutex> lock(settings_mutex);
            if(!settings_registered) {
                auto callback = "pv::Frame::read_from";
                use_differences = GlobalSettings::map().has("use_differences") ? SETTING(use_differences).value<bool>() : false;
                GlobalSettings::map().register_callback(callback, [callback](sprite::Map::Signal signal, sprite::Map&map, const std::string&key, const sprite::PropertyType& value)
                {
                    if(signal == sprite::Map::Signal::EXIT) {
                        map.unregister_callback(callback);
                        return;
                    }
                    
                    if(key == "use_differences")
                        use_differences = value.value<bool>();
                });
                
                settings_registered = true;
            }
        }
        
        if(ref.header().version >= V_6) {
            uchar compression_flag;
            ref.read<uchar>(compression_flag);
            
            if(compression_flag) {
                uint32_t size;
                ref.read<uint32_t>(size);
                uint32_t uncompressed_size;
                ref.read<uint32_t>(uncompressed_size);
                
                
                static DataPackage compressed_block, uncompressed_block;
                compressed_block.resize(size, false);
                uncompressed_block.resize(uncompressed_size, false);
                
                ref.read_data(size, compressed_block.data());
                
                lzo_uint new_len;
                if(lzo1x_decompress((uchar*)compressed_block.data(),size,(uchar*)uncompressed_block.data(),&new_len,NULL) == LZO_E_OK)
                {
                    assert(new_len == uncompressed_size);
                    compressed = new ReadonlyMemoryWrapper((uchar*)uncompressed_block.data(), new_len);
                    ptr = compressed;
                    
                } else {
                    FormatError("Failed to decode frame ", idx," from file ", *this);
                }
            }
        }
        
        // read relative timestamp (relative to timestamp in header)
        if(ref.header().version < V_4) {
            ptr->read_convert<uint32_t>(_timestamp);
        } else {
            ptr->read<uint64_t>(_timestamp);
        }
        
        ptr->read(_n);
        
        _mask.reserve(_n);
        _pixels.reserve(_n);
        
        // declared outside so memory doesnt have to be freed/allocated all the time
        static DataPackage pixels;
        static std::vector<Header::line_type> mask((NoInitializeAllocator<Header::line_type>()));
        static std::vector<LegacyShortHorizontalLine> mask_legacy((NoInitializeAllocator<LegacyShortHorizontalLine>()));
        
        for(int i=0; i<_n; i++) {
            uint16_t start_y, mask_size;
            ptr->read<uint16_t>(start_y);
            ptr->read<uint16_t>(mask_size);
            
            if(ref.header().version < V_7) {
                mask_legacy.resize(mask_size);
                mask.clear();
                mask.reserve(mask_legacy.size());
                
                assert(ref.header().line_size == sizeof(LegacyShortHorizontalLine));
                
                ptr->read_data(mask_size * ref.header().line_size, (char*)mask_legacy.data());
                std::copy(mask_legacy.begin(), mask_legacy.end(), std::back_inserter(mask));
                
            } else {
                mask.resize(mask_size);
                ptr->read_data(mask_size * ref.header().line_size, (char*)mask.data());
            }
            
            uint64_t num_pixels = 0;
            for(auto &l : mask) {
                num_pixels += l.x1() - l.x0() + 1;
            }
            
            pixels.resize(num_pixels, false);
            ptr->read_data(num_pixels, pixels.data());
            
            auto uncompressed = Header::line_type::uncompress(start_y, mask);
            
            if(use_differences) {
                uint64_t idx = 0;
                uchar *ptr = (uchar*)pixels.data();
                for (auto &l : *uncompressed) {
                    for (int x=l.x0; x<=l.x1; x++) {
                        auto &p = ptr[idx++];
                        p = (uchar)saturate(int(ref.average().at<uchar>(l.y, x)) - int(p));
                    }
                }
            }
            
            _mask.push_back(std::move(uncompressed));
            auto v = std::make_unique<std::vector<uchar>>((uchar*)pixels.data(),
                                                 (uchar*)pixels.data()+num_pixels);
            _pixels.emplace_back(std::move(v));
        }
        
        _mask.shrink_to_fit();
        _pixels.shrink_to_fit();
        
        if(compressed)
            delete compressed;
    }
    
    void Frame::add_object(blob::line_ptr_t&& mask, blob::pixel_ptr_t&& pixels) {
        assert(mask->size() < UINT16_MAX);
        
#ifndef NDEBUG
        HorizontalLine prev = mask->empty() ? HorizontalLine() : mask->front();
        
        uint64_t count = 0;
        for (auto &line : *mask) {
            if(!(prev == line) && !(prev < line))
                FormatWarning("Lines not properly ordered, or overlapping in x [",prev.x0,"-",prev.x1,"] < [",line.x0,"-",line.x1,"] (",prev.y,"/",line.y,").");
            prev = line;
            ++count;
        }
#endif
        
        _mask.push_back(std::move(mask));
        _pixels.push_back(std::move(pixels));
        
        _n++;
    }

void Frame::add_object(const std::vector<HorizontalLine>& mask, const std::vector<uchar>& pixels) {
    assert(mask.size() < UINT16_MAX);
    _mask.push_back(std::make_unique<blob::line_ptr_t::element_type>(mask));
    _pixels.push_back(std::make_unique<blob::pixel_ptr_t::element_type>(pixels));
    _n++;
}
    
    void Frame::add_object(const std::vector<HorizontalLine> &mask_, const cv::Mat &full_image) {
        assert(full_image.rows > 0 && full_image.cols > 0);
        assert(!mask_.empty());
        
        auto mask = std::make_unique<std::vector<HorizontalLine>>(mask_);
        
        ptr_safe_t L = (ptr_safe_t)mask_.size();
        //uint64_t offset = 0;
        
        // calculate overall bytes
        ptr_safe_t overall = 0;
        auto line_ptr = mask->data();
        
        assert(full_image.cols-1 < UINT16_MAX);
        assert(full_image.rows-1 < UINT16_MAX);
        
        for(ptr_safe_t i=0; i<L; i++) {
            // restrict to image dimensions
            assert(line_ptr->x1 < full_image.cols);
            assert(line_ptr->y < full_image.rows && line_ptr->x0 < full_image.cols && line_ptr->x1 >= line_ptr->x0);
            /*if(line_ptr->x1 >= full_image.cols) {
                line_ptr->x1 = full_image.cols - 1;
            }
            
            if(line_ptr->y >= full_image.rows || line_ptr->x0 >= full_image.cols || line_ptr->x1 < line_ptr->x0)
            {
                throw U_EXCEPTION("x1 < x0 in ",line_ptr->x1,"-",line_ptr->x0,",",line_ptr->y," ",L,"u 0x",line_ptr,"");
                //mask->erase(mask->begin()+i-offset);
                //offset++;
                continue;
            }*/
            
            overall += ptr_safe_t(line_ptr->x1) - ptr_safe_t(line_ptr->x0) + 1;
            line_ptr++;
        }
        
        // copy grey values to pixels array
        auto pixels = std::make_unique<std::vector<uchar>>();
        pixels->resize(overall);
        //uchar *pixels = (uchar*)malloc(overall);
        
        auto pixel_ptr = pixels->data();
        line_ptr = mask->data(); // reset ptr
        L = mask->size();
        
        for (ptr_safe_t i=0; i<L; i++, line_ptr++) {
            auto ptr = full_image.ptr(line_ptr->y, line_ptr->x0);
            assert(line_ptr->x1 >= line_ptr->x0);
            auto N = ptr_safe_t(line_ptr->x1) - ptr_safe_t(line_ptr->x0) + ptr_safe_t(1);
            memcpy(pixel_ptr, ptr, sign_cast<size_t>(N));
            pixel_ptr += N;
        }
        
        add_object(std::move(mask), std::move(pixels));
        //free(pixels);
    }
    
    uint64_t Frame::size() const {
        uint64_t bytes = sizeof(_timestamp) + sizeof(_n) + _mask.size() * sizeof(uint16_t);
        uint64_t elem_size = sizeof(Header::line_type);
        
        for (auto &m : _mask)
            bytes += sizeof(uint16_t) + m->size() * elem_size;
        
        for (auto &m : _pixels)
            bytes += m->size() * sizeof(char);
        
        return bytes;
    }

    //void Frame::update_timestamp(DataPackage& pack) const {
    //    pack.write(_timestamp, 0u); // just update the timestamp
    //}
    
    void Frame::serialize(DataPackage &pack, bool& compressed) const {
        uint64_t bytes = size();
        uint64_t elem_size = sizeof(Header::line_type);
        compressed = false;
        
        pack.resize(bytes);
        pack.reset_offset();
        
        assert(_timestamp < UINT64_MAX);
        pack.write(_timestamp); // force uint32_t because it should have been turned
        pack.write(_n);         // into a relative timestamp by now // V_4 use uint64_t anyway
        
        for(uint16_t i=0; i<_n; i++) {
            auto &mask = _mask.at(i);
            auto &pixels = _pixels.at(i);
            
            auto compressed = Header::line_type::compress(*mask);
            pack.write(uint16_t(mask->empty() ? 0 : mask->front().y));
            pack.write(uint16_t(compressed.size()));
            pack.write_data(compressed.size() * elem_size, (char*)compressed.data());
            pack.write_data(pixels->size() * sizeof(char), (char*)pixels->data());
        }

        // see whether this frame is worth compressing (size-threshold)
        if (pack.size() >= 1500) {
#define OUT_LEN(L)     (L + L / 16 + 64 + 3)


            /* Work-memory needed for compression. Allocate memory in units
             * of 'lzo_align_t' (instead of 'char') to make sure it is properly aligned.
             */

#define HEAP_ALLOC(var,size) \
    lzo_align_t __LZO_MMODEL var [ ((size) + (sizeof(lzo_align_t) - 1)) / sizeof(lzo_align_t) ]

            HEAP_ALLOC(wrkmem, LZO1X_1_MEM_COMPRESS);

            lzo_uint out_len = 0;
            assert(pack.size() < UINT32_MAX);
            uint32_t in_len = (uint32_t)pack.size();
            uint64_t reserved_size = OUT_LEN(in_len);

            DataPackage out;
            out.resize(reserved_size);

            // lock for wrkmem
            if (lzo1x_1_compress((uchar*)pack.data(), in_len, (uchar*)out.data(), &out_len, wrkmem) == LZO_E_OK)
            {
                uint64_t size = out_len + sizeof(uint32_t) * 2;
                /*if (size < in_len) {
                    if (_compression_samples > 1000) {
                        _compression_value = _compression_value / _compression_samples;
                        _compression_samples = 1;
                    }

                    _compression_value = _compression_value + size / float(in_len);
                    _compression_samples++;
                    _compression_ratio = _compression_value / double(_compression_samples);
                }*/

                if (size < in_len) {
                    pack.reset_offset();
                    compressed = true;

                    assert(out_len < UINT32_MAX);
                    pack.write<uint32_t>((uint32_t)out_len);
                    pack.write<uint32_t>(in_len);
                    pack.write_data(out_len, out.data());
                }

            }
            else {
                print("Compression of ",pack.size()," bytes failed.");
            }
        }
    }
    
    void DataLocation::register_path(std::string purpose, std::function<file::Path (file::Path)> fn)
    {
        purpose = utils::trim(utils::lowercase(purpose));
        
        std::lock_guard<std::mutex> guard(location_mutex);
        if(location_funcs.find(purpose) != location_funcs.end()) {
            auto str = Meta::toStr(extract_keys(location_funcs));
            throw U_EXCEPTION("Purpose '",purpose,"' already found in map with keys ",str,". Cannot register twice.");
        }
        
        location_funcs.insert({purpose, fn});
    }
    
    file::Path DataLocation::parse(const std::string &purpose, file::Path path) {
        std::function<file::Path(file::Path)> fn;
        {
            std::lock_guard<std::mutex> guard(location_mutex);
            auto it = location_funcs.find(utils::trim(utils::lowercase(purpose)));
            if(it == location_funcs.end()) {
                auto str = Meta::toStr(extract_keys(location_funcs));
                throw U_EXCEPTION("Cannot find purpose '",purpose,"' in map with keys ",str," in order to modify path '",path.str(),"'.");
            }
            
            fn = it->second;
        }
        
        return fn(path);
    }
    
    bool DataLocation::is_registered(std::string purpose) {
        purpose = utils::trim(utils::lowercase(purpose));
        
        std::lock_guard<std::mutex> guard(location_mutex);
        return location_funcs.find(purpose) != location_funcs.end();
    }

    void File::_write_header() { 
        _header.write(*this);
    }
    void File::_read_header() {
        _header.read(*this);

        _average = _header.average->get();
        if (has_mask())
            _mask = _header.mask->get();

        //std::chrono::microseconds ns_l, ns_e;
        uint64_t fps_l = 0;

        if (!_open_for_writing) {
            uint64_t idx = length() / 2u;
            //uint64_t edx = length()-1;
            if (idx < length()) {
                pv::Frame lastframe;
                //read_frame(lastframe, edx);

                //ns_e = std::chrono::microseconds(lastframe.timestamp());

                read_frame(lastframe, idx);
                //ns_l = std::chrono::microseconds(lastframe.timestamp());

                if (idx >= 1) {
                    uint64_t last = lastframe.timestamp();

                    read_frame(lastframe, idx - 1);
                    fps_l = last - lastframe.timestamp();
                }
            }

        }

        _header.average_tdelta = fps_l;
    }
    
    void Header::read(DataFormat& ref) {
        std::string version_str;
        ref.read<std::string>(version_str);
        
        if(version_str.length() > 2) {
            auto nr = version_str.at(version_str.length()-1u) - uchar('0');
            version = (Version)(nr-1);
            
        } else {
            version = Version::V_1;
        }
        
        if(version > Version::current)
            throw U_EXCEPTION("Unknown version '",version,"'.");
        
        if(version == Version::V_2) {
            // must read settings from file before loading...
            if(!DataLocation::is_registered("settings"))
                throw U_EXCEPTION("You have to register a DataLocation for 'settings' before using pv files (usually the same folder the video is in + exchange the .pv name with .settings).");
            auto settings_file = DataLocation::parse("settings");
            if (settings_file.exists())
                GlobalSettings::load_from_file({}, settings_file.str(), AccessLevelType::PUBLIC);
        }
        
        
        ref.read<uchar>(channels);
        ref.read<cv::Size>(resolution);
        
        // added offsets
        if (version >= Version::V_3) {
            ref.read(offsets);
            
        } else if(version == Version::V_2) {
            offsets = GlobalSettings::has("crop_offsets") ? SETTING(crop_offsets) : CropOffsets();
        }
        
        ref.read(line_size);
        _num_frames_offset = ref.current_offset();
        ref.read(num_frames);
        _index_offset = ref.current_offset();
        ref.read(index_offset);
        _timestamp_offset = ref.current_offset();
        ref.read(timestamp);
        
        ref.read<std::string>(name);
        
        // check values for sanity
        if(channels != 1)
            throw U_EXCEPTION("Only 1 channel currently supported (",this->channels," provided)");
        
        if(line_size != sizeof(line_type))
            throw U_EXCEPTION("The used line format in this file (",line_size," bytes) differs from the expected ",sizeof(line_type)," bytes.");
        
        if(average)
            delete average;
        
        average = new Image((uint)this->resolution.height, (uint)this->resolution.width, channels);
        _average_offset = ref.current_offset();
        ref.read_data(average->size(), (char*)average->data());
        
        if(mask)
            delete mask;
        mask = NULL;
        
        if(version >= Version::V_2) {
            uint64_t mask_size;
            ref.read<uint64_t>(mask_size);
            if(mask_size) {
                // does it use a mask?
                mask = new Image((uint)this->resolution.height /*- (offsets.y + offsets.height)*/, (uint)this->resolution.width /*- (offsets.x + offsets.width)*/, channels);
                
                ref.read_data(mask->size(), (char*)mask->data());
                
                for(uint64_t i=0; i<mask->size(); i++)
                    if(mask->data()[i] > 1) {
                        mask->get() /= mask->data()[i];
                        break;
                    }
            }
        }
        
        // read the index table
        index_table.resize(num_frames);
        auto len = sizeof(decltype(index_table)::value_type) * num_frames;
        ref.Data::read_data(index_offset, len, (char*)index_table.data());
        
        if(version >= Version::V_5) {
            _meta_offset = index_offset+len;
            ref.read(metadata, _meta_offset);
        }
    }

    void Header::write(DataFormat& ref) {
        /**
         * [HEADER SECTION]
         * (string) "PV" + (version_nr)
         * (byte)   channels
         * (uint16) width
         * (uint16) height
         * (Rect2i) four ushorts with the mask-offsets left,top,right,bottom
         * (uchar)  sizeof(HorizontalLine)
         * (uint32) number of frames
         * (uint64) pointer to index at the end of file
         * (uint64) timestamp (time since 1970 in microseconds)
         * (string) project name
         * (byte*)  average img ([width x height] x channels)
         * (uint64_t) mask present / mask size in bytes (if 0 no mask)
         * [byte*]  mask, but only present if mask_size != NULL
         *
         * [DATA SECTION]
         * for each frame:
         *   (uchar) compression flag (if 1, the whole frame is compressed)
         *   if compressed:
         *      (uint32) original size
         *      (uint32) compressed size
         *      (byte*) lzo1x compressed data (see below for uncompressed)
         *   else:
         *      [UNCOMPRESSED DATA PER FRAME] {
         *          (uint32) timestamp (in microseconds) since start of movie
         *          (uint16) number of individual cropped images
         *
         *          for each individual:
         *              (uint16) number of HorizontalLine structs
         *              (byte*)  n * sizeof(HorizontalLine)
         *              (byte*)  original image pixels ordered exactly as in HorizontalLines (BGR, CV_8UC(n))
         *      }
         *
         * [INDEX TABLE]
         * for each frame
         *   (uint64) frame start position in file
         *
         * [METADATA]
         * (string) JSONized metadata array
         */
        
        // set offsets from global settings
        
        ref.write("PV" + std::to_string((int)Version::current + 1));
        
        ref.write(channels);
        
        if(!resolution.width && !resolution.height)
            throw U_EXCEPTION("Resolution of the video has not been set.");
        ref.write(resolution);
        ref.write(offsets);
        
        ref.write(line_size);
        _num_frames_offset = ref.write(decltype(this->num_frames)(0));
        _index_offset = ref.write(decltype(index_offset)(0));
        _timestamp_offset = ref.write(timestamp);
        
        ref.write<std::string>((std::string)file::Path(name).filename());
        
        if(average)
            _average_offset = ref.write_data(average->size(), (char*)average->data());
        else {
            Image tmp((uint)resolution.height, (uint)resolution.width, 1);
            _average_offset = ref.write_data(tmp.size(), (char*)tmp.data());
        }
        
        if(mask) {
            ref.write(uint64_t(mask->size()));
            ref.write_data(mask->size(), (char*)mask->data());
            print("Written mask with ", mask->cols,"x",mask->rows);
        }
        else {
            ref.write(uint64_t(0));
        }
    }

    void Header::update(DataFormat& ref) {
        // write index table
        index_offset = ref.current_offset();
        print("Index table is ",FileSize(index_table.size() * sizeof(decltype(index_table)::value_type))," big.");
        
        for (auto index : index_table) {
            ref.write<decltype(index_table)::value_type>(index);
        }
        
        metadata = generate_metadata();
        _meta_offset = ref.write(metadata);
        
        ref.write(this->num_frames, _num_frames_offset);
        ref.write(this->index_offset, _index_offset);
        ref.write(this->timestamp, _timestamp_offset);
        
        if(average) {
            ref.Data::write_data(_average_offset, average->size(), (char*)average->data());
        }
        
        print("Updated number of frames with ",this->num_frames,", index offset ",this->index_offset,", timestamp ",this->timestamp,", ", _meta_offset);
    }
    
    std::string Header::generate_metadata() const {
        std::stringstream ss;
        
        std::vector<std::string> write_these = GlobalSettings::map().has("meta_write_these") ? SETTING(meta_write_these) : std::vector<std::string>();
        for (uint64_t i=0; i<write_these.size(); i++) {
            auto &name = write_these.at(i);
            auto val = GlobalSettings::get(name).get().valueString();
            ss << "\""<< name <<"\": "<<val;
            if(i<write_these.size()-1)
                ss << ", ";
        }
        
        std::string ret = ss.str();
        if(ret.empty()) {
            print("Metadata empty.");
        } else {
            ret = "{"+ret+"}";
            print("Metadata: ",ret.c_str());
        }
        
        return ret;
    }
    
    std::string File::get_info(bool full) {
        auto str = get_info_rich_text(full);
        str = utils::find_replace(str, "<b>", "");
        str = utils::find_replace(str, "</b>", "");
        return str;
    }
    
    std::string File::filesize() const {
        uint64_t bytes = 0;
        if(_open_for_writing)
            bytes = current_offset();
        else {
#if defined(__EMSCRIPTEN__)
            bytes = reading_file_size();
#else
            bytes = _filename.add_extension("pv").file_size();
#endif
        }
        
        return Meta::toStr(FileSize(bytes));
    }
    
    std::string File::get_info_rich_text(bool full) {
        std::stringstream ss;
        ss << this->summary() << "\n";
        
        /**
         * Display time related information.
         */
        std::chrono::microseconds ns_l, ns_e;
        
        if(!_open_for_writing){
            uint64_t idx = length() / 2u;
            uint64_t edx = length()-1;
            if(idx < length()) {
                pv::Frame lastframe;
                read_frame(lastframe, edx);
                
                ns_e = std::chrono::microseconds(lastframe.timestamp());
                
                read_frame(lastframe, idx);
                ns_l = std::chrono::microseconds(lastframe.timestamp());
            }
            
        } else {
            ns_l = std::chrono::microseconds(0);
        }
        
        auto ns = std::chrono::microseconds(header().timestamp);
        auto s = std::chrono::duration_cast<std::chrono::seconds>(ns);
        auto now = std::chrono::time_point<std::chrono::system_clock>(s);
        
        auto now_c = std::chrono::system_clock::to_time_t(now);
        
        ss << std::endl;
        
        ss << "<b>crop_offsets:</b> " << Meta::toStr(crop_offsets()) << std::endl;
#ifndef NO_PUT_TIME
        ss << "<b>Time of recording:</b> '" << std::put_time(std::localtime(&now_c), "%c") << "'"
        << std::endl;
#endif
        
        ss << "<b>Length of recording:</b> '";
        duration_to_string(ss, ns_e);
        ss << "'" << std::endl;
        
        ss << "<b>Framerate:</b> " << std::setw(0) << framerate() << "fps (" << float(_header.average_tdelta) / 1000.f << "ms)";
        ss << std::endl;
        
        ss << std::endl;
        
        if(full) {
            if(header().metadata.empty())
                ss << ("<b>Metadata empty.</b>");
            else
                ss << "<b>Metadata:</b> " << header().metadata ;
        }
        
        return ss.str();
    }
    
    void File::set_start_time(std::chrono::time_point<std::chrono::system_clock> tp) {
        assert(!open());
        _header.timestamp = narrow_cast<uint64_t>(std::chrono::time_point_cast<std::chrono::microseconds>(tp).time_since_epoch().count());
    }
    
    const pv::Frame& File::last_frame() {
        std::unique_lock<std::mutex> lock(_lock);
        return _last_frame;
    }

    void File::add_individual(const Frame& frame, DataPackage& pack, bool compressed) {
        assert(_open_for_writing);
        assert(_header.timestamp != 0); // start time has to be set

        std::unique_lock<std::mutex> lock(_lock);

        _header.num_frames++;
        assert(!_prev_frame_time || frame._timestamp > _prev_frame_time);
        //if(frame._timestamp >= _header.timestamp)
        //    frame._timestamp -= _header.timestamp; // make timestamp relative to start of video

        if (_prev_frame_time && frame._timestamp <= _prev_frame_time) {
            throw U_EXCEPTION("Should be dropping frame because ",frame._timestamp," <= ",_prev_frame_time,".");
        }

        _header._running_average_tdelta += frame._timestamp - _prev_frame_time;
        _header.average_tdelta = _header._running_average_tdelta / (_header.num_frames > 0 ? double(_header.num_frames) : 1);
        _prev_frame_time = frame._timestamp;

        // resets the offset and writes frame content to the pack
        auto index = tell();
        if (!compressed)
            this->write<uchar>(0);
        else
            this->write<uchar>(1);
        
        this->write(pack);
        _header.index_table.push_back(index);
        
        // cache last_frame
        _last_frame = frame;
    }

    void File::add_individual(Frame&& frame) {
        static std::mutex pack_mutex;
        static DataPackage pack;

        std::lock_guard g(pack_mutex);
        bool compressed;
        frame.serialize(pack, compressed);
        add_individual(std::move(frame), pack, compressed);
    }
        
    void File::stop_writing() {
        write(uint64_t(0));
        _header.update(*this);
    }
    
    void File::read_frame(Frame& frame, uint64_t frameIndex) {
        assert(!_open_for_writing);
        
        std::unique_lock<std::mutex> guard(_lock);
        if(frameIndex >= _header.num_frames)
           throw U_EXCEPTION("Frame index ", frameIndex," out of range.");
        
        uint64_t pos = _header.index_table.at(frameIndex);
        uint64_t old = current_offset();
        
        if(old != pos)
            seek(pos);
        
        frame.read_from(*this, (long_t)frameIndex);
        frame.get_blobs();
    }
    
    void File::read_next_frame(Frame& frame, uint64_t frame_to_read) {
        assert(!_open_for_writing);
        
        std::unique_lock<std::mutex> guard(_lock);
        if(frame_to_read >= _header.num_frames)
           throw U_EXCEPTION("Frame index ", frame_to_read," out of range.");
        
        frame.read_from(*this, (long_t)frame_to_read);
    }
#ifdef USE_GPU_MAT
    void File::frame(uint64_t frameIndex, gpuMat &output, cmn::source_location loc) {
        cv::Mat local;
        frame_optional_background(frameIndex, local, true);
        local.copyTo(output);
    }
#endif
    void File::frame(uint64_t frameIndex, cv::Mat &output, cmn::source_location loc) {
        frame_optional_background(frameIndex, output, true);
    }
    
    void File::frame_optional_background(uint64_t frameIndex, cv::Mat& output, bool with_background) {
        Frame frame;
        read_frame(frame, frameIndex);
        
        if(with_background)
            average().copyTo(output);
        else
            output = cv::Mat::zeros(header().resolution.height, header().resolution.width, CV_8UC1);
        
        for (uint16_t i=0; i<frame.n(); i++) {
            uint64_t index = 0;
            auto &mask = frame.mask().at(i);
            auto &pixels = frame.pixels().at(i);
            
            for(const HorizontalLine &line : *mask) {
                for(int x=line.x0; x<=line.x1; x++) {
                    output.at<uchar>(line.y, x) = (uchar)pixels->at(index++);
                }
            }
        }
    }

    void fix_file(File& file) {
        print("Starting file copy and fix (",file.filename(),")...");
        
        File copy(file.filename()+"_fix");
        copy.set_resolution(file.header().resolution);
        copy.set_offsets(file.crop_offsets());
        copy.set_average(file.average());
        
        auto keys = sprite::parse_values(file.header().metadata).keys();
        sprite::parse_values(GlobalSettings::map(), file.header().metadata);
        SETTING(meta_write_these) = keys;
        
        if(file.has_mask())
            copy.set_mask(file.mask());
        
        copy.header().timestamp = file.header().timestamp;
        copy.start_writing(true);
        
        uint64_t raw_prev_timestamp = 0;
        uint64_t last_reset = 0;
        uint64_t last_reset_idx = 0;
        uint64_t last_difference = 0;
        
        for (uint64_t idx = 0; idx < file.length(); idx++) {
            pv::Frame frame;
            file.read_frame(frame, idx);
            
            //frame.set_timestamp(file.header().timestamp + frame.timestamp());
            
            if (frame.timestamp() < raw_prev_timestamp) {
                last_reset = raw_prev_timestamp + last_difference;
                last_reset_idx = idx;
                
                FormatWarning("Fixing frame ",idx," because timestamp ",frame.timestamp()," < ",last_reset," -> ",last_reset + frame.timestamp());
            } else {
            	last_difference = frame.timestamp() - raw_prev_timestamp;
            }
            
            raw_prev_timestamp = frame.timestamp();
            
            if(last_reset_idx) {
                frame.set_timestamp(last_reset + frame.timestamp());
            }
            
            copy.add_individual(std::move(frame));
            
            if (idx % 1000 == 0) {
                print("Frame ", idx," / ", file.length()," (",copy.compression_ratio() * 100,"% compression ratio)...");
            }
        }
        
        copy.stop_writing();
        
        print("Written fixed file.");
    }
    
    void File::try_compress() {
        File copy(filename()+"_test");
        copy.set_resolution(header().resolution);
        copy.set_offsets(crop_offsets());
        copy.set_average(average());
        copy.header().timestamp = header().timestamp;
        
        if(has_mask())
            copy.set_mask(mask());
        
        copy.start_writing(true);
        
        std::vector<std::string> save = GlobalSettings::map().has("meta_write_these") ? SETTING(meta_write_these) : std::vector<std::string>{};
        
        sprite::Map map;
        if(!header().metadata.empty())
            sprite::parse_values(map, header().metadata);
        SETTING(meta_write_these) = map.keys();
        
        pv::Frame frame;
        for (uint64_t i=0; i<length(); i++) {
            read_frame(frame, i);
            frame.set_timestamp(header().timestamp + frame.timestamp());
            
            copy.add_individual(std::move(frame));
            
            if (i % 1000 == 0) {
                print("Frame ", i," / ",length(),"...");
            }
        }
        
        copy.stop_writing();
        
        print("Written");
        
        {
            print_info();
            
            File test(filename()+"_test");
            test.start_reading();
        }
        
        SETTING(meta_write_these) = save;
    }
    
    void File::update_metadata() {
        if(!_open_for_modifying)
            throw U_EXCEPTION("Must be open for writing.");
    
        print("Updating metadata...");
        auto metadata = _header.generate_metadata();
        write(metadata, _header.meta_offset());
    }
    
    short File::framerate() const {
        if(_header.average_tdelta == 0)
            return -1;
        return short(1000. * 1000. / _header.average_tdelta);
    }

    std::vector<float> File::calculate_percentiles(const std::initializer_list<float> &percent) {
        if(_open_for_writing)
            throw U_EXCEPTION("Cannot calculate percentiles when file is opened for writing.");
        Timer timer;
        TaskSentinel sentinel(this);
        
        std::vector<float> pixel_values;
#ifdef NDEBUG
        std::unique_lock guard(_lock);
        Image average(_average);
        guard.unlock();
        
        if(average.bounds().size().max() <= 4000) {
            // also take into account background image?
            pixel_values.insert(pixel_values.end(), average.data(), average.data() + average.cols * average.rows);
        }
#endif
        
        uint64_t num_frames = 0;
        uint64_t start_frame = 0;
        std::set<long_t> samples;
        
        while(!sentinel.terminate() && timer.elapsed() < 1 && pixel_values.size() < 10000000 && start_frame < length()) {
            auto range = arange<long_t>((long_t)start_frame, (long_t)length(), max(1, long_t(length() * 0.1)));
            pv::Frame frame;
            uint64_t big_loop_size = samples.size();
            
            for (auto frameIndex : range) {
                if((uint64_t)frameIndex >= length())
                    continue;
                
                uint64_t prev_size = samples.size();
                samples.insert(frameIndex);
                if(samples.size() == prev_size)
                    continue;
                
                read_frame(frame, (uint64_t)frameIndex);
                for(uint64_t i=0; i<frame.n(); ++i) {
                    pixel_values.insert(pixel_values.end(), frame.pixels().at(i)->begin(), frame.pixels().at(i)->end());
                }
                ++num_frames;
                samples.insert(frameIndex);
                
                if(timer.elapsed() >= 2)
                    break;
            }
            
            if(big_loop_size == samples.size())
                break;
            
            ++start_frame;
        }
        
        std::sort(pixel_values.begin(), pixel_values.end());
        auto p = percentile(pixel_values, percent);
        print("Took ", timer.elapsed(),"s to calculate percentiles in ",num_frames," frames.");
        //auto str = Meta::toStr(samples);
        return p;
    }
    
    double File::generate_average_tdelta() {
        if(!_open_for_writing && _header.average_tdelta == 0 && length()>0) {
            // readable
            double average = 0;
            uint64_t samples = 0;
            const uint64_t step = max(1u, (length()-1) / 10u);
            pv::Frame frame;
            for (uint64_t i=1; i<length(); i+=step) {
                read_frame(frame, i);
                double stamp = frame.timestamp();
                if(i < length()-1) {
                    read_frame(frame, i+1);
                    stamp = double(frame.timestamp()) - stamp;
                }
                average += stamp;
                ++samples;
            }
            average /= double(samples);
            header().average_tdelta = average;
        }
        
        return _header.average_tdelta;
    }
    
    timestamp_t File::timestamp(uint64_t frameIndex, cmn::source_location loc) const {
        if(_open_for_writing)
            throw U_EXCEPTION<FormatterType::UNIX, const char*>("Cannot get timestamps for video while writing.", loc);
        
        if(frameIndex >= header().num_frames)
            throw U_EXCEPTION("Access out of bounds ",frameIndex,"/",header().num_frames,". (caller ", loc.file_name(),":", loc.line(),")");
        
        return header().index_table[frameIndex];
    }
    
    timestamp_t File::start_timestamp() const {
        if(_open_for_writing)
            throw U_EXCEPTION("Cannot get timestamps for video while writing.");
        
        return header().timestamp;
    }
    
    std::string File::toStr() const {
        return "pv::File<"+filename().str()+">";
    }
}
