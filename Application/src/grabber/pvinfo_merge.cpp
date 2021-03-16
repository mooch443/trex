#include "pvinfo_merge.h"
#include <pv.h>
#include <tracking/StaticBackground.h>
#include <misc/SpriteMap.h>
#include <misc/GlobalSettings.h>
#include <misc/default_config.h>
#include <misc/PVBlob.h>
#include <processing/CPULabeling.h>

using namespace cmn;

std::string date_time() {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];
    
    time (&rawtime);
    timeinfo = localtime(&rawtime);
    
    strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
    std::string str(buffer);
    return str;
}

void initiate_merging(const std::vector<file::Path>& merge_videos, int argc, char**argv) {
    uint64_t min_length = std::numeric_limits<uint64_t>::max();
    std::vector<std::shared_ptr<pv::File>> files;
    std::vector<std::shared_ptr<track::StaticBackground>> backgrounds;
    std::vector<std::shared_ptr<sprite::Map>> configs;
    
    std::map<pv::File*, float> cms_per_pixel;
    Size2 resolution;
    
    pv::DataLocation::register_path("merge", [](file::Path filename) -> file::Path {
        if(!filename.empty() && filename.is_absolute()) {
#ifndef NDEBUG
            if(!SETTING(quiet))
                Warning("Returning absolute path '%S'. We cannot be sure this is writable.", &filename.str());
#endif
            return filename;
        }
        
        auto path = SETTING(merge_dir).value<file::Path>();
        if(path.empty()) {
            return pv::DataLocation::parse("input", filename);
        } else
            return path / filename;
    });
    
    for(auto name : merge_videos) {
        name = pv::DataLocation::parse("merge", name);
        if((name.has_extension() && name.extension() == "pv" && !name.exists())
           || !name.add_extension("pv").exists())
        { // approximation (this is not a true XOR)
            U_EXCEPTION("File '%S' cannot be found.", &name.str());
        }
        
        if(!name.has_extension() || name.extension() != "pv")
            name = name.add_extension("pv");
        
        auto file = std::make_shared<pv::File>(name);
        file->start_reading();
        files.push_back(file);
        
        if(min_length > file->length())
            min_length = file->length();
        
        resolution += Size2(file->header().resolution);
        backgrounds.push_back(std::make_shared<track::StaticBackground>(std::make_shared<Image>(file->average()), nullptr));
        //cv::imshow(name.str(), backgrounds.back()->image().get());
        //cv::waitKey(1);
        
        SETTING(filename) = name.remove_extension();
        auto settings_file = pv::DataLocation::parse("output_settings");
        if(settings_file.exists()) {
            Debug("settings for '%S' found", &name.str());
            auto config = std::make_shared<sprite::Map>();
            config->set_do_print(false);
            
            GlobalSettings::docs_map_t docs;
            default_config::get(*config, docs, NULL);
            
            GlobalSettings::load_from_string({}, *config, utils::read_file(settings_file.str()), AccessLevelType::STARTUP);
            if(!file->header().metadata.empty())
                sprite::parse_values(*config, file->header().metadata);
            if(!config->has("meta_real_width") || config->get<float>("meta_real_width").value() == 0)
                config->get<float>("meta_real_width").value(30);
            if(!config->has("cm_per_pixel") || config->get<float>("cm_per_pixel").value() == 0)
                config->get<float>("cm_per_pixel") = config->get<float>("meta_real_width").value() / float(file->average().cols);
            
            cms_per_pixel[file.get()] = config->get<float>("cm_per_pixel");
            configs.push_back(config);
            
        } else {
            U_EXCEPTION("Cant find settings for '%S' at '%S'", &name.str(), &settings_file.str());
        }
    }
    
    resolution /= (float)files.size();
    resolution = resolution.map<round>();
    
    cv::Mat average;
    if(SETTING(merge_background).value<file::Path>().empty()) {
        for(auto file : files) {
            if(file->header().resolution.width >= resolution.width
               && file->header().resolution.height >= resolution.height
               && (average.empty() || (file->header().resolution.width > average.cols && file->header().resolution.height > average.rows)))
            {
                file->average()(Bounds(Vec2(), resolution)).copyTo(average);
            }
        }
        
    } else {
        auto path = SETTING(merge_background).value<file::Path>();
        auto raw_path = path;
        path = pv::DataLocation::parse("input", path);
        
        if((!path.has_extension() && path.add_extension("pv").exists()) || (path.has_extension() && path.extension() == "pv")) {
            pv::File file(path);
            file.start_reading();
            
            file.average().copyTo(average);
        } else {
            if(!path.exists()) {
                auto dimensions = Meta::fromStr<Size2>(raw_path.str());
                average = cv::Mat::ones(dimensions.height, dimensions.width, CV_8UC1);
                average *= 255;
                
            } else {
                auto mat = cv::imread(path.str());
                if(mat.channels() > 1) {
                    std::vector<cv::Mat> images;
                    cv::split(mat, images);
                    images[0].copyTo(average);
                } else
                    mat.copyTo(average);
            }
        }
        
        resolution = Size2(average);
    }
    
    track::StaticBackground new_background(std::make_shared<Image>(average), nullptr);
    
    if(SETTING(frame_rate).value<int>() == 0){
        if(!files.front()->header().metadata.empty())
            sprite::parse_values(GlobalSettings::map(), files.front()->header().metadata);
        
        //SETTING(frame_rate) = int(1000 * 1000 / float(frame.timestamp()));
    }
    
    struct Source {
        size_t video_index;
        size_t frame_index;
        uint32_t blob_id;
    };
    
    SETTING(meta_write_these) = std::vector<std::string>{
        "meta_real_width",
        "meta_source_path",
        "meta_cmd",
        "meta_build",
        "meta_conversion_time",
        "meta_number_merged_videos",
        "frame_rate"
    };
    
    SETTING(meta_conversion_time) = std::string(date_time());
    std::stringstream ss;
    for(int i=0; i<argc; ++i) {
        ss << " " << argv[i];
    }
    SETTING(meta_cmd) = ss.str();
    SETTING(meta_source_path) = file::Path();
    SETTING(meta_number_merged_videos) = size_t(files.size());
    
    // frame: {blob : source}
    std::map<long_t, std::map<uint32_t, Source>> meta;
    if(SETTING(merge_output_path).value<file::Path>().empty())
        SETTING(merge_output_path) = file::Path("merged");
    
    file::Path out_path = pv::DataLocation::parse("output", SETTING(merge_output_path).value<file::Path>());
    pv::File output(out_path);
    
    output.set_resolution((cv::Size)resolution);
    output.set_average(average);
    
    output.set_start_time(std::chrono::system_clock::now());
    output.start_writing(true);
    
    //auto start_time = output.header().timestamp;
    auto str = Meta::toStr(files);
    Debug("Writing videos %S to '%S' [0,%lu] with resolution (%f,%f)", &str, &out_path.str(), min_length, resolution.width, resolution.height);
    using namespace track;
    GlobalSettings::map().dont_print("cm_per_pixel");
    const bool merge_overlapping_blobs = SETTING(merge_overlapping_blobs);
    //const float scaled_video_width = floor(sqrt(resolution.width * resolution.height / float(files.size())));
    
    /*for(uint64_t frame=0; frame<min(1000, min_length); ++frame) {
     pv::Frame f, o;
     if(SETTING(terminate))
     break;
     
     std::vector<pv::BlobPtr> ptrs;
     std::vector<size_t> indexes;
     
     for(size_t vdx = 0; vdx < files.size(); ++vdx) {
     auto &file = files.at(vdx);
     file->read_frame(f, frame);
     if(!vdx) o.set_timestamp(f.timestamp());
     
     Vec2 offset = Vec2();
     auto blob_size_range = configs.at(vdx)->get<Rangef>("blob_size_range").value();
     const int track_threshold = configs.at(vdx)->get<int>("track_threshold").value();
     SETTING(cm_per_pixel) = cms_per_pixel[file.get()];
     
     for(size_t i=0; i<f.n(); ++i) {
     auto b = std::make_shared<pv::Blob>(f.mask().at(i), f.pixels().at(i));
     auto recount = b->recount(track_threshold, *backgrounds.at(vdx));
     
     if(recount < blob_size_range.start * 0.1 || recount > blob_size_range.end * 5)
     continue;
     
     auto id = b->blob_id();
     
     b->transfer_backgrounds(*backgrounds.at(vdx), new_background, offset);
     b->add_offset(offset);
     
     if(!new_background.bounds().contains(b->bounds()))
     {
     auto str = Meta::toStr(*b);
     Warning("%S out of bounds for background %fx%f", &str, new_background.bounds().width, new_background.bounds().height);
     
     } else {
     meta[frame][b->blob_id()] = Source{ vdx, frame, id };
     
     
     ptrs.push_back(b);
     indexes.push_back(vdx);
     }
     }
     }
     }*/
    
    uint64_t timestamp_offset = output.length() == 0 ? 0 : output.last_frame().timestamp();
    merge_mode_t::Class merge_mode = SETTING(merge_mode);
    
    for (uint64_t frame=0; frame<min_length; ++frame) {
        pv::Frame f, o;
        if(SETTING(terminate))
            break;
        
        std::vector<pv::BlobPtr> ptrs;
        std::vector<size_t> indexes;
        
        for(size_t vdx = 0; vdx < files.size(); ++vdx) {
            auto &file = files.at(vdx);
            file->read_frame(f, frame);
            if(!vdx) o.set_timestamp(timestamp_offset + f.timestamp());
            //o.set_timestamp(start_time + f.timestamp() - file->start_timestamp());
            
            Vec2 offset = merge_mode == merge_mode_t::centered ? Vec2((Size2(average) - Size2(file->average())) * 0.5) : Vec2(0);
            Vec2 scale = merge_mode == merge_mode_t::centered ? Vec2(1) : Vec2(Size2(average).div(Size2(file->average())));
            auto blob_size_range = configs.at(vdx)->get<Rangef>("blob_size_range").value();
            const int track_threshold = configs.at(vdx)->get<int>("track_threshold").value();
            SETTING(cm_per_pixel) = cms_per_pixel[file.get()];
            
            for(size_t i=0; i<f.n(); ++i) {
                auto b = std::make_shared<pv::Blob>(f.mask().at(i), f.pixels().at(i));
                auto recount = b->recount(track_threshold, *backgrounds.at(vdx));
                
                if(recount < blob_size_range.start * 0.1 || recount > blob_size_range.end * 5)
                    continue;
                
                auto id = b->blob_id();
                
                b->transfer_backgrounds(*backgrounds.at(vdx), new_background, offset);
                b->scale_coordinates(scale);
                b->add_offset(offset);
                
                if(!new_background.bounds().contains(b->bounds()))
                {
                    auto str = Meta::toStr(*b);
                    Warning("%S out of bounds for background %fx%f", &str, new_background.bounds().width, new_background.bounds().height);
                    
                } else {
                    meta[frame][b->blob_id()] = Source{ vdx, frame, id };
                    
                    ptrs.push_back(b);
                    indexes.push_back(vdx);
                }
            }
        }
        
        // collect cliques of potentially overlapping blobs
        std::vector<std::set<pv::BlobPtr>> cliques;
        std::vector<bool> viewed;
        viewed.resize(ptrs.size());
        
        for(size_t i=0; i<ptrs.size(); ++i) {
            if (viewed[i])
                continue;
            
            std::set<pv::BlobPtr> clique{ ptrs.at(i) };
            viewed[i] = true;
            
            for (size_t j=i+1; j<ptrs.size(); ++j) {
                if(viewed.at(j) /*|| indexes.at(j) == indexes.at(i)*/)
                    continue;
                
                if(ptrs.at(i)->bounds().overlaps(ptrs.at(j)->bounds())) {
                    viewed.at(j) = true;
                    clique.insert(ptrs.at(j));
                }
            }
            
            cliques.push_back(clique);
        }
        
        for(auto &clique : cliques) {
            if(clique.size() == 1 || !merge_overlapping_blobs) {
                for(auto &b : clique) {
                    o.add_object(b->lines(), b->pixels());
                }
                
            } else {
                Bounds bounds(FLT_MAX, FLT_MAX, 0, 0);
                Bounds test(FLT_MAX, FLT_MAX, 0, 0);
                for(auto &b : clique) {
                    bounds.combine(b->bounds());
                    test.pos() = min(test.pos(), b->bounds().pos());
                    test.size() = max(test.size(), b->bounds().pos() + b->bounds().size());
                }
                test.size() -= test.pos();
                
                if(bounds != test)
                    Debug("why");
                
                assert(!clique.empty());
                
                cv::Mat mat = cv::Mat::zeros(bounds.height, bounds.width, CV_8UC1);
                
                for(auto &b: clique) {
                    auto [pos, image] = b->image(NULL, Bounds(bounds.pos(), Size2(mat)));
                    pos -= bounds.pos();
                    
                    // blend image into combined image
                    for (uint x=0; x<image->cols; ++x) {
                        for (uint y=0; y<image->rows; ++y) {
                            assert(Rangef(0, mat.cols).contains(x + pos.x));
                            assert(Rangef(0, mat.rows).contains(y + pos.y));
                            
                            assert(image->cols * y + x < image->size());
                            
                            auto &pb = mat.at<uchar>(y + pos.y, x + pos.x);
                            auto &pi = image->data()[image->cols * y + x];
                            if(!pb) pb = pi;
                            else {
                                float alphai = pi > 0 ? 1 - pi / 255.f : 0;
                                float alphab = pb > 0 ? 1 - pb / 255.f : 0;
                                
                                pb = saturate((int)roundf((float(pi) * alphai + float(pb) * alphab) / (alphai + alphab)), 0, 255);
                            }
                        }
                    }
                }
                
                auto blobs = CPULabeling::run(mat);
                for(auto && [lines, pixels] : blobs) {
                    for(auto &line : *lines) {
                        line.x0 += bounds.pos().x;
                        line.x1 += bounds.pos().x;
                        line.y += bounds.pos().y;
                    }
                    o.add_object(lines, pixels);
                    //std::make_shared<pv::Blob>(lines, pixels);
                }
                //cv::imshow("blended", mat);
                //cv::waitKey(10);
            }
        }
        
#ifndef NDEBUG
        for(size_t i=0; i<f.n(); ++i) {
            assert(viewed[i]);
        }
#endif
        
        output.add_individual(o);
        
        if(frame % size_t(min_length * 0.1) == 0) {
            Debug("merging %d/%d", frame, min_length);
        }
    }
    
    output.stop_writing();
    output.close();
    
    output.start_reading();
    output.print_info();
    output.close();
}
