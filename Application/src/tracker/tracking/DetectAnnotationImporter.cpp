#include "DetectAnnotationImporter.h"

#include <file/CSVReader.h>
#include <thirdparty/fkYAML/node.hpp>

namespace track::detect::annotation_import {
namespace {

using namespace cmn;
namespace dataset = track::detect::annotation_dataset;

static constexpr std::array<std::string_view, 4> video_source_extensions{
    "mp4", "avi", "mov", "mkv"
};

std::string source_name_key(std::string_view name) {
    return file::normalized_filename_key(name, video_source_extensions);
}

struct DatasetConfig {
    file::Path yaml_dir;
    file::Path dataset_root;
    std::vector<file::Path> image_inputs;
    cmn::blob::ObjectClass_t class_names;
    std::vector<std::string> keypoint_names;
    size_t keypoint_count{0};
    size_t keypoint_dims{0};
};

struct ImageEntry {
    file::Path path;
    file::Path label_path;
    file::Path relative_path;
};

struct CsvFrameMapping {
    std::string video_source;
    Frame_t source_index;
};

struct ImageMapping {
    std::string video_source;
    Frame_t source_index;
    bool current_source{true};
    bool from_csv{false};
};

struct InferredImageSource {
    std::string video_source;
    Frame_t source_index;
};

struct SourceSelection {
    std::vector<std::string> choices;
    std::string automatic;
    std::string selected;
};

std::string unquote_field(std::string_view input) {
    auto value = std::string(utils::trim(input));
    if(value.size() >= 2
       && ((value.front() == '"' && value.back() == '"')
           || (value.front() == '\'' && value.back() == '\'')))
    {
        return util::unescape(value.substr(1, value.size() - 2));
    }
    return util::unescape(value);
}

using YamlNode = fkyaml::node;

const YamlNode* yaml_find(const YamlNode& node, const std::string& key) {
    if(!node.is_mapping())
        return nullptr;
    const auto& map = node.as_map();
    auto it = map.find(YamlNode(key));
    return it == map.end() ? nullptr : &it->second;
}

uint16_t yaml_class_id_key(const YamlNode& key) {
    std::optional<int64_t> id;
    if(key.is_integer()) {
        id = key.get_value<int64_t>();
    } else if(key.is_string()) {
        auto value = std::string(utils::trim(key.get_value<std::string>()));
        if(!value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isdigit(c); }))
            id = Meta::fromStr<int64_t>(value);
    }

    if(!id)
        throw InvalidArgumentException("data.yaml names mapping keys must be numeric class ids.");
    if(*id < 0 || *id > std::numeric_limits<uint16_t>::max())
        throw InvalidArgumentException("data.yaml names class id ", *id, " is out of range.");
    return narrow_cast<uint16_t>(*id);
}

cmn::blob::ObjectClass_t yaml_names(const YamlNode& node) {
    cmn::blob::ObjectClass_t result;

    if(node.is_mapping()) {
        for(const auto& [key, value] : node.as_map()) {
            result[yaml_class_id_key(key)] = value.get_value<std::string>();
        }
        return result;
    }

    if(node.is_sequence()) {
        uint16_t index = 0;
        for(const auto& value : node.as_seq())
            result[index++] = value.get_value<std::string>();
        return result;
    }

    throw InvalidArgumentException("data.yaml names must be a sequence or class-id mapping.");
}

std::pair<size_t, size_t> yaml_kpt_shape(const YamlNode& node) {
    if(!node.is_sequence() || node.size() < 2)
        throw InvalidArgumentException("data.yaml kpt_shape must be a sequence like [count, dims].");

    auto read_size = [&](size_t index) -> size_t {
        const auto& value = node.at(narrow_cast<int>(index));
        if(!value.is_integer())
            throw InvalidArgumentException("data.yaml kpt_shape values must be integers.");
        const auto number = value.get_value<int64_t>();
        if(number < 0)
            throw InvalidArgumentException("data.yaml kpt_shape values must be non-negative.");
        return narrow_cast<size_t>(number);
    };

    return {read_size(0), read_size(1)};
}

file::Path resolve_path(const file::Path& base, const std::string& value) {
    file::Path path(unquote_field(value));
    if(path.empty())
        return path;
    if(path.is_absolute())
        return path;
    return base / path;
}

std::string filename_of(const file::Path& path) {
    return std::filesystem::path(path.str()).filename().string();
}

// Mapping CSV image keys may be dataset-relative, absolute, or hand-written on
// Windows. Normalize only spelling, not basename, so split paths stay distinct.
std::string normalized_mapping_key(std::string value) {
    value = unquote_field(value);
    std::replace(value.begin(), value.end(), '\\', '/');
    while(value.rfind("./", 0) == 0)
        value.erase(0, 2);
    return value;
}

void add_mapping_key(std::vector<std::string>& keys, std::string key) {
    key = normalized_mapping_key(std::move(key));
    if(key.empty())
        return;
    if(std::find(keys.begin(), keys.end(), key) == keys.end())
        keys.push_back(std::move(key));
}

std::string stem_of(const file::Path& path) {
    return std::filesystem::path(path.str()).stem().string();
}

// Roboflow exports often append the original image extension and hash after
// the useful stem, e.g. "video_mp4-0086_jpg.rf.<hash>".
std::string strip_roboflow_suffix(std::string stem) {
    for(const auto& marker : {"_jpg.rf.", "_jpeg.rf.", "_png.rf.", "_bmp.rf.", "_tif.rf.", "_tiff.rf.", "_webp.rf."}) {
        if(auto pos = stem.find(marker); pos != std::string::npos)
            return stem.substr(0, pos);
    }
    return stem;
}

std::string trim_source_separators(std::string value) {
    while(!value.empty() && (value.back() == '_' || value.back() == '-' || value.back() == '.' || std::isspace(static_cast<unsigned char>(value.back()))))
        value.pop_back();
    while(!value.empty() && (value.front() == '_' || value.front() == '-' || value.front() == '.' || std::isspace(static_cast<unsigned char>(value.front()))))
        value.erase(value.begin());
    return value;
}

// Turn dataset-safe video suffixes back into ordinary basenames so
// "run_mp4" and "run.mp4" match the same selected source.
std::string decode_encoded_video_extension(std::string source) {
    for(const auto& ext : {"mp4", "avi", "mov", "mkv"}) {
        for(const auto& separator : {"_", "-", "."}) {
            const auto suffix = std::string(separator) + ext;
            if(source.size() > suffix.size()
               && utils::lowercase(source).rfind(suffix) == source.size() - suffix.size())
            {
                source.erase(source.size() - suffix.size());
                source += ".";
                source += ext;
                return source;
            }
        }
    }
    return source;
}

// Remove a trailing generic frame marker from the inferred source prefix.
// This keeps "video_frame_0012" associated with "video", not "video_frame".
std::string strip_trailing_frame_marker(std::string source) {
    for(const auto& marker : {"source_index", "source-index", "frame", "source"}) {
        auto lowered = utils::lowercase(source);
        if(lowered.size() > std::strlen(marker)
           && lowered.rfind(marker) == lowered.size() - std::strlen(marker))
        {
            source.erase(source.size() - std::strlen(marker));
            return trim_source_separators(source);
        }
    }
    return source;
}

bool is_generic_source_prefix(const std::string& source) {
    auto lowered = utils::lowercase(source);
    return lowered.empty()
        || lowered == "frame"
        || lowered == "source"
        || lowered == "source_index"
        || lowered == "source-index";
}

// Infer the video source and original source_index from dataset image names.
// The rightmost number is treated as the frame id so dates or run ids in the
// video prefix do not become frame numbers.
std::optional<InferredImageSource> infer_source_from_image_name(const file::Path& image) {
    auto stem = strip_roboflow_suffix(stem_of(image));
    struct NumberSpan {
        size_t begin{};
        size_t end{};
    };

    std::vector<NumberSpan> spans;
    for(size_t i = 0; i < stem.size();) {
        if(!std::isdigit(static_cast<unsigned char>(stem.at(i)))) {
            ++i;
            continue;
        }
        const auto begin = i;
        while(i < stem.size() && std::isdigit(static_cast<unsigned char>(stem.at(i))))
            ++i;
        spans.push_back({begin, i});
    }

    for(auto it = spans.rbegin(); it != spans.rend(); ++it) {
        auto source = strip_trailing_frame_marker(trim_source_separators(stem.substr(0, it->begin)));
        if(is_generic_source_prefix(source))
            continue;
        auto number = stem.substr(it->begin, it->end - it->begin);
        return InferredImageSource{
            .video_source = decode_encoded_video_extension(source),
            .source_index = Frame_t(Meta::fromStr<Frame_t::number_t>(number))
        };
    }
    return std::nullopt;
}

std::string relative_to(const file::Path& path, const file::Path& root) {
    try {
        auto rel = std::filesystem::relative(std::filesystem::path(path.str()), std::filesystem::path(root.str()));
        return rel.string();
    } catch(...) {
        return {};
    }
}

bool is_image_file(const file::Path& path) {
    const auto ext = utils::lowercase(std::string(path.extension()));
    return is_in(ext, "jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp");
}

void collect_images_recursive(const file::Path& folder, std::vector<file::Path>& images) {
    for(const auto& entry : folder.find_files()) {
        if(entry.is_folder()) {
            collect_images_recursive(entry, images);
        } else if(is_image_file(entry)) {
            images.push_back(entry);
        }
    }
}

file::Path label_path_for_image(const file::Path& image_path) {
    auto fs_path = std::filesystem::path(image_path.str());
    auto label_path = fs_path;
    label_path.replace_extension(".txt");

    std::vector<std::filesystem::path> parts;
    for(const auto& part : label_path)
        parts.push_back(part);

    for(auto it = parts.rbegin(); it != parts.rend(); ++it) {
        if(*it == "images") {
            *it = "labels";
            std::filesystem::path rebuilt;
            for(const auto& part : parts)
                rebuilt /= part;
            return file::Path(rebuilt.string());
        }
    }

    return file::Path(label_path.string());
}

std::vector<file::Path> read_image_list(const file::Path& list_file, const file::Path& dataset_root) {
    std::vector<file::Path> images;
    std::istringstream lines(list_file.read_file());
    std::string line;
    while(std::getline(lines, line)) {
        line = std::string(utils::trim(line));
        if(line.empty())
            continue;
        images.push_back(resolve_path(dataset_root, line));
    }
    return images;
}

std::vector<ImageEntry> collect_images(const DatasetConfig& config) {
    std::vector<file::Path> paths;
    for(const auto& input : config.image_inputs) {
        if(input.is_folder()) {
            collect_images_recursive(input, paths);
        } else if(input.has_extension("txt")) {
            auto listed = read_image_list(input, config.dataset_root);
            paths.insert(paths.end(), listed.begin(), listed.end());
        } else if(is_image_file(input)) {
            paths.push_back(input);
        } else {
            throw InvalidArgumentException("Dataset image input ", input, " is neither an image folder, image file, nor .txt list.");
        }
    }

    std::sort(paths.begin(), paths.end());
    paths.erase(std::unique(paths.begin(), paths.end()), paths.end());

    std::vector<ImageEntry> entries;
    entries.reserve(paths.size());
    for(const auto& path : paths) {
        entries.push_back(ImageEntry{
            .path = path,
            .label_path = label_path_for_image(path),
            .relative_path = file::Path(relative_to(path, config.dataset_root))
        });
    }
    return entries;
}

DatasetConfig parse_dataset_config(const file::Path& data_yaml) {
    if(data_yaml.empty())
        throw InvalidArgumentException("Select a data.yaml file to import.");
    if(!data_yaml.exists())
        throw InvalidArgumentException("Cannot find data.yaml at ", data_yaml, ".");
    if(data_yaml.is_folder())
        throw InvalidArgumentException("Expected a data.yaml file but got folder ", data_yaml, ".");

    auto yaml = YamlNode::deserialize(data_yaml.read_file());
    if(!yaml.is_mapping())
        throw InvalidArgumentException("data.yaml must contain a YAML mapping.");

    DatasetConfig config;
    config.yaml_dir = data_yaml.remove_filename();
    config.dataset_root = config.yaml_dir;

    if(auto path = yaml_find(yaml, "path"); path)
        config.dataset_root = resolve_path(config.yaml_dir, path->get_value<std::string>());

    auto add_inputs = [&](const std::string& key) {
        if(auto node = yaml_find(yaml, key); node) {
            const auto items = node->is_sequence()
                ? node->get_value<std::vector<std::string>>()
                : std::vector<std::string>{node->get_value<std::string>()};
            for(const auto& item : items)
                config.image_inputs.push_back(resolve_path(config.dataset_root, item));
        }
    };
    add_inputs("train");
    add_inputs("val");

    if(config.image_inputs.empty())
        throw InvalidArgumentException("data.yaml must define at least a train or val image path.");

    if(auto names = yaml_find(yaml, "names"); names)
        config.class_names = yaml_names(*names);

    if(auto keypoint_names = yaml_find(yaml, "keypoint_names"); keypoint_names)
        config.keypoint_names = keypoint_names->is_sequence()
            ? keypoint_names->get_value<std::vector<std::string>>()
            : std::vector<std::string>{keypoint_names->get_value<std::string>()};

    if(auto kpt_shape = yaml_find(yaml, "kpt_shape"); kpt_shape) {
        std::tie(config.keypoint_count, config.keypoint_dims) = yaml_kpt_shape(*kpt_shape);
        if(config.keypoint_count == 0 || config.keypoint_dims < 2 || config.keypoint_dims > 3)
            throw InvalidArgumentException("Unsupported kpt_shape in data.yaml: ", YamlNode::serialize(*kpt_shape), ".");
    }

    if(config.keypoint_names.empty() && config.keypoint_count > 0) {
        for(size_t i = 0; i < config.keypoint_count; ++i)
            config.keypoint_names.push_back("kp_" + Meta::toStr(i));
    }

    return config;
}

std::unordered_map<std::string, CsvFrameMapping> read_mapping_csv(const file::Path& csv_path) {
    std::unordered_map<std::string, CsvFrameMapping> mapping;
    if(csv_path.empty())
        return mapping;
    if(!csv_path.exists())
        throw InvalidArgumentException("Cannot find frame mapping CSV at ", csv_path, ".");

    auto data = csv_path.read_file();
    CSVReader reader(data, ',', true);
    const auto& header = reader.header();
    auto image_it = std::find(header.begin(), header.end(), "image");
    auto video_it = std::find(header.begin(), header.end(), "video_source");
    auto source_it = std::find(header.begin(), header.end(), "source_index");
    if(image_it == header.end() || video_it == header.end() || source_it == header.end())
        throw InvalidArgumentException("Frame mapping CSV must have image,video_source,source_index columns.");

    const auto image_col = std::distance(header.begin(), image_it);
    const auto video_col = std::distance(header.begin(), video_it);
    const auto source_col = std::distance(header.begin(), source_it);
    while(reader.hasNext()) {
        auto row = reader.nextRow();
        if(row.empty())
            continue;
        if(row.size() <= narrow_cast<size_t>(std::max({image_col, video_col, source_col})))
            throw InvalidArgumentException("Frame mapping CSV row has too few columns.");
        auto image = normalized_mapping_key(row.at(narrow_cast<size_t>(image_col)));
        auto video_source = unquote_field(row.at(narrow_cast<size_t>(video_col)));
        auto source = std::string(utils::trim(row.at(narrow_cast<size_t>(source_col))));
        if(image.empty() || video_source.empty() || source.empty())
            continue;
        mapping[image] = CsvFrameMapping{
            .video_source = file::Path(video_source).filename(),
            .source_index = Frame_t(Meta::fromStr<Frame_t::number_t>(source))
        };
    }
    return mapping;
}

std::vector<std::string> mapping_keys(const ImageEntry& entry, const DatasetConfig& config) {
    std::vector<std::string> keys;

    add_mapping_key(keys, entry.path.str());
    if(!entry.relative_path.empty())
        add_mapping_key(keys, entry.relative_path.str());

    auto rel_root = relative_to(entry.path, config.dataset_root);
    if(!rel_root.empty())
        add_mapping_key(keys, rel_root);
    auto rel_yaml = relative_to(entry.path, config.yaml_dir);
    if(!rel_yaml.empty())
        add_mapping_key(keys, rel_yaml);

    if(!entry.relative_path.empty()) {
        add_mapping_key(keys, (config.dataset_root / entry.relative_path).str());
        add_mapping_key(keys, (config.yaml_dir / entry.relative_path).str());
    }
    if(!entry.path.is_absolute()) {
        add_mapping_key(keys, (config.dataset_root / entry.path).str());
        add_mapping_key(keys, (config.yaml_dir / entry.path).str());
    }

    add_mapping_key(keys, filename_of(entry.path));
    return keys;
}

// CSV mappings are authoritative and may use split-relative paths, absolute
// paths, or legacy basenames. Try path-specific spellings before basename
// fallback so mixed train/val folders with duplicate filenames stay distinct.
std::optional<CsvFrameMapping> lookup_csv_frame(const ImageEntry& entry, const DatasetConfig& config, const std::unordered_map<std::string, CsvFrameMapping>& mapping) {
    for(const auto& key : mapping_keys(entry, config)) {
        if(auto it = mapping.find(key); it != mapping.end())
            return it->second;
    }
    return std::nullopt;
}

void add_source_choice(std::vector<std::string>& choices, std::string source) {
    source = file::Path(trim_source_separators(source)).filename();
    if(source.empty())
        return;
    const auto normalized = source_name_key(source);
    auto exists = std::find_if(choices.begin(), choices.end(), [&](const auto& choice) {
        return source_name_key(choice) == normalized;
    });
    if(exists == choices.end())
        choices.push_back(std::move(source));
}

std::vector<std::string> source_choices_from_mapping(const std::unordered_map<std::string, CsvFrameMapping>& csv_mapping) {
    std::vector<std::string> choices;
    for(const auto& [image, mapping] : csv_mapping) {
        (void)image;
        add_source_choice(choices, mapping.video_source);
    }
    return choices;
}

std::vector<std::string> source_choices_from_images(const std::vector<ImageEntry>& images) {
    std::vector<std::string> choices;
    for(const auto& image : images) {
        if(auto inferred = infer_source_from_image_name(image.path); inferred)
            add_source_choice(choices, inferred->video_source);
    }
    return choices;
}

// Build the GUI source dropdown and select the effective source. A user
// override wins; otherwise prefer the source matching the currently open video.
SourceSelection choose_source(
    const std::vector<ImageEntry>& images,
    const std::unordered_map<std::string, CsvFrameMapping>& csv_mapping,
    const ImportOptions& options,
    bool has_csv_mapping)
{
    SourceSelection selection;
    selection.choices = has_csv_mapping
        ? source_choices_from_mapping(csv_mapping)
        : source_choices_from_images(images);
    std::sort(selection.choices.begin(), selection.choices.end());

    const auto current_normalized = source_name_key(options.current_source_basename);
    auto find_matching = [&](const std::string& source) -> std::optional<std::string> {
        const auto normalized = source_name_key(source);
        for(const auto& choice : selection.choices) {
            if(source_name_key(choice) == normalized)
                return choice;
        }
        return std::nullopt;
    };

    if(!current_normalized.empty()) {
        for(const auto& choice : selection.choices) {
            if(source_name_key(choice) == current_normalized) {
                selection.automatic = choice;
                break;
            }
        }
    }

    if(selection.automatic.empty() && !selection.choices.empty())
        selection.automatic = selection.choices.front();
    if(selection.automatic.empty() && !options.current_source_basename.empty())
        selection.automatic = file::Path(options.current_source_basename).filename();

    if(!options.selected_source_basename.empty()) {
        if(auto matched = find_matching(options.selected_source_basename); matched)
            selection.selected = *matched;
        else
            selection.selected = file::Path(options.selected_source_basename).filename();
    } else {
        selection.selected = selection.automatic;
    }

    add_source_choice(selection.choices, selection.selected);
    return selection;
}

std::optional<std::string> strip_current_source_prefix(const file::Path& image, const std::string& current_source_basename);

std::optional<ImageMapping> map_image_to_source_index(
    const ImageEntry& image,
    const DatasetConfig& config,
    const std::unordered_map<std::string, CsvFrameMapping>& csv_mapping,
    const std::string& selected_source,
    bool has_csv_mapping,
    ImportPreview& preview)
{
    const auto selected_normalized = source_name_key(selected_source);
    // With a CSV, do not guess: unmapped images are skipped so unrelated
    // videos in the dataset can round-trip without being imported into view.
    if(has_csv_mapping) {
        if(auto from_csv = lookup_csv_frame(image, config, csv_mapping); from_csv) {
            const bool current = selected_normalized.empty()
                              || source_name_key(from_csv->video_source) == selected_normalized;
            ++preview.mapped_from_csv;
            if(!current)
                ++preview.skipped_other_sources;
            return ImageMapping{
                .video_source = from_csv->video_source,
                .source_index = from_csv->source_index,
                .current_source = current,
                .from_csv = true
            };
        }
        ++preview.skipped_other_sources;
        return std::nullopt;
    }

    // Without a CSV, first try filenames that include the source basename,
    // then fall back to generic frame_* stems after stripping the selected
    // source prefix.
    if(auto inferred = infer_source_from_image_name(image.path); inferred) {
        const bool current = selected_normalized.empty()
                          || source_name_key(inferred->video_source) == selected_normalized;
        ++preview.mapped_from_filenames;
        if(!current)
            ++preview.skipped_other_sources;
        return ImageMapping{
            .video_source = inferred->video_source,
            .source_index = inferred->source_index,
            .current_source = current,
            .from_csv = false
        };
    }

    auto candidate_stem = strip_current_source_prefix(image.path, selected_source);
    if(!candidate_stem)
        candidate_stem = filename_of(image.path);

    auto parsed = parse_source_index_from_image_stem(*candidate_stem);
    if(parsed.has_value()) {
        ++preview.mapped_from_filenames;
        return ImageMapping{
            .video_source = selected_source,
            .source_index = *parsed.source_index,
            .current_source = true,
            .from_csv = false
        };
    }

    if(!selected_source.empty()) {
        ++preview.skipped_other_sources;
        return std::nullopt;
    }

    preview.errors.push_back(parsed.error + " Provide a frame mapping CSV with image,video_source,source_index.");
    return std::nullopt;
}

std::optional<std::string> strip_current_source_prefix(const file::Path& image, const std::string& current_source_basename) {
    auto stem = stem_of(image);
    auto candidates = file::filename_prefix_candidates(current_source_basename);
    if(candidates.empty())
        return stem;

    auto lower_stem = utils::lowercase(stem);
    for(const auto& source : candidates) {
        auto pos = lower_stem.find(source);
        if(pos == std::string::npos)
            continue;

        auto stripped = stem.substr(pos + source.size());
        while(!stripped.empty() && (stripped.front() == '_' || stripped.front() == '-' || stripped.front() == '.' || std::isspace(static_cast<unsigned char>(stripped.front()))))
            stripped.erase(stripped.begin());
        return stripped.empty() ? stem : stripped;
    }

    return std::nullopt;
}

void require_any_mapped_image(const std::vector<ImageEntry>& images, ImportPreview& preview) {
    if(!images.empty() && preview.mapped_from_filenames + preview.mapped_from_csv == 0) {
        preview.errors.push_back(
            "Could not map any dataset images to the current video. Select a CSV mapping file with image,video_source,source_index columns, or use image names that include the current video basename and a supported frame id."
        );
    }
}

std::string source_key_for_mapping(const ImageMapping& mapping, const ImportOptions& options) {
    if(!mapping.video_source.empty())
        return file::Path(mapping.video_source).filename();
    if(!options.current_source_basename.empty())
        return file::Path(options.current_source_basename).filename();
    return "current";
}

std::optional<Frame_t> to_annotation_frame(Frame_t source, const ImportOptions& options, std::vector<std::string>& errors, std::vector<std::string>& warnings, const file::Path& image) {
    if(!source.valid()) {
        errors.push_back("Invalid source_index for image " + image.str() + ".");
        return std::nullopt;
    }
    if(options.source_start && source < *options.source_start) {
        warnings.push_back("Skipping current-video import for image " + image.str() + " because source_index " + source.toStr() + " is before conversion start " + options.source_start->toStr() + ".");
        return std::nullopt;
    }
    if(options.source_end && source >= *options.source_end) {
        warnings.push_back("Skipping current-video import for image " + image.str() + " because source_index " + source.toStr() + " is outside conversion end " + options.source_end->toStr() + ".");
        return std::nullopt;
    }

    auto frame = source;
    if(options.source_start)
        frame = Frame_t(source.get() - options.source_start->get());

    if(options.converted_length && frame >= *options.converted_length) {
        warnings.push_back("Skipping current-video import for image " + image.str() + " because source_index " + source.toStr() + " maps to converted frame " + frame.toStr() + " outside video length " + options.converted_length->toStr() + ".");
        return std::nullopt;
    }
    return frame;
}

Annotation::Point_t point_from_normalized(double x, double y, const Size2& size) {
    if(round(x * size.width) <= -1 || round(y * size.height) <= -1 || round(x * size.width) > size.width || round(y * size.height) > size.height)
        FormatWarning("Normalized point [", round(x * size.width), ", ", round(y * size.height), "] is outside ", size,".");
    const auto px = saturate(std::round(x * size.width), 0.0, (double)size.width);
    const auto py = saturate(std::round(y * size.height), 0.0, (double)size.height);
    return Annotation::Point_t(narrow_cast<uint16_t>(px), narrow_cast<uint16_t>(py));
}

Annotation parse_label_row(const std::string& row, const DatasetConfig& config, const Size2& video_size, Task& row_task) {
    if(video_size.width <= 0 || video_size.height <= 0)
        throw InvalidArgumentException("Current video_size must be known before importing YOLO annotations.");

    auto normalized = row;
    std::replace(normalized.begin(), normalized.end(), '\t', ' ');
    std::vector<double> values;
    for(const auto& token : utils::split(normalized, ' ', true, true)) {
        values.push_back(Meta::fromStr<double>(token));
    }

    if(values.empty())
        throw InvalidArgumentException("Empty YOLO row.");
    if(values.front() < 0 || values.front() > 255)
        throw InvalidArgumentException("Class id ", values.front(), " is outside the supported Annotation range 0-255.");

    Annotation annotation;
    annotation.clid = narrow_cast<uint8_t>(values.front());

    const bool has_pose_shape = config.keypoint_count > 0;
    const size_t pose_dims = config.keypoint_dims == 0 ? 3 : config.keypoint_dims;
    const size_t pose_fields = has_pose_shape ? 5 + config.keypoint_count * pose_dims : 0;

    if(has_pose_shape && values.size() == pose_fields) {
        annotation.type = AnnotationType::POSE;
        row_task = task_t::pose;
        annotation.points.reserve(config.keypoint_count);
        size_t offset = 5;
        for(size_t i = 0; i < config.keypoint_count; ++i) {
            const double x = values.at(offset);
            const double y = values.at(offset + 1);
            const double v = pose_dims == 3 ? values.at(offset + 2) : 2.0;
            annotation.points.push_back(v <= 0 ? Annotation::Point_t{} : point_from_normalized(x, y, video_size));
            offset += pose_dims;
        }
        return annotation;
    }

    if(values.size() == 5) {
        const double cx = values.at(1);
        const double cy = values.at(2);
        const double w = values.at(3);
        const double h = values.at(4);
        if(w < 0 || h < 0)
            throw InvalidArgumentException("YOLO box width/height cannot be negative.");
        const double x0 = cx - w * 0.5;
        const double y0 = cy - h * 0.5;
        const double x1 = cx + w * 0.5;
        const double y1 = cy + h * 0.5;
        annotation.type = AnnotationType::BOX;
        row_task = task_t::boxes;
        annotation.points = {
            point_from_normalized(x0, y0, video_size),
            point_from_normalized(x1, y0, video_size),
            point_from_normalized(x1, y1, video_size),
            point_from_normalized(x0, y1, video_size)
        };
        return annotation;
    }

    if(values.size() >= 7 && ((values.size() - 1) % 2) == 0) {
        annotation.type = AnnotationType::SEGMENTATION;
        row_task = task_t::segmentation;
        annotation.points.reserve((values.size() - 1) / 2);
        for(size_t i = 1; i + 1 < values.size(); i += 2)
            annotation.points.push_back(point_from_normalized(values.at(i), values.at(i + 1), video_size));
        return annotation;
    }

    throw InvalidArgumentException("Unsupported YOLO row with ", values.size(), " fields.");
}

Task combine_task(Task current, Task next) {
    if(current == task_t::unknown)
        return next;
    if(current == next)
        return current;
    return task_t::mixed;
}

std::optional<detect::ObjectDetectionFormat_t> detect_format_from_task(Task task) {
    switch(task) {
        case task_t::boxes:
            return detect::ObjectDetectionFormat::boxes;
        case task_t::segmentation:
            return detect::ObjectDetectionFormat::masks;
        case task_t::pose:
            return detect::ObjectDetectionFormat::poses;
        case task_t::unknown:
        case task_t::mixed:
            return std::nullopt;
    }
    return std::nullopt;
}

void update_detect_format_metadata(ImportPreview& preview, const ImportOptions& options) {
    auto imported_format = detect_format_from_task(preview.task);
    if(!imported_format)
        return;

    preview.metadata.imported_detect_format = *imported_format;
    preview.metadata.detect_format_changed = options.current_detect_format != *imported_format;
    if(preview.metadata.detect_format_changed)
        preview.warnings.push_back("Import will update detect_format to " + imported_format->str() + ".");
}

const glz::json_t* object_value(const glz::json_t::object_t& object, const std::string& key) {
    auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
}

double required_number(const glz::json_t::object_t& object, const std::string& key, const std::string& context) {
    auto value = object_value(object, key);
    if(!value || !value->is_number())
        throw InvalidArgumentException(context, " is missing numeric field '", key, "'.");
    return value->get_number();
}

bool is_integer_number(double value) {
    return std::isfinite(value) && std::floor(value) == value;
}

uint64_t required_coco_id(const glz::json_t::object_t& object, const std::string& key, const std::string& context) {
    auto value = required_number(object, key, context);
    if(value < 0 || !is_integer_number(value))
        throw InvalidArgumentException(context, " field '", key, "' must be a non-negative integer.");
    return narrow_cast<uint64_t>(value);
}

uint8_t required_coco_class_id(const glz::json_t::object_t& object, const std::string& key, const std::string& context) {
    auto value = required_number(object, key, context);
    if(value < 0 || value > 255 || !is_integer_number(value))
        throw InvalidArgumentException(context, " field '", key, "' must be an integer in the supported Annotation range 0-255.");
    return narrow_cast<uint8_t>(value);
}

std::string required_string(const glz::json_t::object_t& object, const std::string& key, const std::string& context) {
    auto value = object_value(object, key);
    if(!value || !value->is_string())
        throw InvalidArgumentException(context, " is missing string field '", key, "'.");
    return value->get_string();
}

std::vector<double> json_number_array(const glz::json_t& json, const std::string& context) {
    if(!json.is_array())
        throw InvalidArgumentException(context, " must be an array.");
    std::vector<double> values;
    for(const auto& item : json.get_array()) {
        if(!item.is_number())
            throw InvalidArgumentException(context, " contains a non-numeric value.");
        values.push_back(item.get_number());
    }
    return values;
}

std::vector<blob::Pose::Skeleton::Connection> coco_skeleton_connections(const glz::json_t& skeleton, const std::string& context) {
    if(!skeleton.is_array())
        throw InvalidArgumentException(context, " skeleton must be an array.");

    std::vector<blob::Pose::Skeleton::Connection> connections;
    for(const auto& connection_json : skeleton.get_array()) {
        auto values = json_number_array(connection_json, context + " skeleton connection");
        if(values.size() != 2)
            throw InvalidArgumentException(context, " skeleton connections must contain exactly two keypoint ids.");
        if(values.at(0) < 1 || values.at(1) < 1 || !is_integer_number(values.at(0)) || !is_integer_number(values.at(1)))
            throw InvalidArgumentException(context, " skeleton uses invalid COCO keypoint ids.");

        // COCO skeleton ids are 1-based. TRex Pose::Skeleton::Connection is 0-based.
        connections.push_back(blob::Pose::Skeleton::Connection{
            .from = narrow_cast<uint8_t>(values.at(0) - 1),
            .to = narrow_cast<uint8_t>(values.at(1) - 1),
            .name = ""
        });
    }
    return connections;
}

Annotation::Point_t point_from_coco_absolute(double x, double y, const Size2& image_size) {
    if(image_size.width <= 0 || image_size.height <= 0)
        throw InvalidArgumentException("COCO image size must be positive, got ", image_size, ".");
    if(x < -1e-6 || y < -1e-6 || x > image_size.width + 1e-6 || y > image_size.height + 1e-6)
        throw InvalidArgumentException("COCO point [", x, ", ", y, "] is outside image bounds ", image_size, ".");

    // COCO stores keypoints, segmentations, and bboxes as absolute pixels.
    const auto px = saturate(std::round(x), 0.0, double(std::numeric_limits<uint16_t>::max()));
    const auto py = saturate(std::round(y), 0.0, double(std::numeric_limits<uint16_t>::max()));
    return Annotation::Point_t(narrow_cast<uint16_t>(px), narrow_cast<uint16_t>(py));
}

Annotation coco_bbox_annotation(uint8_t clid, const std::vector<double>& bbox, const Size2& image_size) {
    if(bbox.size() != 4)
        throw InvalidArgumentException("COCO bbox must have 4 values.");
    const double x = bbox.at(0);
    const double y = bbox.at(1);
    const double w = bbox.at(2);
    const double h = bbox.at(3);
    if(w < 0 || h < 0)
        throw InvalidArgumentException("COCO bbox width/height cannot be negative.");

    return Annotation{
        .clid = clid,
        .type = AnnotationType::BOX,
        .points = {
            point_from_coco_absolute(x, y, image_size),
            point_from_coco_absolute(x + w, y, image_size),
            point_from_coco_absolute(x + w, y + h, image_size),
            point_from_coco_absolute(x, y + h, image_size)
        }
    };
}

std::optional<std::vector<Annotation::Point_t>> coco_segmentation_points(const glz::json_t& segmentation, const Size2& image_size, const std::string& context) {
    if(segmentation.is_null())
        return std::nullopt;
    if(!segmentation.is_array())
        throw InvalidArgumentException(context, " uses unsupported COCO RLE segmentation. Polygon segmentation is required.");

    const glz::json_t* polygon = nullptr;
    const auto& array = segmentation.get_array();
    if(array.empty())
        return std::nullopt;
    if(array.front().is_array()) {
        polygon = &array.front();
    } else {
        polygon = &segmentation;
    }

    auto coords = json_number_array(*polygon, context + " segmentation");
    if(coords.size() < 6 || coords.size() % 2 != 0)
        throw InvalidArgumentException(context, " segmentation polygon must contain x/y pairs for at least 3 points.");

    std::vector<Annotation::Point_t> points;
    points.reserve(coords.size() / 2);
    for(size_t i = 0; i + 1 < coords.size(); i += 2)
        points.push_back(point_from_coco_absolute(coords.at(i), coords.at(i + 1), image_size));
    return points;
}

std::optional<std::vector<Annotation::Point_t>> coco_keypoints(const glz::json_t& keypoints, const Size2& image_size, const std::string& context) {
    if(keypoints.is_null())
        return std::nullopt;
    auto values = json_number_array(keypoints, context + " keypoints");
    if(values.empty())
        return std::nullopt;
    if(values.size() % 3 != 0)
        throw InvalidArgumentException(context, " keypoints must be x/y/visibility triples.");

    std::vector<Annotation::Point_t> points;
    points.reserve(values.size() / 3);
    bool has_visible = false;
    for(size_t i = 0; i + 2 < values.size(); i += 3) {
        const double v = values.at(i + 2);
        has_visible = has_visible || v > 0;
        points.push_back(v <= 0 ? Annotation::Point_t{} : point_from_coco_absolute(values.at(i), values.at(i + 1), image_size));
    }
    if(!has_visible)
        return std::nullopt;
    return points;
}

}

FrameIndexParseResult parse_source_index_from_image_stem(std::string_view input) {
    auto stem = std::filesystem::path(std::string(input)).stem().string();
    if(stem.empty())
        stem = std::string(input);

    if(std::regex_match(stem, std::regex(R"(^-\d+$|^(frame|source)[_-]-\d+.*$|^source[_-]index[_-]-\d+.*$)")))
        return {.error = "Negative source frame ids are not supported in image name '" + stem + "'."};

    std::vector<std::string> tokens;
    std::string current;
    for(char c : stem) {
        if(c == '_' || c == '-') {
            if(!current.empty())
                tokens.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    if(!current.empty())
        tokens.push_back(current);

    std::vector<size_t> numeric_tokens;
    for(size_t i = 0; i < tokens.size(); ++i) {
        if(!tokens[i].empty()
           && std::all_of(tokens[i].begin(), tokens[i].end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); }))
        {
            numeric_tokens.push_back(i);
        }
    }

    if(numeric_tokens.empty())
        return {.error = "Image name '" + stem + "' does not contain a supported source frame id."};
    if(numeric_tokens.size() > 1)
        return {.error = "Image name '" + stem + "' contains multiple plausible source frame ids."};

    const auto index = numeric_tokens.front();
    bool supported = false;
    if(index == 0) {
        supported = true;
    } else if(index == 1 && is_in(tokens.at(0), "frame", "source")) {
        supported = true;
    } else if(index == 2 && tokens.at(0) == "source" && tokens.at(1) == "index") {
        supported = true;
    }

    if(!supported)
        return {.error = "Image name '" + stem + "' does not match supported frame id patterns."};

    return {
        .source_index = Frame_t(Meta::fromStr<Frame_t::number_t>(tokens.at(index)))
    };
}

ImportPreview preview_yolo_import(const ImportOptions& options, ImportScope scope) {
    ImportPreview preview;
    preview.dataset_file = options.dataset_file;

    try {
        auto config = parse_dataset_config(options.dataset_file);
        auto images = collect_images(config);
        const bool has_csv_mapping = !options.frame_mapping_csv.empty();
        if(!has_csv_mapping)
            preview.warnings.push_back("No frame mapping CSV selected. Filename-based frame mapping is a fallback; use image,video_source,source_index CSV mapping to preserve exact source frames.");
        auto csv_mapping = read_mapping_csv(options.frame_mapping_csv);
        auto source_selection = choose_source(images, csv_mapping, options, has_csv_mapping);
        preview.source_choices = source_selection.choices;
        preview.auto_source_basename = source_selection.automatic;
        preview.selected_source_basename = source_selection.selected;
        preview.image_count = images.size();
        preview.metadata.imported_class_names = config.class_names;
        preview.metadata.imported_keypoint_names = config.keypoint_names;

        if(!config.class_names.empty()) {
            preview.metadata.class_names_changed = !options.current_class_names
                                                || *options.current_class_names != config.class_names;
            if(preview.metadata.class_names_changed)
                preview.warnings.push_back("Import will update detect_classes to " + Meta::toStr(config.class_names) + ".");
        }

        if(!config.keypoint_names.empty()) {
            preview.metadata.keypoint_names_changed = options.current_keypoint_names != config.keypoint_names;
            if(preview.metadata.keypoint_names_changed)
                preview.warnings.push_back("Import will update detect_keypoint_names to " + Meta::toStr(config.keypoint_names) + ".");
        }

        if(images.empty())
            preview.errors.push_back("No images were found from data.yaml train/val paths.");

        for(const auto& image : images) {
            auto mapping = map_image_to_source_index(image, config, csv_mapping, source_selection.selected, has_csv_mapping, preview);
            if(!mapping)
                continue;

            std::optional<Frame_t> frame;
            if(mapping->current_source) {
                frame = to_annotation_frame(mapping->source_index, options, preview.errors, preview.warnings, image.path);
            }
            
            if(not frame
               && scope == import_scope_t::current_video)
                continue;

            if(!image.label_path.exists())
                continue;

            try {
                std::vector<Annotation> parsed;
                Task file_task{task_t::unknown};
                std::istringstream rows(image.label_path.read_file());
                std::string row;
                while(std::getline(rows, row)) {
                    row = std::string(utils::trim(row));
                    if(row.empty())
                        continue;
                    
                    Task row_task{task_t::unknown};
                    auto annotation = parse_label_row(row, config, options.video_size, row_task);
                    file_task = combine_task(file_task, row_task);
                    parsed.push_back(std::move(annotation));
                }

                /// Only commit once the whole label file parsed cleanly. A single
                /// malformed row must not leave already-parsed rows in
                /// source_annotations while the frame is dropped from annotations
                /// (which previously also desynced current_video vs all_videos).
                if(not parsed.empty()) {
                    preview.task = combine_task(preview.task, file_task);
                    auto& source_annotations = preview.source_annotations[source_key_for_mapping(*mapping, options)][mapping->source_index];
                    for(auto& annotation : parsed) {
                        annotation.uid = narrow_cast<uint8_t>(source_annotations.size());
                        source_annotations.push_back(annotation);
                        if(frame) {
                            auto& frame_annotations = preview.annotations[*frame];
                            annotation.uid = narrow_cast<uint8_t>(frame_annotations.size());
                            frame_annotations.push_back(std::move(annotation));
                        }
                    }
                }
                
            } catch(const std::exception& e) {
                
                preview.warnings.push_back("Label " + image.label_path.str() + ": " + e.what());
            }
        }

        require_any_mapped_image(images, preview);

        preview.counts = count_annotation_types(preview.annotations);
        update_detect_format_metadata(preview, options);
        preview.annotated_frames = preview.annotations.size();
        if(preview.source_annotations.empty() && preview.errors.empty())
            preview.warnings.push_back("Dataset contains no label rows to import.");

    } catch(const std::exception& e) {
        preview.errors.push_back(e.what());
    } catch(...) {
        preview.errors.push_back("Unknown YOLO annotation import error.");
    }

    return preview;
}

ImportPreview preview_coco_import(const ImportOptions& options) {
    ImportPreview preview;
    preview.dataset_file = options.dataset_file;

    try {
        if(options.dataset_file.empty())
            throw InvalidArgumentException("Select a COCO annotations JSON file to import.");
        if(!options.dataset_file.exists())
            throw InvalidArgumentException("Cannot find COCO annotations file at ", options.dataset_file, ".");
        if(options.dataset_file.is_folder())
            throw InvalidArgumentException("Expected a COCO annotations JSON file but got folder ", options.dataset_file, ".");

        glz::json_t root;
        auto text = options.dataset_file.read_file();
        if(auto error = glz::read_json(root, text); error != glz::error_code::none)
            throw InvalidArgumentException("Cannot parse COCO JSON: ", glz::format_error(error, text));
        if(!root.is_object())
            throw InvalidArgumentException("COCO annotations file must contain a JSON object.");

        const auto& root_object = root.get_object();
        auto images_json = object_value(root_object, "images");
        auto annotations_json = object_value(root_object, "annotations");
        auto categories_json = object_value(root_object, "categories");
        if(!images_json || !images_json->is_array())
            throw InvalidArgumentException("COCO annotations file must contain an images array.");
        if(!annotations_json || !annotations_json->is_array())
            throw InvalidArgumentException("COCO annotations file must contain an annotations array.");

        DatasetConfig config;
        config.yaml_dir = options.dataset_file.remove_filename();
        config.dataset_root = config.yaml_dir;

        std::map<uint64_t, ImageEntry> image_entries;
        std::map<uint64_t, Size2> image_sizes;
        std::vector<ImageEntry> images;
        for(const auto& image_json : images_json->get_array()) {
            if(!image_json.is_object()) {
                preview.errors.push_back("COCO images array contains a non-object entry.");
                continue;
            }
            try {
                const auto& image = image_json.get_object();
                const auto id = required_coco_id(image, "id", "COCO image");
                const auto file_name = required_string(image, "file_name", "COCO image");
                const auto width = narrow_cast<uint16_t>(std::round(required_number(image, "width", "COCO image")));
                const auto height = narrow_cast<uint16_t>(std::round(required_number(image, "height", "COCO image")));
                ImageEntry entry{
                    .path = file::Path(file_name),
                    .label_path = {},
                    .relative_path = file::Path(file_name)
                };
                image_entries[id] = entry;
                image_sizes[id] = Size2(width, height);
                images.push_back(entry);
            } catch(const std::exception& e) {
                preview.errors.push_back(std::string(e.what()));
            }
        }
        preview.image_count = images.size();

        std::vector<std::string> first_keypoint_names;
        blob::Pose::Skeletons imported_skeletons;
        if(categories_json && categories_json->is_array()) {
            for(const auto& category_json : categories_json->get_array()) {
                if(!category_json.is_object()) {
                    preview.errors.push_back("COCO categories array contains a non-object entry.");
                    continue;
                }
                try {
                    const auto& category = category_json.get_object();
                    const auto id = required_coco_class_id(category, "id", "COCO category");
                    std::string category_name;
                    if(auto name = object_value(category, "name"); name && name->is_string())
                        category_name = name->get_string();
                    if(!category_name.empty())
                        preview.metadata.imported_class_names[id] = category_name;

                    if(auto keypoints = object_value(category, "keypoints"); keypoints && keypoints->is_array()) {
                        std::vector<std::string> names;
                        for(const auto& keypoint : keypoints->get_array()) {
                            if(keypoint.is_string())
                                names.push_back(keypoint.get_string());
                        }
                        if(first_keypoint_names.empty())
                            first_keypoint_names = names;
                        else if(!names.empty() && names != first_keypoint_names)
                            preview.warnings.push_back("COCO categories define different keypoint schemas; using the first schema.");
                    }

                    if(!category_name.empty()) {
                        if(auto skeleton = object_value(category, "skeleton"); skeleton && skeleton->is_array()) {
                            auto connections = coco_skeleton_connections(*skeleton, "COCO category '" + category_name + "'");
                            if(!connections.empty())
                                imported_skeletons._skeletons[category_name] = blob::Pose::Skeleton(std::move(connections));
                        }
                    }
                } catch(const std::exception& e) {
                    preview.errors.push_back(std::string(e.what()));
                }
            }
        }
        preview.metadata.imported_keypoint_names = first_keypoint_names;
        if(!imported_skeletons._skeletons.empty())
            preview.metadata.imported_skeletons = std::move(imported_skeletons);

        if(!preview.metadata.imported_class_names.empty()) {
            preview.metadata.class_names_changed = !options.current_class_names
                                                || *options.current_class_names != preview.metadata.imported_class_names;
            if(preview.metadata.class_names_changed)
                preview.warnings.push_back("Import will update detect_classes to " + Meta::toStr(preview.metadata.imported_class_names) + ".");
        }

        if(!preview.metadata.imported_keypoint_names.empty()) {
            preview.metadata.keypoint_names_changed = options.current_keypoint_names != preview.metadata.imported_keypoint_names;
            if(preview.metadata.keypoint_names_changed)
                preview.warnings.push_back("Import will update detect_keypoint_names to " + Meta::toStr(preview.metadata.imported_keypoint_names) + ".");
        }

        if(preview.metadata.imported_skeletons) {
            preview.metadata.skeletons_changed = !options.current_skeletons
                                              || *options.current_skeletons != *preview.metadata.imported_skeletons;
            if(preview.metadata.skeletons_changed)
                preview.warnings.push_back("Import will update detect_skeleton to COCO skeleton metadata.");
        }

        const bool has_csv_mapping = !options.frame_mapping_csv.empty();
        if(!has_csv_mapping)
            preview.warnings.push_back("No frame mapping CSV selected. Filename-based frame mapping is a fallback; use image,video_source,source_index CSV mapping to preserve exact source frames.");
        auto csv_mapping = read_mapping_csv(options.frame_mapping_csv);
        auto source_selection = choose_source(images, csv_mapping, options, has_csv_mapping);
        preview.source_choices = source_selection.choices;
        preview.auto_source_basename = source_selection.automatic;
        preview.selected_source_basename = source_selection.selected;
        std::map<uint64_t, ImageMapping> image_mappings;
        std::map<uint64_t, Frame_t> mapped_frames;
        for(const auto& [image_id, image] : image_entries) {
            auto mapping = map_image_to_source_index(image, config, csv_mapping, source_selection.selected, has_csv_mapping, preview);
            if(!mapping)
                continue;
            image_mappings[image_id] = *mapping;
            if(mapping->current_source) {
                if(auto frame = to_annotation_frame(mapping->source_index, options, preview.errors, preview.warnings, image.path); frame)
                    mapped_frames[image_id] = *frame;
            }
        }
        require_any_mapped_image(images, preview);

        for(const auto& annotation_json : annotations_json->get_array()) {
            if(!annotation_json.is_object()) {
                preview.errors.push_back("COCO annotations array contains a non-object entry.");
                continue;
            }
            try {
                const auto& object = annotation_json.get_object();
                const auto image_id = required_coco_id(object, "image_id", "COCO annotation");
                auto image_mapping = image_mappings.find(image_id);
                if(image_mapping == image_mappings.end())
                    continue;

                std::optional<Frame_t> frame;
                if(auto mapped = mapped_frames.find(image_id); mapped != mapped_frames.end())
                    frame = mapped->second;

                const auto category_id = required_coco_class_id(object, "category_id", "COCO annotation");

                auto size_it = image_sizes.find(image_id);
                const auto image_size = size_it == image_sizes.end() ? options.video_size : size_it->second;
                const std::string context = "COCO annotation for image_id " + Meta::toStr(image_id);

                Annotation annotation;
                bool parsed = false;

                if(auto keypoints = object_value(object, "keypoints"); keypoints) {
                    if(auto points = coco_keypoints(*keypoints, image_size, context); points) {
                        annotation.points = std::move(*points);
                        annotation.clid = category_id;
                        annotation.type = AnnotationType::POSE;
                        preview.task = combine_task(preview.task, task_t::pose);
                        parsed = true;
                    }
                }

                if(!parsed) {
                    if(auto segmentation = object_value(object, "segmentation"); segmentation) {
                        if(auto points = coco_segmentation_points(*segmentation, image_size, context); points) {
                            annotation.points = std::move(*points);
                            annotation.clid = category_id;
                            annotation.type = AnnotationType::SEGMENTATION;
                            preview.task = combine_task(preview.task, task_t::segmentation);
                            parsed = true;
                        }
                    }
                }

                if(!parsed) {
                    if(auto bbox = object_value(object, "bbox"); bbox) {
                        annotation = coco_bbox_annotation(category_id, json_number_array(*bbox, context + " bbox"), image_size);
                        preview.task = combine_task(preview.task, task_t::boxes);
                        parsed = true;
                    }
                }

                if(!parsed) {
                    preview.errors.push_back(context + " has no supported keypoints, polygon segmentation, or bbox.");
                    continue;
                }

                const auto& mapping = image_mapping->second;
                auto& source_annotations = preview.source_annotations[source_key_for_mapping(mapping, options)][mapping.source_index];
                annotation.uid = narrow_cast<uint8_t>(source_annotations.size());
                source_annotations.push_back(annotation);
                if(frame) {
                    auto& frame_annotations = preview.annotations[*frame];
                    annotation.uid = narrow_cast<uint8_t>(frame_annotations.size());
                    frame_annotations.push_back(std::move(annotation));
                }
            } catch(const std::exception& e) {
                preview.errors.push_back(std::string(e.what()));
            }
        }

        preview.counts = count_annotation_types(preview.annotations);
        update_detect_format_metadata(preview, options);
        preview.annotated_frames = preview.annotations.size();
        if(preview.source_annotations.empty() && preview.errors.empty())
            preview.warnings.push_back("COCO file contains no annotations to import for the current video.");

    } catch(const std::exception& e) {
        preview.errors.push_back(e.what());
    } catch(...) {
        preview.errors.push_back("Unknown COCO annotation import error.");
    }

    return preview;
}

ImportPreview preview_dataset_import(const ImportOptions& options, ImportScope scope) {
    return options.format == dataset::format_t::coco
        ? preview_coco_import(options)
        : preview_yolo_import(options, scope);
}

AnnotationMap apply_dataset_import(const ImportPreview& preview, const AnnotationMap& existing, MergeMode mode, ImportScope scope) {
    if(!preview.can_import())
        throw InvalidArgumentException("Cannot import annotations: ", no_quotes(utils::ShortenText(Meta::toStr(preview.errors), 1000)));

    AnnotationMap result = mode == merge_mode_t::replace ? AnnotationMap{} : existing;
    for(const auto& [frame, annotations] : preview.annotations) {
        auto& destination = result[frame];
        destination.insert(destination.end(), annotations.begin(), annotations.end());
    }

    auto& sources = result.sources();
    if(mode == merge_mode_t::replace)
        sources.clear();
    // The flat map above already holds the current video's annotations (the ones
    // that get displayed/edited). Only fan the dataset out into the per-source
    // store when the user asked to import every video; "current video only" keeps
    // other videos' annotations out of track_detect_annotations entirely.
    if(scope == import_scope_t::all_videos) {
        for(const auto& [source, frames] : preview.source_annotations) {
            auto& destination_source = sources[source];
            for(const auto& [frame, annotations] : frames) {
                auto& destination = destination_source[frame];
                destination.insert(destination.end(), annotations.begin(), annotations.end());
            }
        }
    }

    for(auto& [frame, annotations] : result) {
        (void)frame;
        for(size_t i = 0; i < annotations.size(); ++i)
            annotations[i].uid = narrow_cast<uint8_t>(i);
    }

    for(auto& [source, frames] : sources) {
        (void)source;
        for(auto& [frame, annotations] : frames) {
            (void)frame;
            for(size_t i = 0; i < annotations.size(); ++i)
                annotations[i].uid = narrow_cast<uint8_t>(i);
        }
    }

    return result;
}

AnnotationMap apply_yolo_import(const ImportPreview& preview, const AnnotationMap& existing, MergeMode mode, ImportScope scope) {
    return apply_dataset_import(preview, existing, mode, scope);
}

}
