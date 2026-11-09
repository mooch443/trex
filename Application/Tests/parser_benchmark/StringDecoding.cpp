#include <commons.pc.h>
#include "StringDecoding.h"
#include <misc/metastring.h>

namespace cmn::benchmark {
namespace {

using Decode = void (*)(std::string&, std::string_view, size_t);

void previous_full(std::string& output, std::string_view input, size_t) {
    benchmark_previous::utils::fast_fromstr(output, input);
}

void current_full(std::string& output, std::string_view input, size_t) {
    utils::fast_fromstr(output, input);
}

struct Sample {
    double elapsed_ns;
    uint64_t checksum;
};

Sample measure(Decode decode, std::string_view input, size_t limit, bool reuse, size_t iterations) {
    std::string output;
    if(reuse)
        output.reserve(input.size());
    uint64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for(size_t i = 0; i < iterations; ++i) {
        if(reuse) {
            output.clear();
            decode(output, input, limit);
            checksum += output.size();
            if(not output.empty())
                checksum += static_cast<unsigned char>(output.front()) + static_cast<unsigned char>(output.back());
        } else {
            std::string fresh;
            decode(fresh, input, limit);
            checksum += fresh.size();
            if(not fresh.empty())
                checksum += static_cast<unsigned char>(fresh.front()) + static_cast<unsigned char>(fresh.back());
        }
    }
    const auto end = std::chrono::steady_clock::now();
    return {std::chrono::duration<double, std::nano>(end - start).count(), checksum};
}

}

void CompareStringDecoders() {
    struct Case {
        std::string name;
        std::string input;
        std::string previous;
        std::string current;
    };
    const std::vector<Case> cases{
        {"empty", "", "", ""},
        {"plain_short", "fish #123", "fish #123", "fish #123"},
        {"quoted_short", "'fish #123'", "fish #123", "fish #123"},
        {"raw_windows_path", R"(C:\new\track.mp4)", R"(C:\new\track.mp4)", R"(C:\new\track.mp4)"},
        {"quoted_windows_path", R"("C:\\new\\track.mp4")", R"(C:\new\track.mp4)", R"(C:\new\track.mp4)"},
        {"escaped_apostrophe", R"('can\'t')", "can't", "can't"},
        {"unknown_escape", R"('\q\\')", R"(q\)", R"(q\)"},
        {"plain_4k", std::string(4096, 'x'), std::string(4096, 'x'), std::string(4096, 'x')},
        {"quoted_4k", "'" + std::string(4096, 'x') + "'", std::string(4096, 'x'), std::string(4096, 'x')},
        {"sparse_escapes_4k", "'" + std::string(2048, 'a') + R"(\")" + std::string(2048, 'b') + "'",
            std::string(2048, 'a') + '"' + std::string(2048, 'b'), std::string(2048, 'a') + '"' + std::string(2048, 'b')},
        {"dense_escapes", "'" + utils::repeat(R"(a\\b\"c\/)", 256) + "'",
            utils::repeat(R"(a\b"c/)", 256), utils::repeat(R"(a\b"c/)", 256)},
        {"literal_utf8", "'é🐟'", "é🐟", "é🐟"},
        {"literal_null", std::string("'a\0b'", 5), std::string("a\0b", 3), std::string("a\0b", 3)},
        {"json_controls", R"("\n\t\r\b\f")", "ntrbf", "\n\t\r\b\f"},
        {"dense_newlines", "'" + utils::repeat(R"(\n\t)", 256) + "'",
            utils::repeat("nt", 256), utils::repeat("\n\t", 256)},
        {"unicode_escapes", "'" + utils::repeat(R"(\u00e9)", 256) + "'",
            utils::repeat("u00e9", 256), utils::repeat("é", 256)},
        {"surrogate_pair", R"("\uD83D\uDE00")", "uD83DuDE00", "😀"},
        {"escaped_null", R"("\u0000x")", "u0000x", std::string("\0x", 2)}
    };
    struct Decoder {
        std::string_view name;
        Decode previous;
        Decode current;
        size_t limit;
    };
    const Decoder decoders[]{
        {"fast_fromstr", previous_full, current_full, std::numeric_limits<size_t>::max()},
        {"fast_fromstr_N_1", benchmark_previous::utils::fast_fromstr_N, utils::fast_fromstr_N, 1},
        {"fast_fromstr_N_16", benchmark_previous::utils::fast_fromstr_N, utils::fast_fromstr_N, 16}
    };
    struct Job {
        const Case* input;
        const Decoder* decoder;
        bool reuse;
        size_t iterations;
    };
    std::vector<Job> jobs;
    constexpr size_t max_iterations = 2'000'000;
    constexpr double target_ns = 5'000'000;

    for(const auto& input : cases) {
        for(const auto& decoder : decoders) {
            // Each version is checked against its own contract before any timing.
            for(bool current : {false, true}) {
                std::string output = "prefix:";
                const auto decode = current ? decoder.current : decoder.previous;
                decode(output, input.input, decoder.limit);
                const auto& expected = current ? input.current : input.previous;
                if(output != "prefix:" + expected.substr(0, decoder.limit))
                    throw std::runtime_error("Unexpected decoder output: " + input.name + " / "
                        + std::string(decoder.name) + (current ? " / current" : " / previous"));
            }
            for(bool reuse : {false, true}) {
                // A shared count gives the faster version about 5 ms per sample.
                size_t iterations = 1024;
                double elapsed;
                do {
                    const auto previous = measure(decoder.previous, input.input, decoder.limit, reuse, iterations);
                    const auto current = measure(decoder.current, input.input, decoder.limit, reuse, iterations);
                    elapsed = std::min(previous.elapsed_ns, current.elapsed_ns);
                    if(elapsed >= 1'000'000 || iterations == max_iterations)
                        break;
                    iterations = std::min(iterations * 4, max_iterations);
                } while(true);
                if(elapsed <= 0)
                    throw std::runtime_error("Could not calibrate string decoder timing");
                iterations = static_cast<size_t>(std::clamp(iterations * target_ns / elapsed, 128.0, double(max_iterations)));
                jobs.push_back({&input, &decoder, reuse, iterations});
            }
        }
    }

    std::vector<size_t> order;
    for(size_t i = 0; i < jobs.size(); ++i)
        order.push_back(i);
    std::mt19937 rng(1337);
    std::cout << "version,parser,group,case,ns_per_parse,iterations,checksum,prepared_parts,round,input_bytes,output_bytes,buffer\n";
    for(size_t round = 0; round < 10; ++round) {
        std::shuffle(order.begin(), order.end(), rng);
        for(const auto index : order) {
            const auto& job = jobs[index];
            const auto& input = *job.input;
            const auto& decoder = *job.decoder;
            // Each pair runs in both orders equally often, within the same process.
            for(size_t position = 0; position < 2; ++position) {
                const bool current = (round + index + position) % 2 != 0;
                const auto sample = measure(current ? decoder.current : decoder.previous,
                    input.input, decoder.limit, job.reuse, job.iterations);
                const auto& expected = current ? input.current : input.previous;
                std::cout << (current ? "current" : "previous") << ',' << decoder.name << ','
                    << (input.current == input.previous ? "equivalent" : "policy_change") << ','
                    << input.name << (job.reuse ? " / reuse" : " / fresh") << ','
                    << std::fixed << std::setprecision(4) << sample.elapsed_ns / job.iterations << ','
                    << job.iterations << ',' << sample.checksum << ",0," << round << ','
                    << input.input.size() << ',' << std::min(expected.size(), decoder.limit) << ','
                    << (job.reuse ? "reuse" : "fresh") << '\n';
            }
        }
    }
}

}
