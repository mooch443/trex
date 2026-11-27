#include <commons.pc.h>
#include "StringDecoding.h"

// Frozen pre-unification decoder: commons staged metastring.cpp blob
// dae330fcbb6a3e459ccc3c635ad83de216bcdfab. Function bodies are unchanged.
namespace cmn::benchmark_previous::utils {
using namespace cmn::utils;

void fast_fromstr_N(std::string& output, std::string_view sv, size_t max_output) {
    const std::size_t N = sv.size();
    if (N < 2) {                               // too short to be quoted
        output.append(sv.substr(0, min(N, max_output)));
        return;
    }

    const char quote = sv[0];
    const bool quoted = ((quote == '"' || quote == '\'') && sv[N-1] == quote);
    size_t current_size = output.size();

    if (!quoted) {                             // no outer quotes → verbatim
        output.resize(current_size + min(max_output,N) );
        std::memcpy(output.data() + current_size, sv.data(),  min(max_output,N) );
        return;
    }

    // Payload lies strictly between the two quotes
    output.reserve( current_size +  min(max_output,N - 2) );     // worst‑case growth (every '\' drops)

    const size_t maximum_size = current_size + max_output;
    const char* chunk = sv.data() + 1;         // first byte inside the quotes
    const char* end   = sv.data() + N - 1;     // one‑past‑last payload byte

    while (chunk < end
           && current_size < maximum_size)
    {
        // Find the next backslash in the remaining payload
        const size_t available = static_cast<size_t>(end - chunk);
        const size_t budget = maximum_size - current_size;
        const char* remaining = chunk + std::min(available, budget);
        const char* bs = static_cast<const char*>(std::memchr(chunk, '\\', remaining - chunk));
        if (!bs || bs + 1 >= end)
        {
            // No more escape sequences (or stray trailing '\') – copy the rest verbatim
            size_t L = static_cast<size_t>(remaining - chunk);
            output.resize(current_size + L);
            std::memcpy(output.data() + current_size, chunk, L);
            break;
        }

        // Copy slice before the backslash
        size_t L = min((maximum_size - current_size), size_t(bs - chunk));
        output.resize(current_size + L);
        std::memcpy(output.data() + current_size, chunk, L);
        current_size += L;

        // Copy the character after the backslash, effectively un‑escaping it
        output.push_back(*(bs + 1));
        ++current_size;

        // Advance past the escape sequence
        chunk = bs + 2;
    }
}

void fast_fromstr(std::string& output, std::string_view sv) {
    const std::size_t N = sv.size();
    if (N < 2) {                               // too short to be quoted
        output.append(sv);
        return;
    }

    const char quote = sv[0];
    const bool quoted = ((quote == '"' || quote == '\'') && sv[N-1] == quote);
    size_t current_size = output.size();

    if (!quoted) {                             // no outer quotes → verbatim
        output.resize(current_size + N);
        std::memcpy(output.data() + current_size, sv.data(), N);
        return;
    }

    // Payload lies strictly between the two quotes
    output.reserve(current_size + N - 2);     // worst‑case growth (every '\' drops)

    const char* chunk = sv.data() + 1;         // first byte inside the quotes
    const char* end   = sv.data() + N - 1;     // one‑past‑last payload byte

    while (chunk < end) {
        // Find the next backslash in the remaining payload
        const char* bs = static_cast<const char*>(std::memchr(chunk, '\\', end - chunk));
        if (!bs || bs + 1 >= end) {
            // No more escape sequences (or stray trailing '\') – copy the rest verbatim
            size_t L = end - chunk;
            output.resize(current_size + L);
            std::memcpy(output.data() + current_size, chunk, L);
            break;
        }

        // Copy slice before the backslash
        size_t L = bs - chunk;
        output.resize(current_size + L);
        std::memcpy(output.data() + current_size, chunk, L);
        current_size += L;

        // Copy the character after the backslash, effectively un‑escaping it
        output.push_back(*(bs + 1));
        ++current_size;

        // Advance past the escape sequence
        chunk = bs + 2;
    }
}

}
