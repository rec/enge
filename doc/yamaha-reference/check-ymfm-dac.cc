#include <cstdint>
#include <cstdio>
#include "ymfm.h"

// Exhaustively compare the public output helper with encode/decode, then
// demonstrate overflow and summing order. This is not a hardware measurement.
int main() {
    unsigned mismatches = 0;
    for (int value = -32768; value <= 32767; ++value)
        mismatches += ymfm::roundtrip_fp(value) != ymfm::decode_fp(ymfm::encode_fp(value));
    if (std::printf("in_range_samples,in_range_mismatches\n65536,%u\n", mismatches) < 0) return 2;
    if (mismatches) return 1;
    if (std::printf("input,roundtrip,encode_decode\n") < 0) return 2;
    const int32_t probes[] = {-65536, -32769, -32768, -32767, -513, -512, -511,
                             -1, 0, 1, 511, 512, 513, 32703, 32704, 32767, 32768, 65536};
    for (int value : probes)
        if (std::printf("%d,%d,%d\n", value, ymfm::roundtrip_fp(value),
                        ymfm::decode_fp(ymfm::encode_fp(value))) < 0) return 2;
    if (std::printf("voice_a,voice_b,after_sum,sum_after_conversion\n") < 0) return 2;
    const int pairs[][2] = {{513, 513}, {32767, 1}, {20000, 20000}, {1023, -511}};
    for (const auto &pair : pairs)
        if (std::printf("%d,%d,%d,%d\n", pair[0], pair[1],
                        ymfm::roundtrip_fp(pair[0] + pair[1]),
                        ymfm::roundtrip_fp(pair[0]) + ymfm::roundtrip_fp(pair[1])) < 0) return 2;
    return std::fflush(stdout) != 0;
}
