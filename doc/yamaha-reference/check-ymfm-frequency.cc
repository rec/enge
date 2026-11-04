#include <cmath>
#include <cstdint>
#include <cstdio>
#include "ymfm_opz.h"

// Compile with the pinned ymfm_opz.cpp. No register/core implementation is copied.
int main() {
    constexpr uint32_t ticks = 4096;
    constexpr double clock = 3579545.0;
    std::puts("mode,coarse,fine,range,note,ticks,phase_sum,nominal_hz,reference_hz,cents_error");
    for (int mode = 0; mode < 2; ++mode)
    for (int coarse = 0; coarse < (mode ? 16 : 1); ++coarse)
    for (int fine = 0; fine < (mode ? (coarse ? 16 : 8) : 1); ++fine)
    for (int range = 0; range < (mode ? 8 : 1); ++range)
    for (int note = mode ? 69 : 13; note <= (mode ? 69 : 108); ++note) {
        ymfm::opz_registers regs;
        regs.reset();
        uint32_t channel = 0, mask = 0;
        const int semitone = (note - 13) % 12;
        const int keycode = ((note - 13) / 12) * 16 + semitone + semitone / 3;
        regs.write(0x28, keycode, channel, mask);
        regs.write(0x40, mode ? (range << 4) | coarse : 1, channel, mask);
        regs.write(0x40, 0x80 | fine, channel, mask);
        regs.write(0x80, mode ? 0x20 : 0, channel, mask);
        ymfm::opdata_cache cache{};
        regs.cache_operator_data(0, 0, cache);
        uint64_t sum = 0;
        for (uint32_t tick = 0; tick < ticks; ++tick)
            sum += regs.compute_phase_step(0, 0, cache, 0);
        const double nominal = mode ? ((coarse ? coarse * 16 : 8) + fine) * (1 << range)
                                    : 440.0 * std::exp2((note - 69) / 12.0);
        const double actual = double(sum) * clock / (64.0 * (1 << 20) * ticks);
        std::printf("%s,%d,%d,%d,%d,%u,%llu,%.12f,%.12f,%.9f\n",
            mode ? "fixed" : "ratio", mode ? coarse : 1, fine, range, note, ticks,
            static_cast<unsigned long long>(sum), nominal, actual,
            1200.0 * std::log2(actual / nominal));
    }
    return std::ferror(stdout) ? 1 : 0;
}
