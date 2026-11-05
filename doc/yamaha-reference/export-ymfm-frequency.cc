#include <algorithm>
#include <array>
#include <cstdio>
#include "ymfm_opz.h"
#include "ymfm_fm.ipp"

static bool write_step(FILE *file, uint32_t step) {
    for (int byte = 0; byte < 4; ++byte)
        if (std::fputc((step >> (8 * byte)) & 255, file) == EOF)
            return false;
    return true;
}

// Export little-endian native steps for static controls and multiplier-one PM.
int main(int argc, char **argv) {
    if (argc != 3)
        return 1;
    FILE *base = std::fopen(argv[1], "wb"), *pm = std::fopen(argv[2], "wb");
    if (!base || !pm)
        return 2;
    constexpr int detunes[] = {7, 6, 5, 0, 1, 2, 3};
    constexpr double displays[] = {1.0, 1.41, 1.57, 1.73};
    struct Coarse { double display; int multiple; int detune; };
    std::array<Coarse, 64> controls{};
    for (int multiple = 0; multiple < 16; ++multiple)
        for (int detune = 0; detune < 4; ++detune)
            controls[multiple * 4 + detune] = {
                (multiple ? multiple : 0.5) * displays[detune], multiple, detune};
    std::sort(controls.begin(), controls.end(), [](Coarse a, Coarse b) {
        return a.display < b.display;
    });
    for (int note = 13; note <= 108; ++note)
    for (int coarse = 0; coarse < 64; ++coarse)
    for (int fine = 0; fine < 16; ++fine)
    for (int detune = 0; detune < 7; ++detune) {
        const auto control = controls[coarse];
        ymfm::opz_registers regs;
        regs.reset();
        uint32_t channel = 0, mask = 0;
        const int semitone = (note - 13) % 12;
        regs.write(0x28, ((note - 13) / 12) * 16 + semitone + semitone / 3, channel, mask);
        regs.write(0x40, (detunes[detune] << 4) | control.multiple, channel, mask);
        regs.write(0x40, 0x80 | fine, channel, mask);
        regs.write(0xc0, control.detune << 6, channel, mask);
        ymfm::opdata_cache cache{};
        regs.cache_operator_data(0, 0, cache);
        const auto step = coarse < 4 && fine > 7 ? 0 : cache.phase_step;
        if (!write_step(base, step))
            return 3;
        if (coarse == 4 && fine == 0)
            for (int delta = -512; delta < 512; ++delta)
                if (!write_step(pm, ymfm::opm_key_code_to_phase_step(
                    cache.block_freq, delta) + cache.detune))
                    return 3;
    }
    const int base_result = std::fclose(base), pm_result = std::fclose(pm);
    return base_result || pm_result ? 4 : 0;
}
