#include <cstdint>
#include <cstdio>
#include "ymfm_opz.h"
#include "ymfm_fm.ipp"

// Write 1024 rows of eight signed 13-bit waveform samples as CSV.
// No envelope, modulation, oscillator clock or DAC processing is applied.
int main() {
    ymfm::opz_registers regs;
    regs.reset();
    ymfm::opdata_cache caches[8]{};
    uint32_t channel = 0, mask = 0;
    for (unsigned wave = 0; wave < 8; ++wave) {
        regs.write(0x40, 0x80 | (wave << 4), channel, mask);
        regs.cache_operator_data(0, 0, caches[wave]);
    }
    for (unsigned phase = 0; phase < 1024; ++phase) {
        for (unsigned wave = 0; wave < 8; ++wave) {
            const uint16_t attenuation = caches[wave].waveform[phase];
            int32_t sample = ymfm::attenuation_to_volume(attenuation & 0x7fff);
            if (attenuation & 0x8000) sample = -sample;
            if (std::printf("%s%d", wave ? "," : "", sample) < 0) return 1;
        }
        if (std::putchar('\n') == EOF) return 1;
    }
    return std::ferror(stdout) ? 1 : 0;
}
