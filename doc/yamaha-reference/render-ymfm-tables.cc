#include <cstdio>
#include "ymfm_opz.h"
#include "ymfm_fm.ipp"
int main() {
    ymfm::opz_registers regs;
    regs.reset();
    uint32_t channel = 0, mask = 0;
    for (unsigned wave = 0; wave < 8; ++wave) {
        regs.write(0x40, 0x80 | (wave << 4), channel, mask);
        ymfm::opdata_cache cache{};
        regs.cache_operator_data(0, 0, cache);
        if (std::fwrite(cache.waveform, sizeof(uint16_t), 1024, stdout) != 1024) return 1;
    }
    for (unsigned i = 0; i < 256; ++i) {
        uint16_t value = ymfm::attenuation_to_volume(i);
        if (std::fwrite(&value, sizeof(value), 1, stdout) != 1) return 1;
    }
    return std::fflush(stdout) != 0;
}
