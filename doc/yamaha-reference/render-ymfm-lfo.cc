#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "ymfm_opz.h"

int main(int argc, char **argv) {
    const bool ticks = argc > 1 && std::strcmp(argv[1], "--ticks") == 0;
    if (ticks) { --argc; ++argv; }
    if (argc != 6 && argc != 8) return 2;
    ymfm::opz_registers regs;
    regs.reset();
    uint32_t channel = 0, mask = 0;
    regs.write(0x18, std::atoi(argv[1]), channel, mask);
    regs.write(0x19, 0x80 | std::atoi(argv[2]), channel, mask);
    regs.write(0x19, std::atoi(argv[3]), channel, mask);
    regs.write(0x1b, std::atoi(argv[4]) | (argc == 8 ? std::atoi(argv[7]) << 2 : 0), channel, mask);
    regs.write(0x38, 1, channel, mask);
    if (argc == 8) {
        regs.write(0x16, std::atoi(argv[6]), channel, mask);
        regs.write(0x17, 0xff, channel, mask);
        regs.write(0x17, 0x7f, channel, mask);
        regs.write(0x39, 0x81, channel, mask);
    }
    FILE *output = std::fopen(argv[5], "wb");
    if (!output) return 3;
    uint32_t remainder = 0;
    int32_t pm = 0;
    const auto write_row = [&]() {
        int32_t row[4] = {int8_t(pm), int32_t(regs.lfo_am_offset(0)),
            int8_t(pm >> 8), int32_t(regs.lfo_am_offset(1))};
        size_t count = argc == 8 ? 4 : 2;
        return std::fwrite(row, sizeof(int32_t), count, output) == count;
    };
    for (int frame = 0; frame < 48000; ++frame) {
        remainder += 3579545;
        while (remainder >= 64 * 48000) {
            remainder -= 64 * 48000;
            pm = regs.clock_noise_and_lfo();
            if (ticks && !write_row()) return 4;
        }
        if (!ticks && !write_row()) return 4;
    }
    return std::fclose(output) != 0;
}
