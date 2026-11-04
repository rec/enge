#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "ymfm_opz.h"

int main(int argc, char **argv) {
    if (argc != 6) return 2;
    ymfm::opz_registers regs;
    regs.reset();
    uint32_t channel = 0, mask = 0;
    regs.write(0x18, std::atoi(argv[1]), channel, mask);
    regs.write(0x19, 0x80 | std::atoi(argv[2]), channel, mask);
    regs.write(0x19, std::atoi(argv[3]), channel, mask);
    regs.write(0x1b, std::atoi(argv[4]), channel, mask);
    regs.write(0x38, 1, channel, mask);
    FILE *output = std::fopen(argv[5], "wb");
    if (!output) return 3;
    uint32_t remainder = 0;
    int32_t pm = 0;
    for (int frame = 0; frame < 48000; ++frame) {
        remainder += 3579545;
        while (remainder >= 64 * 48000) {
            remainder -= 64 * 48000;
            pm = int8_t(regs.clock_noise_and_lfo());
        }
        int32_t row[2] = {pm, int32_t(regs.lfo_am_offset(0))};
        if (std::fwrite(row, sizeof(int32_t), 2, output) != 2) return 4;
    }
    return std::fclose(output) != 0;
}
