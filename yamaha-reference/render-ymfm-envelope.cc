#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "ymfm_opz.h"
#include "ymfm_fm.ipp"

// Arguments: AR D1R D2R RR D1L RS keycode reverb output.i32.
// Observe four shifts at 48 kHz; release immediately before frame 24000.
int main(int argc, char **argv) {
    if (argc != 10) return 2;
    ymfm::ymfm_interface interface;
    ymfm::fm_engine_base<ymfm::opz_registers> engine(interface);
    engine.reset();
    auto &regs = engine.regs();
    uint32_t channel = 0, mask = 0;
    const std::array<uint32_t, 4> offsets = {0, 16, 8, 24};
    regs.write(0x28, std::atoi(argv[7]) << 2, channel, mask);
    std::array<ymfm::fm_operator<ymfm::opz_registers>, 4> operators = {
        ymfm::fm_operator<ymfm::opz_registers>(engine, offsets[0]),
        ymfm::fm_operator<ymfm::opz_registers>(engine, offsets[1]),
        ymfm::fm_operator<ymfm::opz_registers>(engine, offsets[2]),
        ymfm::fm_operator<ymfm::opz_registers>(engine, offsets[3])
    };
    for (unsigned i = 0; i < operators.size(); ++i) {
        const auto offset = offsets[i];
        regs.write(0x80 + offset, (std::atoi(argv[6]) << 6) | std::atoi(argv[1]), channel, mask);
        regs.write(0xa0 + offset, std::atoi(argv[2]), channel, mask);
        regs.write(0xc0 + offset, std::atoi(argv[3]), channel, mask);
        regs.write(0xe0 + offset, ((15 - std::atoi(argv[5])) << 4) | std::atoi(argv[4]), channel, mask);
        regs.write(0xc0 + offset, 0x20 | (i << 6) | std::atoi(argv[8]), channel, mask);
        operators[i].keyonoff(1, ymfm::KEYON_NORMAL);
        operators[i].prepare();
    }
    FILE *output = std::fopen(argv[9], "wb");
    if (!output) return 3;
    uint32_t remainder = 0, counter = 0;
    for (int frame = 0; frame < 48000; ++frame) {
        int32_t row[4];
        for (unsigned i = 0; i < operators.size(); ++i) {
            if (frame == 24000) {
                operators[i].keyonoff(0, ymfm::KEYON_NORMAL);
                operators[i].prepare();
            }
            row[i] = operators[i].debug_eg_attenuation() >> operators[i].debug_cache().eg_shift;
        }
        if (std::fwrite(row, sizeof(int32_t), 4, output) != 4) return 4;
        remainder += 3579545;
        while (remainder >= 192 * 48000) {
            remainder -= 192 * 48000;
            ++counter;
            for (auto &op : operators) op.clock(counter << 2, 0);
        }
    }
    return std::fclose(output) != 0;
}
