#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "ymfm_opz.h"
#include "ymfm_fm.ipp"

// Arguments: algorithm (1..8), feedback (0..7), profile (0..1), output.i32.
// Output-frame phase steps, native-rate envelopes, release at frame 24000.
int main(int argc, char **argv) {
    if (argc != 5) return 2;
    const int algorithm = std::atoi(argv[1]);
    const int feedback = std::atoi(argv[2]);
    const int profile = std::atoi(argv[3]);
    if (algorithm < 1 || algorithm > 8 || feedback < 0 || feedback > 7 || profile < 0 || profile > 1) return 2;
    ymfm::ymfm_interface interface;
    ymfm::fm_engine_base<ymfm::opz_registers> engine(interface);
    engine.reset();
    auto &regs = engine.regs();
    uint32_t changed = 0, mask = 0;
    regs.write(0x20, 0xc0 | (feedback << 3) | (algorithm - 1), changed, mask);
    regs.write(0x28, 18 << 2, changed, mask);
    const std::array<uint32_t, 4> offsets = {0, 16, 8, 24};
    const std::array<uint32_t, 4> waves = {0, 1, 4, 7};
    const int carriers[8] = {1, 1, 1, 1, 5, 7, 7, 15};
    const int headroom[8] = {0, 0, 0, 0, 8, 13, 13, 16};
    std::array<ymfm::fm_operator<ymfm::opz_registers>, 4> ops = {
        ymfm::fm_operator<ymfm::opz_registers>(engine, offsets[0]),
        ymfm::fm_operator<ymfm::opz_registers>(engine, offsets[1]),
        ymfm::fm_operator<ymfm::opz_registers>(engine, offsets[2]),
        ymfm::fm_operator<ymfm::opz_registers>(engine, offsets[3])
    };
    ymfm::fm_channel<ymfm::opz_registers> channel(engine, 0);
    channel.reset();
    for (unsigned i = 0; i < 4; ++i) {
        const auto offset = offsets[i];
        regs.write(0x80 + offset, profile ? 20 : 31, changed, mask);
        regs.write(0xa0 + offset, profile ? 16 : 12, changed, mask);
        regs.write(0xc0 + offset, profile ? 12 : 4, changed, mask);
        regs.write(0xe0 + offset, profile ? 0xff : 0x89, changed, mask);
        // Leave both endpoint operators unshifted: ymfm's register-slot
        // exclusion differs from the manual's OP1 restriction.
        regs.write(0xc0 + offset, 0x20 | ((profile && i != 0 ? 3 - i : 0) << 6), changed, mask);
        regs.write(0x40 + offset, 0x80 | ((profile ? waves[i] : 0) << 4), changed, mask);
        const int tl = (carriers[algorithm - 1] & (1 << (3 - i))) ? headroom[algorithm - 1] : 0;
        regs.write(0x60 + offset, tl, changed, mask);
        channel.assign(i, &ops[i]);
        ops[i].keyonoff(1, ymfm::KEYON_NORMAL);
        ops[i].prepare();
        ops[i].debug_cache().phase_step = 9600;
    }
    FILE *output = std::fopen(argv[4], "wb");
    if (!output) return 3;
    uint32_t remainder = 0, counter = 0;
    for (int frame = 0; frame < 48000; ++frame) {
        if (frame == 24000) {
            for (auto &op : ops) {
                op.keyonoff(0, ymfm::KEYON_NORMAL);
                op.prepare();
                op.debug_cache().phase_step = 9600;
            }
        }
        ymfm::fm_engine_base<ymfm::opz_registers>::output_data sample{};
        channel.output_4op(sample, 0, 32767);
        if (std::fwrite(&sample.data[0], sizeof(int32_t), 1, output) != 1) return 4;
        channel.clock(1, 0); // phase and feedback only
        remainder += 3579545;
        while (remainder >= 192 * 48000) {
            remainder -= 192 * 48000;
            ++counter;
            for (auto &op : ops) {
                op.debug_cache().phase_step = 0;
                op.clock(counter << 2, 0);
                op.debug_cache().phase_step = 9600;
            }
        }
    }
    return std::fclose(output) != 0;
}
