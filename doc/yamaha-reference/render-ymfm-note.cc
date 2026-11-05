#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "ymfm_opz.h"

// One ratio-mode channel. Emit one second at 48 kHz.
int main(int argc, char **argv) {
    if (argc != 5) return 2;
    const int algorithm = std::atoi(argv[1]), feedback = std::atoi(argv[2]), profile = std::atoi(argv[3]);
    if (algorithm < 1 || algorithm > 8 || feedback < 0 || feedback > 7 || profile < 0 || profile > 1) return 2;
    FILE *file = std::fopen(argv[4], "wb");
    if (!file) return 3;
    ymfm::ymfm_interface interface;
    ymfm::ym2414 chip(interface);
    chip.reset();
    const auto write = [&](uint8_t address, uint8_t value) {
        chip.write_address(address);
        chip.write_data(value);
    };
    const std::array<int, 8> carriers = {1, 1, 1, 1, 5, 7, 7, 15};
    const std::array<int, 8> headroom = {0, 0, 0, 0, 8, 13, 13, 16};
    const std::array<int, 4> offsets = {0, 16, 8, 24};
    const std::array<int, 4> waves = {0, 1, 4, 7}, fine = {0, 3, 8, 15};
    const std::array<int, 4> detune = {7, 5, 1, 3}, shifts = {0, 2, 1, 0};
    write(0x08, 0);
    write(0x28, 74); // A4: block 4, register note 10 (eight semitones from C#).
    write(0x30, 3);  // both outputs, key fraction zero
    for (int i = 0; i < 4; ++i) {
        const int offset = offsets[i];
        write(0x40 + offset, 1 | (profile ? detune[i] << 4 : 0));
        write(0x40 + offset, 0x80 | (profile ? (waves[i] << 4) | fine[i] : 0));
        write(0x60 + offset, carriers[algorithm - 1] & (1 << (3-i)) ? headroom[algorithm - 1] : 0);
        write(0x80 + offset, profile ? 20 : 31);
        write(0xa0 + offset, profile ? 16 : 12);
        write(0xc0 + offset, profile ? i << 6 : 4);
        write(0xc0 + offset, 0x20 | (profile ? shifts[i] << 6 : 0));
        write(0xe0 + offset, profile ? 0x5f : 0x89);
    }
    const int control = (feedback << 3) | (algorithm - 1);
    write(0x20, control | 0x40);
    constexpr int64_t clock = 3579545, divisor = 64 * 48000;
    int64_t ticks = 0;
    ymfm::ym2414::output_data sample{};
    for (int frame = 0; frame < 48000; ++frame) {
        const int64_t due = (frame + 1) * clock / divisor;
        while (ticks < due) {
            if (ticks == clock / 128) write(0x20, control);
            chip.generate(&sample);
            ++ticks;
        }
        if (sample.data[0] != sample.data[1]) return 4;
        if (std::fwrite(&sample.data[0], sizeof(int32_t), 1, file) != 1) return 5;
    }
    return std::fclose(file) != 0;
}
