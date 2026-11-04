#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "ymfm_opz.h"
#include "ymfm_fm.ipp"

// Arguments: algorithm (1..8), feedback (0..7), profile (0..1), output.i32, graphs.txt.
// Fixed Q20 phase steps and TL isolate the operator graph from patch decoding.
int main(int argc, char **argv) {
    if (argc != 6) return 2;
    const int algorithm = std::atoi(argv[1]);
    const int feedback = std::atoi(argv[2]);
    const int profile = std::atoi(argv[3]);
    if (algorithm < 1 || algorithm > 8 || feedback < 0 || feedback > 7 || profile < 0 || profile > 1) return 2;
    unsigned graphs[8][5];
    FILE *graph_file = std::fopen(argv[5], "r");
    if (!graph_file) return 3;
    for (auto &graph : graphs)
        for (auto &mask : graph)
            if (std::fscanf(graph_file, "%u", &mask) != 1 || mask > 15) return 2;
    if (std::fclose(graph_file)) return 3;
    ymfm::ymfm_interface interface;
    ymfm::fm_engine_base<ymfm::opz_registers> engine(interface);
    engine.reset();
    auto &regs = engine.regs();
    uint32_t changed = 0, mask = 0;
    regs.write(0x20, 0xc0 | (feedback << 3) | (algorithm - 1), changed, mask);
    const std::array<uint32_t, 4> offsets = {0, 16, 8, 24};
    const std::array<uint32_t, 4> waves = {0, 1, 4, 7};
    std::array<ymfm::fm_operator<ymfm::opz_registers>, 4> ops = {
        ymfm::fm_operator<ymfm::opz_registers>(engine, offsets[0]),
        ymfm::fm_operator<ymfm::opz_registers>(engine, offsets[1]),
        ymfm::fm_operator<ymfm::opz_registers>(engine, offsets[2]),
        ymfm::fm_operator<ymfm::opz_registers>(engine, offsets[3])
    };
    ymfm::fm_channel<ymfm::opz_registers> channel(engine, 0);
    channel.reset();
    for (unsigned i = 0; i < 4; ++i) {
        regs.write(0x80 + offsets[i], 31, changed, mask);
        regs.write(0x40 + offsets[i], 0x80 | ((profile ? waves[i] : 0) << 4), changed, mask);
        channel.assign(i, &ops[i]);
        ops[i].keyonoff(1, ymfm::KEYON_NORMAL);
        ops[i].prepare();
        if (ops[i].debug_eg_attenuation() != 0) return 5;
        auto &cache = ops[i].debug_cache();
        cache.phase_step = 9600 + (profile ? i * 4800 : 0);
        cache.total_level = profile ? i * 16 * 8 : 0;
        for (auto &rate : cache.eg_rate) rate = 0;
    }
    FILE *output = std::fopen(argv[4], "wb");
    if (!output) return 3;
    int32_t history[2] = {};
    unsigned mismatches = 0;
    for (int frame = 0; frame < 48000; ++frame) {
        ymfm::fm_engine_base<ymfm::opz_registers>::output_data sample{};
        // No DAC clipping: isolate the raw carrier sum.
        channel.output_4op(sample, 0, 32767);
        const int32_t value = sample.data[0];
        // Execute exported enge edges with the exact same integer operator
        // kernel and phase state, but an independent two-sample feedback path.
        int32_t values[4] = {};
        const auto &graph = graphs[algorithm - 1];
        for (int op = 3; op >= 0; --op) {
            int32_t modulation = 0;
            for (int source = 0; source < 4; ++source)
                if (graph[op] & (1U << source)) modulation += values[source];
            modulation >>= 1;
            if (op == 3 && feedback)
                modulation += (history[0] + history[1]) >> (10 - feedback);
            values[op] = ops[3 - op].compute_volume(ops[3 - op].phase() + modulation, 0);
        }
        int32_t graph_value = 0;
        for (int op = 0; op < 4; ++op)
            if (graph[4] & (1U << op)) graph_value += values[op];
        history[0] = history[1];
        history[1] = values[3];
        mismatches += graph_value != value;
        if (std::fwrite(&value, sizeof(value), 1, output) != 1) return 4;
        channel.clock(0, 0);
    }
    if (std::fclose(output)) return 4;
    std::fprintf(stderr, "%d,%d,%d,48000,%u\n", algorithm, feedback, profile, mismatches);
    return mismatches ? 6 : 0;
}
