#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include "EngineMkI.cpp" // Reuse the pinned integer kernel, not a reimplementation.

// Arguments: exported uFor graph file, output directory (already exists).
// CSV goes to stdout; three graph renderings are retained as raw signed Q24.
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    FILE *input = std::fopen(argv[1], "r");
    if (!input) return 3;
    Sin::init();
    Exp2::init();
    EngineMkI engine;
    std::puts("algorithm,feedback,profile,mismatched_samples,max_absolute_q24_error,relative_rms_error");
    for (int algorithm = 1; algorithm <= 32; ++algorithm) {
        int number, source, destination, count;
        if (std::fscanf(input, "%d%d%d%d", &number, &source, &destination, &count) != 4 || number != algorithm || source < 1 || source > 6 || destination < 1 || destination > 6 || count < 1 || count > 6) return 4;
        bool carriers[6] = {}, edges[6][6] = {};
        for (int i = 0; i < count; ++i) {
            int op;
            if (std::fscanf(input, "%d", &op) != 1 || op < 1 || op > 6) return 4;
            carriers[op - 1] = true;
        }
        if (std::fscanf(input, "%d", &count) != 1 || count < 0 || count > 15) return 4;
        for (int i = 0; i < count; ++i) {
            int from, to;
            if (std::fscanf(input, "%d%d", &from, &to) != 2 || from < 1 || from > 6 || to < 1 || to >= from) return 4;
            edges[from - 1][to - 1] = true;
        }
        for (int feedback : {0, 1, 4, 7}) for (int profile = 0; profile < 4; ++profile) {
            FmOpParams params[6] = {};
            int32_t phases[6], increments[6];
            uint16_t attenuations[6];
            for (int op = 0; op < 6; ++op) {
                const int level = profile == 3 ? 9 + op : std::array<int, 3>{9, 12, 14}[profile];
                phases[op] = profile == 3 ? op * 123456 : 0;
                increments[op] = 153791 * (profile == 3 ? op + 1 : 1);
                attenuations[op] = (16 - level) * 1024;
                auto &param = params[5 - op]; // Dexed OP6 first; uFor OP1 first.
                param.level_in = level * (1 << 24);
                param.gain_out = attenuations[op];
                param.freq = increments[op];
                param.phase = phases[op];
            }
            int32_t reference_history[2] = {}, graph_history[2] = {};
            const int shift = 9 - feedback + ((algorithm == 4 || algorithm == 6 || algorithm == 32) ? 2 : 0);
            int mismatches = 0;
            int64_t maximum = 0;
            double difference_squared = 0, reference_squared = 0;
            FILE *audio = nullptr;
            if (feedback == 7 && profile == 2 && (algorithm == 4 || algorithm == 6 || algorithm == 32)) {
                const std::string path = std::string(argv[2]) + "/graph-" + std::to_string(algorithm) + ".i32";
                audio = std::fopen(path.c_str(), "wb");
                if (!audio) return 5;
            }
            for (int block = 0; block < 750; ++block) {
                int32_t reference[N] = {};
                engine.render(reference, params, algorithm - 1, reference_history, feedback ? 8 - feedback : 16);
                for (int frame = 0; frame < N; ++frame) {
                    int32_t values[6] = {}, mixed = 0;
                    for (int op = 5; op >= 0; --op) {
                        int32_t modulation = 0;
                        for (int from = 5; from > op; --from) if (edges[from][op]) modulation += values[from];
                        if (feedback && op == destination - 1) modulation += (graph_history[0] + graph_history[1]) >> shift;
                        values[op] = mkiSin(phases[op] + modulation, attenuations[op]);
                        if (carriers[op]) mixed += values[op];
                        phases[op] += increments[op];
                    }
                    graph_history[0] = graph_history[1];
                    graph_history[1] = values[source - 1];
                    const int64_t error = int64_t(mixed) - reference[frame];
                    mismatches += error != 0;
                    maximum = std::max(maximum, std::abs(error));
                    difference_squared += double(error) * error;
                    reference_squared += double(reference[frame]) * reference[frame];
                    if (audio && std::fwrite(&mixed, sizeof(mixed), 1, audio) != 1) return 6;
                }
            }
            if (audio && std::fclose(audio) != 0) return 6;
            std::printf("%d,%d,%d,%d,%lld,%.17g\n", algorithm, feedback, profile, mismatches, static_cast<long long>(maximum), std::sqrt(difference_squared / reference_squared));
        }
    }
    return std::fclose(input) != 0 || std::ferror(stdout);
}
