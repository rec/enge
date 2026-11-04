#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "EngineMkI.cpp"

// Arguments: graph file, algorithm, feedback, OP1..OP6 attenuation rows, output.
// Exactly 48000 frames at 48 kHz, with Q24 phase increment 153791.
// The input trajectory and per-frame silence policy are controlled inputs,
// not the upstream Env or EngineMkI block interpolation/admission behavior.
int main(int argc, char **argv) {
    if (argc != 6) return 2;
    const int algorithm = std::atoi(argv[2]), feedback = std::atoi(argv[3]);
    if (algorithm < 1 || algorithm > 32 || feedback < 0 || feedback > 7) return 2;
    FILE *graphs = std::fopen(argv[1], "r");
    if (!graphs) return 3;
    bool carriers[6] = {}, edges[6][6] = {};
    int source = 0, destination = 0;
    for (int row = 1; row <= algorithm; ++row) {
        int number, count;
        if (std::fscanf(graphs, "%d%d%d%d", &number, &source, &destination, &count) != 4 || number != row || source < 1 || source > 6 || destination < 1 || destination > 6 || count < 1 || count > 6) return 4;
        for (int i = 0; i < count; ++i) {
            int op;
            if (std::fscanf(graphs, "%d", &op) != 1 || op < 1 || op > 6) return 4;
            if (row == algorithm) carriers[op - 1] = true;
        }
        if (std::fscanf(graphs, "%d", &count) != 1 || count < 0 || count > 15) return 4;
        for (int i = 0; i < count; ++i) {
            int from, to;
            if (std::fscanf(graphs, "%d%d", &from, &to) != 2 || from < 1 || from > 6 || to < 1 || to >= from) return 4;
            if (row == algorithm) edges[from - 1][to - 1] = true;
        }
    }
    if (std::fclose(graphs) != 0) return 3;
    FILE *input = std::fopen(argv[4], "rb"), *output = std::fopen(argv[5], "wb");
    if (!input || !output) return 3;
    Sin::init();
    Exp2::init();
    EngineMkI engine; // Initializes the pinned logarithmic lookup tables.
    int32_t phases[6] = {}, history[2] = {};
    const int shift = 9 - feedback + ((algorithm == 4 || algorithm == 6 || algorithm == 32) ? 2 : 0);
    for (int frame = 0; frame < 48000; ++frame) {
        int32_t attenuation[6], values[6] = {}, mixed = 0;
        if (std::fread(attenuation, sizeof(int32_t), 6, input) != 6) return 5;
        for (int op = 5; op >= 0; --op) {
            if (attenuation[op] < 0 || attenuation[op] > 16384) return 5;
            int32_t modulation = 0;
            for (int from = 5; from > op; --from) if (edges[from][op]) modulation += values[from];
            if (feedback && op == destination - 1) modulation += (history[0] + history[1]) >> shift;
            // Enge's existing per-frame admission policy is held fixed here.
            values[op] = attenuation[op] > 16284 ? 0 : mkiSin(phases[op] + modulation, attenuation[op]);
            if (carriers[op]) mixed += values[op];
            phases[op] += 153791;
        }
        history[0] = history[1];
        history[1] = values[source - 1];
        if (std::fwrite(&mixed, sizeof(mixed), 1, output) != 1) return 6;
    }
    if (std::fgetc(input) != EOF || std::ferror(input)) return 5;
    return std::fclose(input) != 0 || std::fclose(output) != 0;
}
