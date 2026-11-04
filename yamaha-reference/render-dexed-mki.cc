#include <cstdio>
#include <cstdlib>
#include "EngineMkI.h"
#include "msfa/sin.h"
#include "msfa/exp2.h"

// Arguments: algorithm (1..32), feedback (0..7), log level (0..14), OP6 phase LSB (0..1), output.i32.
// Fixed Q24 frequency, zero phase and constant equal gains isolate the FM core.
int main(int argc, char **argv) {
    if (argc != 6) return 2;
    const int algorithm = std::atoi(argv[1]);
    const int feedback = std::atoi(argv[2]);
    const int level = std::atoi(argv[3]);
    if (algorithm < 1 || algorithm > 32 || feedback < 0 || feedback > 7 || level < 0 || level > 14) return 2;
    const int perturbation = std::atoi(argv[4]);
    if (perturbation < 0 || perturbation > 1) return 2;
    FILE *output = std::fopen(argv[5], "wb");
    if (!output) return 3;
    Sin::init();
    Exp2::init();
    EngineMkI engine;
    FmOpParams params[6] = {};
    for (auto &param : params) {
        param.level_in = level * (1 << 24);
        param.gain_out = (16 - level) * 1024;
        param.freq = 153791; // nearest Q24 increment to 440 Hz at 48 kHz
    }
    params[0].phase = perturbation; // Dexed operator order begins with OP6.
    int32_t history[2] = {};
    for (int block = 0; block < 750; ++block) {
        int32_t samples[N] = {};
        engine.render(samples, params, algorithm - 1, history, feedback ? 8 - feedback : 16);
        if (std::fwrite(samples, sizeof(int32_t), N, output) != N) return 4;
    }
    return std::fclose(output) != 0;
}
