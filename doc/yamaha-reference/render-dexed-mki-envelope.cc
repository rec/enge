#include <cstdio>
#include <cstdlib>
#include "EngineMkI.h"
#include "msfa/env.h"
#include "msfa/sin.h"
#include "msfa/exp2.h"

// Arguments: algorithm, feedback, profile (0=fast, 1=equal-level hold), output.i32.
// Fixed frequency isolates the upstream amplitude envelopes and core together.
int main(int argc, char **argv) {
    if (argc != 5) return 2;
    const int algorithm = std::atoi(argv[1]);
    const int feedback = std::atoi(argv[2]);
    const int profile = std::atoi(argv[3]);
    if (algorithm < 1 || algorithm > 32 || feedback < 0 || feedback > 7 || profile < 0 || profile > 1) return 2;
    FILE *output = std::fopen(argv[4], "wb");
    if (!output) return 3;
    Sin::init();
    Exp2::init();
    Env::init_sr(48000);
    EngineMkI core;
    Env envelopes[6];
    FmOpParams params[6] = {};
    const int rates[4] = {99, profile ? 50 : 99, 99, 99};
    const int levels[4] = {99, 99, profile ? 50 : 99, 0};
    for (int i = 0; i < 6; ++i) {
        envelopes[i].init(rates, levels, 4064, 0);
        params[i].freq = 153791;
    }
    int32_t history[2] = {};
    for (int block = 0; block < 750; ++block) {
        for (int i = 0; i < 6; ++i) {
            if (block == 375) envelopes[i].keydown(false);
            params[i].level_in = envelopes[i].getsample();
        }
        int32_t samples[N] = {};
        core.render(samples, params, algorithm - 1, history, feedback ? 8 - feedback : 16);
        if (std::fwrite(samples, sizeof(int32_t), N, output) != N) return 4;
    }
    return std::fclose(output) != 0;
}
