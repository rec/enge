#include <cstdio>
#include "synth.h"
#include "controllers.h"
#include "dx7note.h"
#include "sin.h"
#include "exp2.h"
#include "freqlut.h"

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char patch[156] = {};
    FILE *input = std::fopen(argv[1], "rb");
    if (!input || std::fread(patch, 1, 155, input) != 155) return 3;
    std::fclose(input);
    FILE *output = std::fopen(argv[2], "wb");
    if (!output) return 4;
    Sin::init();
    Exp2::init();
    Freqlut::init(48000);
    PitchEnv::init(48000);
    Controllers controllers = {};
    controllers.values_[kControllerPitch] = 8192;
    Dx7Note note = {};
    note.init(patch, 69, 100);
    for (int block = 0; block < 750; ++block) {
        if (block == 375) note.keyup();
        int32_t samples[N] = {};
        note.compute(samples, 1 << 23, 0, &controllers);
        if (std::fwrite(samples, sizeof(int32_t), N, output) != N) return 5;
    }
    return std::fclose(output) != 0;
}
