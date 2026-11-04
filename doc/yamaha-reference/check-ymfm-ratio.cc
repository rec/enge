#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include "ymfm_opz.h"

// Sweep the VCED coarse ordering used by enge against unmodified OPZ registers.
int main() {
    struct Coarse { double display; int multiple; int detune; };
    constexpr std::array<double, 4> displays{1.0, 1.41, 1.57, 1.73};
    constexpr std::array<int, 4> cents{0, 600, 781, 950};
    constexpr std::array<int, 4> deltas{0, 384, 500, 608};
    constexpr double clock = 3579545.0 / 64;
    std::array<Coarse, 64> controls{};
    for (int multiple = 0; multiple < 16; ++multiple)
        for (int detune = 0; detune < 4; ++detune)
            controls[multiple * 4 + detune] = {
                (multiple ? multiple : 0.5) * displays[detune], multiple, detune};
    std::sort(controls.begin(), controls.end(), [](Coarse a, Coarse b) {
        return a.display < b.display;
    });
    std::puts("coarse,multiple,detune_cents,cases,clamped_cases,min_cents,max_cents,min_unclamped_cents,max_unclamped_cents,worst_note,worst_fine,top_nominal_hz,top_reference_hz");
    for (int coarse = 0; coarse < 64; ++coarse) {
        const auto control = controls[coarse];
        int cases = 0, clamped = 0, worst_note = 0, worst_fine = 0;
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -minimum, worst = -1, top_nominal = 0, top_reference = 0;
        double min_unclamped = minimum, max_unclamped = maximum;
        for (int fine = 0; fine < (control.multiple ? 16 : 8); ++fine)
        for (int note = 13; note <= 108; ++note) {
            ymfm::opz_registers regs;
            regs.reset();
            uint32_t channel = 0, mask = 0;
            const int semitone = (note - 13) % 12;
            const int block = (note - 13) / 12;
            regs.write(0x28, block * 16 + semitone + semitone / 3, channel, mask);
            regs.write(0x40, control.multiple, channel, mask);
            regs.write(0x40, 0x80 | fine, channel, mask);
            regs.write(0xc0, control.detune << 6, channel, mask);
            ymfm::opdata_cache cache{};
            regs.cache_operator_data(0, 0, cache);
            const auto step = regs.compute_phase_step(0, 0, cache, 0);
            if (step != cache.phase_step)
                return 2;
            const double nominal = 440.0 * std::exp2((note - 69) / 12.0)
                * ((control.multiple ? control.multiple : 0.5) + fine / 16.0)
                * std::exp2(cents[control.detune] / 1200.0);
            const double reference = step * clock / (1 << 20);
            const double error = 1200 * std::log2(reference / nominal);
            minimum = std::min(minimum, error);
            maximum = std::max(maximum, error);
            if (std::abs(error) > worst) {
                worst = std::abs(error);
                worst_note = note;
                worst_fine = fine;
            }
            if (block == 7 && semitone * 64 + deltas[control.detune] >= 768)
                ++clamped;
            else {
                min_unclamped = std::min(min_unclamped, error);
                max_unclamped = std::max(max_unclamped, error);
            }
            if (note == 108 && fine == 0) {
                top_nominal = nominal;
                top_reference = reference;
            }
            ++cases;
        }
        std::printf("%d,%g,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%d,%d,%.12f,%.12f\n",
            coarse, control.multiple ? double(control.multiple) : 0.5,
            cents[control.detune], cases, clamped, minimum, maximum,
            min_unclamped, max_unclamped,
            worst_note, worst_fine, top_nominal, top_reference);
    }
    return std::ferror(stdout) ? 1 : 0;
}
