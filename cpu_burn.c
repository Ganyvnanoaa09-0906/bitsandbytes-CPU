// cpu_burn.c -- hold the CPU busy at a chosen duty cycle, so "power draw" and "being awake" can be
// told apart.
//
// 7.62 concluded that the iGPU drops 5.1x the moment any CPU thread works, and called it power
// arbitration: the SMU feeds the CPU and starves the GPU inside a shared 15 W budget. That conclusion
// was inferred from the SHAPE of the curve -- a cliff rather than a slope -- and a cliff is consistent
// with at least one other mechanism.
//
// The alternative is package state. If a core being awake at all keeps the package out of the state in
// which the iGPU can boost, then the drop has nothing to do with how much power the CPU draws, and the
// fix is not to negotiate for watts but simply to keep the CPU's consumption negligible.
//
// These differ in consequence, so they are worth separating. A power limit is hard; a package-state
// effect might be worked around entirely in software, with no hardware risk and no need to argue with
// the EC.
//
// This program burns a configurable duty cycle: busy for ON microseconds, asleep for OFF. At ON=1000,
// OFF=19000 it is active one twentieth of the time, roughly 2-4% of the package power of a full core,
// while still keeping a core resident and the scheduler from idling the package. If the iGPU drops by
// the same factor as with a fully spinning core, the mechanism is being awake rather than drawing.
//
// Usage: cpu_burn.exe <threads> <on_us> <off_us> <seconds>

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <immintrin.h>

static void burn(double on_us, double off_us, std::atomic<int>* stop) {
    volatile double sink = 0.0;
    double acc = 0.0;
    while (stop->load() == 0) {
        auto t0 = std::chrono::steady_clock::now();
        // busy phase: real FP work, not a spin on a counter, so it draws what a core actually draws
        while (true) {
            for (int k = 0; k < 64; ++k) acc = acc * 1.0000001 + 1e-9;
            auto now = std::chrono::steady_clock::now();
            double us = std::chrono::duration<double, std::micro>(now - t0).count();
            if (us >= on_us) break;
        }
        sink = acc;
        if (off_us > 0.0) {
            auto t1 = std::chrono::steady_clock::now();
            while (true) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                auto now = std::chrono::steady_clock::now();
                double us = std::chrono::duration<double, std::micro>(now - t1).count();
                if (us >= off_us) break;
            }
        }
    }
    if (sink == -1e30) printf("");
}

int main(int argc, char** argv) {
    int nthreads = argc > 1 ? atoi(argv[1]) : 1;
    double on_us = argc > 2 ? atof(argv[2]) : 1000.0;
    double off_us = argc > 3 ? atof(argv[3]) : 0.0;
    double secs = argc > 4 ? atof(argv[4]) : 10.0;

    std::atomic<int> stop{0};
    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; ++t) th.emplace_back(burn, on_us, off_us, &stop);
    std::this_thread::sleep_for(std::chrono::duration<double>(secs));
    stop.store(1);
    for (auto& x : th) x.join();
    return 0;
}
