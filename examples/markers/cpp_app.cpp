// A C++ "inference loop" instrumented with EdgeLens markers — the pattern
// for an inference runtime, an EDDL test program, or any C++ node.
//
//   g++ -O2 -std=c++11 -I"$(edgelens include)" examples/markers/cpp_app.cpp -o cpp_app
//   edgelens record --deadline-ms 50 --period-ms 20 --save cpp.json -- ./cpp_app 500
//   edgelens validate cpp.json --p99-ms 50 --max-miss-ratio 0
//
// With CUDA/TensorRT, end inference spans after synchronising the stream:
//   #define EDGELENS_WITH_CUDA
//   #include "edgelens_mark.h"
//   { EL_SPAN_CUDA("lstm", stream); context->enqueueV3(stream); }
#include "edgelens_mark.h"

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <thread>

// Stop cleanly on Ctrl-C / `edgelens record --duration`: the handler only
// sets a flag; leaving the loop and returning from main lets edgelens_mark.h
// flush pending spans and send its final message (so no loss goes unseen).
static volatile std::sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

static void busy_us(long us) {                       // stand-in for real work
    uint64_t end = el_now_ns() + (uint64_t)us * 1000ull;
    while (el_now_ns() < end) {}
}

int main(int argc, char **argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 300;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    for (int i = 0; i < n && !g_stop; i++) {
        EL_ITERATION();                               // end-to-end latency of one cycle
        { EL_SPAN("build_window"); busy_us(300); }
        { EL_SPAN("lstm"); busy_us(i % 50 == 0 ? 6000 : 2500); }   // occasional slow one
        { EL_SPAN("decide"); busy_us(50); }
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    return 0;                                         // pending spans + final seq flushed at exit
}
