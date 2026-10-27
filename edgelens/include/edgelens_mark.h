/*
 * edgelens_mark.h — EdgeLens inference markers for C and C++ (header-only).
 *
 * Measure spans inside any C/C++ application (an inference runtime, a ROS 2 node, a
 * TensorRT service) and send them to an EdgeLens collector:
 *
 *     edgelens record --deadline-ms 3000 --save run.json -- ./my_app
 *
 * C++:
 *     #include "edgelens_mark.h"
 *     while (running) {
 *         EL_ITERATION();                          // whole iteration (end-to-end)
 *         { EL_SPAN("build_window"); build(); }
 *         { EL_SPAN_GPU("lstm", sync_fn, ctx);     // GPU work: sync before end stamp
 *           infer(); }
 *     }
 *
 * C:
 *     el_span_t s = el_begin("lstm");
 *     infer();
 *     el_end(s);                                   // or el_end_synced(s, sync, ctx)
 *
 * Contract (see docs/markers.md):
 *   - Timestamps: clock_gettime(CLOCK_MONOTONIC), the clock EdgeLens uses
 *     for everything, so spans line up with board power and temperature.
 *   - One UNIX datagram per finished span, sent with MSG_DONTWAIT: the
 *     application never waits for the collector (a short internal lock
 *     guards the overflow batch). No collector / full buffer = span dropped
 *     and counted; the sequence number still advances, so the collector
 *     reports exactly how many spans were lost.
 *   - Thread-safe and fork-safe. No allocation. No dependencies beyond POSIX.
 *   - Disabled at compile time with -DEDGELENS_DISABLE (all calls vanish).
 *   - Socket: $EDGELENS_SOCKET, default /tmp/edgelens.sock.
 *
 * GPU timing: CUDA/TensorRT calls are asynchronous; a span ended on the CPU
 * right after enqueue measures the enqueue, not the inference. Use the
 * *_synced / EL_SPAN_GPU variants with your stream's synchronize function.
 * Define EDGELENS_WITH_CUDA before including to get EL_SPAN_CUDA(name, stream).
 *
 * License: MIT (part of EdgeLens).
 */
#ifndef EDGELENS_MARK_H
#define EDGELENS_MARK_H

#define EDGELENS_MARK_PROTOCOL "EL1"

/* clock_gettime / SOCK_CLOEXEC under strict -std=c99/c11. Only effective
 * when this header comes before other system headers; the default GNU
 * dialects (gcc/clang without -std=c99) need nothing. */
#if !defined(__cplusplus) && defined(__STRICT_ANSI__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t start_ns;
    const char *name;
} el_span_t;

typedef void (*el_sync_fn)(void *ctx);

#ifndef EDGELENS_DISABLE

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

/* Per translation unit state. Each TU has its own sender id, so sequence
 * numbers stay gap-free per sender and loss accounting stays exact. */
static int el__fd = -1;
static int el__init_state = 0;          /* 0 = not yet, 1 = ok, -1 = failed */
static struct sockaddr_un el__addr;
static socklen_t el__addrlen = 0;
static uint64_t el__seq = 0;
static uint64_t el__sent = 0;
static uint64_t el__dropped = 0;
static uint64_t el__retry_at = 0;
static uint64_t el__src = 0;
static int el__pid = 0;

static inline uint64_t el_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void el__atfork_child(void);
/* pthread_atfork lives in libpthread on glibc < 2.28 (e.g. JetPack 4.6 /
 * Ubuntu 18.04): reference it weakly so programs link without -pthread, and
 * fall back to a getpid() check per span when it isn't available. */
#pragma weak pthread_atfork
static int el__atfork_ok = 0;
static __thread long el__tid_cache = 0;
static inline long el__tid(void) {
    if (!el__tid_cache) {
#ifdef SYS_gettid
        el__tid_cache = (long)syscall(SYS_gettid);
#else
        el__tid_cache = (long)el__pid;
#endif
    }
    return el__tid_cache;
}


static inline void el__init(void) {
    int expected = 0;
    if (!__atomic_compare_exchange_n(&el__init_state, &expected, 2, 0,
                                     __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        while (__atomic_load_n(&el__init_state, __ATOMIC_ACQUIRE) == 2) { /* spin */ }
        return;
    }
    const char *path = getenv("EDGELENS_SOCKET");
    if (!path || !*path) path = "/tmp/edgelens.sock";
    memset(&el__addr, 0, sizeof(el__addr));
    el__addr.sun_family = AF_UNIX;
    strncpy(el__addr.sun_path, path, sizeof(el__addr.sun_path) - 1);
    el__addrlen = (socklen_t)sizeof(el__addr);
    el__pid = (int)getpid();
    el__src = (el_now_ns() ^ ((uint64_t)(uintptr_t)&el__seq << 16)) & 0xffffffffffffull;
#ifdef SOCK_CLOEXEC
    el__fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
#else
    el__fd = socket(AF_UNIX, SOCK_DGRAM, 0);
#endif
    /* fork(): the child gets its own sender id and sequence, so its spans
     * neither collide with the parent's nor hide the parent's losses. */
    {
        int (*volatile atfork)(void (*)(void), void (*)(void), void (*)(void)) = pthread_atfork;
#ifndef EDGELENS_TEST_NO_ATFORK
        if (atfork && atfork(NULL, NULL, el__atfork_child) == 0) el__atfork_ok = 1;
#else
        (void)atfork;
#endif
    }
    __atomic_store_n(&el__init_state, el__fd >= 0 ? 1 : -1, __ATOMIC_RELEASE);
}

/* Overflow batching. Normal case: one datagram per span, sent at once. When
 * the collector's queue is full (EAGAIN — on Linux the queue limit is a
 * message COUNT, net.unix.max_dgram_qlen, as low as 10), spans are appended
 * to a pending batch and sent as ONE multi-line datagram on the next call,
 * which lets a burst through a short queue. Only when the batch is full too
 * is a span dropped — and counted. */
#ifndef EDGELENS_PENDING_BYTES
#define EDGELENS_PENDING_BYTES 32768   /* ~400 spans; per translation unit */
#endif
static char el__pend[EDGELENS_PENDING_BYTES];
static size_t el__pend_len = 0;
static uint64_t el__pend_spans = 0;
static uint64_t el__flush_after = 0;   /* back-off deadline after "queue full" */
static int el__lock_flag = 0;

/* Short critical sections only (one non-blocking sendto). Spins briefly,
 * then yields, so threads pinned to one core (SCHED_FIFO) can't live-lock. */
static inline void el__lock(void) {
    unsigned spins = 0;
    while (__atomic_exchange_n(&el__lock_flag, 1, __ATOMIC_ACQUIRE)) {
        if (++spins > 64) { sched_yield(); spins = 0; }
    }
}
static inline int el__trylock_for(uint64_t ns) {
    uint64_t until = el_now_ns() + ns;
    while (__atomic_exchange_n(&el__lock_flag, 1, __ATOMIC_ACQUIRE)) {
        if (el_now_ns() > until) return 0;
        sched_yield();
    }
    return 1;
}
static inline void el__unlock(void) { __atomic_store_n(&el__lock_flag, 0, __ATOMIC_RELEASE); }

static inline int el__send_raw(const char *buf, size_t n) {
    return sendto(el__fd, buf, n, MSG_DONTWAIT | MSG_NOSIGNAL,
                  (const struct sockaddr *)&el__addr, el__addrlen) >= 0 ? 0 : errno;
}

static void el__reset_child_state(void) {
    el__src = (el_now_ns() ^ ((uint64_t)(uintptr_t)&el__seq << 16) ^ (uint64_t)getpid())
              & 0xffffffffffffull;
    __atomic_store_n(&el__seq, 0, __ATOMIC_RELAXED);
    el__sent = el__dropped = el__retry_at = 0;
    el__pend_len = 0;                  /* the parent's pending spans are the parent's */
    el__pend_spans = 0;
    el__flush_after = 0;
    el__tid_cache = 0;
    __atomic_store_n(&el__lock_flag, 0, __ATOMIC_RELEASE);
}

static void el__atfork_child(void) {      /* runs single-threaded in the child */
    el__reset_child_state();
    __atomic_store_n(&el__pid, (int)getpid(), __ATOMIC_RELEASE);
}

/* Fallback when pthread_atfork isn't available: detect a fork by pid, and
 * reset exactly once even if several threads of the child arrive at once. */
static int el__resetting = 0;
static inline void el__fork_check(void) {
    if (el__atfork_ok) return;
    int me = (int)getpid();
    if (me == __atomic_load_n(&el__pid, __ATOMIC_ACQUIRE)) return;
    int zero = 0;
    if (__atomic_compare_exchange_n(&el__resetting, &zero, 1, 0,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        if (__atomic_load_n(&el__pid, __ATOMIC_ACQUIRE) != me) {
            el__reset_child_state();
            __atomic_store_n(&el__pid, me, __ATOMIC_RELEASE);
        }
        __atomic_store_n(&el__resetting, 0, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&el__resetting, __ATOMIC_ACQUIRE)) sched_yield();
    }
}

/* Caller holds the lock. Returns 0 when nothing is pending afterwards. */
#ifndef EDGELENS_CHUNK_BYTES
#define EDGELENS_CHUNK_BYTES 4096      /* one overflow datagram: cheap to allocate */
#endif
#ifndef EDGELENS_BACKOFF_NS
#define EDGELENS_BACKOFF_NS 200000ull  /* after "queue full": no syscall for 200 us */
#endif
static inline int el__is_full(int err) {
    return err == EAGAIN || err == EWOULDBLOCK || err == EINTR || err == ENOBUFS;
}

/* Caller holds the lock. Sends the pending batch in chunks of at most
 * EDGELENS_CHUNK_BYTES, cut at line ends, until done or the queue is full
 * (the kernel copies a whole datagram before it checks the queue, so
 * retrying one huge batch per span would cost a 32 KB copy every time).
 * Returns 0 when nothing is pending afterwards. */
static inline int el__flush_pending_locked(void) {
    size_t off = 0;
    int err = 0;
    while (off < el__pend_len) {
        size_t len = el__pend_len - off;
        if (len > EDGELENS_CHUNK_BYTES) {
            len = EDGELENS_CHUNK_BYTES;
            while (len > 0 && el__pend[off + len - 1] != '\n') len--;
            if (len == 0) len = EDGELENS_CHUNK_BYTES;          /* cannot happen: lines < 320 B */
        }
        err = el__send_raw(el__pend + off, len);
        if (err) break;
        uint64_t lines = 0;
        for (size_t i = off; i < off + len; i++) lines += el__pend[i] == '\n';
        __atomic_fetch_add(&el__sent, lines, __ATOMIC_RELAXED);
        el__pend_spans -= lines;
        off += len;
    }
    if (off) {
        memmove(el__pend, el__pend + off, el__pend_len - off);
        el__pend_len -= off;
    }
    return el__pend_len ? (err ? err : EAGAIN) : 0;
}

/* Send one finished span. flags: "-" or e.g. "g" (end taken after GPU sync). */
static inline int el_record(const char *name, uint64_t start_ns, uint64_t end_ns,
                            const char *flags) {
    if (__atomic_load_n(&el__init_state, __ATOMIC_ACQUIRE) != 1) el__init();
    el__fork_check();
    uint64_t seq = __atomic_fetch_add(&el__seq, 1, __ATOMIC_RELAXED);
    if (el__fd < 0) { __atomic_fetch_add(&el__dropped, 1, __ATOMIC_RELAXED); return 0; }
    uint64_t retry = __atomic_load_n(&el__retry_at, __ATOMIC_RELAXED);
    if (retry && end_ns < retry) {   /* no collector: back off, don't syscall */
        __atomic_fetch_add(&el__dropped, 1, __ATOMIC_RELAXED);
        return 0;
    }
    char buf[320];
    int n = snprintf(buf, sizeof(buf), "EL1 X %llx %llu %d %ld %llu %llu %s %.200s\n",
                     (unsigned long long)el__src, (unsigned long long)seq, el__pid, el__tid(),
                     (unsigned long long)start_ns, (unsigned long long)end_ns,
                     (flags && *flags) ? flags : "-", (name && *name) ? name : "?");
    if (n <= 0) { __atomic_fetch_add(&el__dropped, 1, __ATOMIC_RELAXED); return 0; }
    if (n >= (int)sizeof(buf)) { n = (int)sizeof(buf) - 1; buf[n - 1] = '\n'; }

    int ok = 0, err = 0;
    el__lock();
    /* While backing off after "queue full", skip the syscall entirely. */
    int backoff = el__pend_len && end_ns < el__flush_after;
    if (el__pend_len && !backoff) err = el__flush_pending_locked();
    if (!el__pend_len) {
        err = el__send_raw(buf, (size_t)n);
        if (err == 0) { __atomic_fetch_add(&el__sent, 1, __ATOMIC_RELAXED); ok = 1; }
    } else if (backoff) {
        err = EAGAIN;
    }
    if (!ok && el__is_full(err)) {
        if (!backoff) el__flush_after = end_ns + EDGELENS_BACKOFF_NS;
        if (el__pend_len + (size_t)n <= sizeof(el__pend)) {      /* batch it */
            memcpy(el__pend + el__pend_len, buf, (size_t)n);
            el__pend_len += (size_t)n;
            el__pend_spans++;
            ok = 1;
        }
    }
    el__unlock();
    if (ok) {
        __atomic_store_n(&el__retry_at, 0, __ATOMIC_RELAXED);
        return 1;
    }
    __atomic_fetch_add(&el__dropped, 1, __ATOMIC_RELAXED);
    if (!el__is_full(err))
        __atomic_store_n(&el__retry_at, el_now_ns() + 1000000000ull, __ATOMIC_RELAXED);
    return 0;
}

static inline el_span_t el_begin(const char *name) {
    el_span_t s;
    s.name = name;
    s.start_ns = el_now_ns();
    return s;
}

static inline int el_end(el_span_t s) {
    return el_record(s.name, s.start_ns, el_now_ns(), "-");
}

/* Calls sync(ctx) (e.g. a cudaStreamSynchronize wrapper) BEFORE taking the
 * end timestamp, so asynchronous GPU work is inside the span. */
static inline int el_end_synced(el_span_t s, el_sync_fn sync, void *ctx) {
    if (sync) sync(ctx);
    return el_record(s.name, s.start_ns, el_now_ns(), sync ? "g" : "-");
}

/* Send pending spans and tell the collector the last sequence number used,
 * so losses at the very end of a run are counted too: any span not
 * delivered by then shows up as lost. Called automatically at normal exit
 * (return from main, exit()).
 *
 * Do NOT call it from a signal handler. A process killed by SIGINT/SIGTERM
 * without a handler skips it, and spans still pending inside the process
 * can't be counted — EdgeLens then reports that the sender did not finish
 * cleanly. For clean runs, catch SIGINT/SIGTERM with a handler that only
 * sets a flag, leave your loop, and return from main (see
 * examples/markers/cpp_app.cpp). */
static inline void el_flush(void) {
    if (__atomic_load_n(&el__init_state, __ATOMIC_ACQUIRE) != 1 || el__fd < 0) return;
    el__fork_check();                  /* a forked child never re-sends the parent's spans */
    uint64_t deadline = el_now_ns() + 200000000ull;      /* at most 200 ms at exit */
    for (;;) {
        if (!el__trylock_for(10000000ull)) break;        /* never hang at exit */
        int err = el__flush_pending_locked();
        int left = el__pend_len != 0;
        el__unlock();
        if (!left || (err != EAGAIN && err != EWOULDBLOCK && err != EINTR && err != ENOBUFS))
            break;
        if (el_now_ns() > deadline) break;   /* undelivered spans: counted lost via F */
        usleep(1000);
    }
    uint64_t next = __atomic_load_n(&el__seq, __ATOMIC_RELAXED);
    if (next == 0) return;
    char buf[96];
    int n = snprintf(buf, sizeof(buf), "EL1 F %llx %llu %d", (unsigned long long)el__src,
                     (unsigned long long)(next - 1), el__pid);
    if (n > 0) {
        int tries = 0;
        while (el__send_raw(buf, (size_t)n) == EAGAIN && tries++ < 100) usleep(1000);
    }
}

/* Runs at normal process exit (C and C++), once per translation unit. */
static void el__flush_at_exit(void) __attribute__((destructor, unused));
static void el__flush_at_exit(void) { el_flush(); }

static inline void el_stats(uint64_t *sent, uint64_t *dropped) {
    if (sent) *sent = __atomic_load_n(&el__sent, __ATOMIC_RELAXED);
    if (dropped) *dropped = __atomic_load_n(&el__dropped, __ATOMIC_RELAXED);
}

#else  /* EDGELENS_DISABLE: every call compiles to nothing */

static inline uint64_t el_now_ns(void) { return 0; }
static inline int el_record(const char *n, uint64_t a, uint64_t b, const char *f)
    { (void)n; (void)a; (void)b; (void)f; return 0; }
static inline el_span_t el_begin(const char *name) { el_span_t s; s.name = name; s.start_ns = 0; return s; }
static inline int el_end(el_span_t s) { (void)s; return 0; }
static inline int el_end_synced(el_span_t s, el_sync_fn f, void *c)
    { (void)s; if (f) f(c); return 0; }
static inline void el_flush(void) {}
static inline void el_stats(uint64_t *s, uint64_t *d) { if (s) *s = 0; if (d) *d = 0; }

#endif /* EDGELENS_DISABLE */

#define EL_ITERATION_NAME "@iteration"

#ifdef __cplusplus
}  /* extern "C" */

namespace edgelens {

/* RAII span: measures from construction to destruction. */
class Span {
public:
    explicit Span(const char *name, el_sync_fn sync = nullptr, void *ctx = nullptr)
        : s_(el_begin(name)), sync_(sync), ctx_(ctx) {}
    ~Span() { if (sync_) el_end_synced(s_, sync_, ctx_); else el_end(s_); }
    Span(const Span &) = delete;
    Span &operator=(const Span &) = delete;
private:
    el_span_t s_;
    el_sync_fn sync_;
    void *ctx_;
};

class Iteration : public Span {
public:
    explicit Iteration(el_sync_fn sync = nullptr, void *ctx = nullptr)
        : Span(EL_ITERATION_NAME, sync, ctx) {}
};

}  // namespace edgelens

#define EL__CAT2(a, b) a##b
#define EL__CAT(a, b) EL__CAT2(a, b)
#define EL_SPAN(name) ::edgelens::Span EL__CAT(el_span_, __LINE__)(name)
#define EL_SPAN_GPU(name, sync, ctx) ::edgelens::Span EL__CAT(el_span_, __LINE__)(name, sync, ctx)
#define EL_ITERATION() ::edgelens::Iteration EL__CAT(el_iter_, __LINE__)

#ifdef EDGELENS_WITH_CUDA
#include <cuda_runtime_api.h>
namespace edgelens { namespace detail {
inline void cuda_stream_sync(void *stream) { cudaStreamSynchronize((cudaStream_t)stream); }
}}  // namespace edgelens::detail
#define EL_SPAN_CUDA(name, stream) \
    EL_SPAN_GPU(name, ::edgelens::detail::cuda_stream_sync, (void *)(stream))
#endif

#endif /* __cplusplus */

#endif /* EDGELENS_MARK_H */
