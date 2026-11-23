/**
 * @file threading.hpp
 * @brief Thread count and fork-join helpers shared by the threaded host kernels.
 *
 * @par Theory
 * The host kernels (grid plan build, two-centre integrals, KB projector terms, Libxc evaluation) split independent
 * work into contiguous ranges, one per thread, with no shared mutable state inside a range.
 *
 * @par Algorithm
 * \c thread_count honours \c OMP_NUM_THREADS (capped by the hardware threads) and otherwise uses at most 8 threads;
 * small work runs on one thread. \c parallel_ranges and \c run_threads start plain \c std::thread workers and join
 * them before returning.
 *
 * @par Invariants
 * - Results never depend on the thread count when every range writes only its own outputs.
 * - No global state; safe to call from several host threads.
 *
 * @par SIESTA Mapping
 * Plays the role of SIESTA's OpenMP loops over orbitals and mesh points.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

namespace jsiesta::threading {

/**
 * @brief Number of threads for \p work independent items.
 *
 * @par Theory
 * Threads only pay off above a minimum amount of work.
 *
 * @par Algorithm
 * One thread below \p min_work; else \c OMP_NUM_THREADS capped by \c std::thread::hardware_concurrency, or at most 8
 * hardware threads when the variable is unset.
 *
 * @param work Number of independent work items.
 * @param min_work Smallest amount of work worth threading.
 * @return Thread count, at least 1.
 *
 * @par Invariants
 * Pure function of its arguments, the environment and the hardware.
 *
 * @par SIESTA Mapping
 * \c OMP_NUM_THREADS as read by SIESTA's OpenMP regions.
 */
inline int thread_count(std::int64_t work, std::int64_t min_work = 512) {
    if (work < min_work) return 1;
    const unsigned hc = std::thread::hardware_concurrency();
    const char* env = std::getenv("OMP_NUM_THREADS");
    if (env != nullptr && env[0] != '\0') {
        const int n = std::atoi(env);
        if (n >= 1) return hc == 0 ? n : std::min(n, static_cast<int>(hc));
    }
    return hc == 0 ? 1 : static_cast<int>(std::min<unsigned>(hc, 8U));
}

/**
 * @brief Calls \c body(t, lo, hi) on \p n_threads threads for the contiguous ranges of \c [0, n).
 *
 * @par Theory
 * Static block partition of independent items.
 *
 * @par Algorithm
 * Range \c t is \c [n t / n_threads, n (t + 1) / n_threads); one thread per range, joined before returning.
 *
 * @param n Number of items.
 * @param n_threads Number of threads (1 runs \p body inline).
 * @param body Callable \c (int t, int64_t lo, int64_t hi).
 *
 * @par Invariants
 * Every item is in exactly one range; ranges are ordered by \c t.
 *
 * @par SIESTA Mapping
 * A static-schedule OpenMP loop.
 */
template <class F>
void parallel_ranges(std::int64_t n, int n_threads, F&& body) {
    if (n_threads <= 1) {
        body(0, std::int64_t{0}, n);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
        const std::int64_t lo = n * t / n_threads, hi = n * (t + 1) / n_threads;
        pool.emplace_back([&body, t, lo, hi]() { body(t, lo, hi); });
    }
    for (auto& th : pool) th.join();
}

/**
 * @brief Calls \c fn(t) for \c t = 0 .. n_threads - 1, \c t = 0 on the calling thread.
 *
 * @par Theory
 * Fork-join for work that partitions itself by thread index.
 *
 * @par Algorithm
 * Starts \c n_threads - 1 workers, runs \c fn(0) inline and joins.
 *
 * @param n_threads Number of threads (1 runs \p fn inline).
 * @param fn Callable \c (int t).
 *
 * @par Invariants
 * Each index runs exactly once.
 *
 * @par SIESTA Mapping
 * An OpenMP parallel region with explicit thread ids.
 */
template <class Fn>
void run_threads(int n_threads, Fn&& fn) {
    if (n_threads <= 1) {
        fn(0);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(n_threads - 1));
    for (int t = 1; t < n_threads; ++t) pool.emplace_back([&fn, t] { fn(t); });
    fn(0);
    for (auto& th : pool) th.join();
}

}  // namespace jsiesta::threading
