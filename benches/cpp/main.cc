// C++ contenders for the perf.md scoreboard: tlx::btree_map (the
// canonical C++ B+tree, ex-STX) and absl::btree_map (Google's B-tree)
// on the exact workloads of benches/vs_btreemap.rs and
// benches/blocked_insert.rs. Key sequences are bit-identical to the
// Rust benches (same multiplicative shuffle, same xorshift churn mix).
//
// Timing mirrors criterion's iter_batched_ref: per sample the setup
// (fresh container / prebuilt tree) is untimed and only the op loop is
// timed; each row reports the median of kSamples samples, with
// [min..max] for a noise readout. Both containers run with their
// default node-size traits (256-byte nodes for both).

#include <absl/container/btree_map.h>
#include <tlx/container/btree_map.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <vector>

namespace {

using u64 = std::uint64_t;
using Clock = std::chrono::steady_clock;

using TlxMap = tlx::btree_map<u64, u64>;
using AbslMap = absl::btree_map<u64, u64>;

constexpr int kWarmup = 3;
constexpr int kSamples = 25;

const std::vector<std::size_t> kSizes = {1'000, 100'000};
constexpr std::size_t kBlockedN = 100'000;
const std::vector<std::size_t> kBlocks = {10, 100, 1'000};
constexpr std::size_t kScanLen = 100;

template <class T>
void do_not_optimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

double elapsed_ms(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

/// `n` distinct keys in deterministic shuffled order — same sequence
/// as the Rust benches' `shuffled_keys` (sort by wrapping-mul hash;
/// the odd multiplier is a bijection, so the sort has no ties).
std::vector<u64> shuffled_keys(std::size_t n) {
    std::vector<u64> ks(n);
    for (std::size_t i = 0; i < n; ++i) ks[i] = i;
    std::sort(ks.begin(), ks.end(), [](u64 a, u64 b) {
        return a * 0x9E3779B97F4A7C15ULL < b * 0x9E3779B97F4A7C15ULL;
    });
    return ks;
}

/// Blocks of `block` CONSECUTIVE keys, block order shuffled.
std::vector<u64> blocked_local(std::size_t block) {
    std::vector<u64> keys;
    keys.reserve(kBlockedN);
    for (u64 start : shuffled_keys(kBlockedN / block))
        for (u64 i = 0; i < block; ++i) keys.push_back(start * block + i);
    return keys;
}

/// Blocks of `block` keys STRIDED across the keyspace, sorted within
/// each block, block order shuffled.
std::vector<u64> blocked_strided(std::size_t block) {
    const u64 stride = kBlockedN / block;
    std::vector<u64> keys;
    keys.reserve(kBlockedN);
    for (u64 b : shuffled_keys(kBlockedN / block))
        for (u64 i = 0; i < block; ++i) keys.push_back(b + i * stride);
    return keys;
}

struct ChurnOp {
    bool insert;
    u64 key;
};

/// 60% insert / 40% remove over a small key domain — same xorshift
/// stream as the Rust bench.
std::vector<ChurnOp> churn_ops(std::size_t n) {
    u64 state = 0x5EEDCAFEF00DD00DULL;
    const u64 domain = std::max<u64>(n / 4, 64);
    std::vector<ChurnOp> ops(n);
    for (auto& op : ops) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        op = {((state >> 32) % 5) < 3, state % domain};
    }
    return ops;
}

struct Stats {
    double median, min, max;  // ms
};

/// Run `sample` (returns one timed iteration in ms) kWarmup + kSamples
/// times; report the median of the samples.
template <class F>
Stats run(F&& sample) {
    for (int i = 0; i < kWarmup; ++i) sample();
    std::vector<double> ms(kSamples);
    for (auto& m : ms) m = sample();
    std::sort(ms.begin(), ms.end());
    return {ms[kSamples / 2], ms.front(), ms.back()};
}

template <class Map>
void build_into(Map& m, const std::vector<u64>& keys) {
    for (u64 k : keys) m[k] = k;
}

template <class Map>
Stats bench_insert(const std::vector<u64>& keys) {
    return run([&] {
        Map m;
        const auto t0 = Clock::now();
        for (u64 k : keys) m[k] = k;
        const auto t1 = Clock::now();
        do_not_optimize(m.size());
        return elapsed_ms(t0, t1);
    });
}

template <class Map>
Stats bench_get(const std::vector<u64>& keys, const std::vector<u64>& probes) {
    Map m;
    build_into(m, keys);
    return run([&] {
        u64 sum = 0;
        const auto t0 = Clock::now();
        for (u64 k : probes) {
            auto it = m.find(k);
            if (it != m.end()) sum += it->second;
        }
        const auto t1 = Clock::now();
        do_not_optimize(sum);
        return elapsed_ms(t0, t1);
    });
}

template <class Map>
Stats bench_remove(const std::vector<u64>& keys) {
    return run([&] {
        Map m;
        build_into(m, keys);
        std::size_t erased = 0;
        const auto t0 = Clock::now();
        for (u64 k : keys) erased += m.erase(k);
        const auto t1 = Clock::now();
        do_not_optimize(erased);
        return elapsed_ms(t0, t1);
    });
}

template <class Map>
Stats bench_churn(const std::vector<ChurnOp>& ops) {
    return run([&] {
        Map m;
        const auto t0 = Clock::now();
        for (const auto& op : ops) {
            if (op.insert) {
                m[op.key] = op.key;
            } else {
                m.erase(op.key);
            }
        }
        const auto t1 = Clock::now();
        do_not_optimize(m.size());
        return elapsed_ms(t0, t1);
    });
}

template <class Map>
Stats bench_iterate_all(const std::vector<u64>& keys) {
    Map m;
    build_into(m, keys);
    return run([&] {
        u64 acc = 0;
        const auto t0 = Clock::now();
        for (const auto& kv : m) acc += kv.first + kv.second;
        const auto t1 = Clock::now();
        do_not_optimize(acc);
        return elapsed_ms(t0, t1);
    });
}

template <class Map>
Stats bench_range_scan(const std::vector<u64>& keys, const std::vector<u64>& starts) {
    Map m;
    build_into(m, keys);
    return run([&] {
        u64 acc = 0;
        const auto t0 = Clock::now();
        for (u64 s : starts) {
            auto it = m.lower_bound(s);
            for (std::size_t i = 0; i < kScanLen && it != m.end(); ++i, ++it)
                acc += it->first + it->second;
        }
        const auto t1 = Clock::now();
        do_not_optimize(acc);
        return elapsed_ms(t0, t1);
    });
}

template <class Map>
Stats bench_drop(const std::vector<u64>& keys) {
    return run([&] {
        std::optional<Map> m;
        m.emplace();
        build_into(*m, keys);
        const auto t0 = Clock::now();
        m.reset();
        const auto t1 = Clock::now();
        return elapsed_ms(t0, t1);
    });
}

void row(const char* bench, std::size_t param, Stats tlx, Stats absl) {
    std::printf("%-24s %7zu   tlx %9.3f ms [%8.3f..%9.3f]   absl %9.3f ms [%8.3f..%9.3f]\n",
                bench, param, tlx.median, tlx.min, tlx.max, absl.median, absl.min, absl.max);
    std::fflush(stdout);
}

}  // namespace

int main() {
    std::printf("tlx::btree_map vs absl::btree_map, u64 keys; median of %d samples\n\n", kSamples);

    for (std::size_t n : kSizes) {
        std::vector<u64> seq(n);
        for (std::size_t i = 0; i < n; ++i) seq[i] = i;
        row("insert_sequential", n, bench_insert<TlxMap>(seq), bench_insert<AbslMap>(seq));
    }
    for (std::size_t n : kSizes) {
        const auto keys = shuffled_keys(n);
        row("insert_shuffled", n, bench_insert<TlxMap>(keys), bench_insert<AbslMap>(keys));
    }
    for (std::size_t n : kSizes) {
        const auto keys = shuffled_keys(n);
        row("get_hit", n, bench_get<TlxMap>(keys, keys), bench_get<AbslMap>(keys, keys));
    }
    for (std::size_t n : kSizes) {
        // Store the EVEN keys, probe the ODD ones (shuffled): every
        // probe is a miss landing BETWEEN stored keys.
        auto keys = shuffled_keys(n);
        auto probes = keys;
        for (auto& k : keys) k *= 2;
        for (auto& k : probes) k = 2 * k + 1;
        row("get_miss", n, bench_get<TlxMap>(keys, probes), bench_get<AbslMap>(keys, probes));
    }
    for (std::size_t n : kSizes) {
        const auto keys = shuffled_keys(n);
        row("remove_shuffled", n, bench_remove<TlxMap>(keys), bench_remove<AbslMap>(keys));
    }
    for (std::size_t n : kSizes) {
        const auto ops = churn_ops(n);
        row("churn", n, bench_churn<TlxMap>(ops), bench_churn<AbslMap>(ops));
    }
    for (std::size_t n : kSizes) {
        const auto keys = shuffled_keys(n);
        row("drop", n, bench_drop<TlxMap>(keys), bench_drop<AbslMap>(keys));
    }
    for (std::size_t n : kSizes) {
        const auto keys = shuffled_keys(n);
        row("iterate_all", n, bench_iterate_all<TlxMap>(keys), bench_iterate_all<AbslMap>(keys));
    }
    for (std::size_t n : kSizes) {
        // Seek a random present key, read the next kScanLen pairs;
        // n / kScanLen scans touch ~n pairs per timed iteration.
        const auto keys = shuffled_keys(n);
        const std::vector<u64> starts(keys.begin(), keys.begin() + n / kScanLen);
        row("range_scan", n, bench_range_scan<TlxMap>(keys, starts),
            bench_range_scan<AbslMap>(keys, starts));
    }
    for (std::size_t block : kBlocks) {
        const auto keys = blocked_local(block);
        row("insert_blocked_local", block, bench_insert<TlxMap>(keys), bench_insert<AbslMap>(keys));
    }
    for (std::size_t block : kBlocks) {
        const auto keys = blocked_strided(block);
        row("insert_blocked_strided", block, bench_insert<TlxMap>(keys),
            bench_insert<AbslMap>(keys));
    }
    return 0;
}
