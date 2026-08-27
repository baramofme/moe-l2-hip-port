// test_hybrid.cpp
// [moe-l2 HIP] 하이브리드 hot/cold 실험.
// 실측 상수로 3개 스킴을 시뮬레이션해 VRAM 캐시 용량별 winner 를 판정한다.
//   A. bounce-only     : miss -> memcpy+H2D, hit -> D2D (mmap 방출로 RSS 상수)
//   B. zero-copy-only  : 모든 접근 -> 커널 직접 읽기 (VRAM 불필요, RSS 상수)
//   C. hybrid          : hot expert -> mapped zero-copy, cold expert -> bounce
// D2D hit 비용은 실시간 측정, miss/zero-copy 비용은 앞선 실험의 실측값 사용.
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

__global__ void read_sum_kernel(const uint32_t * p, uint64_t n, uint64_t * partial) {
    __shared__ uint64_t sh[256];
    uint64_t s = 0;
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) s += p[i];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < (unsigned)off) sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sh[0];
}

static double measure_d2d_hit_us(void* d_src, void* d_dst, size_t bytes) {
    double best = 1e9;
    for (int i = 0; i < 30; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        hipMemcpyAsync(d_dst, d_src, bytes, hipMemcpyDeviceToDevice, 0);
        hipDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (us < best) best = us;
    }
    return best;
}

struct SimResult {
    double total_us;
    long   misses;
    long   hits;
};

// bounce-only / hybrid 의 공통 VRAM LRU 시뮬레이션
// - bounce: hit 시 D2D(hit_us), miss 시 memcpy+H2D(bounce_miss_us), set 후 방출(무료)
// - hybrid: hot 만 zero-copy(zc_us, VRAM 미사용), cold 는 위와 동일
static SimResult sim(const std::vector<int>& access, const std::vector<bool>& is_hot,
                     int vram_cap, double hit_us, double miss_us, double zc_us, bool hybrid) {
    std::vector<int> lru;              // expert id, 최근이 뒤
    lru.reserve(vram_cap);
    long hits = 0, misses = 0;
    double total = 0;
    for (size_t t = 0; t < access.size(); t++) {
        int e = access[t];
        if (hybrid && is_hot[e]) {
            total += zc_us;            // zero-copy: VRAM/캐시 무관, 항상 커널 읽기
            continue;
        }
        auto it = std::find(lru.begin(), lru.end(), e);
        if (it != lru.end() && vram_cap > 0) {  // hit (캐시 없으면 항상 miss)
            hits++;
            total += hit_us;
            lru.erase(it);
            lru.push_back(e);
        } else {                        // miss
            misses++;
            total += miss_us;
            if ((int)lru.size() >= vram_cap && vram_cap > 0) lru.erase(lru.begin());
            if (vram_cap > 0) lru.push_back(e);
        }
    }
    return {total, misses, hits};
}

int main() {
    const size_t ES = 1625292;  // 1.55MB expert
    const int    N_EXP = 256;
    const int    N_TOK = 2000;
    const int    K     = 6;
    const double P_HOT = 0.85;
    const double ZC_US = 213.0;   // 실측: zero-copy 커널 읽기 min
    const double MCPY  = 78.0;    // 실측: 1.55MB CPU memcpy (콜드 ~1700us, hot ~78us)
    const double H2D   = 131.0;   // 실측: pinned H2D 1.55MB min
    const double MISS  = MCPY + H2D;  // bounce miss = memcpy + H2D

    // 실시간 D2D hit 비용 측정 (VRAM -> VRAM, 1.55MB)
    void* d_a = nullptr; void* d_b = nullptr;
    hipMalloc(&d_a, ES); hipMalloc(&d_b, ES);
    hipMemset(d_a, 1, ES);
    double d2d = measure_d2d_hit_us(d_a, d_b, ES);
    printf("measured D2D hit (1.55MB): %.1fus | bounce miss: %.1fus | zero-copy: %.1fus\n",
           d2d, MISS, ZC_US);

    // 접근 패턴 생성 (hot set 30/50/100)
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> hd(0, 1);
    std::uniform_int_distribution<int> hot_draw(0, 0), cold_draw(0, 0);

    printf("\n%8s %6s | %12s %12s %12s | %s\n", "hotset", "vram", "bounce-only", "zc-only", "hybrid", "winner");

    for (int hot_size : {30, 50, 100}) {
        // access pattern
        std::vector<int> acc(N_TOK * K);
        std::vector<bool> is_hot(N_EXP, false);
        std::vector<int> hot_ids;
        for (int i = 0; i < hot_size; i++) { hot_ids.push_back(i); is_hot[i] = true; }
        std::uniform_int_distribution<int> hd2(0, hot_size - 1), cd2(hot_size, N_EXP - 1);
        for (int t = 0; t < N_TOK * K; t++) {
            acc[t] = hd(rng) < P_HOT ? hot_ids[hd2(rng)] : cd2(rng);
        }

        for (int vram_cap : {0, 5, 10, 30, 60, 100}) {
            auto A = sim(acc, is_hot, vram_cap, d2d, MISS, ZC_US, false);
            auto B = sim(acc, is_hot, vram_cap, d2d, MISS, ZC_US, false);
            B.total_us = 0; for (int i = 0; i < (int)acc.size(); i++) B.total_us += ZC_US;  // zc-only: 전부 213us
            auto C = sim(acc, is_hot, vram_cap, d2d, MISS, ZC_US, true);

            double tA = A.total_us, tB = B.total_us, tC = C.total_us;
            const char* w = (tA <= tB && tA <= tC) ? "bounce" : (tB <= tC ? "zc-only" : "hybrid");
            printf("%8d %6d | %9.1fms %9.1fms %9.1fms | %s\n",
                   hot_size, vram_cap, tA/1000.0, tB/1000.0, tC/1000.0, w);
        }
    }

    hipFree(d_a); hipFree(d_b);
    printf("\nDONE\n");
    return 0;
}