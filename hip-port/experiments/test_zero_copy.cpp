// test_zero_copy.cpp
// [moe-l2 HIP] Zero-copy fallback candidates (in case the bounce engine
// underperforms). Verifies on the current ROCm/RDNA3:
//   A. hipHostMalloc(hipHostMallocMapped) -> hipHostGetDevicePointer -> kernel read
//   B. mmap + hipHostRegister(hipHostRegisterMapped) -> hipHostGetDevicePointer -> kernel read
//   C. RSS right after register (eager pin vs lazy fault-in)
//   D. adjacent unpinned-page H2D after mapped register (poison re-test)
//   E. baseline: pinned H2D copy bandwidth
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
#include <string>
#include <vector>

__global__ void sum_kernel(const uint32_t * p, uint64_t n, uint64_t * partial) {
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

// reports mismatch count and first mismatch index into out[0], out[1]
__global__ void verify_kernel(const uint32_t * p, uint64_t n, unsigned long long * out) {
    uint64_t cnt = 0;
    uint64_t first_bad = 0;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        uint32_t expect = (uint32_t)((i * 2654435761u) >> 7);
        if (p[i] != expect) {
            cnt++;
            if (first_bad == 0) first_bad = i + 1;
        }
    }
    if (cnt > 0) {
        atomicAdd(&out[0], (unsigned long long)cnt);
        if (first_bad != 0) {
            atomicMin(&out[1], first_bad);
        }
    }
}

static size_t vmrss_kb() {
    std::ifstream s("/proc/self/status");
    std::string line;
    while (std::getline(s, line)) {
        if (line.rfind("VmRSS:", 0) == 0) return (size_t)std::stoul(line.substr(6));
    }
    return 0;
}

static uint64_t host_sum(const uint32_t * p, size_t n) {
    uint64_t s = 0;
    for (size_t i = 0; i < n; i++) s += p[i];
    return s;
}

// verify against the deterministic pattern; reports count and first bad index
static void verify_pattern(const uint32_t * d_ptr, size_t n_elem, unsigned long long * d_out) {
    unsigned long long init[2] = {0, 0};
    hipMemcpy(d_out, init, sizeof(init), hipMemcpyHostToDevice);
    verify_kernel<<<256, 256>>>((const uint32_t *)d_ptr, (uint64_t)n_elem, d_out);
    hipDeviceSynchronize();
    unsigned long long res[2] = {0, 0};
    hipMemcpy(res, d_out, sizeof(res), hipMemcpyDeviceToHost);
    printf("   pattern verify        : mismatches=%llu first_bad_idx=%llu\n", res[0], res[1]);
}

// host-computed reference for one BLOCK's full range: all threads of the
// block, grid-stride, same layout as sum_kernel (64 blocks x 256 threads)
static uint64_t block_sum_ref(const uint32_t * p, size_t n, int block, int blocks, int threads) {
    uint64_t s = 0;
    size_t stride = (size_t)blocks * threads;
    for (size_t t = 0; t < (size_t)threads; t++) {
        for (size_t i = (size_t)block * threads + t; i < n; i += stride) s += p[i];
    }
    return s;
}

// control: run the SAME sum_kernel on a real device buffer (not mapped).
// isolates "kernel/compiler bug" from "mapped-path bug".
static bool sum_on_device_buffer(const uint32_t * host_src, size_t bytes, size_t n_elem, uint64_t expect) {
    void * d = nullptr;
    hipMalloc(&d, bytes);
    hipMemcpy(d, host_src, bytes, hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    uint64_t * d_part = nullptr;
    hipMalloc(&d_part, 4096);
    sum_kernel<<<64, 256>>>((const uint32_t *)d, (uint64_t)n_elem, d_part);
    hipDeviceSynchronize();
    uint64_t part[64];
    hipMemcpy(part, d_part, sizeof(part), hipMemcpyDeviceToHost);
    uint64_t got = 0;
    for (int b = 0; b < 64; b++) got += part[b];
    // per-block compare (0..3) to localize
    for (int b = 0; b < 4; b++) {
        uint64_t ref = block_sum_ref(host_src, n_elem, b, 64, 256);
        printf("   [ctrl diag] block %d: kernel=%llu host=%llu %s\n", b,
               (unsigned long long)part[b], (unsigned long long)ref,
               part[b] == ref ? "OK" : "DIFF");
    }
    hipFree(d);
    hipFree(d_part);
    return got == expect;
}

static double kernel_read_bw_us(const uint32_t * d_ptr, size_t n_elem, uint64_t * d_partial, int blocks, bool * ok) {
    auto t0 = std::chrono::high_resolution_clock::now();
    sum_kernel<<<blocks, 256>>>(d_ptr, n_elem, d_partial);
    hipError_t se = hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    hipError_t le = hipGetLastError();
    if (ok) *ok = (se == hipSuccess && le == hipSuccess);
    if (ok && !*ok) printf("   [kernel err] sync=%s last=%s\n", hipGetErrorString(se), hipGetErrorString(le));
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

// D2D sanity: copy from the (mapped) device pointer to a fresh device buffer,
// then back to host; compares the first 64 elements byte-for-byte.
static bool d2d_sanity(const void * d_ptr, size_t bytes, const uint32_t * expect) {
    void * d_buf = nullptr;
    hipMalloc(&d_buf, bytes);
    hipError_t e = hipMemcpyAsync(d_buf, d_ptr, bytes, hipMemcpyDeviceToDevice, 0);
    hipDeviceSynchronize();
    if (e != hipSuccess) {
        printf("   [d2d err] %s\n", hipGetErrorString(e));
        hipFree(d_buf);
        return false;
    }
    std::vector<uint32_t> back(bytes / 4);
    hipMemcpy(back.data(), d_buf, bytes, hipMemcpyDeviceToHost);
    hipFree(d_buf);
    bool same = true;
    for (size_t i = 0; i < 64 && i < back.size(); i++) {
        if (back[i] != expect[i]) { same = false; break; }
    }
    return same;
}

static double h2d_copy_bw_us(void * d_dst, const void * h_src, size_t bytes) {
    auto t0 = std::chrono::high_resolution_clock::now();
    hipMemcpyAsync(d_dst, h_src, bytes, hipMemcpyHostToDevice, 0);
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "/tmp/test_gguf.bin";
    const size_t SZ = 64 * 1024 * 1024;          // 64MB test region
    const size_t N  = SZ / sizeof(uint32_t);     // elements

    // fill pattern
    std::vector<uint32_t> host(SZ / 4);
    for (size_t i = 0; i < N; i++) host[i] = (uint32_t)((i * 2654435761u) >> 7);
    uint64_t expect = host_sum(host.data(), N);

    void * d_partial = nullptr;
    hipMalloc(&d_partial, 4096);

    // control: sum kernel on a real device buffer
    printf("-- CONTROL: sum_kernel on real device buffer --\n");
    printf("   device-buffer sum_match=%s\n",
           sum_on_device_buffer(host.data(), SZ, N, expect) ? "YES" : "NO");
    // also verify the host `expect` itself against a plain CPU sum
    uint64_t cpu_sum = 0;
    for (size_t i = 0; i < N; i++) cpu_sum += host[i];
    printf("   expect=%llu cpu_sum=%llu (%s)\n", (unsigned long long)expect,
           (unsigned long long)cpu_sum, expect == cpu_sum ? "SAME" : "DIFF");

    // E. baseline: pinned H2D copy bandwidth
    {
        void * h_pin = nullptr, * d_dst = nullptr;
        hipHostMalloc(&h_pin, SZ, hipHostMallocDefault);
        memcpy(h_pin, host.data(), SZ);
        hipMalloc(&d_dst, SZ);
        double us = h2d_copy_bw_us(d_dst, h_pin, SZ);
        printf("E  pinned H2D copy     : %6.1f us  -> %6.1f GB/s\n", us, SZ / us / 1e3);
        hipFree(d_dst);
        hipHostFree(h_pin);
    }

    // A. hipHostMallocMapped zero-copy
    {
        printf("-- A. hipHostMallocMapped --\n");
        void * h = nullptr, * d = nullptr;
        hipError_t e1 = hipHostMalloc(&h, SZ, hipHostMallocMapped);
        printf("   hipHostMalloc(mapped) : %s\n", hipGetErrorString(e1));
        if (e1 == hipSuccess) {
            hipError_t e2 = hipHostGetDevicePointer(&d, h, 0);
            printf("   hipHostGetDevicePointer: %s d=%p\n", hipGetErrorString(e2), d);
            if (e2 == hipSuccess && d) {
                memcpy(h, host.data(), SZ);
                bool k_ok = false;
                double us = kernel_read_bw_us((const uint32_t *)d, N, (uint64_t *)d_partial, 64, &k_ok);
                // verify
                uint64_t * part = new uint64_t[64];
                hipError_t d2h_err = hipMemcpy(part, d_partial, 64 * sizeof(uint64_t), hipMemcpyDeviceToHost);
                uint64_t got = 0;
                for (int b = 0; b < 64; b++) got += part[b];
                uint64_t ref0 = block_sum_ref(host.data(), N, 0, 64, 256);
                printf("   [diag A] d2h_err=%s part[0]=%llu ref_block0=%llu\n",
                       hipGetErrorString(d2h_err), (unsigned long long)part[0], (unsigned long long)ref0);
                delete[] part;
                bool d2d_ok = d2d_sanity(d, SZ, host.data());
                printf("   kernel zero-copy read: %6.1f us -> %6.1f GB/s  sum_match=%s kernel_ok=%s d2d_sanity=%s\n",
                       us, SZ / us / 1e3, got == expect ? "YES" : "NO", k_ok ? "YES" : "NO", d2d_ok ? "YES" : "NO");
                verify_pattern((const uint32_t *)d, N, (unsigned long long *)d_partial);
                // first-read staleness probe: rerun the sum AFTER other GPU
                // work touched the mapping; if it now matches, first read was stale
                {
                    bool k2 = false;
                    kernel_read_bw_us((const uint32_t *)d, N, (uint64_t *)d_partial, 64, &k2);
                    uint64_t * part2 = new uint64_t[64];
                    hipMemcpy(part2, d_partial, 64 * sizeof(uint64_t), hipMemcpyDeviceToHost);
                    uint64_t got2 = 0;
                    for (int b = 0; b < 64; b++) got2 += part2[b];
                    delete[] part2;
                    printf("   rerun-sum-after-verify : sum_match=%s\n", got2 == expect ? "YES" : "NO");
                }
            }
            hipHostFree(h);
        }
    }

    // B. mmap + hipHostRegister(mapped)
    printf("-- B. mmap + hipHostRegister(hipHostRegisterMapped) --\n");
    {
        int fd = open(path, O_RDWR);
        if (fd < 0) { perror("open"); return 1; }
        struct stat st;
        if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
        size_t fsz = (size_t)st.st_size;
        void * m = mmap(NULL, fsz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) { perror("mmap"); return 1; }

        // fill the region so the kernel can verify values
        memcpy(m, host.data(), SZ);

        size_t rss_before = vmrss_kb();
        hipError_t e1 = hipHostRegister(m, SZ, hipHostRegisterMapped);
        size_t rss_after  = vmrss_kb();
        printf("   hipHostRegister(mapped): %s\n", hipGetErrorString(e1));
        printf("   RSS before=%zuKB after=%zuKB (delta=%+zdKB)\n",
               rss_before, rss_after, (ssize_t)rss_after - (ssize_t)rss_before);

        if (e1 == hipSuccess) {
            void * d = nullptr;
            hipError_t e2 = hipHostGetDevicePointer(&d, m, 0);
            printf("   hipHostGetDevicePointer: %s d=%p\n", hipGetErrorString(e2), d);
            if (e2 == hipSuccess && d) {
                bool k_ok = false;
                double us = kernel_read_bw_us((const uint32_t *)d, N, (uint64_t *)d_partial, 64, &k_ok);
                uint64_t * part = new uint64_t[64];
                hipMemcpy(part, d_partial, 64 * sizeof(uint64_t), hipMemcpyDeviceToHost);
                uint64_t got = 0;
                for (int b = 0; b < 64; b++) got += part[b];
                delete[] part;
                bool d2d_ok = d2d_sanity(d, SZ, host.data());
                printf("   kernel zero-copy read: %6.1f us -> %6.1f GB/s  sum_match=%s kernel_ok=%s d2d_sanity=%s\n",
                       us, SZ / us / 1e3, got == expect ? "YES" : "NO", k_ok ? "YES" : "NO", d2d_ok ? "YES" : "NO");
                verify_pattern((const uint32_t *)d, N, (unsigned long long *)d_partial);
            }

            // D. adjacent unpinned-page H2D after mapped register (poison re-test)
            printf("   D. adjacent unpinned H2D after mapped register:\n");
            void * d_dst = nullptr;
            hipMalloc(&d_dst, SZ);
            hipError_t e3 = hipMemcpyAsync(d_dst, (const uint8_t *)m + SZ, SZ, hipMemcpyHostToDevice, 0);
            hipDeviceSynchronize();
            printf("      H2D from adjacent unpinned page: %s\n", hipGetErrorString(e3));
            hipFree(d_dst);
            hipHostUnregister(m);
        }
        munmap(m, fsz);
        close(fd);
    }

    hipFree(d_partial);
    printf("DONE\n");
    return 0;
}