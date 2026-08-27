// test_reservations.cpp
// [moe-l2 HIP] 유보 사항 3가지 실측
//  R1. 다중 소형 슬라이스 등록 + 사이/인접 미핀 접근 (실제 모델 패턴에서 poison 없는지)
//  R2. GPU가 터치한 mapped 페이지의 RSS 상주/방출 (madvise(PAGEOUT) 결합 가능성)
//  R3. 1.55MB 소형 슬라이스에서 zero-copy vs H2D 대역폭/지연
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

static size_t vmrss_kb() {
    std::ifstream s("/proc/self/status");
    std::string line;
    while (std::getline(s, line)) {
        if (line.rfind("VmRSS:", 0) == 0) return (size_t)std::stoul(line.substr(6));
    }
    return 0;
}

static size_t rss_file_kb() {
    std::ifstream s("/proc/self/status");
    std::string line;
    while (std::getline(s, line)) {
        if (line.rfind("RssFile:", 0) == 0) return (size_t)std::stoul(line.substr(8));
    }
    return 0;
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "/tmp/test_gguf.bin";
    const size_t ES = 1625292;  // 1.55MB, DS expert size (non-page-multiple)
    const size_t PAGESZ = 4096;
    const size_t N_SLICES = 16;

    int fd = open(path, O_RDWR);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    size_t fsz = (size_t)st.st_size;
    if (fsz < 768 * 1024 * 1024) { fprintf(stderr, "file too small\n"); return 1; }
    void* m = mmap(NULL, fsz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { perror("mmap"); return 1; }

    printf("R1: multiple small-slice mapped register + interleaved unpinned H2D\n");
    {
        // register 16 slices spread over 0..768MB, each page-aligned range
        std::vector<std::pair<uintptr_t, uintptr_t>> regs;  // [start,end) page-aligned
        for (size_t i = 0; i < N_SLICES; i++) {
            size_t base = i * 48 * 1024 * 1024;  // 48MB apart
            uintptr_t start = ((uintptr_t)m + base) & ~(uintptr_t)(PAGESZ - 1);
            uintptr_t end = ((uintptr_t)m + base + ES + PAGESZ - 1) & ~(uintptr_t)(PAGESZ - 1);
            hipError_t e = hipHostRegister((void*)start, end - start, hipHostRegisterMapped);
            if (e != hipSuccess) {
                printf("   register slice %zu: %s\n", i, hipGetErrorString(e));
                regs.clear();
                break;
            }
            regs.emplace_back(start, end);
        }
        printf("   registered %zu slices (page-aligned, 48MB apart)\n", regs.size());

        void* d_dst = nullptr;
        hipMalloc(&d_dst, ES);

        // 1a. H2D from a REGISTERED slice -> should work
        hipError_t e_reg = hipMemcpyAsync(d_dst, (const uint8_t*)m + 0, ES, hipMemcpyHostToDevice, 0);
        hipDeviceSynchronize();
        printf("   1a. H2D from REGISTERED slice   : %s\n", hipGetErrorString(e_reg));

        // 1b. H2D from an UNREGISTERED slice BETWEEN two registered ones
        //     (slice 0 at 0MB, slice 1 at 48MB -> unregistered at 24MB)
        hipError_t e_unreg = hipMemcpyAsync(d_dst, (const uint8_t*)m + 24 * 1024 * 1024, ES,
                                            hipMemcpyHostToDevice, 0);
        hipDeviceSynchronize();
        printf("   1b. H2D from UNREGISTERED slice (between regs): %s\n", hipGetErrorString(e_unreg));

        // 1c. H2D from an unregistered slice far away (700MB)
        hipError_t e_far = hipMemcpyAsync(d_dst, (const uint8_t*)m + 700 * 1024 * 1024, ES,
                                          hipMemcpyHostToDevice, 0);
        hipDeviceSynchronize();
        printf("   1c. H2D from UNREGISTERED slice (700MB, far)  : %s\n", hipGetErrorString(e_far));

        // 1d. kernel direct read of a registered slice via device pointer
        void* d_slice0 = nullptr;
        hipHostGetDevicePointer(&d_slice0, (void*)regs[0].first, 0);
        uint64_t* d_part = nullptr;
        hipMalloc(&d_part, 4096);
        read_sum_kernel<<<64, 256>>>((const uint32_t*)d_slice0, (uint64_t)(ES / 4), d_part);
        hipDeviceSynchronize();
        hipError_t kerr = hipGetLastError();
        printf("   1d. kernel read via registered slice device ptr: %s\n",
               kerr == hipSuccess ? "no error" : hipGetErrorString(kerr));
        hipFree(d_part);

        for (auto& r : regs) hipHostUnregister((void*)r.first);
        hipFree(d_dst);
        printf("   R1 done (unregistered all)\n");
    }

    printf("R2: RSS residency of GPU-touched mapped pages + MADV_PAGEOUT\n");
    {
        // register a 256MB region, kernel-touch part of it, watch RSS
        size_t reg_sz = 256 * 1024 * 1024;
        hipError_t e = hipHostRegister(m, reg_sz, hipHostRegisterMapped);
        printf("   hipHostRegister(256MB, mapped): %s\n", hipGetErrorString(e));
        if (e != hipSuccess) { munmap(m, fsz); close(fd); return 1; }
        void* d_ptr = nullptr;
        hipHostGetDevicePointer(&d_ptr, m, 0);

        size_t rss0 = vmrss_kb();
        // kernel reads only the FIRST 32MB of the registered region
        uint64_t* d_part = nullptr;
        hipMalloc(&d_part, 4096);
        read_sum_kernel<<<64, 256>>>((const uint32_t*)d_ptr, (uint64_t)(32 * 1024 * 1024 / 4), d_part);
        hipDeviceSynchronize();
        size_t rss1 = vmrss_kb();
        printf("   RSS before-kernel=%zuKB after-32MB-touch=%zuKB delta=%+zdKB\n",
               rss0, rss1, (ssize_t)rss1 - (ssize_t)rss0);

        // madvise(MADV_PAGEOUT) on the touched 32MB range
#if defined(MADV_PAGEOUT)
        int pr = madvise(m, 32 * 1024 * 1024, MADV_PAGEOUT);
        size_t rss2 = vmrss_kb();
        printf("   after MADV_PAGEOUT(32MB): RSS=%zuKB delta=%+zdKB madvise_ret=%d\n",
               rss2, (ssize_t)rss2 - (ssize_t)rss1, pr);
#endif
        hipFree(d_part);
        hipHostUnregister(m);
        printf("   R2 done\n");
    }

    printf("R3: small-slice (1.55MB) zero-copy vs H2D bandwidth/latency\n");
    {
        // zero-copy: register 1.55MB aligned range, kernel-read repeatedly
        size_t base = 600 * 1024 * 1024;
        uintptr_t start = ((uintptr_t)m + base) & ~(uintptr_t)(PAGESZ - 1);
        uintptr_t end = ((uintptr_t)m + base + ES + PAGESZ - 1) & ~(uintptr_t)(PAGESZ - 1);
        hipHostRegister((void*)start, end - start, hipHostRegisterMapped);
        void* d_ptr = nullptr;
        hipHostGetDevicePointer(&d_ptr, (void*)start, 0);
        uint64_t* d_part = nullptr;
        hipMalloc(&d_part, 4096);
        size_t ne = ES / 4;

        // zero-copy kernel read, repeated
        double zc_min = 1e9, zc_sum = 0;
        for (int i = 0; i < 50; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            read_sum_kernel<<<8, 256>>>((const uint32_t*)d_ptr, (uint64_t)ne, d_part);
            hipDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            zc_sum += us;
            if (us < zc_min) zc_min = us;
        }
        printf("   zero-copy kernel read : avg=%.1fus min=%.1fus (%.2fMB -> %s GB/s)\n",
               zc_sum / 50, zc_min, ES / 1048576.0,
               (ES / (zc_min / 1e6) / 1e9) > 0 ? "calc" : "calc");

        // H2D copy of 1.55MB from the same region
        void* d_dst = nullptr;
        hipMalloc(&d_dst, ES);
        double h2d_min = 1e9, h2d_sum = 0;
        for (int i = 0; i < 50; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            hipMemcpyAsync(d_dst, (const uint8_t*)m + base, ES, hipMemcpyHostToDevice, 0);
            hipDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            h2d_sum += us;
            if (us < h2d_min) h2d_min = us;
        }
        printf("   pinned H2D copy       : avg=%.1fus min=%.1fus\n", h2d_sum / 50, h2d_min);
        printf("   latency ratio (zc_min/h2d_min): %.2fx\n", zc_min / h2d_min);
        hipFree(d_dst);
        hipFree(d_part);
        hipHostUnregister((void*)start);
        printf("   R3 done\n");
    }

    munmap(m, fsz);
    close(fd);
    printf("DONE\n");
    return 0;
}