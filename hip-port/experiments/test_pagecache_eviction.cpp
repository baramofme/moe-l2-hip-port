// test_pagecache_eviction.cpp
// [moe-l2 HIP] 검증 실험 2: MAP_SHARED pagecache 회수
// PROT_READ-only mmap(실제 GGUF 시맨틱)에서 posix_fadvise(DONTNEED) / madvise(PAGEOUT) 이
// RssFile 을 실제로 줄이는지, 그리고 각각의 refault 지연을 메서드별 독립 서브레인지로 측정한다.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

static size_t rss_file_kb() {
    std::ifstream s("/proc/self/status");
    std::string line;
    while (std::getline(s, line)) {
        if (line.rfind("RssFile:", 0) == 0) return (size_t)std::stoul(line.substr(8));
        if (line.rfind("VmRSS:", 0) == 0) return (size_t)std::stoul(line.substr(6));
    }
    return 0;
}

static void touch_range(const char* p, size_t start, size_t len) {
    volatile unsigned char sum = 0;
    for (size_t i = start; i < start + len; i += 4096) sum += ((volatile unsigned char*)p)[i];
    (void)sum;
}

static double refault_ns(const char* p, size_t off) {
    auto t0 = std::chrono::high_resolution_clock::now();
    volatile unsigned char c = ((volatile unsigned char*)p)[off];
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)c;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file>\n", argv[0]); return 1; }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    size_t size = (size_t)st.st_size;
    if (size < 768 * 1024 * 1024) { fprintf(stderr, "need >= 768MB file, have %zu\n", size); return 1; }

    char* ptr = (char*)mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) { perror("mmap"); return 1; }

    const size_t MB = 1024 * 1024;
    printf("file=%s size=%zuMB  RssFile_initial=%zuKB\n", argv[1], size / MB, rss_file_kb());

    // A: posix_fadvise(DONTNEED)  — 0..256MB
    touch_range(ptr, 0, 256 * MB);
    printf("A [fadvise] after touch : RssFile=%zuKB\n", rss_file_kb());
    posix_fadvise(fd, 0, 256 * MB, POSIX_FADV_DONTNEED);
    printf("A [fadvise] after evict : RssFile=%zuKB  refault_1page=%.0fns\n",
           rss_file_kb(), refault_ns(ptr, 0));

    // B: madvise(MADV_PAGEOUT)     — 256..512MB
    touch_range(ptr, 256 * MB, 256 * MB);
    printf("B [pageout] after touch : RssFile=%zuKB\n", rss_file_kb());
#if defined(MADV_PAGEOUT)
    int pr = madvise(ptr + 256 * MB, 256 * MB, MADV_PAGEOUT);
    printf("B [pageout] after evict : RssFile=%zuKB  madvise_ret=%d  refault_1page=%.0fns\n",
           rss_file_kb(), pr, refault_ns(ptr, 256 * MB));
#else
    printf("B [pageout] NOT SUPPORTED (kernel < 5.4)\n");
#endif

    // C: 대조군 — 아무것도 안 함 (커널 자발적 회수 관측)  512..768MB
    touch_range(ptr, 512 * MB, 256 * MB);
    printf("C [control] after touch : RssFile=%zuKB  refault_1page=%.0fns (resident)\n",
           rss_file_kb(), refault_ns(ptr, 512 * MB));

    // 연속 1.55MB refault (DS expert 크기) — 방출 후 연속 재접근 비용
    double t_start = 0, t_end = 0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        volatile unsigned char sum = 0;
        for (size_t i = 0; i < 1.55 * MB; i += 4096) sum += ((volatile unsigned char*)ptr)[i];
        auto t1 = std::chrono::high_resolution_clock::now();
        (void)sum;
        t_start = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }
    // A 영역(이미 방출됨)을 1.55MB 연속 재접근 → cold refault 비용
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        volatile unsigned char sum = 0;
        for (size_t i = 0; i < 1.55 * MB; i += 4096) sum += ((volatile unsigned char*)ptr)[i];
        auto t1 = std::chrono::high_resolution_clock::now();
        (void)sum;
        t_end = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }
    printf("1.55MB warm touch=%.0fus  cold refault(evicted A)=%.0fus\n", t_start, t_end);

    munmap(ptr, size);
    close(fd);
    return 0;
}
