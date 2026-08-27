// test_cold_memcpy_latency.cpp
// [moe-l2 HIP] 검증 실험 3: Cold/Warm/Hit 상태의 memcpy 지연
//  - Cold: MADV_PAGEOUT 로 방출 후 memcpy (major fault + NVMe I/O 지배)
//        (실험 2 에서 posix_fadvise 가 O_RDONLY/PROT_READ/MAP_SHARED 에서 no-op 임을 확인
//         → 방출은 반드시 madvise(MADV_PAGEOUT) 사용)
//  - Warm: touch 만 하고 방출 안 함 (RAM hit)
//  - Hit : 직전에 읽음 (resident)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static double bench_memcpy(const char* src, size_t len, void* dst) {
    auto t0 = std::chrono::high_resolution_clock::now();
    std::memcpy(dst, src, len);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <file> <offset> <size>\n", argv[0]); return 1; }
    const char* path = argv[1];
    size_t off  = (size_t)strtoull(argv[2], nullptr, 10);
    size_t len  = (size_t)strtoull(argv[3], nullptr, 10);

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    size_t size = (size_t)st.st_size;
    if (off + len > size) { fprintf(stderr, "range out of file\n"); return 1; }

    char* ptr = (char*)mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) { perror("mmap"); return 1; }

    void* dst = malloc(len + 4096); // 핀 풀 대용 (host RAM)
    const char* src = ptr + off;

    // 페이지 정렬: madvise(MADV_PAGEOUT) 는 정렬 len 필요 (미정렬은 조용히 no-op)
    size_t len_align = (len + 4095) & ~(size_t)4095;
    if (len_align > size - off) len_align = (size - off) & ~(size_t)4095;

    // Hit: warm-up (resident)
    for (int i = 0; i < 10; i++) bench_memcpy(src, len, dst);
    double hit_sum = 0;
    for (int i = 0; i < 100; i++) hit_sum += bench_memcpy(src, len, dst);

    // Warm: 방출만 하고 touch (RAM hit)
#if defined(MADV_PAGEOUT)
    madvise((void*)src, len_align, MADV_PAGEOUT);
#else
    posix_fadvise(fd, off, len, POSIX_FADV_DONTNEED);
#endif
    volatile unsigned char s = 0;
    for (size_t i = 0; i < len; i += 4096) s += ((volatile unsigned char*)src)[i];
    (void)s;
    double warm_sum = 0;
    for (int i = 0; i < 100; i++) warm_sum += bench_memcpy(src, len, dst);

    // Cold: 방출 후 바로 memcpy (major fault)
#if defined(MADV_PAGEOUT)
    madvise((void*)src, len_align, MADV_PAGEOUT);
#else
    posix_fadvise(fd, off, len, POSIX_FADV_DONTNEED);
#endif
    double cold_sum = 0;
    for (int i = 0; i < 100; i++) {
        cold_sum += bench_memcpy(src, len, dst);
#if defined(MADV_PAGEOUT)
        madvise((void*)src, len_align, MADV_PAGEOUT); // 매 반복 cold 유지
#else
        posix_fadvise(fd, off, len, POSIX_FADV_DONTNEED);
#endif
    }

    printf("file=%s off=%zu len=%zu (%.2fMB)\n", path, off, len, len / 1048576.0);
    printf("HIT  memcpy avg=%.1fus\n", hit_sum / 100);
    printf("WARM memcpy avg=%.1fus\n", warm_sum / 100);
    printf("COLD memcpy avg=%.1fus\n", cold_sum / 100);

    free(dst);
    munmap(ptr, size);
    close(fd);
    return 0;
}
