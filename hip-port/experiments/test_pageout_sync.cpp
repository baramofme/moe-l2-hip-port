// test_pageout_sync.cpp
// [moe-l2 HIP] MADV_PAGEOUT 동기성/정렬 검증
//  - PAGEOUT 이 RssFile 을 동기적으로 떨어뜨리는지 확인
//  - 페이지 정렬 len vs 미정렬 len 의 madvise 반환값 비교
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

static size_t rss_file_kb() {
    std::ifstream s("/proc/self/status");
    std::string line;
    while (std::getline(s, line)) {
        if (line.rfind("RssFile:", 0) == 0) return (size_t)std::stoul(line.substr(8));
    }
    return 0;
}

static void touch(const char* p, size_t len) {
    volatile unsigned char sum = 0;
    for (size_t i = 0; i < len; i += 4096) sum += ((volatile unsigned char*)p)[i];
    (void)sum;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file>\n", argv[0]); return 1; }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    size_t size = (size_t)st.st_size;
    char* ptr = (char*)mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) { perror("mmap"); return 1; }

    const size_t MB = 1024 * 1024;
    printf("initial RssFile=%zuKB\n", rss_file_kb());

    // 정렬 1MB 영역 (512MB offset)
    char* p = ptr + 512 * MB;
    touch(p, 1 * MB);
    printf("after touch 1MB(aligned)  : RssFile=%zuKB\n", rss_file_kb());
    int r1 = madvise(p, 1 * MB, MADV_PAGEOUT);
    printf("madvise(PAGEOUT) 1MB ret=%d RssFile=%zuKB\n", r1, rss_file_kb());

    // 미정렬 1.55MB 영역 (600MB offset)
    size_t len_un = (size_t)(1.55 * MB);
    char* q = ptr + 600 * MB;
    touch(q, len_un);
    printf("after touch 1.55MB(un)    : RssFile=%zuKB\n", rss_file_kb());
    int r2 = madvise(q, len_un, MADV_PAGEOUT);
    printf("madvise(PAGEOUT) 1.55MB ret=%d RssFile=%zuKB\n", r2, rss_file_kb());

    // 미정렬 len 을 페이지 상향 정렬 후
    size_t len_align = (len_un + 4095) & ~(size_t)4095;
    int r3 = madvise(q, len_align, MADV_PAGEOUT);
    printf("madvise(PAGEOUT) 1.55MB->%zu ret=%d RssFile=%zuKB\n", len_align, r3, rss_file_kb());

    munmap(ptr, size);
    close(fd);
    return 0;
}