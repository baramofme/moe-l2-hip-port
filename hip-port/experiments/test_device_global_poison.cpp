// test_device_global_poison.cpp
// [moe-l2 HIP] 검증 실험 1: Device-Global Poisoning
// hipHostMalloc(mini pool) 등록 상태에서, unpinned mmap 영역의 hipMemcpyAsync H2D 가
// 동작하는지 판별한다.
//   - 성공(hipSuccess)          : poison 없음 → 로딩/비-expert direct H2D 유지
//   - 실패(invalid argument 등) : poison 존재 → 모든 H2D staging 경유
// 수정 사항: 파일 사전 존재 확인 + fstat 로 mmap 크기 clamp (SIGBUS 방지).
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "/tmp/dummy_gguf.bin";
    const size_t POOL = 16 * 1024 * 1024; // 16MB mini pinned pool

    // 1. 미니 핀 풀 할당
    void* pinned_pool = nullptr;
    hipError_t h = hipHostMalloc(&pinned_pool, POOL, hipHostMallocDefault);
    printf("[1] hipHostMalloc(16MB pool): %s\n", hipGetErrorString(h));
    if (h != hipSuccess) return 1;

    // 2. 임의 파일 mmap (미핀 영역) — 실제 파일 크기로 clamp
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    size_t size = (size_t)st.st_size;
    if (size <= 0) { fprintf(stderr, "empty file\n"); return 1; }
    printf("[2] mmap file=%s size=%zu bytes\n", path, size);

    void* mmap_ptr = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_ptr == MAP_FAILED) { perror("mmap"); return 1; }

    // touch 페이지 (실제 읽기 → 페이지폴트)
    volatile unsigned char sum = 0;
    for (size_t i = 0; i < size; i += 4096) sum += ((volatile unsigned char*)mmap_ptr)[i];
    printf("[3] touched %zu pages (sum=%u)\n", size/4096, (unsigned)sum);

    // 3. GPU VRAM 할당
    void* d_ptr = nullptr;
    h = hipMalloc(&d_ptr, size);
    printf("[4] hipMalloc(VRAM %zu): %s\n", size, hipGetErrorString(h));
    if (h != hipSuccess) return 1;

    // 4. 핀되지 않은 mmap 영역에서 direct async H2D 시도
    hipStream_t stream;
    hipStreamCreate(&stream);
    hipError_t err = hipMemcpyAsync(d_ptr, mmap_ptr, size, hipMemcpyHostToDevice, stream);
    printf("[5] async unpinned mmap H2D: %s\n", hipGetErrorString(err));
    hipError_t sync = hipStreamSynchronize(stream);
    printf("[6] stream sync: %s\n", hipGetErrorString(sync));

    // 5. 보조 probe: sync hipMemcpy (unpinned) — 문서화된 스테이징 동작 대조
    hipError_t err_sync = hipMemcpy(d_ptr, mmap_ptr, size, hipMemcpyHostToDevice);
    printf("[7] sync unpinned mmap H2D: %s\n", hipGetErrorString(err_sync));

    hipFree(d_ptr);
    munmap(mmap_ptr, size);
    close(fd);
    hipHostFree(pinned_pool);
    printf("DONE\n");
    return 0;
}
