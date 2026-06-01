import time
from collections import defaultdict


class CodeProfiler:
    """
    Đo thời gian trung bình từng bước xử lý theo ms/frame.
    Dùng cặp start(name) / stop(name) bao quanh đoạn cần đo.

    Lưu ý: Mỗi name phải có đúng 1 start và 1 stop tương ứng trong mỗi vòng lặp.
    Gọi start() hai lần liên tiếp với cùng name sẽ reset timer — stop() chỉ
    ghi nhận lần start() cuối cùng.
    """

    TARGET_FPS_MS = 33.3   # ms tương ứng 30 FPS

    def __init__(self):
        self.times:       defaultdict = defaultdict(float)
        self.counts:      defaultdict = defaultdict(int)
        self.start_times: dict        = {}

    def start(self, name: str) -> None:
        self.start_times[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if name not in self.start_times:
            return
        elapsed_ms = (time.perf_counter() - self.start_times.pop(name)) * 1000
        self.times[name]  += elapsed_ms
        self.counts[name] += 1

    def avg_ms(self, name: str) -> float:
        if self.counts[name] == 0:
            return 0.0
        return self.times[name] / self.counts[name]

    def report(self) -> None:
        if not self.times:
            return
        print("\n" + "=" * 55)
        print(f"  BÁO CÁO HIỆU NĂNG (ms / frame trung bình)")
        print(f"  Mục tiêu 30 FPS: tổng < {self.TARGET_FPS_MS:.1f} ms")
        print("=" * 55)
        total = 0.0
        for name in self.times:
            avg = self.avg_ms(name)
            total += avg
            flag = "  ⚠" if avg > 10 else ""
            print(f"  {name:<30}: {avg:>6.2f} ms{flag}")
        print("-" * 55)
        over = " ⚠  VƯỢT MỤC TIÊU" if total > self.TARGET_FPS_MS else " ✓"
        print(f"  {'TỔNG CỘNG':<30}: {total:>6.2f} ms{over}")
        print("=" * 55 + "\n")

    def reset(self) -> None:
        self.times.clear()
        self.counts.clear()
        self.start_times.clear()