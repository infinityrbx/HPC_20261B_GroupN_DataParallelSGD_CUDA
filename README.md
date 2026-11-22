# SGD song song dữ liệu trên CUDA với allreduce tự cài đặt: kết quả

_English version: [README.en.md](README.en.md)_

Đề tài 10 (Tối ưu truyền thông đa GPU), dựa trên Hegde & Usmani, *Parallel and Distributed Deep
Learning* (Stanford). Mạng MLP được huấn luyện trên MNIST bằng SGD song song dữ liệu đồng bộ trên
k GPU; gradient được lấy trung bình bằng các thuật toán allreduce tự viết (không dùng NCCL, cuBLAS
hay MPI). Bài báo chỉ phân tích chi phí truyền thông (O(L·k) + O(k·p/B)) về mặt lý thuyết; đề tài
này đo nó trên thực tế.

Mọi số liệu dưới đây đo trên Kaggle, 2× Tesla T4, ngày 22/11/2026. Log và CSV gốc nằm trong
[`runs/kaggle-2026-11-22/`](runs/kaggle-2026-11-22/) (huấn luyện và allreduce) và
[`runs/kaggle-profile-2026-11-22/`](runs/kaggle-profile-2026-11-22/) (profile kernel).

## Tóm tắt

| Câu hỏi | Trả lời |
|---|---|
| Các thuật toán allreduce trên GPU có đúng không? | Có. tree, ring và rd qua mọi bài kiểm tra; các bản sao tham số giống nhau từng bit; loss khớp với 1 GPU sai khác 3,4e-6; độ chính xác 92,70 % sau 2 epoch. |
| GPU có nhanh hơn CPU khi huấn luyện không? | Có. 1 T4: 0,208 s/epoch; bản C tuần tự trên một nhân CPU: 7,99 s. Nhanh khoảng **38 lần**. |
| Phép nhân ma trận có phải nút thắt cổ chai không? | Có. Chiếm 99,7 % số FLOP và **94,5–95,5 % thời gian kernel trên GPU** (đo bằng nsys). |
| Truyền thông tốn bao nhiêu? | Khoảng **một nửa mỗi bước khi dùng 2 GPU**: 0,41–0,46 ms để allreduce gradient 1,63 MB trên GPU, 1,27 ms (68,5 % bước) nếu đi qua CPU. |
| 2 GPU có nhanh hơn 1 GPU không? | Chỉ nhỉnh hơn một chút: 1,06–1,18 lần mỗi bước, vì thời gian truyền thông gần bằng phần tính toán tiết kiệm được khi chia batch. |
| Thuật toán allreduce nào nhanh nhất? | tree ở 4–8 KB, recursive doubling từ 16 KB đến 2 MB, ring từ 4 MB trở lên. Cả ba đều đạt 4,5–4,7 GB/s. |

## Cấu hình thí nghiệm

| Mục | Giá trị |
|---|---|
| Phần cứng | Kaggle, 2× Tesla T4 (70 W, sm_75), driver 580.178, CUDA 13.0 |
| Kết nối | PCIe qua host bridge (topology PHB); lần chạy này có truy cập P2P |
| Mô hình | MLP 784–512–10, 407.050 tham số (gradient 1,63 MB fp32) |
| Huấn luyện | seed 42, batch toàn cục 256 (128 mỗi GPU khi k = 2), lr 0,1, 2 epoch |
| Benchmark | 4 KB đến 64 MB, 5 lần khởi động + 20 lần đo, lấy trung vị; busbw = bytes/time × 2(k−1)/k |

Các phương pháp (mỗi phương pháp một file trong `src/allreduce/`):

| Phương pháp | Cách hoạt động |
|---|---|
| `host_ref` | Chép mọi gradient về CPU, lấy trung bình, chép ngược lên GPU. Dùng làm chuẩn để kiểm tra tính đúng |
| `tree` | Phương pháp của bài báo: gom (reduce) về GPU 0, lấy trung bình, phát (broadcast) lại |
| `ring` | Reduce-scatter rồi allgather (chưa chia nhỏ khối để pipeline) |
| `rd` | Recursive doubling; GPU dư được gộp vào khi k không phải lũy thừa của 2 |
| `--overlap` | Bắt đầu allreduce gradient lớp 2 trong khi lan truyền ngược của lớp 1 vẫn đang chạy |

## 1. Tính đúng đắn

| Kiểm tra | host_ref | tree | ring | rd |
|---|---|---|---|---|
| `make simcheck` (CUDA giả lập, thứ tự stream ngẫu nhiên, k = 2..8) | không áp dụng | đạt | đạt | đạt |
| `bench_allreduce --check-only`, mọi kích thước kể cả n lẻ | đạt | đạt | đạt | đạt |
| Sai khác tham số lớn nhất giữa các bản sao | 0 (từng bit) | 0 (từng bit) | 0 (từng bit) | 0 (từng bit) |
| Loss so với 1 GPU, sai khác tuyệt đối lớn nhất (ngưỡng 1e-4) | 3,36e-6 | 3,36e-6 | 3,36e-6 | 3,36e-6 |
| Độ chính xác trên tập test, 2 epoch | 92,70 % | 92,70 % | 92,70 % | 92,70 % |

Bản C tuần tự cũng khớp với 1 GPU (sai khác loss lớn nhất 3,67e-6, 92,70 %).

## 2. GPU so với CPU

| Phiên bản | Thời gian mỗi epoch | Tăng tốc |
|---|---|---|
| C tuần tự, một nhân CPU (`mlp_seq`) | 7,99 s | 1× |
| 1× T4 (`train_dp --gpus 1`), epoch 2 | 0,208 s | ~38× |

Cả hai đạt cùng loss (0,55714 rồi 0,30289) và cùng độ chính xác 92,70 %.

## 3. Phép nhân ma trận là nút thắt tính toán

Mô hình là MLP, nhưng một lớp kết nối đầy đủ chính là một phép nhân ma trận, và một lớp tích chập
sau khi biến đổi im2col cũng trở thành một phép nhân ma trận, nên lập luận này áp dụng được cho CNN.

**Đếm FLOP** (`scripts/flops.py`, H = 512, batch 256): năm phép nhân ma trận mỗi bước tốn
418,9 MFLOP; mọi phần còn lại (bias + ReLU, softmax, ReLU ngược, gradient bias, cập nhật SGD) chỉ
1,4 MFLOP. Nhân ma trận chiếm **99,7 % số FLOP**, ở mọi kích thước lớp ẩn. Để so sánh, một lớp tích
chập 3×3 từ 32 lên 64 kênh trên ảnh 28×28 là phép nhân (784×288)·(288×64), tức 28,9 MFLOP mỗi ảnh.

**Thời gian đo được** (nsys, một epoch trên 1 T4, [`kernel_share.csv`](runs/kaggle-profile-2026-11-22/kernel_share.csv)):

| Kernel | H = 512 | H = 1024 | H = 2048 |
|---|---|---|---|
| `k_matmul_tiled` | **94,5 %** | **95,0 %** | **95,5 %** |
| `k_sgd` | 1,6 % | 2,4 % | 2,5 % |
| `k_colsum` | 1,8 % | 0,9 % | 0,5 % |
| `k_bias_relu` | 0,9 % | 0,8 % | 0,8 % |
| `k_softmax_xent` | 0,8 % | 0,5 % | 0,2 % |
| `k_relu_backward` | 0,4 % | 0,4 % | 0,5 % |
| Tổng thời gian kernel | 220,7 ms | 368,1 ms | 715,2 ms |

Tỉ lệ thời gian thấp hơn tỉ lệ FLOP một chút vì các kernel theo từng phần tử bị giới hạn bởi băng
thông bộ nhớ: chúng gần như không tính toán nhưng vẫn phải đọc và ghi mọi phần tử. Kernel nhân ma
trận dạng tile đạt khoảng 490 GFLOP/s, chỉ 6 % hiệu năng FP32 đỉnh của T4, nên tăng tốc phép nhân
ma trận sẽ giảm trực tiếp phần tính toán của mỗi bước.

## 4. Chi phí truyền thông và tăng tốc trên 2 GPU

Thời gian mỗi bước theo từng pha, ms (lấy giá trị lớn nhất giữa các GPU), và tỉ lệ truyền thông
không được che trong một bước:

| Cấu hình | fwd | bwd | comm | step | tỉ lệ comm | Tăng tốc so với 1 GPU (theo bước) |
|---|---|---|---|---|---|---|
| 1 GPU | 0,383 | 0,472 | 0,004 | 0,964 | 0,4 % | 1,00× |
| host_ref | 0,237 | 0,249 | 1,266 | 1,801 | 68,5 % | 0,54× |
| tree | 0,189 | 0,199 | 0,427 | 0,862 | 46,8 % | 1,12× |
| tree + overlap | 0,166 | 0,178 | 0,411 | 0,825 | 49,9 % | 1,17× |
| ring | 0,212 | 0,223 | 0,460 | 0,908 | 50,3 % | 1,06× |
| ring + overlap | 0,166 | 0,181 | 0,416 | 0,828 | 50,3 % | 1,16× |
| rd | 0,208 | 0,218 | 0,452 | 0,899 | 48,4 % | 1,07× |
| rd + overlap | 0,167 | 0,181 | 0,407 | 0,815 | 49,9 % | 1,18× |

Thời gian nạp dữ liệu (khoảng 0,17 ms) và cập nhật không đưa vào bảng; chúng như nhau ở mọi dòng.

Thời gian mỗi epoch (s):

| Cấu hình | Trung bình 2 epoch | Chỉ epoch 2 |
|---|---|---|
| 1 GPU | 0,267 | 0,208 |
| host_ref | 0,463 | 0,436 |
| tree / ring / rd | 0,243 / 0,250 / 0,251 | 0,234 / 0,231 / 0,233 |
| tree / ring / rd + overlap | 0,234 / 0,234 / 0,232 | 0,233 / 0,234 / 0,231 |

Nhận xét:

- Chia batch làm phần tính toán giảm một nửa (forward + backward từ 0,86 ms xuống khoảng
  0,35–0,43 ms mỗi GPU), nhưng allreduce 1,63 MB tốn gần bằng phần tiết kiệm được. Truyền thông là
  pha lớn nhất trong mọi lần chạy 2 GPU.
- Thực hiện phép gom trên GPU thay vì CPU giảm truyền thông khoảng 3 lần (1,27 ms xuống
  0,41–0,46 ms) và giảm một nửa thời gian mỗi epoch.
- `--overlap` giảm thời gian bước 4–9 %, nhưng tỉ lệ truyền thông không được che vẫn khoảng 50 %.
  Phần lớn lợi ích đến từ việc thời gian giữa hai GPU đều hơn (độ mất cân bằng từ 1,06–1,08 xuống
  1,02); MLP hai lớp còn quá ít việc lan truyền ngược để che truyền thông.
- Mức tăng tốc phụ thuộc cách đo. Theo bước, 2 GPU nhanh hơn 6–18 %. Theo epoch, trung bình 2 epoch
  cho 1,06–1,15 lần, nhưng chỉ vì epoch đầu của lần chạy 1 GPU có thời gian khởi động; nếu chỉ xét
  epoch 2, 2 GPU chậm hơn 1 GPU khoảng 10 %.

## 5. Allreduce thuần: thuật toán nào thắng ở kích thước nào

Thời gian trung vị mỗi lần allreduce, k = 2, ms (`allreduce.csv`); chữ đậm là nhanh nhất:

| Kích thước | tree | ring | rd |
|---|---|---|---|
| 4 KB | **0,031** | 0,053 | 0,034 |
| 8 KB | **0,031** | 0,054 | 0,035 |
| 16 KB | 0,041 | 0,053 | **0,035** |
| 64 KB | 0,053 | 0,060 | **0,037** |
| 256 KB | 0,101 | 0,100 | **0,076** |
| 1 MB | 0,261 | 0,258 | **0,238** |
| 2 MB | 0,486 | 0,481 | **0,456** |
| 4 MB | 0,971 | **0,927** | 0,955 |
| 16 MB | 3,776 | **3,608** | 3,750 |
| 64 MB | 15,009 | **14,301** | 14,981 |

Băng thông bus ở 64 MB: tree 4,47, ring 4,69, rd 4,48 GB/s. Một lần chép P2P một chiều đạt 9,9 GB/s.

- **Thông điệp nhỏ bị giới hạn bởi độ trễ.** Ring có mức sàn khoảng 0,053 ms vì phải chạy nhiều bước
  phụ thuộc nhau; tree và rd bắt đầu từ khoảng 0,031–0,035 ms.
- **Thông điệp vừa (16 KB–2 MB) thì recursive doubling nhanh nhất**, hơn tới 30 % ở 64 KB và 5–10 %
  ở 1–2 MB. Gradient 1,63 MB của mô hình nằm trong khoảng này.
- **Thông điệp lớn bị giới hạn bởi băng thông và ring nhanh nhất**, nhưng chỉ hơn 4–5 %.
- **Cả ba chỉ đạt khoảng một nửa tốc độ đường truyền.** Tree phải gửi dữ liệu đi một chiều rồi gửi
  về, nên một nửa là giới hạn của nó. Ring và rd về lý thuyết có thể dùng cả hai chiều cùng lúc
  nhưng không vượt được mức đó, cho thấy hai chiều không chạy chồng lên nhau và ring cần pipeline
  theo khối.
- Với k = 2, ba thuật toán truyền cùng một lượng dữ liệu. Khác biệt lý thuyết giữa chúng chỉ thể hiện
  khi k ≥ 4.

## Hạn chế

- **Chỉ một lần chạy, 2 epoch.** Khác biệt giữa tree, ring và rd khi huấn luyện (dưới 4 % thời gian
  epoch) nằm trong mức nhiễu giữa các lần chạy. Số liệu cuối cùng nên dùng mặc định 10 epoch, báo
  cáo riêng epoch 1, và lặp mỗi cấu hình ít nhất 3 lần.
- **Chỉ có 2 GPU.** Kaggle cung cấp tối đa 2× T4, nên không kiểm chứng được sự tăng theo k của
  O(L·k) + O(k·p/B).
- **Truy cập P2P khác nhau giữa các máy Kaggle.** Lần chạy này có P2P. Một máy Kaggle trước đó không
  có (ma trận P2P toàn 0); khi đó việc chép giữa GPU phải đi qua bộ nhớ host và số liệu không so sánh
  được (1 GPU 0,24 s/epoch, `host_ref` 0,41 s/epoch với 64 % truyền thông). Luôn ghi lại ma trận P2P
  cùng kết quả.
- **T4 hạ xung khi nóng hoặc chạm giới hạn công suất.** T4 là card 70 W tản nhiệt thụ động, tự giảm
  xung khi tải kéo dài. Xung nhịp không được ghi lại trong lúc chạy nên không loại trừ được khả năng
  này.
- **Phiên bản mã nguồn.** Log huấn luyện 2 GPU được build từ nhánh `ltnaphixay/allreduce-cuda` của
  `infinityrbx/High-Performance-Computing`, có các tùy chọn `--per-layer` và `--phase-csv` được dùng
  ở đó; `train_dp` trong repo này chưa có các tùy chọn đó.

## Build và chạy

```bash
make data          # tải MNIST vào ./data
make seq           # bản C tuần tự (chỉ cần gcc)
make ARCH=sm_75    # build tất cả; sm_75 = T4, sm_80 = A100
make simcheck      # kiểm tra tree/ring/rd với thứ tự stream ngẫu nhiên (chỉ cần g++, không cần GPU)

./bin/mlp_seq                                     # bản tuần tự (CPU)
./bin/train_dp --gpus 1                           # 1 GPU
./bin/train_dp --gpus 2 --method ring [--overlap] # song song dữ liệu
./bin/bench_allreduce --method ring --gpus 2 [--check-only]
bash scripts/run_experiments.sh 2 "tree ring rd"  # so sánh đầy đủ
python3 scripts/flops.py --hidden 512             # đếm FLOP, nhân ma trận so với phần còn lại
```

Trên Kaggle (điền username vào `kernel-metadata.json` trước):

```bash
kaggle kernels push -p scripts/kaggle/check --accelerator NvidiaTeslaT4    # build, kiểm tra, huấn luyện, benchmark
kaggle kernels push -p scripts/kaggle/profile --accelerator NvidiaTeslaT4  # profile thời gian kernel bằng nsys
```

Các thiết lập cố định nằm trong `src/common/config.h`. Mọi file CSV có cùng các cột:
`method,k,msg_bytes,time_ms,busbw_GBs,epoch_s,comm_pct,test_acc` (`-1` = không áp dụng;
`comm_pct` = truyền thông không được che, tính theo % thời gian bước).

## Cấu trúc thư mục

```
src/
  common/     config.h (thiết lập cố định), đọc MNIST, tham số/CLI, timer, CSV
  seq/        mlp_seq.c: MLP tuần tự bằng C (chuẩn kiểm tra tính đúng)
  cuda/       kernels.cu (nhân ma trận dạng tile, softmax, relu, ...), train_dp.cu (vòng huấn luyện)
  allreduce/  allreduce.h (giao diện), ar_sync.h, common.cu, registry.cu, mỗi phương pháp một file
bench/        bench_allreduce.cu: kiểm tra tính đúng + đo độ trễ/băng thông
tests/        sim_allreduce.cpp + CUDA runtime giả lập (make simcheck)
scripts/      run_experiments.sh, flops.py, kernel_share.py, get_mnist.sh
  kaggle/     check/ (build, kiểm tra, huấn luyện, benchmark), profile/ (profile kernel bằng nsys)
runs/         log và CSV gốc từ Kaggle
```
