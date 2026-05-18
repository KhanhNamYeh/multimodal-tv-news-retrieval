# AIC — Vietnamese News Video Retrieval Demo

Demo retrieval video tin tức tiếng Việt (kiến trúc Variant 3).

## 1. Tải checkpoint

Tải file `pretrain_v3_gate_neg1.pth` từ link sau:

> **Link Google Drive:** https://drive.google.com/file/d/1kBpQKjlE_xMIW6ZsFjLoen_3SV_QWlfG/view?usp=sharing

Đặt file vào **thư mục gốc** của repo:

```
aic/
└── pretrain_v3_gate_neg1.pth   ← đặt ở đây
```

## 2. Cài đặt dependencies

```bash
pip install -r source/demo/requirements.txt
```

## 3. Chạy demo

```bash
python source/demo/app.py
```

Chạy CPU-only (nếu không có GPU):

```bash
python source/demo/app.py cpu
```

Sau khi khởi động, mở trình duyệt tại địa chỉ Gradio in ra (mặc định `http://127.0.0.1:7860`).
