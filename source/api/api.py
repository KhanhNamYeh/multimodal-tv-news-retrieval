import os
import requests

# Lay HF token tu bien moi truong: export HF_TOKEN=hf_xxx
HF_TOKEN = os.environ.get("HF_TOKEN", "REDACTED")

# Qwen3 models tren HF (chon model phu hop voi free tier)
# - Qwen/Qwen3-0.6B  -> nhe nhat, nhanh nhat
# - Qwen/Qwen3-1.7B
# - Qwen/Qwen3-4B
# - Qwen/Qwen3-8B    -> khuyen dung cho chat
MODEL_ID = "Qwen/Qwen3-8B"  # Qwen3 co tren HF qua Nscale provider

API_URL = "https://router.huggingface.co/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}


def chat(
    prompt: str,
    system: str = "You are a helpful assistant.",
    max_tokens: int = 512,
    temperature: float = 0.7,
    thinking: bool = False,  # Qwen3 ho tro thinking mode
) -> str:
    """
    Goi Qwen3 qua HF Inference API (mien phi).

    Args:
        prompt: noi dung nguoi dung
        system: system prompt
        max_tokens: so token toi da tra ve
        temperature: do ngau nhien (0.0 - 1.0)
        thinking: bat thinking mode cua Qwen3 (yeu cau them token)

    Returns:
        Chuoi phan hoi cua model
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }

    # Qwen3: thinking=False -> dung temperature binh thuong
    #         thinking=True  -> khong truyen temperature (model tu dieu chinh)
    if thinking:
        payload["thinking"] = {"type": "enabled", "budget_tokens": 1024}
    else:
        payload["temperature"] = temperature

    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


def batch_chat(
    prompts: list[str],
    system: str = "You are a helpful assistant.",
    max_tokens: int = 512,
) -> list[str]:
    """Goi nhieu prompt lien tiep."""
    results = []
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Processing...")
        try:
            result = chat(prompt, system=system, max_tokens=max_tokens)
            results.append(result)
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                print("Rate limit! Cho 60s...")
                import time
                time.sleep(60)
                result = chat(prompt, system=system, max_tokens=max_tokens)
                results.append(result)
            else:
                raise
    return results


SYSTEM_PROMPT = """Bạn tạo training data cho mô hình embedding retrieval tìm kiếm tài liệu.

Nhiệm vụ: Từ passage cho sẵn, sinh các câu truy vấn tìm tài liệu dạng "tìm tài liệu/thông tin về [nội dung cụ thể]". Người dùng chưa có tài liệu, họ đang tìm kiếm tài liệu chứa thông tin đó.

- positive (3 mẫu): câu truy vấn tìm tài liệu mà passage này là kết quả phù hợp. Bắt đầu bằng "tài liệu về...", "thông tin về...", "số liệu về...", "nghiên cứu về...", v.v. Nêu chi tiết cụ thể (số liệu, năm, tổ chức) có trong passage.
- negative (2 mẫu): câu truy vấn tương tự nhưng passage này KHÔNG chứa thông tin đó (hard negative).

Ví dụ positive: "tài liệu về dự báo mực nước biển dâng đến năm 2100 theo báo cáo IPCC"
Ví dụ positive: "thông tin về cam kết Net Zero của Việt Nam tại COP26 và mục tiêu năm 2050"
Ví dụ negative: "tài liệu về cam kết giảm phát thải của Hoa Kỳ theo hiệp định Paris"

Chỉ trả về JSON thuần túy, không giải thích:
{"positive": ["query 1", "query 2", "query 3"], "negative": ["query 1", "query 2"]}"""

SAMPLE_PASSAGE = """Trí tuệ nhân tạo (AI) đang trải qua giai đoạn phát triển bùng nổ chưa từng có trong lịch sử công nghệ.
Năm 2023, OpenAI ra mắt GPT-4 với khả năng lập luận vượt trội, đạt điểm số nằm trong top 10% kỳ thi
luật sư Hoa Kỳ và có thể xử lý cả văn bản lẫn hình ảnh. Cùng năm đó, Google DeepMind giới thiệu Gemini,
mô hình đa phương thức được huấn luyện trên dữ liệu văn bản, âm thanh, hình ảnh và video, đánh dấu
bước ngoặt trong phát triển AI tổng quát.

Thị trường AI toàn cầu được định giá 142 tỷ USD năm 2023 và dự báo tăng trưởng với tốc độ CAGR 37%
đến năm 2030, đạt 1,8 nghìn tỷ USD. Riêng lĩnh vực AI sinh tạo (Generative AI) thu hút 25 tỷ USD
đầu tư mạo hiểm chỉ trong năm 2023, gấp 8 lần so với năm 2022. Các công ty dẫn đầu bao gồm Microsoft
(đầu tư 13 tỷ USD vào OpenAI), Google, Amazon và Meta với các mô hình nền tảng mã nguồn mở như LLaMA.

Tuy nhiên, sự phát triển nhanh chóng của AI cũng đặt ra nhiều thách thức về đạo đức và quản trị.
Liên minh châu Âu đã thông qua Đạo luật AI (EU AI Act) vào năm 2024 — bộ luật toàn diện đầu tiên
trên thế giới điều chỉnh AI, phân loại rủi ro thành 4 cấp độ và cấm hoàn toàn một số ứng dụng như
hệ thống chấm điểm xã hội và nhận diện sinh trắc học thời gian thực nơi công cộng. Hoa Kỳ tiếp cận
theo hướng tự nguyện thông qua Executive Order của Tổng thống Biden năm 2023 yêu cầu các công ty AI
lớn phải báo cáo kết quả kiểm thử an toàn trước khi phát hành mô hình."""


def generate_retrieval_data(passage: str) -> dict:
    """Sinh 3 cau hoi retrieval + 2 hard negative tu doan van."""
    import json

    user_prompt = f"Đoạn văn:\n\n{passage}\n\nHãy sinh đúng 5 câu hỏi theo yêu cầu."

    import re

    raw = chat(
        prompt=user_prompt,
        system=SYSTEM_PROMPT,
        max_tokens=2048,   # du cho thinking + JSON
        temperature=0.7,
    )

    # Bo <think>...</think> neu Qwen3 tu them
    raw_clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    start = raw_clean.find("{")
    end = raw_clean.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"Khong tim thay JSON:\n{raw_clean}")
    return json.loads(raw_clean[start:end])


if __name__ == "__main__":
    if not HF_TOKEN:
        print("Thiet lap HF_TOKEN truoc: set HF_TOKEN=hf_xxx (Windows CMD)")
        exit(1)

    print("=== Sinh du lieu Retrieval + Hard Negative ===")
    print(f"Model: {MODEL_ID}\n")
    print("--- PASSAGE (~1000 tokens) ---")
    print(SAMPLE_PASSAGE)
    print("\n--- GENERATING... ---\n")

    result = generate_retrieval_data(SAMPLE_PASSAGE)

    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
