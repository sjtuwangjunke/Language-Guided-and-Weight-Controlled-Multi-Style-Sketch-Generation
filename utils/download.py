#!/usr/bin/env python
import os
import time
from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import HfHubHTTPError
from requests.exceptions import ReadTimeout, ConnectionError, RequestException, Timeout

# 设置镜像端点（必须在导入 huggingface_hub 之前设置）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 设置更长的超时时间
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # 10分钟
# 设置并发下载数（可以提高下载速度）
os.environ['HF_HUB_DOWNLOAD_MAX_WORKERS'] = '4'

model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
local_dir = os.path.expanduser("~/.cache/torch/sentence_transformers/paraphrase-multilingual-MiniLM-L12-v2")

print(f"Downloading {model_id}...")
print(f"Using mirror: {os.environ['HF_ENDPOINT']}")
print(f"Target directory: {local_dir}")

# 验证镜像是否可用
try:
    api = HfApi(endpoint=os.environ['HF_ENDPOINT'])
    print(f"✓ 镜像连接成功: {os.environ['HF_ENDPOINT']}")
except Exception as e:
    print(f"⚠ 镜像连接测试失败: {e}")
    print("  将继续尝试下载...")

# 配置重试参数
max_retries = 20  # 增加重试次数
retry_delay = 5  # 秒

for attempt in range(1, max_retries + 1):
    try:
        print(f"\n尝试 {attempt}/{max_retries}...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,  # 支持断点续传
            endpoint=os.environ['HF_ENDPOINT'],  # 明确指定端点
            max_workers=4,  # 增加并发下载线程数
        )
        print("✓ Download completed!")
        break
    except (ReadTimeout, Timeout, ConnectionError, RequestException, HfHubHTTPError) as e:
        error_msg = str(e)
        # 检查是否是超时错误
        is_timeout = any(keyword in error_msg.lower() for keyword in ['timeout', 'timed out', 'read timeout'])
        
        if attempt < max_retries:
            if is_timeout:
                print(f"✗ 下载超时 (尝试 {attempt}/{max_retries})")
            else:
                print(f"✗ 连接错误 (尝试 {attempt}/{max_retries})")
            print(f"  错误信息: {error_msg[:200]}...")
            print(f"  等待 {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
            # 每次重试后增加等待时间（指数退避，但不超过60秒）
            retry_delay = min(int(retry_delay * 1.5), 60)
        else:
            print(f"✗ 下载失败: 经过 {max_retries} 次尝试后仍然失败")
            print(f"  最后错误: {error_msg}")
            raise
    except Exception as e:
        print(f"✗ 发生未知错误: {str(e)}")
        raise