import os
import requests
import m3u8
from urllib.parse import urljoin
import subprocess


def download_ts_files(m3u8_url, output_dir):
  """解析 m3u8 文件并下载 .ts 文件."""
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  response = requests.get(m3u8_url)

  if response.status_code != 200:
    print("m3u8 文件下载失败！")
    return
  
  m3u8_content = response.text

  m3u8_obj = m3u8.loads(m3u8_content)
  
  base_uri = m3u8_url.rsplit('/', 1)[0]
  
  for segment in m3u8_obj.segments:
    ts_url = urljoin(base_uri, segment.uri)
    filename = os.path.join(output_dir, os.path.basename(segment.uri))
    
    print(f"下载: {ts_url}")
    
    try:
      response = requests.get(ts_url, stream=True)
      response.raise_for_status()
      
      with open(filename, 'wb') as f:
          for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

      print(f"已保存: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"下载 {ts_url} 失败: {e}")


def concat_ts_to_mp4(ts_dir, output_mp4_file):
  """拼接 .ts 文件并转换为 MP4."""

  ts_files = sorted([
    os.path.join(ts_dir, f) for f in os.listdir(ts_dir) if f.endswith(".ts")
  ])

  if not ts_files:
    print("没有找到 ts 文件")
    return

  
  ts_concat_list = "concat:" + "|".join(ts_files)

  try:
      command = [
          "ffmpeg",
          "-i", ts_concat_list,
          "-c", "copy",  # 使用 copy 方式以避免重编码，提高效率
          output_mp4_file
      ]
      subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      print("MP4 文件转换完成！")
  except subprocess.CalledProcessError as e:
      print(f"FFmpeg 转换失败: {e}")


if __name__ == '__main__':

    m3u8_url = "https://book.englishbook2023.com/20240227/qK4aOrdn/1000kb/hls/playlist.m3u8" #请替换为实际的 m3u8 地址
    ts_output_dir = "ts_files"
    final_mp4_file = "output.mp4"

    download_ts_files(m3u8_url, ts_output_dir)
    concat_ts_to_mp4(ts_output_dir, final_mp4_file)