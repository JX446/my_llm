@echo off
REM ----------------------------------------
REM 下载 seq-monkey 数据集和 BelleGroup 数据集
REM ----------------------------------------

REM 设置临时环境变量
set HF_ENDPOINT=https://hf-mirror.com

REM 数据集目录（修改为你本地路径）
set dataset_dir=C:\Users\djy\Desktop\work\USTC\jackaroo\my_llm\data

REM 如果目录不存在就创建
if not exist "%dataset_dir%" mkdir "%dataset_dir%"

REM 下载 seq-monkey 数据集（确保 modelscope 安装在当前环境）
python -m modelscope download --dataset ddzhu123/seq-monkey mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 --local_dir "%dataset_dir%"

REM 解压数据集（Windows 10+ 自带 tar）
tar -xvf "%dataset_dir%\mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2" -C "%dataset_dir%"

REM 下载 BelleGroup SFT 数据集
huggingface-cli download ^
  --repo-type dataset ^
  --resume-download ^
  BelleGroup/train_3.5M_CN ^
  --local-dir "%dataset_dir%\BelleGroup"

echo 数据集下载完成！
pause
