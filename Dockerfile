# 代码运行语言/框架
FROM python:3.8
# 将当前目录文件拷贝到镜像中/app(“.”表示当前目录)
COPY . /app
# 代码运行的工作目录
WORKDIR app/
# 代码运行环境/库
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 暴露的接口
EXPOSE 7861
# 代码运行命令
CMD ["python","test_new.py"]