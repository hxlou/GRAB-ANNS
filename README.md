# lightCagra

一个简化版本的cagra代码，方便后续进一步修改

依赖`faiss`库，可以通过以下指令安装：

```bash
conda create -n lightCagra
conda activate lightCagra
conda install conda-forge::faiss-gpu        # 包含所有必须环境
```

构建方法

```bash
mkdir build & cd build
cmake ..
make -j10
```