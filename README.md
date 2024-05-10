# Pipeline-Parallelism

Code thực hiện chạy Pipeline Parallelism để trên môi trường mạng nhiều thiết bị khác nhau. Sử dụng với framework `PyTorch` và các thư viện Distributed đi kèm.

Mô hình dựng đang chạy với ví dụ về mô hình mạng neural DenseNet với đặc điểm các block như trên hình:
![image](https://user-images.githubusercontent.com/95759699/203720137-46eb8e12-57cb-496e-b2d2-90277321483f.png)

## Hướng dẫn cài đặt

Yêu cầu các thư viện Torch, TorchVision

- Với thiết bị chạy cuda:

```python
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```

- Với thiết bị chạy cpu:

```python
pip install torch==1.10.1 torchvision==0.11.2
```

## Mô hình thực hiện

Trong ví dụ, mô hình mạng neural được chia ra trên hai thiết bị riêng biệt.

Các thư viện python đảm bảo được cài đúng yêu cầu trong file `requirements.txt`.

Các loại mô hình được hướng dẫn trong các folder riêng