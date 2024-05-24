# Cách chạy

```python
python pipeline_5.py -r 1 -b 192.168.101.234 -p 3456 -s 1 -i eth0
```

Trong đó các trường:
- `-r`: Rank của thiết bị, với r=0 là master controller
- `-b`: Địa chỉ ip của master
- `-p`: Port của master
- `-s`: Số lượng split 
- `-i`: Interface mạng hoạt động

Mô hình chạy:

![image](https://github.com/future-internet-lab/Pipeline-Parallelism-Testbed/assets/95759699/d886ef0b-4648-4daa-9444-de7802a4dc0a)
