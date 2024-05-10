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

![image](https://github-production-user-asset-6210df.s3.amazonaws.com/95759699/325295167-d886ef0b-4648-4daa-9444-de7802a4dc0a.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240425%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240425T021316Z&X-Amz-Expires=300&X-Amz-Signature=afa4ce3ad544182bdb8fdee1759c38bd9b53fcaa90cf5b91c4ea06000a895186&X-Amz-SignedHeaders=host&actor_id=83150815&key_id=0&repo_id=570033267)
