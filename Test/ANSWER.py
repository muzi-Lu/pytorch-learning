from Crypto.Cipher import AES
import base64
import json

key = '1234567891234567'  # 密钥需为16/24/32位字符
iv = '1234567891234567'  # 初始向量需为16位字符

# 加载gltf文件
with open('/media/benben/0ECABB60A248B50C/Test/gltf1.gltf', 'r') as f:
    gltf = json.load(f)

# 将gltf文件序列化为字符串，并转换为bytes类型的数据
gltf_str = json.dumps(gltf)
data = str(gltf_str)

iv = '1234567887654321'
key = 'miyaoxuyao16ziji'
#data = 'hello world'


# 将原始的明文用空格填充到16字节
def pad(data):
    pad_data = data
    for i in range(0, 16 - len(data)):
        pad_data = pad_data + ' '
    return pad_data


# 将明文用AES加密
def AES_en(key, data):
    # 将长度不足16字节的字符串补齐
    if len(data) < 16:
        data = pad(data)
    # 创建加密对象
    AES_obj = AES.new(key.encode("utf-8"), AES.MODE_CBC, iv.encode("utf-8"))
    # 完成加密
    AES_en_str = AES_obj.encrypt(data.encode("utf-8"))
    # 用base64编码一下
    AES_en_str = base64.b64encode(AES_en_str)
    # 最后将密文转化成字符串
    AES_en_str = AES_en_str.decode("utf-8")
    return AES_en_str


def AES_de(key, data):
    # 解密过程逆着加密过程写
    # 将密文字符串重新编码成二进制形式
    data = data.encode("utf-8")
    # 将base64的编码解开
    data = base64.b64decode(data)
    # 创建解密对象
    AES_de_obj = AES.new(key.encode("utf-8"), AES.MODE_CBC, iv.encode("utf-8"))
    # 完成解密
    AES_de_str = AES_de_obj.decrypt(data)
    # 去掉补上的空格
    AES_de_str = AES_de_str.strip()
    # 对明文解码
    AES_de_str = AES_de_str.decode("utf-8")
    return AES_de_str


data = AES_en(key, data)
print(data)
data = AES_de(key, data)
print(data)



