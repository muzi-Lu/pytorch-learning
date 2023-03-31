import json
import base64
from Crypto.Cipher import AES

# 加密gltf文件
def encrypt_gltf(gltf_file_path, key):
    # 读取gltf文件内容
    with open(gltf_file_path, 'r') as f:
        gltf_content = json.load(f)
    # 将gltf内容转换为bytes类型
    gltf_bytes = bytes(json.dumps(gltf_content), 'utf-8')
    #print(gltf_bytes)

    # 设置加密算法
    cipher = AES.new(key.encode('utf-8'), AES.MODE_EAX)
    # 加密gltf内容
    ciphertext, tag = cipher.encrypt_and_digest(gltf_bytes)
    # 将加密后的内容和tag转换为base64编码的字符串并返回
    return base64.b64encode(ciphertext).decode('utf-8') + ':' + base64.b64encode(tag).decode('utf-8')

# 解密gltf文件
def decrypt_gltf(gltf_encrypted_str, key):
    # 将加密字符串和tag还原为bytes类型
    ciphertext, tag = map(base64.b64decode, gltf_encrypted_str.split(':'))
    # 设置加密算法
    cipher = AES.new(key.encode('utf-8'), AES.MODE_EAX, nonce=tag)
    # 解密gltf内容
    gltf_bytes = cipher.decrypt(ciphertext)
    # 将bytes类型转换为gltf内容并返回
    return json.loads(gltf_bytes.decode('utf-8'))

# 使用示例
key = '1234567891234567'
gltf_file_path = '/media/benben/0ECABB60A248B50C/Test/gltf1.gltf'
# 加密gltf文件
gltf_encrypted_str = encrypt_gltf(gltf_file_path, key)
print(gltf_encrypted_str)
# with open('/media/benben/0ECABB60A248B50C/Test/gltf1_en.txt', 'w') as f:
    # f.write(gltf_encrypted_str)
# 解密gltf文件
gltf_content = decrypt_gltf(gltf_encrypted_str, key)
print(gltf_content)