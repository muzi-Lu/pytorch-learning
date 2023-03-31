import json
from pathlib import Path
from secrets import token_bytes
import argparse

def random_key(length):
    key = token_bytes(nbytes=length)            #根据指定长度生成随机密钥
    key_int = int.from_bytes(key, 'big')        #将byte转换为int
    return key_int


def encrypt(raw):                                    #加密单元
    raw_bytes = raw.encode()                         #将字符串编码成字节串
    raw_int = int.from_bytes(raw_bytes, 'big')       #将byte转换成int
    key_int = random_key(len(raw_bytes))             #根据长度生成密钥
    return raw_int ^ key_int, key_int         #将密钥与文件异或，返回异或后的结果和密钥


def decrypt(encrypted, key_int):                             #解密单元
    decrypted = encrypted ^ key_int                          #将加密后的文件与密钥异或
    length = (decrypted.bit_length() + 7) // 8               #计算所占比特大小
    decrypted_bytes = int.to_bytes(decrypted, length, 'big') #将int转换回byte
    return decrypted_bytes.decode()


def encrypt_file(path, key_path=None, *, encoding='utf-8'):  # 参数path指定文件地址
    path = Path(path)
    cwd = path.cwd() / path.name.split('.')[0]
    path_encrypted = cwd / path.name
    if key_path is None:
        key_path = cwd / 'key'
    if not cwd.exists():
        cwd.mkdir()
        path_encrypted.touch()
        key_path.touch()
        with path.open('rt', encoding=encoding) as f1, \
                path_encrypted.open('wt', encoding=encoding) as f2, \
                key_path.open('wt', encoding=encoding) as f3:
            encrypted, key = encrypt(f1.read())
            json.dump(encrypted, f2)
            json.dump(key, f3)

def decrypt_file(path_encrypted, key_path=None, *, encoding='utf-8'):
    path_encrypted = Path(path_encrypted)
    cwd = path_encrypted.cwd()
    path_decrypted = cwd / 'decrypted'
    if not path_decrypted.exists():
        path_decrypted.mkdir()
        path_decrypted /= path_encrypted.name
        path_decrypted.touch()
    if key_path is None:
        key_path = cwd / 'key'
    with path_encrypted.open('rt', encoding=encoding) as f1, \
            key_path.open('rt', encoding=encoding) as f2, \
            path_decrypted.open('wt', encoding=encoding) as f3:
        decrypted = decrypt(json.load(f1), json.load(f2))
        f3.write(decrypted)

if __name__ == '__main__':
    encrypt_file("/media/benben/0ECABB60A248B50C/Test/gltf1.txt")
    # decrypt_file("/media/benben/0ECABB60A248B50C/Download/gltf1.gltf")