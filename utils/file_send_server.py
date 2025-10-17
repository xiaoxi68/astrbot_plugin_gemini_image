import asyncio
import os
import struct
from astrbot.api import logger

async def send_file(filename, host, port):
    reader = None
    writer = None
    try:
        reader, writer = await asyncio.open_connection(host, port)
        file_name = os.path.basename(filename)
        file_name_bytes = file_name.encode("utf-8")

        # 发送文件名长度和文件名
        writer.write(struct.pack(">I", len(file_name_bytes)))
        writer.write(file_name_bytes)

        # 发送文件大小
        file_size = os.path.getsize(filename)
        writer.write(struct.pack(">Q", file_size))

        # 发送文件内容
        await writer.drain()
        with open(filename, "rb") as f:
            while True:
                data = f.read(4096)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        logger.info(f"文件 {file_name} 发送成功")

        # 接收接收端发送的文件绝对路径
        try:
            file_abs_path_len_data = await recv_all(reader, 4)
            if not file_abs_path_len_data:
                logger.error("无法接收文件绝对路径长度")
                return None
            file_abs_path_len = struct.unpack(">I", file_abs_path_len_data)[0]

            file_abs_path_data = await recv_all(reader, file_abs_path_len)
            if not file_abs_path_data:
                logger.error("无法接收文件绝对路径")
                return None
            file_abs_path = file_abs_path_data.decode("utf-8")
            logger.info(f"接收端文件绝对路径: {file_abs_path}")
            return file_abs_path
        except (struct.error, UnicodeDecodeError) as e:
            logger.error(f"解析服务器响应失败: {e}")
            return None
        
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"网络连接失败: {e}")
        return None
    except (OSError, IOError) as e:
        logger.error(f"文件操作失败: {e}")
        return None
    except Exception as e:
        logger.error(f"传输失败: {e}")
        return None
    finally:
        # 确保资源被正确释放
        if writer:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as e:
                logger.warning(f"关闭连接时出错: {e}")

async def recv_all(reader, n):
    try:
        data = bytearray()
        while len(data) < n:
            packet = await reader.read(n - len(data))
            if not packet:
                logger.warning(f"连接意外关闭，已接收 {len(data)}/{n} 字节")
                return None
            data.extend(packet)
        return data
    except Exception as e:
        logger.error(f"接收数据时出现错误: {e}")
        return None

