from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, sp
from astrbot.api.all import *
from astrbot.core.message.components import Reply, Plain, Image
from typing import Optional
import time
import os
from pathlib import Path
import shutil
import importlib

try:
    _command_module = importlib.import_module("astrbot.core.star.filter.command")
    GreedyStr = getattr(_command_module, "GreedyStr")
except Exception:  # AstrBot 未安装时的开发环境降级
    class GreedyStr(str):
        pass

from .utils.gemini_images_api import generate_or_edit_image_gemini
from .utils.file_send_server import send_file


@register("gemini-image", "薄暝", "对接 gcli2api 的 Gemini 生图/改图并发送到 QQ", "0.4.0")
class GeminiImagePlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)

        # 仅 gcli2api 后端
        default_base = (config.get("gcli2api_base_url") or "http://127.0.0.1:7861").strip()
        # 固定端点（强制 v1beta），不再提供配置项
        self.api_base = default_base
        self._GEN_PATH = "/v1beta/models/{model}:generateContent"
        self._STREAM_GEN_PATH = "/v1beta/models/{model}:streamGenerateContent"

        # 模型与重试
        self.model_name = (config.get("model_name") or "gemini-2.5-flash-image").strip()
        self.max_retry_attempts = int(config.get("max_retry_attempts", 3))
        # 固定策略：默认启用流式，附带 alt=sse；不提供开关
        self.use_stream = True
        # 温度参数（重新加入配置）
        try:
            self.temperature = float(config.get("temperature", 1.0))
        except Exception:
            self.temperature = 1.0

        # 请求超时时间（适配大图耗时场景）
        try:
            self.request_timeout_seconds = int(config.get("request_timeout_seconds", 300))
        except Exception:
            self.request_timeout_seconds = 300

        # gcli2api 鉴权（默认 pwd）
        self.gcli2api_api_password = (config.get("gcli2api_api_password") or "pwd").strip()

        # 群控制与限流
        self.group_control_mode = (config.get("group_control_mode") or "off").strip().lower()
        self.group_list = list(config.get("group_list", []))
        try:
            self.group_rate_window_seconds = int(config.get("group_rate_window_seconds", 3600))
        except Exception:
            self.group_rate_window_seconds = 3600
        try:
            self.group_rate_max_calls = int(config.get("group_rate_max_calls", 10))
        except Exception:
            self.group_rate_max_calls = 10
        # 运行时计数：group_id -> {"window_start": float, "count": int}
        self._group_call_bucket = {}

        # Napcat 文件转发（可选）
        self.nap_server_address = config.get("nap_server_address")
        self.nap_server_port = config.get("nap_server_port")

        self._global_config_loaded = False

    async def _load_global_config(self):
        if self._global_config_loaded:
            return
        try:
            plugin_config = await sp.global_get("gemini-image", {})
            if "gcli2api_base_url" in plugin_config:
                self.api_base = str(plugin_config["gcli2api_base_url"]).strip() or self.api_base
                logger.info(f"从全局配置加载 gcli2api_base_url: {self.api_base}")
            if "model_name" in plugin_config:
                self.model_name = str(plugin_config["model_name"]).strip() or self.model_name
                logger.info(f"从全局配置加载 model_name: {self.model_name}")
            # 不再加载端点与流式相关配置项（固定策略）
            if "gcli2api_api_password" in plugin_config:
                self.gcli2api_api_password = str(plugin_config["gcli2api_api_password"]).strip() or self.gcli2api_api_password
            # 群控制
            if "group_control_mode" in plugin_config:
                self.group_control_mode = str(plugin_config.get("group_control_mode", self.group_control_mode) or "").strip().lower()
            if "group_list" in plugin_config and isinstance(plugin_config.get("group_list"), list):
                self.group_list = list(plugin_config.get("group_list", self.group_list))
            if "group_rate_window_seconds" in plugin_config:
                try:
                    self.group_rate_window_seconds = int(plugin_config.get("group_rate_window_seconds", self.group_rate_window_seconds))
                except Exception:
                    pass
            if "group_rate_max_calls" in plugin_config:
                try:
                    self.group_rate_max_calls = int(plugin_config.get("group_rate_max_calls", self.group_rate_max_calls))
                except Exception:
                    pass
            # 重新加载温度配置（其余生成参数固定不提供）
            if "temperature" in plugin_config:
                try:
                    self.temperature = float(plugin_config.get("temperature", self.temperature))
                except Exception:
                    pass
            if "request_timeout_seconds" in plugin_config:
                try:
                    self.request_timeout_seconds = int(plugin_config.get("request_timeout_seconds", self.request_timeout_seconds))
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"加载全局配置失败: {e}")
        finally:
            self._global_config_loaded = True

    def _check_group_access(self, event: AstrMessageEvent) -> Optional[str]:
        """检查群白/黑名单与限流，返回错误提示或 None 允许通过"""
        try:
            gid = None
            try:
                gid = event.get_group_id()  # 群聊返回群号，私聊返回 None
            except Exception:
                gid = None

            # 白/黑名单
            mode = self.group_control_mode
            if gid:
                if mode == "whitelist" and gid not in self.group_list:
                    return "当前群未被授权使用本插件"
                if mode == "blacklist" and gid in self.group_list:
                    return "当前群已被限制使用本插件"

                # 限流：仅对群聊生效
                import time
                now = time.time()
                b = self._group_call_bucket.get(gid, {"window_start": now, "count": 0})
                window_start = b.get("window_start", now)
                count = int(b.get("count", 0))
                if now - window_start >= self.group_rate_window_seconds:
                    window_start = now
                    count = 0
                if count >= self.group_rate_max_calls:
                    return "本群调用已达上限，请稍后再试"
                # 预占位+1（通过后真正执行业务）
                b["window_start"], b["count"] = window_start, count + 1
                self._group_call_bucket[gid] = b
            else:
                # 私聊不做名单与限流限制
                pass
        except Exception:
            # 出错不拦截
            return None
        return None

    async def send_image_with_callback_api(self, image_path: str) -> Image:
        callback_api_base = self.context.get_config().get("callback_api_base")
        if not callback_api_base:
            return Image.fromFileSystem(image_path)
        try:
            image_component = Image.fromFileSystem(image_path)
            download_url = await image_component.convert_to_web_link()
            return Image.fromURL(download_url)
        except Exception as e:
            logger.warning(f"回退本地文件发送，原因: {e}")
            return Image.fromFileSystem(image_path)

    async def gemini_image_tool(self, event: AstrMessageEvent, image_description: str, use_reference_images: bool = True, mode: str = "auto"):
        """
        Generate or edit images via gcli2api endpoints.
        If images exist in the message/reply and use_reference_images=True, will include them.
        mode: "auto" | "generate" | "edit". When "auto", edit if references provided else generate.
        """
        await self._load_global_config()
        # 尝试执行 images 目录定期清理
        await self._maybe_cleanup_images()

        # 为提示词添加图片生成目标提示，避免多模态模型返回纯文本
        image_generation_prefix = "【本次任务目标：生成图片】请根据以下描述生成一张图片，必须输出图像而非文本描述：\n"
        image_description = image_generation_prefix + image_description

        # gcli2api 模式：仅需 gcli2api_api_password（默认 pwd），无需官方 API Key

        # 收集参考图片（当前消息与引用消息）
        input_images = []
        if use_reference_images and hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message'):
            for comp in event.message_obj.message:
                if isinstance(comp, Image):
                    try:
                        base64_data = await comp.convert_to_base64()
                        input_images.append(base64_data)
                    except Exception as e:
                        logger.warning(f"参考图片转 base64 失败: {e}")
                elif isinstance(comp, Reply) and comp.chain:
                    for reply_comp in comp.chain:
                        if isinstance(reply_comp, Image):
                            try:
                                base64_data = await reply_comp.convert_to_base64()
                                input_images.append(base64_data)
                            except Exception as e:
                                logger.warning(f"引用图片转 base64 失败: {e}")

        # 模式与端点选择（流式优先），编辑与生成均走 generateContent，仅差别为是否带参考图
        endpoint_path = self._STREAM_GEN_PATH if self.use_stream else self._GEN_PATH

        # 记录开始时间
        start_time = time.time()
        
        try:
            if self.use_stream:
                from .utils.gemini_images_api import generate_or_edit_image_gemini_stream
                image_url, image_path = await generate_or_edit_image_gemini_stream(
                    prompt=image_description,
                    api_keys=[self.gcli2api_api_password] if self.gcli2api_api_password else [""],
                    model=self.model_name,
                    api_base=self.api_base,
                    endpoint_path=endpoint_path,
                    input_images_b64=input_images,
                    max_retry_attempts=self.max_retry_attempts,
                    temperature=self.temperature,
                    timeout_seconds=self.request_timeout_seconds,
                )
                # 流式失败则回退非流式
                if not image_path:
                    from .utils.gemini_images_api import generate_or_edit_image_gemini
                    image_url, image_path = await generate_or_edit_image_gemini(
                        prompt=image_description,
                        api_keys=[self.gcli2api_api_password] if self.gcli2api_api_password else [""],
                        model=self.model_name,
                        api_base=self.api_base,
                        endpoint_path=self._GEN_PATH,
                        input_images_b64=input_images,
                        max_retry_attempts=self.max_retry_attempts,
                        temperature=self.temperature,
                        timeout_seconds=self.request_timeout_seconds,
                    )
            else:
                from .utils.gemini_images_api import generate_or_edit_image_gemini
                image_url, image_path = await generate_or_edit_image_gemini(
                    prompt=image_description,
                    api_keys=[self.gcli2api_api_password] if self.gcli2api_api_password else [""],
                    model=self.model_name,
                    api_base=self.api_base,
                    endpoint_path=endpoint_path,
                    input_images_b64=input_images,
                    max_retry_attempts=self.max_retry_attempts,
                    temperature=self.temperature,
                    timeout_seconds=self.request_timeout_seconds,
                )

            if not image_path:
                yield event.plain_result("图像生成失败，请检查 API 配置与模型名称。")
                return

            # 计算耗时
            elapsed = time.time() - start_time

            # 可选：通过 Napcat 文件服务器中转
            if self.nap_server_address and self.nap_server_port:
                try:
                    new_path = await send_file(image_path, self.nap_server_address, self.nap_server_port)
                    if new_path:
                        image_path = new_path
                except Exception as e:
                    logger.warning(f"Napcat 文件中转失败，回退为本地发送: {e}")

            image_component = await self.send_image_with_callback_api(image_path)
            
            # 构建成功消息
            success_msg = f"✅ 生成成功 ({elapsed:.2f}s)"
            
            yield event.chain_result([image_component, Plain(success_msg)])
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Gemini 生图/改图异常: {e}")
            yield event.plain_result(f"❌ 生成失败 ({elapsed:.2f}s)\n原因: {str(e)}")

    async def _maybe_cleanup_images(self):
        """按配置每隔 N 天清理一次 images 目录（清空目录）。"""
        try:
            cfg = self.context.get_config() or {}
            if not bool(cfg.get("images_cleanup_enabled", True)):
                return
            days = int(cfg.get("images_cleanup_interval_days", 3))
            days = max(1, days)
            interval = days * 86400
            meta = await sp.global_get("gemini-image", {})
            last_ts = float(meta.get("images_cleanup_last_ts", 0))
            now = time.time()
            if now - last_ts < interval:
                return
            # 执行清理
            await self._cleanup_images_dir()
            meta["images_cleanup_last_ts"] = now
            await sp.global_put("gemini-image", meta)
        except Exception as e:
            logger.warning(f"images 清理调度失败: {e}")

    async def _cleanup_images_dir(self):
        try:
            images_dir = Path(__file__).parent / "images"
            if not images_dir.exists() or not images_dir.is_dir():
                return
            removed = 0
            for p in images_dir.iterdir():
                try:
                    if p.is_file():
                        p.unlink()
                        removed += 1
                    elif p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                        removed += 1
                except Exception as ie:
                    logger.debug(f"删除 {p} 失败: {ie}")
            if removed > 0:
                logger.info(f"已清理 images 目录，共删除 {removed} 项")
        except Exception as e:
            logger.warning(f"清理 images 目录失败: {e}")

    @filter.command("生图")
    async def cmd_generate(self, event: AstrMessageEvent, *, prompt: GreedyStr):
        """生图：/生图 <提示词>"""
        # 群控制与限流
        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        
        # 先返回生成中提示
        display_prompt = prompt[:20] + '...' if len(prompt) > 20 else prompt
        yield event.plain_result(f"🎨 收到请求，正在生成 [{display_prompt}]...")
        
        # 然后执行生成并发送结果
        async for res in self.gemini_image_tool(event, image_description=prompt, use_reference_images=False, mode="generate"):
            yield res

    @filter.command("改图")
    async def cmd_edit(self, event: AstrMessageEvent, *, prompt: GreedyStr):
        """改图（需携带/引用图片）：/改图 <提示词>"""
        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        # 如果未携带/引用图片，提示用户
        has_image = False
        if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message'):
            for comp in event.message_obj.message:
                if isinstance(comp, Image):
                    has_image = True
                    break
                if isinstance(comp, Reply) and comp.chain:
                    for reply_comp in comp.chain:
                        if isinstance(reply_comp, Image):
                            has_image = True
                            break
                if has_image:
                    break
        if not has_image:
            yield event.plain_result("请先携带或引用一张图片后，再使用：/改图 <提示词>")
            return
        
        # 先返回生成中提示
        display_prompt = prompt[:20] + '...' if len(prompt) > 20 else prompt
        yield event.plain_result(f"🎨 收到请求，正在生成 [{display_prompt}]...")
        
        # 然后执行生成并发送结果
        async for res in self.gemini_image_tool(event, image_description=prompt, use_reference_images=True, mode="edit"):
            yield res

    @filter.command("手办化")
    async def cmd_figure(self, event: AstrMessageEvent, prompt: GreedyStr = None):
        """手办化（需携带/引用图片）：/手办化 [描述]"""
        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        default_prompt = (
            "将画面中的角色重塑为顶级收藏级树脂手办，全身动态姿势，置于角色主题底座；"
            "高精度材质，手工涂装，肌肤纹理与服装材质真实分明。"
            "戏剧性硬光为主光源，凸显立体感，无过曝；强效补光消除死黑，细节完整可见。"
            "背景为窗边景深模糊，侧后方隐约可见产品包装盒。"
            "博物馆级摄影质感，全身细节无损，面部结构精准。"
            "禁止：任何2D元素或照搬原图、塑料感、面部模糊、五官错位、细节丢失。"
        )
        user_prompt = str(prompt) if prompt else ""
        final_prompt = f"{default_prompt}\n用户补充要求：{user_prompt}" if user_prompt.strip() else default_prompt
        
        # 检查是否包含图片
        has_image = self._check_has_image(event)
        if not has_image:
            yield event.plain_result("手办化需要携带或引用图片，请附图后再发送：/手办化")
            return
        
        # 先返回生成中提示
        yield event.plain_result("🎨 收到请求，正在生成 [手办化]...")
        
        # 然后执行生成并发送结果
        async for res in self.gemini_image_tool(event, image_description=final_prompt, use_reference_images=True, mode="edit"):
            yield res

    @filter.command("aiimg帮助")
    async def cmd_help(self, event: AstrMessageEvent):
        """帮助：/aiimg帮助"""
        help_text = (
            "🎨 AI 图像生成帮助\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📝 基础指令：\n"
            "• 生图 <提示词>  → 纯文本生图\n"
            "• 改图 <提示词>  → 携带/引用图片进行改图\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "⚡ 快速指令（携带/引用图片）：\n"
            "• 手办化   → 将角色转为收藏级树脂手办风格\n"
            "• 海报    → 生成16:9宣传海报风格\n"
            "• 壁纸    → 生成高清桌面壁纸\n"
            "• 卡片    → 生成精美卡片/名片风格\n"
            "• 手机壁纸 → 生成9:16竖版手机壁纸\n"
            "• 表情包   → 生成Q版LINE风格表情包\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "💡 提示：快速指令后可追加描述以自定义效果"
        )
        yield event.plain_result(help_text)

    @filter.command("生图帮助")
    async def cmd_help_alias(self, event: AstrMessageEvent):
        """帮助（别名）：/生图帮助"""
        async for res in self.cmd_help(event):
            yield res

    @filter.command("海报")
    async def cmd_poster(self, event: AstrMessageEvent, prompt: GreedyStr = None):
        """海报（可携带/引用图片）：/海报 [描述]"""
        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        default_prompt = (
            "将画面转换为专业电影海报风格，16:9宽屏比例。"
            "采用电影级构图和光影效果，突出主体视觉冲击力。"
            "色彩饱满鲜明，层次分明，具有商业宣传海报的精致质感。"
            "高清细节，专业排版美感，适合用作宣传展示。"
        )
        user_prompt = str(prompt) if prompt else ""
        final_prompt = f"{default_prompt}\n用户补充要求：{user_prompt}" if user_prompt.strip() else default_prompt
        
        # 检查是否包含图片
        has_image = self._check_has_image(event)
        
        # 先返回生成中提示
        yield event.plain_result("🎨 收到请求，正在生成 [海报]...")
        
        async for res in self.gemini_image_tool(event, image_description=final_prompt, use_reference_images=has_image, mode="edit" if has_image else "generate"):
            yield res

    @filter.command("壁纸")
    async def cmd_wallpaper(self, event: AstrMessageEvent, prompt: GreedyStr = None):
        """壁纸（可携带/引用图片）：/壁纸 [描述]"""
        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        default_prompt = (
            "将画面转换为高清桌面壁纸风格，16:9宽屏比例，4K超高清质量。"
            "构图优美，色彩和谐，视觉舒适。"
            "画面干净整洁，适合作为电脑桌面背景。"
            "细节丰富，光影自然，具有艺术美感和沉浸感。"
        )
        user_prompt = str(prompt) if prompt else ""
        final_prompt = f"{default_prompt}\n用户补充要求：{user_prompt}" if user_prompt.strip() else default_prompt
        
        has_image = self._check_has_image(event)
        
        yield event.plain_result("🎨 收到请求，正在生成 [壁纸]...")
        
        async for res in self.gemini_image_tool(event, image_description=final_prompt, use_reference_images=has_image, mode="edit" if has_image else "generate"):
            yield res

    @filter.command("卡片")
    async def cmd_card(self, event: AstrMessageEvent, prompt: GreedyStr = None):
        """卡片（可携带/引用图片）：/卡片 [描述]"""
        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        default_prompt = (
            "将画面转换为精美卡片风格，3:2比例。"
            "设计简洁大方，排版美观，适合用作名片、贺卡或收藏卡片。"
            "色彩搭配和谐，具有精致的印刷品质感。"
            "边框与装饰元素得当，整体风格统一协调。"
        )
        user_prompt = str(prompt) if prompt else ""
        final_prompt = f"{default_prompt}\n用户补充要求：{user_prompt}" if user_prompt.strip() else default_prompt
        
        has_image = self._check_has_image(event)
        
        yield event.plain_result("🎨 收到请求，正在生成 [卡片]...")
        
        async for res in self.gemini_image_tool(event, image_description=final_prompt, use_reference_images=has_image, mode="edit" if has_image else "generate"):
            yield res

    @filter.command("手机壁纸")
    async def cmd_phone_wallpaper(self, event: AstrMessageEvent, prompt: GreedyStr = None):
        """手机壁纸（可携带/引用图片）：/手机壁纸 [描述]"""
        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        default_prompt = (
            "将画面转换为手机壁纸风格，9:16竖版比例，2K高清质量。"
            "构图适合竖屏展示，主体位置考虑手机图标和时间显示区域。"
            "色彩鲜明但不刺眼，适合日常使用。"
            "画面简洁有层次，细节精致，具有现代感。"
        )
        user_prompt = str(prompt) if prompt else ""
        final_prompt = f"{default_prompt}\n用户补充要求：{user_prompt}" if user_prompt.strip() else default_prompt
        
        has_image = self._check_has_image(event)
        
        yield event.plain_result("🎨 收到请求，正在生成 [手机壁纸]...")
        
        async for res in self.gemini_image_tool(event, image_description=final_prompt, use_reference_images=has_image, mode="edit" if has_image else "generate"):
            yield res

    @filter.command("表情包")
    async def cmd_sticker(self, event: AstrMessageEvent, prompt: GreedyStr = None):
        """表情包（可携带/引用图片）：/表情包 [描述]"""
        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        default_prompt = (
            "将画面转换为Q版可爱表情包风格，LINE贴纸风格。"
            "角色Q版化，大头小身，表情夸张生动有趣。"
            "线条简洁流畅，色彩明快活泼。"
            "背景简单或透明，适合用作聊天表情。"
            "整体风格可爱萌系，富有表现力和感染力。"
        )
        user_prompt = str(prompt) if prompt else ""
        final_prompt = f"{default_prompt}\n用户补充要求：{user_prompt}" if user_prompt.strip() else default_prompt
        
        has_image = self._check_has_image(event)
        
        yield event.plain_result("🎨 收到请求，正在生成 [表情包]...")
        
        async for res in self.gemini_image_tool(event, image_description=final_prompt, use_reference_images=has_image, mode="edit" if has_image else "generate"):
            yield res

    def _check_has_image(self, event: AstrMessageEvent) -> bool:
        """检查消息中是否包含图片"""
        if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message'):
            for comp in event.message_obj.message:
                if isinstance(comp, Image):
                    return True
                if isinstance(comp, Reply) and comp.chain:
                    for reply_comp in comp.chain:
                        if isinstance(reply_comp, Image):
                            return True
        return False

    # 已移除 gconf 指令组，配置请在 AstrBot 插件设置中修改
