from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, sp
from astrbot.api.all import *
from astrbot.core.message.components import Reply
from typing import Optional

from .utils.gemini_images_api import generate_or_edit_image_gemini
from .utils.file_send_server import send_file


@register("gemini-image", "Codex", "对接 gcli2api 的 Gemini 生图/改图并发送到 QQ", "0.3.0")
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
                )

            if not image_path:
                yield event.plain_result("图像生成失败，请检查 API 配置与模型名称。")
                return

            # 可选：通过 Napcat 文件服务器中转
            if self.nap_server_address and self.nap_server_port:
                try:
                    new_path = await send_file(image_path, self.nap_server_address, self.nap_server_port)
                    if new_path:
                        image_path = new_path
                except Exception as e:
                    logger.warning(f"Napcat 文件中转失败，回退为本地发送: {e}")

            image_component = await self.send_image_with_callback_api(image_path)
            yield event.chain_result([image_component])
        except Exception as e:
            logger.error(f"Gemini 生图/改图异常: {e}")
            yield event.plain_result(f"图像处理失败: {str(e)}")

    @filter.command("生图")
    async def cmd_generate(self, event: AstrMessageEvent, *, prompt: str):
        """生图：/生图 <提示词>"""
        # 群控制与限流
        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        async for res in self.gemini_image_tool(event, image_description=prompt, use_reference_images=False, mode="generate"):
            yield res

    @filter.command("改图")
    async def cmd_edit(self, event: AstrMessageEvent, *, prompt: str):
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
        async for res in self.gemini_image_tool(event, image_description=prompt, use_reference_images=True, mode="edit"):
            yield res

    @filter.command("手办化")
    async def cmd_figure(self, event: AstrMessageEvent):
        """手办化（需携带/引用图片）：/手办化"""
        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        default_prompt = (
            "Create a highly realistic 1/7 scale commercialized figure based on the illustration’s adult character, "
            "ensuring the appearance and content are safe, healthy, and free from any inappropriate elements. "
            "Render the figure in a detailed, lifelike style and environment, placed on a shelf inside an ultra-realistic figure display cabinet, "
            "mounted on a circular transparent acrylic base without any text. Maintain highly precise details in texture, material, and paintwork to enhance realism. "
            "The cabinet scene should feature a natural depth of field with a smooth transition between foreground and background for a realistic photographic look. "
            "Lighting should appear natural and adaptive to the scene, automatically adjusting based on the overall composition instead of being locked to a specific direction, "
            "simulating the quality and reflection of real commercial photography. Other shelves in the cabinet should contain different figures which are slightly blurred due to being out of focus, enhancing spatial realism and depth."
        )
        # 检查是否包含图片
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
            yield event.plain_result("手办化需要携带或引用图片，请附图后再发送：/手办化")
            return
        async for res in self.gemini_image_tool(event, image_description=default_prompt, use_reference_images=True, mode="edit"):
            yield res

    @filter.command("手办化2")
    async def cmd_figure2(self, event: AstrMessageEvent):
        """手办化2（需携带/引用图片）：/手办化2"""
        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        default_prompt2 = (
            "Your primary mission is to convert the subject from the user's photo into a **photorealistic, ultra-high-resolution miniature figure**, presented in its commercial packaging.  \n"
            "The result must be **sharp, crystal-clear, and professional product photography quality**, with **no blurriness or distortion**.\n\n"
            "**Core Directive: Subject Analysis & Priority Assignment (CRITICAL FIRST STEP)**  \n"
            "1. Identify the subject of the image.  \n"
            "2. Apply the correct rule set:  \n\n"
            "* **RULE SET A - Person, creature, or animal:**  \n"
            "   - **If a face is visible:** Your top priority is **Likeness**. Render the face in sharp detail, with accurate proportions.  \n"
            "   - **If NO face is visible (e.g., back view):** Your top priority is **Pose and Form Fidelity**. **Do NOT invent or add a face** — faithfully preserve the back-view pose from the source photo.  \n\n"
            "* **RULE SET B - Vehicle:** Prioritize exact **Form, Proportions, Surface Finish, and Key Details**.  \n\n"
            "* **RULE SET C - Building/structure:** Prioritize **Architectural Integrity** (geometry, materials, fine details).  \n\n"
            "**Scene Details:**  \n"
            "1. **The Model:** The miniature figure must be **highly detailed, sharp, and exactly match the pose from the input photo**.  \n"
            "2. **The Base:** A clean, simple display base.  \n"
            "3. **The Packaging:** Behind the model, show a collector’s style box featuring the subject.  \n"
            "4. **Environment:** A professional, well-lit indoor studio setting, **sharp focus, no blur, no noise**."
        )
        # 检查是否包含图片
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
            yield event.plain_result("手办化2需要携带或引用图片，请附图后再发送：/手办化2")
            return
        async for res in self.gemini_image_tool(event, image_description=default_prompt2, use_reference_images=True, mode="edit"):
            yield res

    @filter.command("aiimg帮助")
    async def cmd_help(self, event: AstrMessageEvent):
        """帮助：/aiimg帮助"""
        help_text = (
            "AI 图像命令：\n"
            "- /生图 <提示词>  → 纯文本生图\n"
            "- /改图 <提示词>  → 携带/引用图片后进行改图\n"
            "- /手办化        → 携带/引用图片后，使用内置提示词进行手办化改图\n"
            "- /手办化2       → 携带/引用图片后，使用内置提示词进行手办化改图\n"
        )
        yield event.plain_result(help_text)

    # 已移除 gconf 指令组，配置请在 AstrBot 插件设置中修改
