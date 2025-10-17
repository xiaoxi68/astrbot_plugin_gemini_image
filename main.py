from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, sp
from astrbot.api.all import *
from astrbot.core.message.components import Reply

from .utils.gemini_images_api import generate_or_edit_image_gemini
from .utils.file_send_server import send_file


@register("gemini-image", "Codex", "对接 gcli2api 的 Gemini 生图/改图并发送到 QQ", "0.3.0")
class GeminiImagePlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)

        # 仅 gcli2api 后端
        default_base = (config.get("gcli2api_base_url") or "http://127.0.0.1:7861").strip()
        default_gen = "/v1beta/models/{model}:generateContent"
        default_edit = "/v1beta/models/{model}:generateContent"
        default_stream_gen = "/v1beta/models/{model}:streamGenerateContent"

        self.api_base = default_base
        self.generation_path = (config.get("generation_path") or default_gen).strip()
        self.edit_path = (config.get("edit_path") or default_edit).strip()
        self.stream_generation_path = (config.get("stream_generation_path") or default_stream_gen).strip()
        self.append_alt_sse = bool(config.get("append_alt_sse", True))

        # 模型与重试
        self.model_name = (config.get("model_name") or "gemini-2.5-flash-image").strip()
        self.max_retry_attempts = int(config.get("max_retry_attempts", 3))
        self.use_stream = bool(config.get("use_stream", True))
        # 生成参数与系统指令（可选）
        self.temperature = config.get("temperature", 1.0)
        self.top_p = config.get("top_p", 0.95)
        self.max_output_tokens = int(config.get("max_output_tokens", 0))
        self.system_instruction = (config.get("system_instruction") or "").strip()

        # gcli2api 鉴权（默认 pwd）
        self.gcli2api_api_password = (config.get("gcli2api_api_password") or "pwd").strip()

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
            if "generation_path" in plugin_config:
                self.generation_path = str(plugin_config["generation_path"]).strip() or self.generation_path
            if "edit_path" in plugin_config:
                self.edit_path = str(plugin_config["edit_path"]).strip() or self.edit_path
            if "stream_generation_path" in plugin_config:
                self.stream_generation_path = str(plugin_config["stream_generation_path"]).strip() or self.stream_generation_path
            if "use_stream" in plugin_config:
                self.use_stream = bool(plugin_config["use_stream"]) if plugin_config["use_stream"] is not None else self.use_stream
            if "gcli2api_api_password" in plugin_config:
                self.gcli2api_api_password = str(plugin_config["gcli2api_api_password"]).strip() or self.gcli2api_api_password
            if "append_alt_sse" in plugin_config:
                self.append_alt_sse = bool(plugin_config["append_alt_sse"]) if plugin_config["append_alt_sse"] is not None else self.append_alt_sse
            if "temperature" in plugin_config:
                self.temperature = plugin_config.get("temperature", self.temperature)
            if "top_p" in plugin_config:
                self.top_p = plugin_config.get("top_p", self.top_p)
            if "max_output_tokens" in plugin_config:
                try:
                    self.max_output_tokens = int(plugin_config.get("max_output_tokens", self.max_output_tokens))
                except Exception:
                    pass
            if "system_instruction" in plugin_config:
                self.system_instruction = str(plugin_config.get("system_instruction", self.system_instruction) or "")
        except Exception as e:
            logger.error(f"加载全局配置失败: {e}")
        finally:
            self._global_config_loaded = True

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

    @llm_tool(name="gemini-image")
    async def gemini_image_tool(self, event: AstrMessageEvent, image_description: str, use_reference_images: bool = True, mode: str = "auto"):
        """
        Generate or edit images via Gemini Images API (official format).
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
        endpoint_path = self.stream_generation_path if self.use_stream else self.generation_path

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
                    append_alt_sse=self.append_alt_sse,
                    extra_generation_config={
                        "temperature": self.temperature,
                        "topP": self.top_p,
                        "maxOutputTokens": self.max_output_tokens or None,
                    },
                    system_instruction=self.system_instruction or None,
                )
                # 流式失败则回退非流式
                if not image_path:
                    from .utils.gemini_images_api import generate_or_edit_image_gemini
                    image_url, image_path = await generate_or_edit_image_gemini(
                        prompt=image_description,
                        api_keys=[self.gcli2api_api_password] if self.gcli2api_api_password else [""],
                        model=self.model_name,
                        api_base=self.api_base,
                        endpoint_path=self.generation_path,
                        input_images_b64=input_images,
                        max_retry_attempts=self.max_retry_attempts,
                        extra_generation_config={
                            "temperature": self.temperature,
                            "topP": self.top_p,
                            "maxOutputTokens": self.max_output_tokens or None,
                        },
                        system_instruction=self.system_instruction or None,
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
                    extra_generation_config={
                        "temperature": self.temperature,
                        "topP": self.top_p,
                        "maxOutputTokens": self.max_output_tokens or None,
                    },
                    system_instruction=self.system_instruction or None,
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

    @filter.command_group("gimg")
    def gimg(self):
        """Gemini 生图/改图命令组"""
        pass

    @gimg.command("gen")
    async def cmd_generate(self, event: AstrMessageEvent, *, prompt: str):
        """生成图片：/gimg gen <描述>"""
        async for res in self.gemini_image_tool(event, image_description=prompt, use_reference_images=False, mode="generate"):
            yield res

    @gimg.command("edit")
    async def cmd_edit(self, event: AstrMessageEvent, *, prompt: str):
        """根据消息中的图片改图：/gimg edit <修改描述>（需携带/引用图片）"""
        async for res in self.gemini_image_tool(event, image_description=prompt, use_reference_images=True, mode="edit"):
            yield res

    @filter.command_group("gconf")
    def gconf(self):
        """Gemini 插件快速配置命令组"""
        pass

    @gconf.command("model")
    async def switch_model(self, event: AstrMessageEvent, new_model: str = None, save_global: str = "false"):
        """切换模型：/gconf model <模型名> [true 保存为全局]"""
        await self._load_global_config()
        if not new_model:
            yield event.plain_result(f"当前模型: {self.model_name}\n示例：/gconf model gemini-2.5-flash-image")
            return
        self.model_name = new_model.strip()
        if save_global.lower() == "true":
            try:
                plugin_config = await sp.global_get("gemini-image", {})
                plugin_config["model_name"] = self.model_name
                await sp.global_put("gemini-image", plugin_config)
                yield event.plain_result(f"已将模型保存为全局：{self.model_name}")
                return
            except Exception as e:
                logger.warning(f"保存全局模型失败: {e}")
        yield event.plain_result(f"本会话已切换模型：{self.model_name}")

    @gconf.command("baseurl")
    async def switch_baseurl(self, event: AstrMessageEvent, new_base: str = None, save_global: str = "false"):
        """切换 API Base：/gconf baseurl <URL> [true 保存为全局]"""
        await self._load_global_config()
        if not new_base:
            yield event.plain_result(f"当前 API Base: {self.api_base}")
            return
        self.api_base = new_base.strip()
        if save_global.lower() == "true":
            try:
                plugin_config = await sp.global_get("gemini-image", {})
                plugin_config["gcli2api_base_url"] = self.api_base
                await sp.global_put("gemini-image", plugin_config)
                yield event.plain_result(f"已将 gcli2api Base 保存为全局：{self.api_base}")
                return
            except Exception as e:
                logger.warning(f"保存全局 API Base 失败: {e}")
        yield event.plain_result(f"本会话已切换 API Base：{self.api_base}")
